from __future__ import annotations
import gc
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from .core import add_roi_probabilities, extract_official_candidates, lowres_reference, masked_pdac_heatmap, predicted_roi, resample
from .io import atomic_json, read_image, write_image
from .model import load_predictor, verify_bundle

class FrozenPredictor:

    def __init__(self, manifest: Path, *, device: str='cuda'):
        if device not in ('cuda', 'cpu'):
            raise ValueError('device must be cuda or cpu')
        manifest = manifest.resolve()
        self.root = manifest.parent.parent
        self.bundle = verify_bundle(manifest)
        self.threshold = float(self.bundle['threshold'])
        self.device = device

    def _predict(self, image: sitk.Image, model: dict, probabilities: bool):
        import torch
        predictor = load_predictor(self.root, model, self.device)
        data = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)[None]
        try:
            return predictor.predict_single_npy_array(data, {'spacing': tuple(image.GetSpacing()[::-1])}, save_or_return_probabilities=probabilities)
        finally:
            del predictor
            gc.collect()
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def predict_case(self, image_path: Path, output: Path) -> dict:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError('Choose a new or empty prediction directory.')
        started = time.perf_counter()
        image = read_image(image_path)
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError('Inference requires one three-dimensional, single-channel CT volume')
        if not np.isfinite(sitk.GetArrayViewFromImage(image)).all():
            raise ValueError('CT values must be finite')
        output.mkdir(parents=True, exist_ok=True)
        low_spacing, low_size = lowres_reference(image)
        low = resample(image, low_spacing, low_size, False)
        with tempfile.TemporaryDirectory(prefix='pdac-') as scratch:
            shape = tuple(image.GetSize()[::-1])
            mean = np.memmap(Path(scratch) / 'probabilities.dat', dtype=np.float32, mode='w+', shape=(7, *shape))
            try:
                mean[:] = 0
                mean[0] = 1
                for model in self.bundle['models']:
                    fold = model['fold']
                    print(f'Fold {fold + 1}/5: localizing pancreas', flush=True)
                    localizer = np.asarray(self._predict(low, model['localizer'], False), dtype=np.uint8)
                    if localizer.shape != tuple(low.GetSize()[::-1]) or set(np.unique(localizer)) - {0, 1}:
                        raise RuntimeError('Localizer output shape or labels are invalid')
                    mask = sitk.GetImageFromArray(localizer)
                    mask.CopyInformation(low)
                    if not localizer.any():
                        result = {
                            'status': 'localizer_failed',
                            'failed_folds': [fold],
                            'patient_score': None,
                            'threshold': self.threshold,
                            'predicted_positive': None,
                            'failure_reason': 'The pancreas localizer returned an empty mask.',
                            'elapsed_seconds': time.perf_counter() - started,
                        }
                        atomic_json(output / 'prediction.json', result)
                        return result
                    start, size = predicted_roi(image, mask)
                    roi = sitk.RegionOfInterest(image, size, start)
                    print(f'Fold {fold + 1}/5: detecting PDAC', flush=True)
                    segmentation, probabilities = self._predict(roi, model['detector'], True)
                    if np.asarray(segmentation).shape != tuple(size[::-1]):
                        raise RuntimeError('Detector output does not match native ROI geometry')
                    add_roi_probabilities(mean, np.asarray(probabilities, dtype=np.float32), start, 5)
                    del probabilities, segmentation, roi, localizer, mask
                segmentation = np.empty(shape, dtype=np.uint8)
                for start_z in range(0, shape[0], 16):
                    segmentation[start_z:start_z + 16] = np.argmax(mean[:, start_z:start_z + 16], axis=0).astype(np.uint8)
                raw_pdac = np.asarray(mean[1]).copy()
            finally:
                mean._mmap.close()
                del mean
        heatmap = masked_pdac_heatmap(segmentation, raw_pdac)
        candidate_map, candidates = extract_official_candidates(heatmap)
        voxel_ml = float(np.prod(image.GetSpacing()) / 1000)
        candidates = [{**asdict(row), 'volume_ml': float(row.voxel_count * voxel_ml)} for row in candidates]
        write_image(segmentation, image, output / 'segmentation.nii.gz')
        write_image(np.isin(segmentation, (1, 4)).astype(np.uint8), image, output / 'pancreas.nii.gz')
        write_image(raw_pdac, image, output / 'pdac_probability.nii.gz')
        write_image(heatmap, image, output / 'pdac_likelihood.nii.gz')
        write_image(candidate_map.astype(np.uint8), image, output / 'candidates.nii.gz')
        score = float(heatmap.max(initial=0))
        result = {
            'status': 'complete',
            'patient_score': score,
            'threshold': self.threshold,
            'predicted_positive': score >= self.threshold,
            'candidates': candidates,
            'candidate_count': len(candidates),
            'elapsed_seconds': time.perf_counter() - started,
        }
        atomic_json(output / 'prediction.json', result)
        return result
