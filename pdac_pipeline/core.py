from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
DILATION_RADIUS_XYZ = (5, 5, 1)
MARGINS_XYZ = (80, 50, 10)
OFFICIAL_RELATIVE_PEAK_THRESHOLD = 0.4
OFFICIAL_MAX_CANDIDATES = 5
CONNECTIVITY_26 = ndimage.generate_binary_structure(3, 3)

def lowres_reference(image: sitk.Image) -> tuple[tuple[float, ...], tuple[int, ...]]:
    size = image.GetSize()
    spacing = image.GetSpacing()
    scale = max(float(size[0]) / 256.0, 1.0)
    output_spacing = tuple((float(value) * scale for value in spacing))
    output_size = tuple((max(1, round(size[index] * spacing[index] / output_spacing[index])) for index in range(3)))
    return (output_spacing, output_size)

def resample(image: sitk.Image, spacing: tuple[float, ...], size: tuple[int, ...], is_label: bool) -> sitk.Image:
    filter_ = sitk.ResampleImageFilter()
    filter_.SetOutputSpacing(spacing)
    filter_.SetSize(size)
    filter_.SetOutputDirection(image.GetDirection())
    filter_.SetOutputOrigin(image.GetOrigin())
    filter_.SetTransform(sitk.Transform())
    filter_.SetDefaultPixelValue(0 if is_label else -1024.0)
    filter_.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return filter_.Execute(image)

def largest_contiguous_run(values: np.ndarray) -> tuple[int, int]:
    values = np.unique(values.astype(int))
    if len(values) == 0:
        raise RuntimeError('empty_stage1_prediction')
    breaks = np.where(np.diff(values) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(values) - 1]
    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    return (int(values[starts[best]]), int(values[ends[best]]))

def predicted_roi(full_image: sitk.Image, lowres_mask: sitk.Image) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    dilated = sitk.BinaryDilate(lowres_mask > 0, DILATION_RADIUS_XYZ)
    nonzero_zyx = np.nonzero(sitk.GetArrayViewFromImage(dilated))
    runs_zyx = [largest_contiguous_run(values) for values in nonzero_zyx]
    start_xyz = (runs_zyx[2][0], runs_zyx[1][0], runs_zyx[0][0])
    finish_xyz = (runs_zyx[2][1], runs_zyx[1][1], runs_zyx[0][1])
    physical_start = lowres_mask.TransformIndexToPhysicalPoint(start_xyz)
    physical_finish = lowres_mask.TransformIndexToPhysicalPoint(finish_xyz)
    full_start = full_image.TransformPhysicalPointToIndex(physical_start)
    full_finish = full_image.TransformPhysicalPointToIndex(physical_finish)
    full_size = full_image.GetSize()
    roi_start = tuple((max(0, min(full_start[axis], full_finish[axis]) - MARGINS_XYZ[axis]) for axis in range(3)))
    roi_finish = tuple((min(full_size[axis], max(full_start[axis], full_finish[axis]) + MARGINS_XYZ[axis]) for axis in range(3)))
    roi_size = tuple((roi_finish[axis] - roi_start[axis] for axis in range(3)))
    if any((value <= 0 for value in roi_size)):
        raise RuntimeError(f'invalid_predicted_roi:{roi_start}:{roi_finish}')
    return (roi_start, roi_size)

def masked_pdac_heatmap(segmentation: np.ndarray, pdac_probability: np.ndarray) -> np.ndarray:
    anatomical = np.isin(segmentation, (1, 4, 5)).astype(np.uint8)
    anatomical_image = sitk.GetImageFromArray(anatomical)
    dilated = sitk.BinaryDilate(anatomical_image, DILATION_RADIUS_XYZ)
    mask = sitk.GetArrayViewFromImage(dilated) > 0
    return np.where(mask, pdac_probability, 0.0).astype(np.float32, copy=False)

def add_roi_probabilities(mean: np.ndarray, probabilities: np.ndarray, start_xyz: tuple[int, int, int], folds: int) -> None:
    if folds < 1 or probabilities.ndim != 4 or probabilities.shape[0] != mean.shape[0]:
        raise ValueError('Invalid probability map shape or ensemble size')
    if not np.isfinite(probabilities).all() or probabilities.min() < -1e-05 or probabilities.max() > 1 + 1e-05:
        raise ValueError('Invalid class probabilities')
    if not np.allclose(probabilities.sum(axis=0), 1, atol=0.002, rtol=0):
        raise ValueError('Class probabilities do not sum to one')
    start = tuple((int(v) for v in start_xyz[::-1]))
    stop = tuple((a + b for a, b in zip(start, probabilities.shape[1:])))
    if any((a < 0 or b > n for a, b, n in zip(start, stop, mean.shape[1:]))):
        raise ValueError('ROI lies outside original image geometry')
    region = tuple((slice(a, b) for a, b in zip(start, stop)))
    mean[0, *region] -= 1 / folds
    mean[slice(None), *region] += probabilities / folds

@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    score: float
    voxel_count: int

def extract_official_candidates(heatmap: np.ndarray, *, relative_peak_threshold: float=OFFICIAL_RELATIVE_PEAK_THRESHOLD, max_candidates: int=OFFICIAL_MAX_CANDIDATES) -> tuple[np.ndarray, list[Candidate]]:
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f'Expected a 3-D heatmap, got shape {values.shape}')
    if not 0 < relative_peak_threshold <= 1:
        raise ValueError('relative_peak_threshold must be in (0, 1]')
    if max_candidates < 1:
        raise ValueError('max_candidates must be positive')
    work = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).copy()
    np.maximum(work, 0.0, out=work)
    candidate_map = np.zeros(work.shape, dtype=np.uint8)
    candidates: list[Candidate] = []
    for candidate_id in range(1, max_candidates + 1):
        flat_index = int(np.argmax(work))
        peak = float(work.flat[flat_index])
        if peak <= 0:
            break
        seed = np.unravel_index(flat_index, work.shape)
        thresholded = work >= relative_peak_threshold * peak
        components, _ = ndimage.label(thresholded, structure=CONNECTIVITY_26)
        component_id = int(components[seed])
        if component_id == 0:
            raise RuntimeError('Peak voxel was not assigned to a candidate component')
        component = components == component_id
        voxel_count = int(component.sum())
        candidate_map[component] = candidate_id
        candidates.append(Candidate(candidate_id, peak, voxel_count))
        work[component] = 0.0
    return (candidate_map, candidates)
