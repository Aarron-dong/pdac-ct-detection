# PDAC detection on contrast-enhanced CT

[中文说明](README.zh-CN.md)

Inference code and trained weights for *Development and external validation of a fully automated deep learning model for pancreatic ductal adenocarcinoma detection on contrast-enhanced CT*. The model localizes the pancreas, predicts seven anatomical classes and combines five folds to produce a PDAC score and segmentation maps.

## Installation

Use Python 3.13 in a separate environment. Run these commands from the repository directory with the environment activated:

```bash
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -r requirements-inference.txt
```

The first command installs PyTorch with CUDA 12.8 support for NVIDIA GPUs. For CPU inference, install only `requirements-inference.txt` and select `--device cpu`. CPU inference is slower. The requirements files specify the reference dependency versions.

## Weights

The ten inference checkpoints occupy about 1.23 GB and use Git LFS. After cloning the repository, run:

```bash
git lfs install
git lfs pull
```

Keep this directory structure; each stage also requires its `plans.json` and `dataset.json`:

```text
weights/
  frozen_bundle.json
  localizer/
    plans.json
    dataset.json
    fold_0/checkpoint.pth
    ...
    fold_4/checkpoint.pth
  detector/
    plans.json
    dataset.json
    fold_0/checkpoint.pth
    ...
    fold_4/checkpoint.pth
```

The program checks the checkpoints and model configurations when loading them. A small Git LFS pointer file must be replaced by the downloaded checkpoint before inference.

## Inference

Provide one de-identified, three-dimensional, single-channel contrast-enhanced CT volume in NIfTI format (`.nii` or `.nii.gz`). Intensities must be in Hounsfield units, with the original spacing, orientation and origin preserved. Convert DICOM to NIfTI before running this command; preprocessing is performed by the model.

```bash
python infer.py --image CT.nii.gz --output runs/case --device cuda
```

Use a new or empty output directory for each examination. `--device cpu` runs the same model on the CPU. Inference processes all five folds and uses temporary disk space for probability maps.

## Outputs

| File | Contents |
|---|---|
| `prediction.json` | Status, PDAC score, decision threshold, classification and lesion candidates |
| `segmentation.nii.gz` | Seven-class segmentation in the original CT geometry |
| `pancreas.nii.gz` | Binary mask of pancreatic parenchyma and PDAC |
| `pdac_probability.nii.gz` | Averaged PDAC probability map |
| `pdac_likelihood.nii.gz` | PDAC probability within the anatomical support region |
| `candidates.nii.gz` | Up to five candidate regions, labeled 1–5 |

Segmentation labels: **0** background, **1** PDAC, **2** veins, **3** arteries, **4** pancreatic parenchyma, **5** pancreatic duct, **6** common bile duct.

After successful inference, `status` is `complete` and `patient_score` is the maximum value in `pdac_likelihood.nii.gz`. The classification threshold is **0.112**; `predicted_positive` is `true` when the score is at or above this value.

The tumor region is label **1** in `segmentation.nii.gz`. Overlay this segmentation on the original CT in a NIfTI-compatible medical image viewer.

If a pancreas localizer returns an empty mask, `prediction.json` reports `localizer_failed`, the score and classification are `null`, and the command exits with a nonzero status. Use `status == "complete"` before interpreting predictions.

## Citation and license

For research use. Cite the accompanying manuscript and the upstream methods in [NOTICE.md](NOTICE.md); citation metadata are in [CITATION.cff](CITATION.cff). Code uses [Apache 2.0](LICENSE). Trained weights are available for noncommercial research under [WEIGHTS_LICENSE.txt](WEIGHTS_LICENSE.txt).
