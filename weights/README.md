# Model weights

`localizer/fold_0` through `fold_4` contain five pancreas localizers. `detector/fold_0` through `fold_4` contain five seven-class detectors. Together, the ten checkpoints occupy about 1.23 GB. Each stage requires its `plans.json` and `dataset.json`.

Run `git lfs pull` after cloning to download the checkpoints. `frozen_bundle.json` records their locations, checksums and decision threshold. Keep the `weights/` directory beside `infer.py` and `pdac_pipeline/`.

Usage: [English](../README.md) · [中文](../README.zh-CN.md).
