# 增强 CT 胰腺导管腺癌检测模型

[English](README.md)

本仓库提供论文 *Development and external validation of a fully automated deep learning model for pancreatic ductal adenocarcinoma detection on contrast-enhanced CT* 的推理代码和训练权重。模型先定位胰腺，再进行七分类分割，综合五折模型生成 PDAC 分数和分割图。

## 安装

使用 Python 3.13，建议为本项目建立独立环境：

```bash
python -m venv .venv
```

Windows PowerShell 中运行 `.venv\Scripts\Activate.ps1` 激活环境；Linux 中运行 `source .venv/bin/activate`。随后在仓库目录执行：

```bash
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -r requirements-inference.txt
```

第一条命令安装支持 CUDA 12.8 的 PyTorch，适用于 NVIDIA 显卡。仅使用 CPU 时，只安装 `requirements-inference.txt`，推理时加上 `--device cpu`。CPU 运行较慢。依赖文件记录了参考环境的具体版本。

## 获取权重

推理需要 10 个权重文件，总计约 1.23 GB，使用 Git LFS 管理。克隆仓库后执行：

```bash
git lfs install
git lfs pull
```

保留以下目录结构，每个阶段的 `plans.json` 和 `dataset.json` 也需要一起保留：

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

`localizer` 是胰腺定位模型，`detector` 是七分类检测模型，每组各含五折权重。程序加载时会核对权重和模型配置的完整性。只有几百字节的 Git LFS 指针文件需要通过上述命令下载为实际权重。

## 运行推理

输入为一例已去标识化的三维、单通道增强 CT，格式为 `.nii` 或 `.nii.gz`。图像数值应为 HU，保留原始体素间距、方向和原点。DICOM 需要先转换成 NIfTI；后续预处理由模型完成。

```bash
python infer.py --image CT.nii.gz --output runs/case --device cuda
```

将 `CT.nii.gz` 替换成自己的图像路径，含空格的路径用引号括起。`runs/case` 是保存结果的位置，每例使用一个新目录或空目录。没有 NVIDIA 显卡时，将 `cuda` 改成 `cpu`。

一次推理依次运行五折模型。概率图会使用临时磁盘空间，运行过程中会显示当前折数和处理阶段。

## 查看输出

| 文件 | 内容 |
|---|---|
| `prediction.json` | 运行状态、PDAC 分数、判断阈值、分类结果和候选病灶 |
| `segmentation.nii.gz` | 与原始 CT 几何位置一致的七分类分割图 |
| `pancreas.nii.gz` | 胰腺实质与 PDAC 区域合并的二值掩膜 |
| `pdac_probability.nii.gz` | 五折平均后的 PDAC 概率图 |
| `pdac_likelihood.nii.gz` | 保留解剖支持区域内的 PDAC 概率图 |
| `candidates.nii.gz` | 最多五个候选区域，编号为 1–5 |

分割标签：**0** 背景，**1** PDAC，**2** 静脉，**3** 动脉，**4** 胰腺实质，**5** 胰管，**6** 胆总管。

运行完成后，`prediction.json` 中的 `status` 为 `complete`，`patient_score` 为 `pdac_likelihood.nii.gz` 的最大值。分类阈值为 **0.112**；分数大于或等于该值时，`predicted_positive` 为 `true`。

肿瘤区域对应 `segmentation.nii.gz` 中的标签 **1**，可在支持 NIfTI 的医学图像软件中与原始 CT 叠加查看。

若某一折未定位到胰腺，状态为 `localizer_failed`，分数和分类结果均为空值 `null`，命令以非零状态退出。

## 引用与许可

本模型用于科研。使用时请引用配套论文及 [NOTICE.md](NOTICE.md) 所列原方法；引用信息见 [CITATION.cff](CITATION.cff)。代码采用 [Apache 2.0](LICENSE)，权重依照 [WEIGHTS_LICENSE.txt](WEIGHTS_LICENSE.txt) 用于非商业科研。
