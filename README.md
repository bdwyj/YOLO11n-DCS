# YOLO11n-DCS: Dataset Splits and Evaluation Support

Reproducibility materials for YOLO11n-DCS ship detection in optical remote
sensing imagery: fixed image IDs, DOTA source-to-tile mappings, construction
scripts, file hashes and a test-set evaluation entry point.

| Dataset | Training images/tiles | Validation | Test | Valid ship boxes (train / val / test) |
|---|---:|---:|---:|---|
| DOTA ship subset | 2,108 | 603 | 301 | 39,521 / 17,622 / 6,327 |
| DIOR ship subset | 1,891 | 540 | 271 | 44,517 / 12,414 / 5,465 |

All baseline, full-model and ablation configurations use the same saved splits.
The DIOR split is the study's seed-42 custom allocation. DOTA is grouped by
source-image ID. Image IDs retain their original spelling and leading zeros.

- [Preparation instructions](docs/data_preparation.md)
- [Evaluation rules and execution](docs/evaluation.md)
- [DOTA split records](splits/dota)
- [DIOR split records](splits/dior)
- [File checksums](SHA256.json)

Original images are obtained from [DOTA](https://captain-whu.github.io/DOTA/dataset.html)
and [DIOR](https://gcheng-nwpu.github.io/). Dataset rights remain with their
providers; follow the original terms and cite the original dataset papers.
Code in this repository is distributed under AGPL-3.0; see LICENSE.

## 中文说明

本仓库提供论文实验采用的固定图像 ID、DOTA 原图与切片对应关系、子集构建脚本、
图像及标签哈希，以及测试集评价入口。DIOR 训练、验证、测试图像数分别为
1,891、540 和 271。各模型与消融配置使用同一划分。

原始图像从 DOTA、DIOR 官方来源获取。按固定清单复建时，脚本核对图像和标签
SHA-256，保持实验所用文件版本。构建方式、标签处理与评价规则见 docs/。
