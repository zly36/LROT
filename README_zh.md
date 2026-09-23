# LROT — ligand–receptor-guided optimal transport

空间转录组切片对齐方法 LROT 的代码与复现材料，对应论文
**LROT: ligand–receptor-guided optimal transport for spatial transcriptomics
slice alignment**。

> **直接看 [`RUN_REPRO_zh.md`](RUN_REPRO_zh.md)** —— 里面有环境安装、数据获取、
> 逐图复现命令、预期数值与已知偏差。

> English version: [`README.md`](README.md) · [`RUN_REPRO.md`](RUN_REPRO.md)

## 快速开始

```bash
pip install -r requirements.txt
python experiments/dlpfc_layer_ratio.py     # 例：复现图S13
```

* 脚本用「向上查找 `lrot_core.py`」自定位仓库根，**解压到任意路径就能跑**，无需改任何路径；
* 随包已附图纸脚本所需的中间结果，**大多数图件不需要重跑重实验**（分钟级）；
* 需要原始数据/超大中间数组的步骤在 `RUN_REPRO_zh.md` §4 标为 **Tier 2**，含实测耗时；
* 需要 **Python 3.10**（实测 3.10.11 / Windows 11）；原始数据按 `RUN_REPRO_zh.md` §2 的公开来源
  下载到 `cancer_data/`、`dlpfc_data/`。

## 包内容速览

| 路径 | 内容 |
|---|---|
| `lrot_core.py` / `real_st_loader.py` | 核心库与数据加载（LR 强度矩阵、FGW/LROT 求解器、指标） |
| `lrot_families.py` | LR 家族划分的唯一权威来源（11 个信号家族）；导入时对 LR 库做完备性校验 |
| `lrot_paths.py` | 统一路径解析：`ROOT/OUT/FIG/NPZ` + `find_input()` 多候选查找 |
| `experiments/` | 实验脚本（结果写入 `lrot_output/`） |
| `figures/` | 绘图脚本（含 `fig_crop_title.py`：顶部总标题裁切） |
| `tools/` | `verify_env.py`（环境与输入自检）、`build_cellchat_db.py`（LR 数据库构建） |
| `lrot_output/figures_png/` | 论文与补充材料成图及零件 52 张 PNG（其中 27 张被 docx 直接内嵌） |
| `lrot_output/*.txt / *.csv / *.json` | 实验数值结果（可直接与重跑逐行比对） |
| `data/` | LR 数据库源件 |

## 命名规则

* Python 文件一律小写 `snake_case`，不带 `_v2` 这类版本号后缀；
* 脚本用小写图号（`experiments/supp/_supp_s17_plot.py` → 图S17），同一张图的 PNG 图件保留稿件标签（`lrot_supp_S17_*.png`），与补充图一一对应；
* 数值结果统一命名为 `<图或实验>_results.txt`。

## 结果可信度（实测）

* `experiments/lr_db_table.py` / `figures/gen_scaling_from_archive.py`：**101/101 个产物字节一致**
* `experiments/run_cancer_visium_truecoord.py`：数值全部复现，`.npz` 15 个数组**位级一致**
* `figures/gen_fig7_composite.py`：产物与论文内嵌图 **md5 相同**
* `experiments/dlpfc_layer_ratio.py`：一次运行即得 `3919×1442 @360 dpi`，与论文图**逐像素相同**
* 图件总览：11 张逐像素一致 + 2 张（含裁切步骤）一致；其余 16 张数值不变、版式属另一修订版

## 引用与归档

* 代码仓库：<https://github.com/zly36/LROT>
* Zenodo 归档（v1.0.3，永久 DOI）：<https://doi.org/10.5281/zenodo.22908788>
* 全部版本的概念 DOI：<https://doi.org/10.5281/zenodo.22898129>

## 许可协议

见 [`LICENSE`](LICENSE)。
