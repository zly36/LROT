# LROT 代码包 — 运行与复现说明

本包对应论文 **LROT: ligand–receptor-guided optimal transport for spatial transcriptomics
slice alignment**。目标是让审稿人在**不修改任何路径**的情况下复现论文图件与关键数值。

**归档 DOI**：<https://doi.org/10.5281/zenodo.22908788>（对应 v1.0.3 快照）；概念 DOI <https://doi.org/10.5281/zenodo.22898129> 始终指向最新版本。

> 英文版见 [`RUN_REPRO.md`](RUN_REPRO.md)。

---

## 0. 一句话说明

```
pip install -r requirements.txt
python tools/verify_env.py                     # ① 先自检：解释器/依赖/自定位/输入/最小复现
python experiments/dlpfc_layer_ratio.py        # ② 例：复现图S13
```

`tools/verify_env.py` 会依次检查：解释器与 12 个关键包版本、仓库自定位是否指向本包、
绘图所需的 Tier 1 输入是否就位，并**实跑 2 个秒级脚本做逐字节比对**；
全部通过时退出码为 0，有缺项会明确列出"缺什么、怎么补"。

* 所有脚本用「向上查找 `lrot_core.py`」自定位仓库根，**解压到任意路径都能跑**；
* 随包已附 **Tier 1** 中间结果，因此**绝大多数图件不需要重跑重实验**（分钟级）；
* 需要原始数据或超大中间数组的步骤在 §3 标为 **Tier 2**（含耗时实测）。

## 1. 目录结构

```
LROT_code/
├── README.md                 英文总览
├── README_zh.md              中文总览
├── RUN_REPRO.md              英文复现清单（命令 + 预期结果 + 已知偏差）
├── RUN_REPRO_zh.md           本文件（中文复现清单）
├── requirements.txt          Python 依赖（本机实测版本）
├── CITATION.cff              引用元数据（CFF 1.2.0）
├── LICENSE                   MIT
├── MANIFEST.sha256           全部文件校验和
├── MANIFEST_size.txt         文件清单与体积
├── lrot_core.py              核心库：LR 强度矩阵、FGW/LROT 求解器、指标
├── real_st_loader.py         真实/真实感数据加载与生成
├── lrot_families.py          LR 家族划分（唯一权威来源；导入时校验 LR 库完备性）
├── lrot_paths.py             统一路径解析（ROOT/OUT/FIG/NPZ + find_input 多候选查找）
├── experiments/              实验脚本（结果写入 lrot_output/）
│   └── supp/               图S17/S18 复现脚本、DLPFC 数据下载脚本
├── figures/                  绘图脚本（含 fig_crop_title.py）
├── tools/                    LR 数据库构建等
├── data/                     LR 数据库源（CellChatDB.mouse.rda）
├── cancer_data/  dlpfc_data/ ← 空目录，放下载的原始数据（见 §2）
└── lrot_output/
    ├── figures_png/          论文与补充材料成图及零件（52 张 PNG，其中 27 张被 docx 直接内嵌）
    ├── *.txt / *.csv / *.json 实验结果数值
    ├── 数据_npz/             Tier 1 中间数组（绘图脚本的输入）
    ├── dlpfc_donor2/        图S18 供体2 复现结果
    ├── 验证_CellChatDB重建/    LR 数据库重建与校验记录
    └── _中间产物/
        ├── 结果txt/            图9 渲染所需的 results.txt
        ├── S17_support/        图S17 所需 3 个 json
        └── _img_orig/          图S14 合成所需的原始面板
```

## 2. 环境与数据

### 2.1 Python 环境

```
python -m pip install -r requirements.txt
```

本机实测：Python 3.10.11（Windows 11 / AMD64）。

> ⚠ **不要用 Python 3.12 环境**（如 Anaconda 默认 base）。实测 `matplotlib 3.8.4`
> 会在 `figures/gen_paste2_multiseed_figure.py` 直接报
> `Axes.boxplot() got an unexpected keyword argument 'tick_labels'`，且产图与论文版不同。

三个外部基线是**独立仓库**，需按其官方说明单独安装（本包不包含其源码）：

| 基线 | 包/仓库 | 用途 |
|---|---|---|
| 官方 PASTE | `POT` + PASTE 源码（改名导入） | 图4 / 图S6 等 |
| 官方 PASTE2 | `pip install paste2==1.0.1` | 图S8 / 图S17 |
| STAligner | `pip install STAligner==1.0.0`（或其源码） | 图S6 |

### 2.2 原始数据（Tier 2 才需要）

```bash
python experiments/supp/download_dlpfc_data.py     # DLPFC（spatialLIBD / HumanPilot）
```

| 数据集 | 获取方式 | 落地目录 |
|---|---|---|
| 小鼠脑 Visium CytAssist | 10x Genomics 官网 `Visium_Mouse_Olfactory_Bulb` 等 | `cancer_data/mouse_brain.h5` |
| 人乳腺癌 Visium Block A | 10x Genomics 官网 `V1_Breast_Cancer_Block_A_Section_1/2` | `cancer_data/V1_Breast_Cancer_Block_A_Section_*.h5` |
| DLPFC（spatialLIBD） | `experiments/supp/download_dlpfc_data.py` | `dlpfc_data/` |
| GO 本体库 | `go-basic.obo`、`mgi.gaf.gz`（Gene Ontology 官方） | `data/go/` |

## 3. 复现图件（Tier 1：用随包结果，分钟级）

`lrot_output/figures_png/` 里的成图覆盖论文与补充材料全部图件，可直接用于比对：其中 27 张与 docx 内嵌图逐像素一致，
其余为组图零件（供 `figures/rebuild_*.py` 拼合）或正文图的降采样母版。
下表为**实测**状态（逐图像素级比对）：

| 图号 | 论文内嵌 | 生成脚本 | 实测状态 | 备注 |
|---|---|---|---|---|
| 图1 | image1 | ``figures/gen_method_flow.py`` | 逐像素一致 |  |
| 图2 | image2 | ``experiments/lrot_prototype.py`` | 不一致 | 脚本需 lrot_synthetic_results.txt（随包于 lrot_output/）；论文版 5220x2973 为 lrot_full_panel.png(9482x5359) 的降采样，重跑得 9486x5572 |
| 图3 | image3 | ``experiments/lrot_real_data.py`` | 不一致 | 论文版=重跑图去顶部 280 px 总标题带；重跑后仍有版式差 |
| 图4 | image4 | ``figures/gen_sota_figure.py`` | 不一致 | 重跑高 +84 px；裁切后仍有 6.2% 像素差（标注/版面不同版） |
| 图5 | image5 | ``experiments/lrot_real_data.py`` | 不一致 | dpi=380 已保留；重跑高 +52 px，裁切后仍有差 |
| 图6 | image6 | ``experiments/_render_mouse_visium_fig6.py`` | 不一致 | 重跑 5200x1631 vs 论文 5154x1546 |
| 图7 | image7 | ``figures/gen_fig7_composite.py`` | 逐像素一致 |  |
| 图8 | image42 | ``experiments/_render_breast_realcoord_official.py`` | 不一致 | 重跑得 11337x6537 母版（即 figures_png/lrot_breast_cancer_realcoord.png）；论文版 5220x2939 为其降采样 |
| 图9 | image11 | ``experiments/_render_dlpfc_fig9.py`` | 不一致 | 脚本需 lrot_dlpfc_results.txt（随包于 _中间产物/结果txt/）；论文版 4950x2840 为 figures_png/lrot_dlpfc.png(17007x9735) 的降采样 |
| 图10 | image12 | ``experiments/run_dlpfc_3d.py`` | 不一致 | dpi=240 已保留；重跑高 +127 px，裁切后仍有差 |
| 图11 | image13 | ``experiments/run_lrot_entropy_calibration.py`` | 不一致 | dpi=270 已保留；重跑高 +176 px，裁切后仍有差 |
| 图S1 | image14 | ``experiments/lrot_prototype.py`` | 不一致 | 同图2：脚本产物写入 figures_png，尺寸相同但内容有差 |
| 图S2 | image15 | ``figures/rebuild_supp_composites.py`` | 逐像素一致 |  |
| 图S3 | image37 | ``experiments/go_enrichment_fig.py`` | 逐像素一致 |  |
| 图S4 | image17 | ``figures/rebuild_supp_composites.py`` | 逐像素一致 |  |
| 图S5 | image38 | ``experiments/lr_parameter_sensitivity.py`` | 不一致 | 重跑 3600x1440 vs 论文 3600x1220（高 +220 px） |
| 图S6 | image19 | ``figures/polish_three_figures.py`` | 逐像素一致 | 中英两版同一脚本产出；中版比对磁盘 canonical 也一致 |
| 图S7 | image21 | ``figures/gen_scaling_from_archive.py`` | 逐像素一致 |  |
| 图S8 | image22 | ``figures/gen_paste2_multiseed_figure.py`` | 重跑即无标题（99.77%） | dpi=540 已保留；2026-09-21 删掉图级 ax.set_title 并加 rect=[0,0,1,0.957] 占位，重跑产物与论文版 99.77% 逐像素相同（残差为文字亚像素抗锯齿）；论文原图未改 |
| 图S9 | image23 | ``figures/polish_three_figures.py`` | 逐像素一致 |  |
| 图S10 | image43 | ``experiments/run_cancer_calibration_truecoord.py`` | 不一致 | dpi=360 已保留；重跑高 +105 px，裁切后仍有差 |
| 图S11 | image25 | ``figures/rebuild_supp_composites.py`` | 逐像素一致 |  |
| 图S12 | image35 | ``experiments/official_paste_convergence.py`` | 不一致 | dpi=360 已保留；重跑高 +104 px，裁切后仍有差 |
| 图S13 | image29 | ``experiments/dlpfc_layer_ratio.py`` | 重跑即无标题且逐像素一致 | dpi=360 已保留；suptitle 仅作布局占位，渲染后由 crop_baked_title 裁掉（expect_title=True，裁不到即报错），裁 194 px 后与论文原图逐像素完全相同 |
| 图S14 | image44 | ``figures/rebuild_s14.py`` | 逐像素一致 |  |
| 图S15 | image45 | ``figures/rebuild_supp_composites.py`` | 逐像素一致 |  |
| 图S16 | image36 | ``experiments/lr_gene_set_fairness.py`` | 不一致 | dpi=360 已保留；重跑高 +105 px，裁切后仍有 7.7% 像素差 |
| 图S17 | image40 | ``experiments/supp/_supp_s17_plot.py`` | 不一致 | 脚本需 3 个 json（随包于 lrot_output/_中间产物/S17_support/，由 experiments/supp/_supp_s17_checks.py 生成）；重跑 4604x2730 vs 论文 3936x2367 |
| 图S18 | image41 | ``experiments/supp/_supp_s18_plot.py`` | 不一致 | 重跑 4770x2789 vs 论文 4131x2410 |
| 图S19 | image46 | ``experiments/_render_dlpfc_fullscale_tissue.py`` | 本次未跑 | 全量 DLPFC(4221x4381) 求解，耗时很长；输入 lrot_data_dlpfc_fullscale.npz 存在 |

**合计：11 张逐像素一致 + 2 张「多一步顶部裁切」后一致 = 13/30 一键可复现；**
其余 16 张的**数值口径不变**，差异仅在版式/标注（属不同修订版），另有 1 张全量规模图耗时过长未跑。

### 3.1 顶部总标题裁切（已烘焙进脚本）
论文要求图内不留总标题。历史做法是「渲染后再裁」，导致单跑脚本得到的图与论文版不一致。
现已把该步骤做成 `figures/fig_crop_title.py:crop_baked_title()`，绘图脚本在 `savefig` 后调用：

* 裁切规则与当年一致：顶部 15% 内的首段墨迹判为标题带 → 找其下方首个 ≥12 行空白 → 裁到
  `max(标题下缘, 面板内容起点 − 50 px)`；
* **保留原图 dpi**（不写死 300——论文图按 360 dpi 渲染，NAR 要求线稿 ≥600 dpi、彩色 ≥300 dpi）；
* 找不到标题带会**抛错**而不是静默跳过（避免把带标题的图放进论文）。

例：`experiments/dlpfc_layer_ratio.py` 一次运行即得 `3919×1442 @360 dpi`，与论文 `image29.png` 逐像素相同。

### 3.2 净室实跑实测（2026-09-21）

把本包复制到**另一块盘、全 ASCII 路径** `D:\repro_test\LROT_code` 后，用本机解释器实跑 17 个脚本：

* **15/17 跑通**。失败 2 个均因缺少**原始数据或上游结果**（属 Tier 2，不是代码问题）：

| 脚本 | 图 | 缺什么 |
|---|---|---|
| `experiments/dlpfc_layer_ratio.py` | 图S13 | `dlpfc_data/151507_filtered_feature_bc_matrix.h5` 等 DLPFC 原始数据（§2.2 下载） |
| `experiments/go_enrichment_fig.py` | 图S3 | 需先跑 `experiments/go_enrichment.py` 并准备 `data/go/` |

* 跑完后把整个包与随包原件**逐文件 sha256 比对**：

| 结果 | 数量 | 说明 |
|---|---|---|
| 字节完全相同 | **239 / 240** | 论文版图件与结果文件在**异路径**下逐字节复现 |
| 字节不同 | 1 | `figures_png/lrot_paste2_multiseed.png`（图S8），差 0.1%，为文字抗锯齿的亚像素差异 |
| 新增 | 16 | 重跑产物写到 `lrot_output/` 根目录（与 `figures_png/` 下论文版并列，便于比对），另有 `__pycache__` |
| 缺失 | 0 | — |

> 结论：**路径问题已消除** —— 解压到任意路径无需改一行代码，且绝大多数图件达到字节级复现。

## 4. 复现实验数值（Tier 2：需原始数据，含实测耗时）
按论文顺序列出主结果脚本。`→` 后为**实测复现结果**（本机 i5-11300H / 8 线程 / 16 GB）。

| 步骤 | 命令 | 实测耗时 | 产出与核对点 |
|---|---|---|---|
| 合成数据主实验 | `python experiments/lrot_prototype.py` | 数分钟 | `lrot_output/lrot_synthetic_results.txt`、图2/图S1 |
| 真实感数据 | `python experiments/lrot_real_data.py` | 数分钟 | 图3/图5 输入 |
| 消融与统计 | `python experiments/lrot_control_experiment.py`、`experiments/pathway_significance_test.py` | 分钟级 | 表1、图S5 |
| 小鼠脑 Visium | `python experiments/run_real_visium.py` | 分钟级 | 图6 |
| 人乳腺癌 | `python experiments/run_cancer_visium_truecoord.py` | 171 s | FGW 熵 5.8465 / LROT 5.8176（−0.49%）；成本 0.1144→0.1143；有效 LR 对 10/68；9 次迭代；`lrot_data_breast_cancer_truecoord.npz`（186 MB，未随包） |
| DLPFC 主实验 | `python experiments/run_dlpfc_visium.py` | 分钟级 | 图9、图S13 |
| DLPFC 全量规模 | `python experiments/run_dlpfc_full_scale.py` | 178–194 s/对，峰值 ≈4.3 GB | 熵降 14.4%；`lrot_data_dlpfc_fullscale.npz`（141 MB，未随包） |
| DLPFC 两位供体 | `python experiments/run_dlpfc_visium.py`（供体2）/ `experiments/supp/_supp_s18_dlpfc_donor2.py` | 分钟级 | 图S18 |
| 3D 堆叠重建 | `python experiments/lrot_3d_reconstruction.py`、`experiments/lrot_3d_multi_seed.py` | 分钟级 | 图7 |
| 消融：权重/预处理 | `python experiments/lr_weight_sensitivity.py`、`experiments/lr_parameter_sensitivity.py`、`experiments/preproc_sensitivity.py` | 分钟级 | 图S5、图S11、补充表S15 |
| 基线对比 | `python experiments/run_dlpfc_baselines.py`、`experiments/official_paste_convergence.py`、`experiments/lr_cellchat_comparison.py` | 分钟级（需 §2.1 外部基线） | 图4、图S2、图S6、图S12 |
| 富集与通路 | `python experiments/go_enrichment.py`、`experiments/pathway_activity_dlpfc.py` | 分钟级（需 `data/go/`） | 图S3、图S4 |
| LR 数据库表 | `python experiments/lr_db_table.py` | 秒级 | 补充表S11–S13；**实测 101/101 个产物字节一致** |
| 运行时间/规模 | `python experiments/run_runtime_benchmark.py`、`figures/gen_scaling_from_archive.py` | 分钟级 | 图S6、图S7 |

> 说明：`lrot_output/` 内的 `*_results.txt` 已随包，可直接与重跑结果**逐行比对**，
> 不需要先跑重实验。

## 5. 已知偏差（诚实清单）

1. **墙上时钟**：`lrot_output/lrot_cancer_truecoord_results.txt` 里记录的耗时（43.7 s → 68.8 s 等）
   因机器负载而异，**不属于可复现量**；其余数值（熵、成本、有效对数、迭代数）逐位一致。
2. **16/30 图版式不同**：重跑得到的是同一数值口径下的**另一修订版**，与论文版差异为
   字体/边距/标注位置，不是数值差异（详见 §3 表「实测状态」列）。
3. **论文版可能为降采样母版**：图2/图8/图9 的论文版是在超大幅母版上降采样得到的，
   直接重跑会得到更大的原始尺寸。
4. **dpi 差异已修**：早期脚本的顶部裁切步骤会把 360 dpi 静默降成 300 dpi；现已改为沿用原图 dpi。

## 6. 自检方法

```bash
# 校验包完整性
python -c "import hashlib,pathlib;[print(hashlib.sha256(p.read_bytes()).hexdigest(), p.as_posix()) for p in pathlib.Path('.').rglob('*') if p.is_file()]" > /tmp/now.txt
# 与 MANIFEST.sha256 比对（顺序无关，逐行 diff 即可）

# 图件与论文比对：lrot_output/figures_png/*.png 应与 docx 内嵌 media 逐像素相同
```

**行尾**：本包以 `* text=auto eol=lf` 存储，并在 `.gitattributes` 中显式标记二进制类型，
因此无论平台与 `core.autocrlf` 设置如何，`git clone` 检出的字节都与 `MANIFEST.sha256` 一致。

**数值核对**：任一 `experiments/*.py` 重跑后，其 `lrot_output/*_results.txt` 应与随包同名文件
在数值上一致（时间字段除外）。

## 7. 常见问题

**Q1：编辑器里脚本有红色波浪线，但实际能跑通？**

那是 **Pylance 静态分析**的提示，不是运行错误。常见原因：

* `experiments/`、`figures/` 下的脚本在运行时用 `sys.path.insert` 动态加入仓库根，
  静态分析看不到，于是报 `Import "lrot_3d_multi_seed" could not be resolved`；
* 编辑器选中的解释器不是你装依赖的那个（常见于目录里残留一个空的 `.venv`）。

处理：① 命令面板 `Python: Select Interpreter` 选对解释器；② 若还有残留，执行
`Developer: Reload Window`（或 Pylance 的 `Restart Language Server`）刷新缓存。
**别用波浪线判断环境是否正常**——用 `python tools/verify_env.py`，它靠实跑得出结论。

**Q2：脚本跑了一半只有数值表格，没有图？**

`experiments/*.py` 是实验脚本：先算数值、打印表格，**最后一步**才画图并存盘；
`figures/*.py` 才是秒级直接出图。图的落盘路径脚本会打印（`图已保存: ...`）。

**Q3：终端里为什么看不到图片？**

终端只能输出文本。PNG 写在磁盘上，用图片查看器打开；
`lrot_output/figures_png/` 里已经放了论文版成图，无需重跑即可查看。

## 8. 许可协议

见 `LICENSE`。
