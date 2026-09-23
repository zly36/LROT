# lrot_output/ — 实验结果与成图

本目录是**实验脚本的输出目录**，也是**绘图脚本的输入目录**。

| 子目录/文件 | 内容 |
|---|---|
| `figures_png/` | 论文与补充材料成图及零件（52 张 PNG，其中 27 张被 docx 直接内嵌） |
| `*.txt` | 各实验的数值结果（可直接与重跑结果逐行比对） |
| `*.csv` | 表格类结果（补充表S11–S13 等） |
| `数据_npz/` | 绘图脚本所需的中间数组（Tier 1 已随包） |
| `_中间产物/结果txt/` | 图9 渲染所需的 `*_results.txt` |
| `dlpfc_donor2/` | 图S18 供体 2 复现结果（json + txt） |
| `_中间产物/S17_support/` | 图S17 所需的 3 个 json |
| `_中间产物/_img_orig/` | 图S14 合成所需的原始面板 |
| `验证_CellChatDB重建/` | LR 数据库重建与校验记录 |

未随包（>60 MB，重跑生成）：`lrot_data_breast_cancer_truecoord.npz`（186 MB）、
`lrot_data_dlpfc_fullscale.npz`（141 MB）——生成命令见 `RUN_REPRO.md` §4。
