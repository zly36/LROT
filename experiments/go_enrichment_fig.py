# -*- coding: utf-8 -*-
"""go_enrichment_fig.py — 重绘 图S3（数据取自 lrot_go_enrichment_data.pkl，排版沿用既有格式）

排版参数：figsize=(16,7)@300dpi；
近条形图字号 7/9/12/13；条端标注 f'{n}*  [{-log10(padj):.0f}]'；
colorbar RdYlGn + p=0.05 阈值虚线；右图 Set3 饼图 + 'Family (n)' 图例。
术语筛选（与正文 §2.7 一致）：前景命中≥5、padj<1e-5、背景占比≤2%，按 padj 取前 15。

输出：lrot_output/figures_png/lrot_go_enrichment.png
用法：python -X utf8 experiments/go_enrichment_fig.py
"""
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
import matplotlib as mpl            # noqa: E402
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


sys.path.insert(0, _R)
from lrot_core import ALL_LR_GENES                            # noqa: E402
from lrot_families import LR_FAMILY_GROUPS, group_gene_counts  # noqa: E402

OUT = os.path.join(_R, 'lrot_output')
PNG = os.path.join(OUT, 'figures_png', 'lrot_go_enrichment.png')

d = pickle.load(open(os.path.join(OUT, 'lrot_go_enrichment_data.pkl'), 'rb'))
tests, n_bg, n_fg = d['tests'], d['n_bg'], d['n_fg']

# ---------- (A) 术语筛选 ----------
# 宽度上限 2%：排除覆盖全基因组 1/50 以上的上位泛术语。
# 注：放宽到 5% 会让 Top-15 被 "regulation of …"/"tube morphogenesis" 这类上位术语占据，
#     血管形态发生、轴突导向、趋化性等具体过程会被挤出，故取 2%。
BG_FRAC_CAP = 0.02
sel = [r for r in tests if r['a'] >= 5 and r['padj'] < 1e-5
       and r['c'] <= n_bg * BG_FRAC_CAP]
sel.sort(key=lambda r: r['padj'])
top = sel[:15]
print('候选术语 %d 个（背景占比上限 %.0f%%）；取前 15'
      % (len(sel), BG_FRAC_CAP * 100))
for r in top:
    print('   %-58s fg=%3d bg=%4d padj=%.1e'
          % (r['term'][:58], r['a'], r['c'], r['padj']))

terms = [r['term'][:45] for r in top]
counts = [r['a'] for r in top]
p_adj = [r['padj'] for r in top]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

ax1 = axes[0]
log_p = -np.log10(np.maximum(np.array(p_adj, dtype=float), 1e-300))
max_log = np.max(log_p)
min_log = np.min(log_p)
norm_log = (log_p - min_log) / max(max_log - min_log, 1e-10)   # 0=红, 1=绿
colors = matplotlib.colormaps['RdYlGn'](norm_log)

bars = ax1.barh(range(len(terms)), counts, color=colors,
                edgecolor='gray', linewidth=0.5)
ax1.set_yticks(range(len(terms)))
ax1.set_yticklabels(terms, fontsize=9)
ax1.set_xlabel('Number of LR Genes', fontsize=12)
ax1.set_title('(A) GO biological-process enrichment', fontsize=13,
              fontweight='bold')
ax1.invert_yaxis()

# 条形末端标注：基因数 * [−log10(padj)]
for bar, count, p, lp in zip(bars, counts, p_adj, log_p):
    sig_mark = '*' if p < 0.05 else ''
    ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
             f'{count}{sig_mark}  [{lp:.0f}]', va='center', fontsize=7)

# 颜色条 + 显著性阈值标注
# 注：v2 数据的 padj 全部远小于 0.05，阈值线落在色标范围之外（与旧图同样的情形），
#     故画在色标底端并紧贴其下方标注，视觉与旧图一致，且不会撑大保存区。
norm_cb = mpl.colors.Normalize(vmin=min_log, vmax=max_log)
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm_cb)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, orientation='vertical', shrink=0.7, pad=0.02)
cbar.set_label('-log10(adjusted p-value)', fontsize=9)
thr = -np.log10(0.05)
if min_log <= thr <= max_log:
    frac = (thr - min_log) / (max_log - min_log)
    cbar.ax.axhline(y=frac, color='gray', linestyle='--', linewidth=0.8)
    cbar.ax.text(0.5, frac - 0.01, 'p=0.05', transform=cbar.ax.transAxes,
                 fontsize=7, color='gray', ha='center', va='top')
else:
    cbar.ax.axhline(y=0.0, color='gray', linestyle='--', linewidth=0.8)
    cbar.ax.text(0.5, -0.02, 'p=0.05', transform=cbar.ax.transAxes,
                 fontsize=7, color='gray', ha='center', va='top')

# --- (B) 通路家族饼图 ---
# 家族划分的唯一权威来源是 lrot_families.py（导入时已对 LR_DB 做完备性校验：
# 61 对恰好分属 11 个家族、基因并集 == ALL_LR_GENES == 109）。此处不再硬编码
# 家族成员表，避免与正文“11 个信号家族”的口径脱节。
cat_names = list(LR_FAMILY_GROUPS)              # 11 个家族归并后的 7 类
cat_sizes = list(group_gene_counts().values())
print('\n饼图家族计数（11 个家族归并为 7 类）：\n   %s\n   合计 %d'
      % ('\n   '.join('%s = %d' % (n, s)
                      for n, s in zip(cat_names, cat_sizes)),
         sum(cat_sizes)))
assert sum(cat_sizes) == len(ALL_LR_GENES), (sum(cat_sizes),
                                             len(ALL_LR_GENES))

ax2 = axes[1]
colors_pie = matplotlib.colormaps['Set3'](np.linspace(0, 1, len(cat_names)))
wedges, texts, autotexts = ax2.pie(cat_sizes, labels=None, autopct='%1.0f%%',
                                   colors=colors_pie, startangle=90,
                                   textprops={'fontsize': 8})
ax2.set_title('(B) LR gene pathway distribution', fontsize=13, fontweight='bold')

legend_labels = [f'{name} ({size})' for name, size in zip(cat_names, cat_sizes)]
ax2.legend(wedges, legend_labels, title='Pathway Families',
           loc='center left', bbox_to_anchor=(-0.4, 0.5),
           fontsize=7, title_fontsize=8)

plt.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs(os.path.dirname(PNG), exist_ok=True)
fig.savefig(PNG, dpi=300, bbox_inches='tight')
plt.close()
from PIL import Image    # noqa: E402
print('\nSaved: %s  %s' % (PNG, Image.open(PNG).size))
