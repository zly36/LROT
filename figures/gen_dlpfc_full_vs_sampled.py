# -*- coding: utf-8 -*-
"""
gen_dlpfc_full_vs_sampled.py — 图S10: DLPFC 全量 vs 1,000-spot 采样验证对比
数据来源: lrot_output/dlpfc_full_scale_results.txt (全量) 与 lrot_output/dlpfc_baselines_results.txt (采样对1)
面板: (a)传输熵 (b)同层质量 (c)运行时间(对数)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "lrot_output")
SAVE = os.path.join(OUT_DIR, "figures_png", "lrot_dlpfc_full_vs_sampled.png")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

FGW_C = '#E68A8A'    # 红 - FGW
LROT_C = '#2E8B57'   # 绿 - LROT

# (FGW, LROT) —— 采样(对1) 与 全量(对1)
ent = {'s': (6.0119, 4.9524), 'f': (7.4938, 6.4135)}      # 传输熵
same = {'s': (0.3766, 0.3742), 'f': (0.3882, 0.3886)}     # 同层质量
# 运行时间(s)：采样(对1) 取 dlpfc_baselines_results.txt 的 FGW 10.2 / LROT 18.5；
#              全量(对1) 取 dlpfc_full_scale_results.txt 的 FGW 307.5 / LROT 225.4
times = {'s': (10.2, 18.5), 'f': (307.5, 225.4)}

fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.3))
groups = ['Sampled\n(1,000 spots)', 'Full-scale\n(4,221/4,381)']


def grouped(ax, data, ylab, fmt='{:.2f}', log=False, annot=None, ymax_scale=1.22, title=''):
    x = np.arange(2)
    w = 0.34
    fgw = [data[k][0] for k in ('s', 'f')]
    lrot = [data[k][1] for k in ('s', 'f')]
    b1 = ax.bar(x - w / 2, fgw, w, label='FGW (γ=0)', color=FGW_C,
                alpha=0.85, edgecolor='#333333', lw=0.6)
    b2 = ax.bar(x + w / 2, lrot, w, label='LROT (γ=0.1)', color=LROT_C,
                alpha=0.85, edgecolor='#333333', lw=0.6)
    for xi, (fv, lv) in enumerate(zip(fgw, lrot)):
        ax.text(xi - w / 2, fv * 1.04, fmt.format(fv), ha='center', va='bottom', fontsize=8.5)
        ax.text(xi + w / 2, lv * 1.04, fmt.format(lv), ha='center', va='bottom', fontsize=8.5)
    if annot:
        for xi, txt in annot.items():
            top = max(fgw[xi], lrot[xi])
            ax.text(xi, top * 1.13, txt, ha='center', fontsize=9, color='#B22222', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel(ylab, fontsize=10)
    ax.set_title(title, fontsize=10)
    if log:
        ax.set_yscale('log')
        ax.set_ylim(1, 500)
    else:
        vmax = max(fgw + lrot)
        ax.set_ylim(0, vmax * ymax_scale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return b1, b2


# (a) 传输熵 + 熵降标注
grouped(axes[0], ent, 'Transport entropy', fmt='{:.2f}',
        annot={0: '−17.6%', 1: '−14.4%'}, title='(A) Transport entropy')
axes[0].legend(fontsize=8, loc='upper right', frameon=False)

# (b) 同层质量
grouped(axes[1], same, 'Same-layer mass', fmt='{:.3f}',
        title='(B) Layer concordance')

# (c) 运行时间（对数轴）
grouped(axes[2], times, 'Runtime (s, log scale)', fmt='{:.0f}', log=True,
        title='(C) Runtime (peak mem. 4.3 GB)')

plt.tight_layout()
plt.savefig(SAVE, dpi=300, bbox_inches='tight')
print("saved:", SAVE)
