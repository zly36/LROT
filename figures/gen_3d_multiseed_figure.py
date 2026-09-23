import os
# -*- coding: utf-8 -*-
"""图7 面板 C:3D 重建多 seed 误差棒(重写版,匹配 experiments/lrot_3d_multi_seed.py 现行接口)
统计口径与原图一致: N_SEEDS 个独立 seed,每个 seed 生成 N_SLICES 张切片,
相邻切片两两对齐,逐对精度取 10 seed 的均值±标准差;另存 per-seed 散点图。
NAR 规范: 面板用大写字母 (C)。
输出: lrot_output/lrot_3d_multiseed_errorbar.png (图7 面板 C, 前缀 "C: ")
      lrot_output/lrot_3d_multiseed_scatter.png (备用)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'experiments'))
from lrot_3d_multi_seed import generate_slices, sequential_align, N_SEEDS, N_SLICES, N_SPOTS

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = Path(_R, 'lrot_output')

print(f"Multi-seed 3D validation: {N_SEEDS} seeds x {N_SLICES} slices x {N_SPOTS} spots")
lrot_all, fgw_all = [], []
for s in range(N_SEEDS):
    slices = generate_slices(42 + s * 100)
    lrot_all.append(sequential_align(slices, 0.1, True))
    fgw_all.append(sequential_align(slices, 0.0, False))
    print(f"  seed {s+1}/{N_SEEDS}: LROT={np.mean(lrot_all[-1]):.4f}  FGW={np.mean(fgw_all[-1]):.4f}")

lrot_all = np.array(lrot_all)
fgw_all = np.array(fgw_all)
per_mean, per_std = lrot_all.mean(axis=0), lrot_all.std(axis=0)
seed_means = lrot_all.mean(axis=1)
overall_mean, overall_std = lrot_all.mean(), lrot_all.std()
print("LROT overall %.4f +/- %.4f | FGW overall %.4f +/- %.4f"
      % (overall_mean, overall_std, fgw_all.mean(), fgw_all.std()))

# ---------- 面板 C: 相邻切片对精度误差棒 ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
pairs = [f'Slice {i}-{i+1}' for i in range(N_SLICES - 1)]
x = np.arange(len(pairs))
ax.bar(x, per_mean * 100, yerr=per_std * 100, capsize=6, color='#4C72B0', alpha=0.85,
       error_kw={'elinewidth': 1.5, 'ecolor': '#2A3D5C'})
ax.axhline(overall_mean * 100, color='#C44E52', linestyle='--', linewidth=1.5,
           label='Overall mean = %.1f%%' % (overall_mean * 100))
ax.axhspan((overall_mean - overall_std) * 100, (overall_mean + overall_std) * 100,
           color='#C44E52', alpha=0.12, label='+/- 1 SD')
for i, (m, sd) in enumerate(zip(per_mean, per_std)):
    ax.text(i, m * 100 + sd * 100 + 2, '%.1f%%' % (m * 100), ha='center', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=11)
ax.set_ylabel('Alignment accuracy (%)', fontsize=12)
ax.set_ylim(0, 105)
ax.set_title('C: Adjacent-slice alignment accuracy (mean $\\pm$ SD over %d seeds)' % N_SEEDS,
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(str(OUT_DIR / 'lrot_3d_multiseed_errorbar.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('  Saved:', OUT_DIR / 'lrot_3d_multiseed_errorbar.png')

# ---------- 备用: per-seed 散点 ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
sx = np.arange(1, N_SEEDS + 1)
ax.scatter(sx, seed_means * 100, s=60, color='#4C72B0', zorder=3, label='Per-seed mean accuracy')
ax.axhline(overall_mean * 100, color='#C44E52', linestyle='--', linewidth=1.5,
           label='Overall mean = %.1f%%' % (overall_mean * 100))
ax.fill_between(sx, (overall_mean - overall_std) * 100, (overall_mean + overall_std) * 100,
                color='#C44E52', alpha=0.12, label='+/- 1 SD')
for i, m in enumerate(seed_means):
    ax.text(float(sx[i]), m * 100 + 2, '%.1f%%' % (m * 100), ha='center', fontsize=8.5)
ax.set_xlabel('Random seed index', fontsize=12)
ax.set_ylabel('Mean alignment accuracy (%)', fontsize=12)
ax.set_xticks(sx); ax.set_ylim(0, 105)
ax.set_title('3D reconstruction accuracy across %d random seeds' % N_SEEDS,
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower left'); ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(str(OUT_DIR / 'lrot_3d_multiseed_scatter.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('  Saved:', OUT_DIR / 'lrot_3d_multiseed_scatter.png')
