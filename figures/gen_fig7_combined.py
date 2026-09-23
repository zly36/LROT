# -*- coding: utf-8 -*-
"""图7 合成图(NAR 规范):把 3D 重建、跨切片一致性、多 seed 相邻精度三个面板
合成**一张**图,面板编号用大写 A/B/C。

设计: 画布 15.92 cm × 21.0 cm(= 正文文本宽 × 单页可容纳高度), 字号按最终印刷尺寸给定
      (标题 9 pt / 轴标 8 pt / 刻度 7 pt), 避免"三张独立大图"跨 1.5 页且字体偏大的问题。

数据: 面板 A/B 由 `experiments/lrot_3d_reconstruction.py` 现算(确定性 seed),
      面板 C 由 `experiments/lrot_3d_multi_seed.py` 现算(10 个独立 seed)。
输出: lrot_output/figures_png/lrot_fig7_combined.png
"""
import os
import sys
import runpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


ROOT = _R
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\experiments")

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7})

OUT = Path(ROOT) / 'lrot_output' / 'figures_png' / 'lrot_fig7_combined.png'

# ---------- 数据: 面板 A/B ----------
print('[1/2] 现算 3D 重建(面板 A/B)...')
ns = runpy.run_path(ROOT + r"\experiments\lrot_3d_reconstruction.py")
coords3d = ns['all_coords_3d']
labels3d = ns['all_labels']
pairwise = np.asarray(ns['pairwise_acc'])
n_regions = int(ns['gen'].n_regions)
n_slices = int(ns['N_SLICES'])
print('    coords3d', coords3d.shape, 'pairwise', pairwise.shape, 'regions', n_regions)

# ---------- 数据: 面板 C ----------
CACHE = Path(ROOT) / 'lrot_output' / '数据_npz' / 'lrot_fig7_multiseed_cache.npz'
print('[2/2] 多 seed 相邻切片精度(面板 C)...')
from lrot_3d_multi_seed import generate_slices, sequential_align, N_SEEDS, N_SLICES as MS_SLICES

if CACHE.exists():
    accs = np.load(CACHE)['accs']
    print('    由缓存读取:', CACHE.name)
else:
    accs = np.array([sequential_align(generate_slices(42 + s * 100), 0.1, True)
                     for s in range(N_SEEDS)])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, accs=accs)
    print('    已缓存:', CACHE)
per_mean, per_std = accs.mean(axis=0), accs.std(axis=0)
overall_mean, overall_std = accs.mean(), accs.std()
print('    overall %.4f +/- %.4f' % (overall_mean, overall_std))

# ================= 绘图 =================
fig = plt.figure(figsize=(6.27, 2.20))          # 15.92 x 5.59 cm(横排一行三面板)
gs = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.00, 1.18], wspace=0.62,
                      left=0.062, right=0.985, top=0.795, bottom=0.175)

def letter(ax, s, title, dx=0.0, dy=1.135):
    """面板字母(大写, 粗体) + 短标题, 同一行放在面板上方。"""
    put = ax.text2D if hasattr(ax, 'text2D') else ax.text
    put(dx, dy, s, transform=ax.transAxes, fontsize=10.5, fontweight='bold',
        va='bottom', ha='left', clip_on=False)
    put(dx + 0.16, dy, title, transform=ax.transAxes, fontsize=8.2,
        va='bottom', ha='left', clip_on=False)

# ---- 面板 A: 3D 重建 ----
axA = fig.add_subplot(gs[0], projection='3d')
cmap = matplotlib.colormaps['tab10'](np.linspace(0, 1, 10))
for r in range(n_regions):
    m = labels3d == r
    axA.scatter(coords3d[m, 0], coords3d[m, 1], coords3d[m, 2], c=[cmap[r]], s=1.1,
                alpha=0.65, linewidths=0, label='R%d' % r)
axA.set_xlabel('X', labelpad=-4); axA.set_ylabel('Y', labelpad=-4)
axA.set_zlabel('Z (slice)', labelpad=-6)
axA.tick_params(pad=-2.5, labelsize=5.8)
axA.legend(loc='upper right', fontsize=5.6, ncol=2, frameon=True, framealpha=0.9,
           borderpad=0.25, columnspacing=0.5, handletextpad=0.25, markerscale=2.6,
           bbox_to_anchor=(1.06, 1.04))
axA.view_init(elev=18, azim=-62)
letter(axA, 'A', 'LROT 3-D reconstruction')

# ---- 面板 B: 跨切片一致性矩阵 ----
axB = fig.add_subplot(gs[1])
im = axB.imshow(pairwise, cmap='YlOrRd', vmin=0, vmax=1)
for i in range(pairwise.shape[0]):
    for j in range(pairwise.shape[1]):
        axB.text(j, i, '%.2f' % pairwise[i, j], ha='center', va='center', fontsize=5.8,
                 color='black' if pairwise[i, j] < 0.72 else 'white')
axB.set_xticks(range(n_slices)); axB.set_yticks(range(n_slices))
axB.tick_params(labelsize=6.2)
axB.set_xlabel('Slice B', fontsize=6.8, labelpad=1)
axB.set_ylabel('Slice A', fontsize=6.8, labelpad=1)
cb = fig.colorbar(im, ax=axB, fraction=0.042, pad=0.03)
cb.set_label('Accuracy', fontsize=6.2); cb.ax.tick_params(labelsize=5.8)
letter(axB, 'B', 'Cross-slice consistency')

# ---- 面板 C: 多 seed 相邻精度 ----
axC = fig.add_subplot(gs[2])
x = np.arange(len(per_mean))
axC.bar(x, per_mean * 100, yerr=per_std * 100, capsize=2.5, color='#4C72B0', alpha=0.9,
        error_kw={'elinewidth': 0.9, 'ecolor': '#2A3D5C'})
axC.axhline(overall_mean * 100, color='#C44E52', linestyle='--', linewidth=1.0,
            label='Overall %.1f%%' % (overall_mean * 100))
axC.axhspan((overall_mean - overall_std) * 100, (overall_mean + overall_std) * 100,
            color='#C44E52', alpha=0.12, label='$\\pm$ 1 SD')
for i, (m, sd) in enumerate(zip(per_mean, per_std)):
    axC.text(i, (m + sd) * 100 + 1.0, '%.1f' % (m * 100), ha='center', fontsize=5.8)
axC.set_xticks(x)
axC.set_xticklabels(['%d-%d' % (i, i + 1) for i in range(len(per_mean))], fontsize=6.2)
axC.set_xlabel('Adjacent slice pair', fontsize=6.8, labelpad=1)
axC.set_ylabel('Accuracy (%)', fontsize=6.8, labelpad=1)
axC.set_ylim(0, 100)
axC.tick_params(labelsize=6.2)
axC.legend(loc='lower left', fontsize=5.6, frameon=False)
axC.grid(axis='y', alpha=0.25, linewidth=0.5)
letter(axC, 'C', 'Adjacent-slice accuracy')

fig.savefig(OUT, dpi=600, facecolor='white')
print('Saved:', OUT)
from PIL import Image
im2 = Image.open(OUT)
print('size %dx%d  aspect H/W = %.4f  (display 15.92 cm -> %.2f cm tall)'
      % (im2.width, im2.height, im2.height / im2.width, 15.92 * im2.height / im2.width))
