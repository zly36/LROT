# -*- coding: utf-8 -*-
"""论文图6(真实小鼠脑 Visium)正式渲染 v5：位移着色(亮色) + 面板标注
说明: 该数据为单张真实切片 + 人工刚体变换(1:1 对应), 无细胞类型/分区标注,
因此颜色只编码“每点相对其匹配参考点的位移”(非细胞类型):
(a) 对齐前: 位移中心小→外缘大, 亮青→黄梯度; (b) 对齐后: 位移≈0, 整片亮青。
面板内标注中位位移与说明。输出: lrot_output/lrot_real_visium.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
# 自定位仓库根：向上找到含 lrot_core.py 的目录（代码包解压到任意路径均可运行）
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)
sys.path.insert(0, _R)
from lrot_paths import find_input


OUT = os.path.join(_R, 'lrot_output')
# 输入位于 lrot_output/数据_npz/，由 find_input 多候选查找
z = np.load(find_input('lrot_data_real_visium.npz'), allow_pickle=True)
A, B = z['A_coords'], z['B_coords']
P = z['P']; loss = z['loss_hist']

Pc = P / np.maximum(P.sum(0, keepdims=True), 1e-10)
B_on_A = Pc.T @ A

d_before = np.linalg.norm(B - A, axis=1)
d_after = np.linalg.norm(B_on_A - A, axis=1)

fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6))
xs = np.concatenate([A[:,0], B[:,0], B_on_A[:,0]])
ys = np.concatenate([A[:,1], B[:,1], B_on_A[:,1]])
cx, cy = (xs.min()+xs.max())/2, (ys.min()+ys.max())/2
hy = max(xs.max()-xs.min(), ys.max()-ys.min())/2*1.08
win = ([cx-hy, cx+hy], [cy-hy, cy+hy])

def frame(ax):
    ax.set_xlim(*win[0]); ax.set_ylim(*win[1])
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor('0.25'); sp.set_linewidth(1.0)

vmax = float(np.percentile(d_before, 99))
cmap = LinearSegmentedColormap.from_list('viridis_bright',
                                         matplotlib.colormaps['viridis'](np.linspace(0.4, 1.0, 256)))

def disp_panel(ax, coords, d, title, note):
    ax.scatter(A[:,0], A[:,1], c='0.88', s=7, alpha=0.8, linewidths=0, zorder=1)
    sc = ax.scatter(coords[:,0], coords[:,1], c=d, cmap=cmap, vmin=0, vmax=vmax,
                    s=14, alpha=1.0, linewidths=0, zorder=2)
    frame(ax)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('per-spot displacement (px)', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=11.5)
    ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=7.5, va='bottom',
            ha='left', color='0.25', style='italic')

med_b = np.median(d_before); med_a = np.median(d_after)
disp_panel(axes[0], B, d_before,
           'A: Before LROT (12\u00b0 rotation, 0.97 scale)\n(color = displacement of each spot to its matched reference spot)',
           'median = %.2f px (%.1f%% of span)\nno cell-type labels: color = registration displacement' % (med_b, med_b/(xs.max()-xs.min())*100))
disp_panel(axes[1], B_on_A, d_after,
           'B: After LROT (\u03b3=0.1)\n(displacement \u2248 0 \u2192 uniform bright teal = aligned)',
           'median = %.2f px (%.2f%% of span)' % (med_a, med_a/(xs.max()-xs.min())*100))

ax = axes[2]
ax.plot(loss, 'b-', linewidth=1.8)
ax.set_xlabel('Iteration', fontsize=11); ax.set_ylabel('Loss', fontsize=11)
ax.set_title('C: LROT Loss ({} iter)'.format(len(loss)), fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(OUT, 'lrot_real_visium.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print('saved:', out)
print('median before/after: %.2f / %.2f px' % (med_b, med_a))
