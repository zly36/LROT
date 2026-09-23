# -*- coding: utf-8 -*-
import os
"""
_render_dlpfc_fullscale_tissue.py — 全量 DLPFC 组织对齐可视化(补充图 S19/Figure S19)
对1: 151507(4,221) -> 151508(4,381), 不采样; B 施加旋转2°/平移0.5 (与主实验/全量验证一致)。
面板: (a) 对齐前 (151507 实心 + 151508 描边, 共享坐标窗); (b) LROT(γ=0.1) 对齐后
      (151508 经列归一化传输计划的 OT 重心软映射到 151507 帧, 灰点为 151507 参考组织)。
仅用于可视化; 指标数字见 lrot_output/dlpfc_full_scale_results.txt (正文数字不变)。
输出: lrot_output/lrot_dlpfc_fullscale_tissue.png, lrot_output/lrot_data_dlpfc_fullscale.npz
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'experiments'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_dlpfc_visium import load_dlpfc_slice
from run_cancer_visium import HUMAN_LR_DB, apply_misalignment, compute_lr_strength_human
from lrot_core import fgw_lr_solver

OUT = os.path.join(_R, 'lrot_output')
LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]

# ---------- 1. 数据加载(全量, 不采样) ----------
A = load_dlpfc_slice("151507", target_spots=10 ** 6, seed=42)
B = load_dlpfc_slice("151508", target_spots=10 ** 6, seed=99)
print(f"151507: {A['n_spots']} spots; 151508: {B['n_spots']} spots", flush=True)
B['coords'] = apply_misalignment(B['coords'], rotation_deg=2.0, translation=0.5)

# ---------- 2. LR 强度 + LROT(γ=0.1) 全量求解 ----------
ligs = sorted(set(k[0] for k in HUMAN_LR_DB) | set(k[1] for k in HUMAN_LR_DB))
print("compute LR strength (full)...", flush=True)
t0 = time.time()
S, top_pairs = compute_lr_strength_human(A, B, HUMAN_LR_DB)
print(f"  S {S.shape} top {len(top_pairs)}; {time.time()-t0:.1f}s", flush=True)
print("solve LROT (gamma=0.1, full)...", flush=True)
t0 = time.time()
P, loss = fgw_lr_solver(A, B, S, gamma=0.1, beta=0.3, alpha=0.5,
                        max_iter=200, verbose=False, lr_gene_list=ligs)
print(f"  P {P.shape}; {time.time()-t0:.1f}s", flush=True)

np.savez_compressed(os.path.join(OUT, "lrot_data_dlpfc_fullscale.npz"),
                    P=P, A_coords=A['coords'], B_coords=B['coords'],
                    A_region_labels=A['region_labels'], B_region_labels=B['region_labels'],
                    loss_hist=np.array(loss))
print("npz saved", flush=True)

# ---------- 3. 渲染 (a)/(b) 两面板, 风格同论文图9 ----------
coords_A, coords_B = A['coords'], B['coords']
labels_A, labels_B = A['region_labels'], B['region_labels']
Pc = P / np.maximum(P.sum(0, keepdims=True), 1e-10)      # 列归一 (nA, nB)
B_al = Pc.T @ coords_A                                     # B -> A 帧 (nB, 2)

layer_colors = matplotlib.colormaps['tab10'](np.linspace(0, 1, len(LAYER_ORDER)))

_xmin = min(coords_A[:, 0].min(), coords_B[:, 0].min(), B_al[:, 0].min())
_xmax = max(coords_A[:, 0].max(), coords_B[:, 0].max(), B_al[:, 0].max())
_ymin = min(coords_A[:, 1].min(), coords_B[:, 1].min(), B_al[:, 1].min())
_ymax = max(coords_A[:, 1].max(), coords_B[:, 1].max(), B_al[:, 1].max())
cx, cy = (_xmin + _xmax) / 2, (_ymin + _ymax) / 2
_hy = max(_xmax - _xmin, _ymax - _ymin) / 2 * 1.04
_hx = _hy * 1.0  # 两面板并排等宽
win = ([cx - _hx, cx + _hx], [cy - _hy, cy + _hy])

def frame_ax(ax):
    ax.set_xlim(*win[0]); ax.set_ylim(*win[1])
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor('0.25'); sp.set_linewidth(1.0)

def draw_layer(ax, X, labs, s, alpha=0.9, ec=None, lw=0.0, zorder=2):
    for l in range(len(LAYER_ORDER)):
        mk = labs == l
        if mk.sum() == 0:
            continue
        ax.scatter(X[mk, 0], X[mk, 1], s=s, c=[layer_colors[l]], alpha=alpha,
                   linewidths=lw, edgecolors=ec, zorder=zorder)

fig, axes = plt.subplots(1, 2, figsize=(16, 8.6))

# (a) Before
ax = axes[0]
draw_layer(ax, coords_A, labels_A, 4.5, alpha=0.55, zorder=2)
draw_layer(ax, coords_B, labels_B, 3.0, alpha=0.95, ec='black', lw=0.25, zorder=3)
frame_ax(ax)
ax.set_title('(A) Before alignment\n(solid = 151507, outlined = 151508, color = layer)', fontsize=13)

# (b) After (LROT)
ax = axes[1]
ax.scatter(coords_A[:, 0], coords_A[:, 1], c='0.78', s=3.0, alpha=0.6, linewidths=0, zorder=1)
draw_layer(ax, B_al, labels_B, 8.0, alpha=0.95, zorder=2)
frame_ax(ax)
ax.set_title('(B) After LROT (γ=0.1)\n(151508 soft-mapped onto 151507, shared frame)', fontsize=13)

# 层色图例 (bottom)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=layer_colors[l],
                      markersize=8, label=LAYER_ORDER[l]) for l in range(len(LAYER_ORDER))]
fig.legend(handles=handles, loc='lower center', ncol=7, frameon=False, fontsize=11,
           bbox_to_anchor=(0.5, 0.005))
plt.tight_layout(rect=(0, 0.03, 1, 0.95))
out_png = os.path.join(OUT, 'lrot_dlpfc_fullscale_tissue.png')
fig.savefig(out_png, dpi=500, bbox_inches='tight')
plt.close()
print('saved:', out_png)
