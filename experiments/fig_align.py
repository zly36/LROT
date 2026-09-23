# -*- coding: utf-8 -*-
"""
fig_align.py — 增强 Before/After 对齐面板的共享工具
策略(不动物理数据): 高对比叠加 + 自适应放大插图(标出小错位) + 位移量化标注
用法: 各画图脚本 import fig_align; 调用 draw_before_after(...)
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def matched_offsets(coords_A, coords_mov, P):
    """每个 A spot 的 argmax 匹配到 mov 的坐标位移(欧氏)"""
    j = P.argmax(1)
    return np.linalg.norm(coords_mov[j] - coords_A, axis=1)


def pick_zoom_window(coords_A, disp, frac=0.16, grid=24):
    """在 disp 较大且局部密集的区域选放大窗口"""
    ext_x = coords_A[:, 0].max() - coords_A[:, 0].min()
    ext_y = coords_A[:, 1].max() - coords_A[:, 1].min()
    th = np.percentile(disp, 82)
    cand = np.where(disp >= th)[0]
    if len(cand) == 0:
        cand = np.arange(len(coords_A))
    gx = np.clip(((coords_A[:, 0] - coords_A[:, 0].min()) / max(ext_x, 1e-9) * (grid - 1)).astype(int), 0, grid - 1)
    gy = np.clip(((coords_A[:, 1] - coords_A[:, 1].min()) / max(ext_y, 1e-9) * (grid - 1)).astype(int), 0, grid - 1)
    dens = np.zeros((grid, grid)); np.add.at(dens, (gy, gx), 1)
    best = cand[np.argmax(dens[gy[cand], gx[cand]])]
    half = max(ext_x, ext_y) * frac / 2
    cx = np.clip(coords_A[best, 0], coords_A[:, 0].min() + half, coords_A[:, 0].max() - half)
    cy = np.clip(coords_A[best, 1], coords_A[:, 1].min() + half, coords_A[:, 1].max() - half)
    return (cx - half, cx + half, cy - half, cy + half), best


def draw_panel(ax, coords_A, coords_mov, disp, tag, disp_units='px', disp_med=None,
               labels_A=None, labels_mov=None, layer_cmap=None, win=None,
               A_name='Slice A', B_name='Slice B', show_legend=True, sA=7, sB=7):
    """高对比叠加: A 为实心蓝色圆圈 / B 为红色空心圆环; 若提供 layer 标注则按层着色。
    位移显示像素差: 真实数据坐标常非像素, 用范围归一化为百分比标注。"""
    ext = max(coords_A[:, 0].max() - coords_A[:, 0].min(),
              coords_A[:, 1].max() - coords_A[:, 1].min())
    dmed = np.median(disp) if disp_med is None else disp_med
    pct = dmed / ext * 100

    if labels_A is not None and layer_cmap is not None:
        for l in np.unique(labels_A):
            mk = labels_A == l
            ax.scatter(coords_A[mk, 0], coords_A[mk, 1], c=[layer_cmap(l)], s=sA,
                       alpha=0.55, linewidths=0, label=f'{A_name} L{l+1}' if l < 6 else f'{A_name} WM')
        for l in np.unique(labels_mov) if labels_mov is not None else []:
            mk = labels_mov == l
            ax.scatter(coords_mov[mk, 0], coords_mov[mk, 1], marker='o',
                       c=[layer_cmap(l)], s=sB * 1.2, edgecolors='k',
                       linewidths=0.3, label=f'{B_name} L{l+1}' if l < 6 else f'{B_name} WM')
    else:
        ax.scatter(coords_A[:, 0], coords_A[:, 1], c='#2166AC', s=sA, alpha=0.6,
                   linewidths=0, label=A_name)
        ax.scatter(coords_mov[:, 0], coords_mov[:, 1], c='#C0392B', s=sB * 1.1,
                   alpha=0.6, linewidths=0, label=B_name)
    ax.set_aspect('equal')
    ax.set_title(f'{tag}\nmedian matched-spot shift = {dmed:.1f} ({pct:.1f}% of span)',
                 fontsize=11, linespacing=1.25)
    if show_legend:
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    # 自适应放大插图: 放大 disp 最大且密集的区域, 让小错位清楚可见
    if win is None:
        # 放大窗口选在“被位移的 mov 云”上位移最大且密集的区域
        win, _ = pick_zoom_window(coords_mov, disp) if len(disp) == len(coords_mov) \
            else pick_zoom_window(coords_A, disp)
    axin = inset_axes(ax, width='40%', height='40%', loc='lower left', borderpad=0.5)
    axin.scatter(coords_A[:, 0], coords_A[:, 1], c='#2166AC', s=sA * 2.4, alpha=0.75, linewidths=0)
    axin.scatter(coords_mov[:, 0], coords_mov[:, 1], c='#C0392B', s=sB * 1.1 * 2.6,
                 alpha=0.8, linewidths=0)
    axin.set_xlim(win[0], win[1]); axin.set_ylim(win[2], win[3])
    axin.set_aspect('equal'); axin.set_xticks([]); axin.set_yticks([])
    axin.tick_params(length=0)
    for sp in axin.spines.values():
        sp.set_linewidth(0.6)
    mark_inset(ax, axin, loc1=1, loc2=3, fc='none', ec='gray', lw=0.7)
    axin.set_title('zoom', fontsize=7)
    return dmed


def draw_before_after(grid, before_slot, after_slot, coords_A, coords_B_raw, P,
                      labels_A=None, labels_B=None, layer_cmap=None,
                      A_name='Slice A', B_name='Slice B', title_before='Before LROT',
                      title_after='After LROT', show_legend=True, sA=7, sB=7):
    """增强 Before/After: 在 grid 两个坐标格作画。
    Before 画 A + B_raw(原始错位);  After 把 B 传输到 A 帧(colnorm(P).T @ A)后再画。
    位移(逐 B 点): before=|B_raw[j]-A[col argmax j]|, after=|B_on_A[j]-A[col argmax j]|。
    假设 nA==nB(本论文所有数据均为 1000×1000)。"""
    Pc = P / np.maximum(P.sum(0, keepdims=True), 1e-10)      # 列归一 (nA,nB)
    coords_B_on_A = Pc.T @ coords_A                            # B 映射到 A 帧 (nB,2)
    im = P.argmax(0)                                           # B_j -> A_im
    d_b = np.linalg.norm(coords_B_raw - coords_A[im], axis=1)
    d_a = np.linalg.norm(coords_B_on_A - coords_A[im], axis=1)
    draw_panel(grid[before_slot], coords_A, coords_B_raw, d_b, title_before,
               labels_A=labels_A, labels_mov=labels_B, layer_cmap=layer_cmap,
               A_name=A_name, B_name=B_name, show_legend=show_legend, sA=sA, sB=sB)
    draw_panel(grid[after_slot], coords_A, coords_B_on_A, d_a, title_after,
               labels_A=labels_A, labels_mov=None, layer_cmap=layer_cmap,
               A_name=A_name, B_name=B_name, show_legend=show_legend, sA=sA, sB=sB)
    return d_b, d_a
