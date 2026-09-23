# -*- coding: utf-8 -*-
import os
"""DLPFC 论文图9 正式渲染脚本（PASTE2 风格，共享坐标窗 + 软映射）
面板: A 对齐前 / B 对齐后(软映射) / C 损失 / D 熵(↓17.6%) / E 软ARI / F γ权衡。
输出: lrot_output/lrot_dlpfc.png
"""
import os, re, sys, warnings
warnings.filterwarnings('ignore')
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)
from lrot_paths import find_input
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(_R, 'lrot_output')
LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]

# ---------- 数据 ----------
# 输入不在 OUT 根目录时（位于 数据_npz/、_中间产物/结果txt/ 等子目录），
# find_input 会抱预先声明的候选目录逐个找，并在命中时打印实际位置；找不到则列出全部已查找路径。
z = np.load(find_input("lrot_data_dlpfc.npz"), allow_pickle=True)
coords_A, coords_B = z['A_coords'], z['B_coords']
labels_A, labels_B = z['A_region_labels'], z['B_region_labels']
P = z['P']
loss_hist = z['loss_hist']
# LROT 对齐: 151508 经质心(OT 重心)软映射到 151507 帧 —— 1000 个 B spot 各占一位,
# 不塌缩(硬 argmax 快照会让 1000 个点挤到仅 539 个 A 位置上, 画面出现空洞/稀疏)
Pc = P / np.maximum(P.sum(0, keepdims=True), 1e-10)   # 列归一 (nA, nB)
B_al = Pc.T @ coords_A                                  # B 映射到 A 帧 (nB, 2)

txt = open(find_input("lrot_dlpfc_results.txt"), encoding='utf-8').read()
block = txt.split("Pair 151507-151508")[1].split("Pair 151508")[0]
def grab(pattern):
    m = re.search(pattern, block)
    if m is None:                       # 显式判空：re.search 可能返回 None，直接 .group 会崩
        raise ValueError('未在结果文件中匹配到: %s' % pattern)
    return float(m.group(1))
fgw_ent, lrot_ent = grab(r'FGW.*?熵=([\d.]+)'), grab(r'LROT.*?熵=([\d.]+)')
fgw_ari, lrot_ari = grab(r'FGW.*?软ARI=([\d.]+)'), grab(r'LROT.*?软ARI=([\d.]+)')
sweep = [{'gamma': float(m.group(1)), 'entropy': float(m.group(2)),
          'soft_ari': float(m.group(3)), 'same_mass': float(m.group(4))}
         for m in re.finditer(r'γ=([\d.]+)\s+熵=([\d.]+)\s+软ARI=([\d.]+)\s+同层质量=([\d.]+)', block)]

span = max(coords_A[:, 0].max() - coords_A[:, 0].min(), coords_A[:, 1].max() - coords_A[:, 1].min())
layer_colors = matplotlib.colormaps['tab10'](np.linspace(0, 1, len(LAYER_ORDER)))

fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))

def draw_layer(ax, X, labs, s):
    for l in range(len(LAYER_ORDER)):
        mk = labs == l
        if mk.sum() == 0:
            continue
        ax.scatter(X[mk, 0], X[mk, 1], s=s, c=[layer_colors[l]], alpha=0.9,
                   linewidths=0, zorder=2)

def frame_ax(ax, win=None):
    """A/B 散点面板: 同一方形轴框(与论文图9 一致), 只隐藏刻度.
    win=(xlim_pair, ylim_pair) 给定时 A/B 共用同一数据窗口 → 边框同尺寸、标题同高.
    """
    if win is not None:
        ax.set_xlim(*win[0]); ax.set_ylim(*win[1])
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor('0.25')
        sp.set_linewidth(1.0)

# ---------- A: Before (共享坐标系叠加, 与 B 同一方形窗口 → 六格同尺寸) ----------
# 共享数据窗口: 覆盖 A/B 原始坐标与 B 映射结果, 取方形
_xmin = min(coords_A[:, 0].min(), coords_B[:, 0].min(), B_al[:, 0].min())
_xmax = max(coords_A[:, 0].max(), coords_B[:, 0].max(), B_al[:, 0].max())
_ymin = min(coords_A[:, 1].min(), coords_B[:, 1].min(), B_al[:, 1].min())
_ymax = max(coords_A[:, 1].max(), coords_B[:, 1].max(), B_al[:, 1].max())
cx, cy = (_xmin + _xmax) / 2, (_ymin + _ymax) / 2
_hy = max(_xmax - _xmin, _ymax - _ymin) / 2 * 1.06
# 窗口宽高比 = 子图格宽高比(实测 532/445) → A/B 框与 CDEF 精确同尺寸铺满
_hx = _hy * (532.0 / 445.0)
win = ([cx - _hx, cx + _hx], [cy - _hy, cy + _hy])

ax = axes[0, 0]
# 151507 实心按层(作为图例来源)
for l in range(len(LAYER_ORDER)):
    mk = labels_A == l
    if mk.sum() == 0:
        continue
    ax.scatter(coords_A[mk, 0], coords_A[mk, 1], s=13, c=[layer_colors[l]],
               alpha=0.5, linewidths=0, zorder=2, label=LAYER_ORDER[l])
# 151508 原始描边(黑边层色) → 层边界错位可见
for l in range(len(LAYER_ORDER)):
    mk = labels_B == l
    if mk.sum() == 0:
        continue
    ax.scatter(coords_B[mk, 0], coords_B[mk, 1], s=9, c=[layer_colors[l]],
               alpha=0.95, linewidths=0.4, edgecolors='black', zorder=3)
frame_ax(ax, win)
ax.legend(loc='upper left', fontsize=6.5, ncol=2, framealpha=0.85, markerscale=0.8,
          handletextpad=0.4, columnspacing=1.0, borderpad=0.4, labelspacing=0.3)
ax.set_title('A: Before LROT\n(solid=151507, outlined=151508, color=layer)', fontsize=12)

# ---------- B: After (映射到 A 帧; 灰底参考 + 彩色映射点) ----------
ax = axes[0, 1]
# 淡灰 A(151507)参考底: 稍大/稍深/更不透明, 让组织轮廓更饱满
ax.scatter(coords_A[:, 0], coords_A[:, 1], c='0.8', s=12, alpha=0.7,
           linewidths=0, zorder=1)
# 151508 质心软映射结果, 按自身层色着色; 点加大使层带更醒目
draw_layer(ax, B_al, labels_B, 36)
frame_ax(ax, win)
ax.set_title('B: After LROT (γ=0.1)\n(151508 mapped onto 151507, shared frame)', fontsize=12)

# ---------- C: loss ----------
ax = axes[0, 2]
ax.plot(loss_hist, 'b-', linewidth=1.8)
ax.set_xlabel('Iteration', fontsize=11); ax.set_ylabel('Loss', fontsize=11)
ax.set_title(f"C: LROT Loss ({len(loss_hist)} iter)", fontsize=12)
ax.grid(True, alpha=0.3)

# ---------- D: entropy ----------
ax = axes[1, 0]
ent = [fgw_ent, lrot_ent]
bars = ax.bar(['FGW (γ=0)', 'LROT (γ=0.1)'], ent, color=['#E74C3C', '#2ECC71'],
              edgecolor='gray', alpha=0.9, width=0.55)
ax.set_ylim(min(ent) * 0.98, max(ent) * 1.02)
for b, v in zip(bars, ent):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.4f}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Entropy', fontsize=11)
imp0 = (fgw_ent - lrot_ent) / fgw_ent * 100
ax.set_title(f"D: Alignment Entropy (↓{imp0:.1f}%)", fontsize=12)

# ---------- E: soft ARI ----------
ax = axes[1, 1]
x = np.arange(2); w = 0.35
ari = [fgw_ari, lrot_ari]
bars = ax.bar(x, ari, w, color=['#E74C3C', '#2ECC71'], edgecolor='gray', alpha=0.9)
for b, v in zip(bars, ari):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f'{v:.3f}',
            ha='center', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(['FGW (γ=0)', 'LROT (γ=0.1)'], fontsize=11)
ax.set_ylabel('Soft ARI', fontsize=11)
ax.set_title('E: Layer Concordance (Soft ARI)', fontsize=12)

# ---------- F: gamma sweep ----------
gs = [s['gamma'] for s in sweep]
ents = [s['entropy'] for s in sweep]
masses = [s['same_mass'] for s in sweep]
ax1 = axes[1, 2]
ax1.plot(gs, ents, 'o-', color='#2980B9', label='Entropy', linewidth=2.0, ms=5)
ax1.set_xlabel('γ (LR guidance weight)', fontsize=11)
ax1.set_ylabel('Entropy', fontsize=11, color='#2980B9')
ax1.tick_params(axis='y', labelcolor='#2980B9')
ax1.set_xticks(gs)
ax2 = ax1.twinx()
ax2.plot(gs, masses, 's--', color='#C0392B', label='Same-layer mass', linewidth=2.0)
ax2.set_ylabel('Same-layer mass', fontsize=11, color='#C0392B')
ax2.tick_params(axis='y', labelcolor='#C0392B')
ax1.set_title('F: Entropy–Accuracy Trade-off vs γ', fontsize=12)
lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc='center right')

# 层色图例已内嵌到 A 面板(不再需要整图底部图例)
plt.tight_layout(rect=(0, 0, 1, 0.96))
out = os.path.join(OUT, 'lrot_dlpfc.png')
fig.savefig(out, dpi=900, bbox_inches='tight')   # 高分辨率: 900 dpi (论文印刷级)
plt.close()
print('saved:', out)
