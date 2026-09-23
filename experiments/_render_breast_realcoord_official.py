# -*- coding: utf-8 -*-
import os
"""
Render official Figure 8 (real-coordinate version) - 6-panel layout matching the
original lrot_breast_cancer.png but driven by the TRUE tissue-coordinate run:
  lrot_data_breast_cancer_truecoord.npz + lrot_cancer_truecoord_results.txt
A: Before (S1 solid + S2 outlined, matched-sector color)
B: After  (S1 gray base + S2 soft-mapped (OT-centroid) onto S1 frame, matched-sector color)
  透明度说明: After 使用 OT 质心软映射呈现(可视化), 不改变熵/成本等指标口径
C: LROT loss history
D: Transport plan heatmap (downsampled)
E: Top LR pairs (barh)
F: Entropy bars FGW vs LROT
Output: lrot_output/lrot_breast_cancer_realcoord.png
"""
import os, re, sys, warnings
warnings.filterwarnings('ignore')
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(_R, 'lrot_output')
z = np.load(os.path.join(OUT, 'lrot_data_breast_cancer_truecoord.npz'), allow_pickle=True)
P = z['P']
A = z['A_coords'].astype(float)
Braw = z['B_coords'].astype(float)
loss_hist = z['loss_hist']
top_pairs = list(z['top_pairs'])
N_SAMPLE = len(A)

txt = open(os.path.join(OUT, 'lrot_cancer_truecoord_results.txt'), encoding='utf-8').read()
mm = re.findall(r'Entropy=([\d.]+)\s+Cost=([\d.]+)\s+Time=([\d.]+)s', txt)
fgw_ent, lrot_ent = float(mm[0][0]), float(mm[1][0])
imp = float(re.search(r'Entropy reduction: ([\d.]+)%', txt).group(1))

im = P.argmax(0)                     # B -> A 硬匹配目标(仅用于 secB 目标扇区着色; 不作为 After 绘图)
Pc = P / np.maximum(P.sum(0, keepdims=True), 1e-10)
B_al = Pc.T @ A                      # B soft centroid on A frame (After 绘图, 与 DLPFC 图9 B 同风格)

cen = A.mean(0)
ang = np.arctan2(A[:, 1] - cen[1], A[:, 0] - cen[0])
NSEC = 8
secA = np.mod(np.floor((ang + np.pi) / (2 * np.pi / NSEC)), NSEC).astype(int)
secB = secA[im]                       # S2 对齐后的目标扇区(用于 After 面板)
# S2 在自身(原始错位)坐标下的扇区 —— Before 面板用, 使 2° 错位在色带上如实可见
cenB = Braw.mean(0)
angB = np.arctan2(Braw[:, 1] - cenB[1], Braw[:, 0] - cenB[0])
secB_own = np.mod(np.floor((angB + np.pi) / (2 * np.pi / NSEC)), NSEC).astype(int)
palette = [matplotlib.colormaps['tab10'](i) for i in range(NSEC)]
cA = np.array([palette[q] for q in secA])
cB = np.array([palette[q] for q in secB])
cB_own = np.array([palette[q] for q in secB_own])

allpts = np.vstack([A, Braw, B_al])
_xmin, _xmax = allpts[:, 0].min(), allpts[:, 0].max()
_ymin, _ymax = allpts[:, 1].min(), allpts[:, 1].max()
cx, cy = (_xmin + _xmax) / 2, (_ymin + _ymax) / 2
_hy = max(_xmax - _xmin, _ymax - _ymin) / 2 * 1.05
win = ([cx - _hy, cx + _hy], [cy - _hy, cy + _hy])

def frame_ax(ax):
    ax.set_xlim(*win[0]); ax.set_ylim(*win[1])
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor('0.25'); sp.set_linewidth(1.0)

fig, axes = plt.subplots(2, 3, figsize=(19, 11.5))

# A: Before (S1 按自身扇区实心; S2 按自身扇区描边 → 原始错位可见)
ax = axes[0, 0]
for q in range(NSEC):
    mk = secA == q
    if mk.sum() == 0:
        continue
    ax.scatter(A[mk, 0], A[mk, 1], s=13, c=[palette[q]], alpha=0.55,
               linewidths=0, zorder=2, label='Sector %d' % (q + 1))
for q in range(NSEC):
    mk = secB_own == q
    if mk.sum() == 0:
        continue
    ax.scatter(Braw[mk, 0], Braw[mk, 1], s=9.5, c=[palette[q]], alpha=0.95,
               linewidths=0.4, edgecolors='black', zorder=3)
frame_ax(ax)
ax.legend(loc='upper left', fontsize=6.0, ncol=2, framealpha=0.85, markerscale=0.9,
          handletextpad=0.4, columnspacing=1.0, borderpad=0.4, labelspacing=0.25)
ax.set_title('A: Before LROT\n(solid=Section 1, outlined=Section 2, color=sector)', fontsize=12)

# B: After (soft/OT-质心投影; 灰 S1 参考底)
ax = axes[0, 1]
ax.scatter(A[:, 0], A[:, 1], c='0.82', s=11, alpha=0.65, linewidths=0, zorder=1)
for q in range(NSEC):
    mk = secB == q
    if mk.sum() == 0:
        continue
    ax.scatter(B_al[mk, 0], B_al[mk, 1], s=30, c=[palette[q]], alpha=0.9,
               linewidths=0, zorder=2)
frame_ax(ax)
ax.set_title('B: After LROT (gamma=0.1)\n(Section 2 soft-mapped onto Section 1 frame)', fontsize=12)

# C: loss
axes[0, 2].plot(loss_hist, 'b-', linewidth=1.8)
axes[0, 2].set_xlabel('Iteration', fontsize=11)
axes[0, 2].set_ylabel('Loss', fontsize=11)
axes[0, 2].set_title('C: LROT Loss (' + str(len(loss_hist)) + ' iter)', fontsize=12)
axes[0, 2].grid(True, alpha=0.3)

# D: transport plan
step = max(1, N_SAMPLE // 100)
axd = axes[1, 0]
imd = axd.imshow(P[::step, ::step], cmap='hot', aspect='auto', interpolation='nearest')
axd.set_title('D: Transport Plan (%d x %d, shown @1/%d)' % (N_SAMPLE, N_SAMPLE, step), fontsize=11)
axd.set_xlabel('Section 2 spots', fontsize=10)
axd.set_ylabel('Section 1 spots', fontsize=10)
plt.colorbar(imd, ax=axd, shrink=0.8)

# E: top LR pairs
ax = axes[1, 1]
pair_labels = ['%s-%s' % (l, r) for l, r, w in top_pairs[:8]]
pair_values = [float(w) for _, _, w in top_pairs[:8]]
colors_bar = matplotlib.colormaps['RdYlGn'](np.linspace(0.3, 0.9, len(pair_labels)))
bars = ax.barh(range(len(pair_labels)), pair_values, color=colors_bar)
for bar, val in zip(bars, pair_values):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            '%.2f' % val, va='center', fontsize=8)
ax.set_yticks(range(len(pair_labels)))
ax.set_yticklabels(pair_labels, fontsize=9)
ax.set_xlabel('DB Weight', fontsize=10)
ax.set_title('E: Top LR Pairs (Breast Cancer)', fontsize=12)
ax.invert_yaxis()

# F: entropy
ax = axes[1, 2]
ent = [fgw_ent, lrot_ent]
bars = ax.bar(['FGW (gamma=0)', 'LROT (gamma=0.1)'], ent, width=0.5,
              color=['#E74C3C', '#2ECC71'], edgecolor='gray', linewidth=0.8, alpha=0.9)
ax.set_ylim(min(ent) * 0.985, max(ent) * 1.02)
for bar, v in zip(bars, ent):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            '%.4f' % v, ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Entropy', fontsize=11)
ax.set_title('F: Alignment Uncertainty (Entropy)', fontsize=11.5, fontweight='bold')
ax.annotate('down %.2f%%' % imp, xy=(1, lrot_ent), xytext=(0.5, max(ent) * 1.01),
            ha='center', fontsize=10.5, color='#2E8B57', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=(0, 0, 1, 0.96))
out_png = os.path.join(OUT, 'lrot_breast_cancer_realcoord.png')
fig.savefig(out_png, dpi=600, bbox_inches='tight')
plt.close()
print('saved:', out_png)
