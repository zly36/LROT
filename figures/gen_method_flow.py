# -*- coding: utf-8 -*-
"""
LROT 方法总览图（论文图1）的生成脚本。

版面约定
--------
 · 画布 16.0 cm 宽（字号即印刷字号），高度留足留白；
 · 多行文本行距 pitch >= 2.2 单位（约 10 pt），图注统一单行；
 · A 区两切片以"叠片"符号表示；B 区的变换在箭头上以短标注表示；
 · 底部信息条为三栏（默认参数 / 求解设置 / 复杂度）+ 图例行。

方法内容与正文 §2.1–2.9 及算法 1 一致。
输出: lrot_output/figures_png/lrot_method_flow.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)



OUT = os.path.join(_R, 'lrot_output', 'figures_png', 'lrot_method_flow.png')
DEBUG = os.environ.get('LROT_DEBUG') == '1'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.unicode_minus'] = False

INK = '#20272E'; SUB = '#6E7883'; HAIR = '#DCE3E9'; ARRC = '#5A6672'
BLUE = '#2B6CB0'; ORG = '#D98431'; PINK = '#C2537A'; PUR = '#7B5EA7'; RED = '#C0392B'; GRN = '#2E8B57'
CM_BLUE = LinearSegmentedColormap.from_list('bl', ['#FFFFFF', '#D6E5F4', '#7FAAD6', '#2B6CB0', '#173A61'])
CM_BLUE_L = LinearSegmentedColormap.from_list('bll', ['#FFFFFF', '#EDF4FB', '#D3E3F2', '#B3CFE9', '#93BADE'])
CM_ORG = LinearSegmentedColormap.from_list('og', ['#FFFFFF', '#FBE4C7', '#EFBB77', '#D98431', '#8A4A0E'])
CM_PINK = LinearSegmentedColormap.from_list('pk', ['#FFFFFF', '#F6DCE4', '#E1A0B6', '#C2537A', '#7E2743'])

WU, HU = 100.0, 124.0
fig, ax = plt.subplots(figsize=(6.30, 6.30 * HU / WU))
ax.set_xlim(0, WU); ax.set_ylim(0, HU); ax.axis('off')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
TB = []

# ----------------------------------------------------------------- helpers
def t(x, y, s, fs=6.4, bold=False, color=INK, ha='center', va='center', z=6, rot=0):
    o = ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, rotation=rot,
                fontweight='bold' if bold else 'normal', zorder=z)
    if DEBUG:
        bb = o.get_window_extent(fig.canvas.get_renderer())
        inv = ax.transData.inverted()
        p0, p1 = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        TB.append((p0[0], p0[1], p1[0], p1[1], s))
        if p0[0] < 0.2 or p1[0] > WU - 0.2 or p0[1] < 0.2 or p1[1] > HU - 0.2:
            print(f'  [out-of-canvas] {s!r} x[{p0[0]:.2f},{p1[0]:.2f}] y[{p0[1]:.2f},{p1[1]:.2f}]')

def arrow(x1, y1, x2, y2, color=ARRC, lw=0.8, ls='-', ms=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))

def larrow(verts, color=ARRC, lw=0.8, ls='-', ms=7):
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    ax.add_patch(FancyArrowPatch(path=Path(verts, codes), arrowstyle='-|>', mutation_scale=ms,
                                 lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0, zorder=4))

def pmat(x, y, w, h, data, cmap, sk=0.20, z=3.0, elw=0.18, frame=True):
    nr, nc = data.shape
    ch, cw = h / nr, w / nc
    vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
    for i in range(nr):
        for j in range(nc):
            yb = y + h - (i + 1) * ch
            x0 = x + j * cw + (yb - y) * sk
            pts = [(x0, yb), (x0 + cw, yb), (x0 + cw + ch * sk, yb + ch), (x0 + ch * sk, yb + ch)]
            ax.add_patch(Polygon(pts, closed=True,
                                 fc=cmap((data[i, j] - vmin) / max(vmax - vmin, 1e-9)),
                                 ec='white', lw=elw, zorder=z))
    if frame:
        ax.add_patch(Polygon([(x, y), (x + w, y), (x + w + h * sk, y + h), (x + h * sk, y + h)],
                             closed=True, fc='none', ec=INK, lw=0.5, zorder=z + 0.4))

def slabs(x, y, w, h, n, dx, dy, mats, cmap, sk=0.18, z=3.0):
    for k in range(n - 1, -1, -1):
        pmat(x + k * dx, y + k * dy, w, h, mats[k], cmap, sk=sk, z=z + 0.02 * (n - k))

def spotgrid(cx, cy, w, h, nx, ny, seed=1, connect=False, colors=None, s=2.2, jit=0.10,
             off=(0.0, 0.0), z=3.0):
    rng = np.random.RandomState(seed)
    xs, ys = np.meshgrid(np.linspace(cx - w / 2, cx + w / 2, nx),
                         np.linspace(cy - h / 2, cy + h / 2, ny))
    xs = xs.ravel() + rng.uniform(-jit, jit, xs.size) + off[0]
    ys = ys.ravel() + rng.uniform(-jit, jit, ys.size) + off[1]
    if connect:
        for i in range(xs.size):
            d = (xs - xs[i]) ** 2 + (ys - ys[i]) ** 2
            for j in np.argsort(d)[1:3]:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color='#C3CCD4', lw=0.35, zorder=z)
    ax.scatter(xs, ys, s=s, c=(colors if colors is not None else '#3E6FB0'),
               edgecolors='white', linewidths=0.25, zorder=z + 0.3)

def tissuetile(cx, cy, size, seed=3, cmap=None, z=2.5):
    rng = np.random.RandomState(seed)
    m = 0.45 + 0.55 * rng.rand(14, 14)
    m = (m + np.roll(m, 1, 0) + np.roll(m, 1, 1)) / 3.0
    ax.imshow(m, extent=[cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2],
              aspect='auto', cmap=(cmap if cmap is not None else CM_PINK),
              interpolation='bilinear', zorder=z)
    ax.add_patch(Rectangle((cx - size / 2, cy - size / 2), size, size, fc='none', ec=INK,
                           lw=0.5, zorder=z + 0.4))

def lrpair(cx, cy, r=0.55, z=3.2):
    ax.add_patch(Circle((cx - r, cy), r * 0.72, fc='#F0C3D1', ec='#B25C7E', lw=0.5, zorder=z))
    ax.add_patch(Circle((cx + r, cy), r * 0.72, fc='#C6DCF0', ec='#3E77AE', lw=0.5, zorder=z))
    ax.plot([cx - r * 0.3, cx + r * 0.3], [cy, cy], color='#7E2743', lw=0.6, zorder=z)

def chips(x, y, w, h, items, col, fs=6.1, gap=0.75, z=3.0):
    for i, s in enumerate(items):
        yy = y + (len(items) - 1 - i) * (h + gap)
        ax.add_patch(FancyBboxPatch((x, yy), w, h, boxstyle='round,pad=0.02,rounding_size=0.30',
                                    fc='white', ec=col, lw=0.55, zorder=z))
        t(x + 0.6, yy + h / 2, s, fs=fs, color=INK, ha='left', z=z + 0.3)

def gauss(cx, cy, w, h, mu=0.0, sd=1.0, color=BLUE, lw=1.0, z=3.2):
    x = np.linspace(-3.4, 3.4, 160)
    yv = np.exp(-0.5 * ((x - mu) / sd) ** 2)
    xx, yy = cx + x * w / 6.8, cy + yv * h
    ax.fill_between(xx, cy, yy, color=color, alpha=0.14, zorder=z, lw=0)
    ax.plot(xx, yy, color=color, lw=lw, zorder=z + 0.2)

def header(x0, x1, ytop, letter, num, title, col):
    t(x0 + 1.5, ytop, letter, fs=10.5, bold=True, color=col, ha='center')
    cx = x0 + 4.0
    ax.add_patch(FancyBboxPatch((cx - 0.9, ytop - 0.9), 1.8, 1.8,
                                boxstyle='round,pad=0.02,rounding_size=0.45',
                                fc=col, ec='none', zorder=5))
    t(cx, ytop, str(num), fs=6.2, bold=True, color='white', z=6)
    t(x0 + 5.6, ytop, title, fs=7.6, bold=True, ha='left')
    ax.plot([x0 + 1.2, x1 - 1.2], [ytop - 2.0, ytop - 2.0], color=HAIR, lw=0.8, zorder=1)

def key(cx, cy, cm, lb, w=1.5, gap=1.7, fs=5.8):
    ax.add_patch(Rectangle((cx, cy - 0.55), w, 1.1, fc=cm(0.75), ec=INK, lw=0.3, zorder=3))
    t(cx + gap, cy, lb, fs=fs, ha='left')

BAND = np.arange(14)
def bandplan(sig):
    m = np.exp(-((BAND[:, None] - BAND[None, :]) ** 2) / (2.0 * sig ** 2))
    return m / m.sum(1, keepdims=True)

# ============================================================ A  inputs
AX0, AX1 = 2.0, 37.0
header(AX0, AX1, 121.6, 'A', 1, 'Inputs', BLUE)
rngA = np.random.RandomState(4)
t(19.5, 115.6, 'two adjacent sections:  Slice A + Slice B', fs=5.8, color=SUB)
t(8.2, 108.0, 'expression', fs=6.0, color=SUB)
t(18.0, 108.0, 'spatial location', fs=6.0, color=SUB)
t(27.8, 108.0, 'H&E', fs=6.0, color=SUB)
# 叠片符号: back = Slice B (淡色), front = Slice A
pmat(AX0 + 4.2, 109.4, 6.4, 4.4, rngA.rand(8, 8), CM_BLUE_L, sk=0.16, z=2.9)
pmat(AX0 + 3.4, 108.8, 6.4, 4.4, rngA.rand(8, 8), CM_BLUE, sk=0.16, z=3.1)
spotgrid(18.0, 111.6, 5.4, 4.2, 5, 4, seed=11, off=(0.6, 0.6), colors='#E4574C', s=2.0, z=3.0)
spotgrid(18.0, 111.0, 5.4, 4.2, 5, 4, seed=21, colors='#3E6FB0', s=2.0, z=3.2)
tissuetile(28.0, 111.9, 4.6, seed=31, cmap=CM_BLUE_L, z=2.9)
tissuetile(27.6, 111.2, 4.6, seed=12, z=3.1)
t(3.6, 104.2, 'LR database: CellChatDB / CellPhoneDB', fs=6.4, bold=True, ha='left')
for xx in (6.4, 12.4, 18.4):
    lrpair(xx, 101.6)
t(6.4, 99.9, 'ligand', fs=6.0, color=SUB)
t(13.0, 99.9, 'receptor', fs=6.0, color=SUB)
t(20.0, 97.6, r'61 pairs  $\cdot$  weight $w$  $\cdot$  109 genes  $\cdot$  11 families', fs=6.0)
arrow(AX1 + 0.6, 111.6, AX1 + 3.6, 111.6)

# ============================================================ B  cost construction
BX0, BX1 = 41.0, 98.0
header(BX0, BX1, 121.6, 'B', 2, 'Cost construction', ORG)
t(BX0 + 2.0, 116.6, 'three complementary cost channels', fs=6.0, color=SUB, ha='left')
rows = [112.4, 106.6, 100.8]
rngB = np.random.RandomState(7)

# row 1 — expression
t(BX0 + 0.4, rows[0], 'expression (non-LR)', fs=6.4, bold=True, ha='left')
pmat(BX0 + 14.4, rows[0] - 1.8, 6.2, 3.6, rngB.rand(6, 6), CM_BLUE, sk=0.16)
arrow(BX0 + 21.2, rows[0], BX0 + 25.0, rows[0])
t(BX0 + 23.1, rows[0] + 2.0, 'cosine', fs=6.0, color=SUB)
pmat(BX0 + 25.4, rows[0] - 1.8, 6.2, 3.6,
     np.abs(np.subtract.outer(rngB.rand(5), rngB.rand(5))), CM_BLUE, sk=0.16)
t(BX0 + 28.5, rows[0] - 2.9, r'$C_{expr}$', fs=6.2, bold=True)
t(BX0 + 33.0, rows[0] + 1.6, 'over non-LR genes only', fs=6.0, ha='left')
t(BX0 + 33.0, rows[0] - 1.0, r'$O(n_A n_B d^{\prime})$', fs=6.0, ha='left', color=SUB)

# row 2 — geometry
t(BX0 + 0.4, rows[1], 'geometry (GW)', fs=6.4, bold=True, ha='left')
spotgrid(BX0 + 17.5, rows[1], 6.0, 3.4, 5, 4, seed=31, connect=True)
arrow(BX0 + 21.2, rows[1], BX0 + 25.0, rows[1])
t(BX0 + 23.1, rows[1] + 2.0, 'KDTree / sqeucl', fs=6.0, color=SUB)
pmat(BX0 + 25.4, rows[1] - 1.8, 6.2, 3.6,
     np.abs(np.subtract.outer(np.sort(rngB.rand(6)), np.sort(rngB.rand(6)))), CM_ORG, sk=0.16)
t(BX0 + 28.5, rows[1] - 2.9, r'$C_{gw}$', fs=6.2, bold=True)
t(BX0 + 33.0, rows[1] + 1.6, r'$2\,D_A P D_B^{\top}$: depends on $P$', fs=6.0, ha='left')
t(BX0 + 33.0, rows[1] - 1.0, r'$O(n^3)$ per outer iteration', fs=6.0, ha='left', color=SUB)

# row 3 — ligand–receptor
t(BX0 + 0.4, rows[2], 'ligand-receptor', fs=6.4, bold=True, ha='left')
lrpair(BX0 + 16.0, rows[2] + 0.9, r=0.56); lrpair(BX0 + 19.0, rows[2] - 0.9, r=0.56)
arrow(BX0 + 21.2, rows[2], BX0 + 25.0, rows[2])
t(BX0 + 23.1, rows[2] + 2.0, 'min-max, $-\\log$', fs=6.0, color=SUB)
sp2 = rngB.rand(6, 6); sp2[sp2 < 0.6] = 0.0
pmat(BX0 + 25.4, rows[2] - 1.8, 6.2, 3.6, sp2, CM_PINK, sk=0.16)
t(BX0 + 28.5, rows[2] - 2.9, r'$C_{lr}$', fs=6.2, bold=True)
t(BX0 + 33.0, rows[2] + 1.6, r'$S_{lr}=\sum_{LR}w\,g_L^{A}g_R^{B}$', fs=6.0, ha='left')
t(BX0 + 33.0, rows[2] - 1.0, r'$O(n_A n_B|LR|)$', fs=6.0, ha='left', color=SUB)

# ============================================================ C  fusion & solver
header(2.0, 98.0, 96.0, 'C', 3, 'Cost fusion & Sinkhorn solver', GRN)
rngC = np.random.RandomState(15)

for yy, cm, lb in [(87.0, CM_BLUE, r'$\hat{C}_{expr}$'), (81.6, CM_PINK, r'$\hat{C}_{lr}$'),
                   (76.2, CM_ORG, r'$\hat{C}_{gw}$')]:
    pmat(3.0, yy - 1.7, 4.6, 3.4, rngC.rand(6, 6), cm, sk=0.18)
    t(8.3, yy, lb, fs=6.0, bold=True, ha='left')
ax.plot([13.9, 13.9], [76.2, 87.0], color='#C3CCD4', lw=0.6, zorder=2.5)
for yy in (87.0, 81.6, 76.2):
    ax.plot([13.6, 13.9], [yy, yy], color='#C3CCD4', lw=0.6, zorder=2.5)
arrow(13.9, 81.6, 14.35, 81.6)
ax.add_patch(Circle((15.4, 81.6), 1.0, fc='white', ec=INK, lw=0.55, zorder=4))
t(15.4, 81.6, r'$\oplus$', fs=7.0, z=5)
arrow(16.4, 81.6, 17.05, 81.6)

pmat(17.4, 76.0, 8.4, 12.0, np.abs(np.subtract.outer(np.sort(rngC.rand(8)),
                                                     np.sort(rngC.rand(8)))), CM_ORG, sk=0.18)
t(21.6, 74.0, r'fused cost $C_{tot}$', fs=6.4, bold=True)
arrow(25.9, 82.0, 27.6, 82.0)

pmat(29.2, 76.0, 7.4, 12.0,
     np.clip(np.exp(-2.6 * np.abs(np.subtract.outer(np.linspace(0, 1, 8),
                                                    np.linspace(0, 1, 8)))), 0, 1), CM_BLUE, sk=0.18)
t(33.0, 74.0, 'kernel', fs=6.4, bold=True)
t(33.0, 71.8, r'$K=\exp(-C_{tot}/\lambda)$', fs=6.0)
arrow(36.7, 82.0, 38.4, 82.0)

for k, p in enumerate([bandplan(s) for s in (2.60, 1.80, 1.15, 0.62)]):
    pmat(39.8 + k * 3.9, 80.2, 3.4, 3.4, p, CM_BLUE, sk=0.18)
    t(41.5 + k * 3.9, 78.4, r'$P^{(%s)}$' % ['0', '1', '2', 't'][k], fs=5.8)
arrow(39.8, 85.0, 54.9, 85.0, color=PUR, ls=(0, (3, 1.6)))
t(47.6, 86.2, 'alternating scaling until marginals match', fs=6.0, color=PUR)
t(47.0, 76.2, 'Sinkhorn inner loop ($\\leq$200)', fs=6.4, bold=True)
t(47.0, 74.0, r'$P\!\leftarrow\!\mathrm{diag}(s)\,K\,\mathrm{diag}(t)$', fs=5.8)
t(47.0, 72.0, r'init $P^{(0)}=ab^{\top}$, uniform $a,b$', fs=6.0, color=SUB)

pmat(58.6, 75.6, 15.0, 13.6, bandplan(2.0), CM_BLUE, sk=0.16)
arrow(56.4, 82.0, 58.2, 82.0)
t(66.9, 73.6, r'soft transport plan $P$', fs=7.0, bold=True)
t(66.9, 71.6, r'row-normalised $\bar{P}$ $\rightarrow$ confidence', fs=6.0, color=SUB)
larrow([(58.6, 89.2), (58.6, 91.4), (1.2, 91.4), (1.2, 76.2), (2.6, 76.2)], color=PUR, ls=(0, (4, 1.8)))
t(30.0, 92.2, r'outer loop: $C_{gw}$ recomputed from $P$', fs=6.0, color=PUR, z=6.5)

ax.add_patch(FancyBboxPatch((78.4, 72.2), 18.2, 18.0, boxstyle='round,pad=0.3,rounding_size=0.5',
                            fc='#FBFCFD', ec=HAIR, lw=0.7, zorder=1.5))
t(79.6, 87.2, 'cost fusion', fs=6.4, bold=True, ha='left')
t(79.6, 84.6, r'$C_{tot}=\alpha\,\hat{C}_{expr}$', fs=5.8, ha='left')
t(79.6, 82.0, r'$+\;\gamma\,\hat{C}_{lr}+(1-\alpha)\beta\,\hat{C}_{gw}$', fs=5.8, ha='left')
t(79.6, 79.4, 'entropic OT', fs=6.4, bold=True, ha='left')
t(79.6, 76.8, r'$\min_P\langle P,C_{tot}\rangle-\lambda H(P)$', fs=5.8, ha='left')
t(79.6, 74.2, r'$\lambda=\mathrm{reg}_{ot}+\beta\,\mathrm{reg}_{gw}$', fs=5.8, ha='left')

# ============================================================ D  calibration
header(2.0, 98.0, 69.6, 'D', 4, 'Uncertainty calibration & alignment', PUR)

ax.plot([4.0, 33.0], [47.6, 47.6], color='#B7C1CB', lw=0.5, zorder=3)
ax.fill_between([4.6, 10.6], 47.6, 58.6, color=GRN, alpha=0.09, lw=0, zorder=2)
gauss(18.5, 47.6, 26.0, 8.8, mu=-1.5, sd=1.00, color='#9AA6B2', lw=0.9)
gauss(18.5, 47.6, 26.0, 8.8, mu=-0.9, sd=0.72, color=BLUE, lw=1.1)
t(18.5, 62.6, r'row entropy $H_i$', fs=6.4, bold=True)
t(4.6, 55.2, 'low uncertainty', fs=5.8, color=GRN, ha='left')
t(18.5, 45.2, r'$\downarrow$ 12.3-17.6%   (mean 14.8%)', fs=6.0, color=SUB)
ax.plot([23.8, 26.2], [58.4, 58.4], color='#9AA6B2', lw=1.0, zorder=3)
t(26.8, 58.4, 'baseline', fs=5.8, color=SUB, ha='left')
ax.plot([23.8, 26.2], [56.0, 56.0], color=BLUE, lw=1.1, zorder=3)
t(26.8, 56.0, 'LROT', fs=5.8, color=BLUE, ha='left')

ax.plot([36.0, 65.0], [47.6, 47.6], color='#B7C1CB', lw=0.5, zorder=3)
ax.fill_between([57.2, 65.0], 47.6, 58.2, color=ORG, alpha=0.10, lw=0, zorder=2)
xs = np.linspace(36.0, 65.0, 160)
conf = 0.30 + 0.62 / (1 + np.exp(-(xs - 51.0) / 3.2))
ax.plot(xs, 47.6 + conf * 9.0, color=RED, lw=1.1, zorder=3.2)
ax.scatter([36.0, 65.0], [47.6 + 0.30 * 9.0, 47.6 + 0.92 * 9.0], s=6, c=[RED], zorder=3.4,
           edgecolors='white', linewidths=0.3)
t(50.5, 62.6, 'per-spot confidence', fs=6.4, bold=True)
t(50.5, 60.2, r'confidence = max layer prob. of $\bar{P}$', fs=6.0, color=SUB)
t(61.5, 54.6, 'retain top-20%', fs=6.0, color=ORG, ha='right')
t(50.5, 45.2, r'top-20%: 0.51 $\rightarrow$ 0.78', fs=6.0)
larrow([(41.2, 52.4), (44.6, 50.4)])

spotgrid(73.0, 52.2, 6.8, 6.2, 7, 6, seed=61, colors='#3E6FB0', s=2.4, jit=0.22)
spotgrid(74.9, 50.6, 6.8, 6.2, 7, 6, seed=62, colors='#E4574C', s=2.4, jit=0.22)
ax.add_patch(Rectangle((69.4, 48.8), 7.0, 6.6, fc='none', ec='#C3CCD4', lw=0.5, zorder=2))
t(73.0, 46.8, 'before', fs=6.0, color=SUB)
arrow(81.4, 52.2, 84.8, 52.2, color=RED, ls=(0, (3, 1.5)))
t(83.1, 54.0, 'LROT', fs=6.0, color=RED, bold=True)
spotgrid(91.0, 52.2, 6.8, 6.2, 7, 6, seed=61, colors='#3E6FB0', s=2.4, jit=0.22)
spotgrid(91.0, 52.2, 6.8, 6.2, 7, 6, seed=62, colors='#E4574C', s=2.4, jit=0.22)
ax.add_patch(Rectangle((87.4, 48.8), 7.0, 6.6, fc='none', ec='#C3CCD4', lw=0.5, zorder=2))
t(91.0, 46.8, 'after (aligned)', fs=6.0, color=SUB)

# ============================================================ E  downstream
header(2.0, 98.0, 42.6, 'E', 5, 'Downstream tasks & evaluation', RED)
rngE = np.random.RandomState(33)

t(4.4, 33.4, '3D reconstruction', fs=6.4, bold=True, ha='left')
slabs(11.0, 21.6, 9.0, 6.0, 6, 1.5, 1.35,
      [np.clip(0.3 + 0.7 * rngE.rand(7, 7), 0, 1) for _ in range(6)], CM_BLUE, sk=0.30)
t(17.6, 19.6, '6 serial sections', fs=6.0, color=SUB)
t(17.6, 17.4, r'$\rightarrow$ layer-consistent volume', fs=6.0, color=SUB)

bl, bv = ['Ncam1', 'Slit2', 'Erbb2', 'Bdnf'], [1.0, 0.72, 0.55, 0.38]
bx, bw = 39.0, 3.6
ax.plot([37.6, 62.0], [21.6, 21.6], color='#B7C1CB', lw=0.5, zorder=3)
for i, (lb, v) in enumerate(zip(bl, bv)):
    ax.add_patch(Rectangle((bx + i * (bw + 1.6), 21.6), bw, v * 10.2,
                           fc=[PINK, BLUE, ORG, PUR][i], ec='white', lw=0.3, zorder=3.2))
    t(bx + i * (bw + 1.6) + bw / 2, 20.2, lb, fs=5.8, color=SUB)
t(49.8, 33.4, 'top LR couplings', fs=6.4, bold=True)
t(38.2, 26.6, 'enrichment', fs=5.8, color=SUB, rot=90)
t(49.8, 17.4, 'top pairs: Ncam1-Ncam1, Slit2-Robo2, Bdnf-Ntrk2', fs=6.0, color=SUB)

t(68.0, 34.8, 'evaluation', fs=6.4, bold=True, ha='left')
chips(68.0, 28.2, 27.0, 2.1, [r'mapping accuracy ($\bar{P}$ argmax)', 'transport cost (cosine)'], BLUE)
chips(68.0, 23.0, 27.0, 2.1, ['transport entropy', 'runtime < 3 s / 1,000 spots'], GRN)
chips(68.0, 17.8, 27.0, 2.1, ['layer concordance (DLPFC)', r'training-free $\cdot$ no GPU'], RED)

# ============================================================ F  footer
ax.plot([3.2, 96.8], [12.6, 12.6], color=HAIR, lw=0.8, zorder=1)
t(3.2, 10.6, 'defaults', fs=6.4, bold=True, ha='left')
t(36.0, 10.6, 'solver settings', fs=6.4, bold=True, ha='left')
t(69.0, 10.6, 'complexity', fs=6.4, bold=True, ha='left')
t(3.2, 8.4, r'$\alpha=0.5$   expression-geometry', fs=6.0, ha='left')
t(3.2, 6.2, r'$\beta=0.3$   GW regularisation', fs=6.0, ha='left')
t(3.2, 4.0, r'$\gamma=0.1$   LR strength;  $\mathrm{reg}_{ot}=\mathrm{reg}_{gw}=0.01$', fs=6.0, ha='left')
t(36.0, 8.4, r'uniform marginals $a,b$', fs=6.0, ha='left')
t(36.0, 6.2, r'inner Sinkhorn scaling $\leq$200', fs=6.0, ha='left')
t(36.0, 4.0, r'outer stop $\|\Delta P\|/\|P\|<10^{-5}$ ($t>5$)', fs=6.0, ha='left')
t(69.0, 8.4, r'preprocessing $O(n_A n_B(d^{\prime}+|LR|))$', fs=6.0, ha='left')
t(69.0, 6.2, r'each outer iteration $O(n^3)$', fs=6.0, ha='left')
t(69.0, 4.0, r'Sinkhorn $O(S\,n_A n_B)$;  $\sim$3 s per pair', fs=6.0, ha='left')
key(3.2, 1.6, CM_BLUE, 'expression'); key(14.0, 1.6, CM_PINK, 'ligand-receptor'); key(26.0, 1.6, CM_ORG, 'geometry')
t(96.8, 1.6, 'reproducible: fixed seeds, no training, no GPU', fs=6.0, ha='right', color=SUB)

# ============================================================ cross-panel arrows
arrow(56.0, 99.6, 56.0, 97.2); arrow(50.0, 70.6, 50.0, 69.0); arrow(50.0, 43.6, 50.0, 41.6)

if DEBUG:
    n = 0
    for i in range(len(TB)):
        for j in range(i + 1, len(TB)):
            a, b = TB[i], TB[j]
            ox = min(a[2], b[2]) - max(a[0], b[0]); oy = min(a[3], b[3]) - max(a[1], b[1])
            if ox > 0 and oy > 0:
                aa = (a[2] - a[0]) * (a[3] - a[1]); ab = (b[2] - b[0]) * (b[3] - b[1])
                if ox * oy > 0.25 * min(aa, ab):
                    n += 1; print(f'  [text-overlap] {a[4]!r} <-> {b[4]!r} ({ox:.2f}x{oy:.2f})')
    print(f'  text overlaps: {n}')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=650, facecolor='white')
print('Saved:', OUT, '(%.2f x %.2f cm)' % (6.30 * 2.54, 6.30 * HU / WU * 2.54))
