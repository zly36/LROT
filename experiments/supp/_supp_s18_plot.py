# -*- coding: utf-8 -*-
"""Plot the two-donor DLPFC replication as Supplementary Fig. S18. (restyled to match main-figure look)"""
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 自定位仓库根：向上找到含 lrot_core.py 的目录（代码包解压到任意路径均可运行）
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

OUT = Path(_R, 'lrot_output')

# (label, donor, FGW ent, LROT ent, red%, FGW mass, LROT mass, FGW ari, LROT ari, sweep[(g,ent,ari,mass)])
D = [
    ("151507-151508", "Donor 1 (Br8325)", 6.0119, 4.9524, 17.62, 0.3766, 0.3742, 0.416, 0.344,
     [(0,6.0119,0.416,0.3766),(0.02,5.9380,0.415,0.3786),(0.05,5.6125,0.397,0.3790),(0.1,4.9524,0.344,0.3742),(0.15,4.5389,0.310,0.3683),(0.2,4.3150,0.292,0.3634)]),
    ("151508-151509", "Donor 1 (Br8325)", 6.0230, 5.0401, 16.32, 0.3478, 0.3476, 0.191, 0.152,
     [(0,6.0230,0.191,0.3478),(0.02,5.9554,0.183,0.3496),(0.05,5.6544,0.167,0.3503),(0.1,5.0401,0.152,0.3476),(0.15,4.6643,0.134,0.3435),(0.2,4.4717,0.118,0.3399)]),
    ("151669-151670", "Donor 2 (Br6423)", 6.2253, 5.4151, 13.01, 0.55745, 0.56281, 0.34468, 0.39972,
     [(0,6.22532,0.34468,0.55745),(0.02,6.15533,0.37273,0.56114),(0.05,5.90079,0.37731,0.56353),(0.1,5.41510,0.39972,0.56281),(0.15,5.09535,0.39157,0.56002),(0.2,4.90479,0.37612,0.55743)]),
    ("151670-151671", "Donor 2 (Br6423)", 6.3551, 5.5757, 12.26, 0.50086, 0.49797, 0.54273, 0.54112,
     [(0,6.35507,0.54273,0.50086),(0.02,6.29419,0.55057,0.50302),(0.05,6.04835,0.56300,0.50311),(0.1,5.57569,0.54112,0.49797),(0.15,5.27652,0.49930,0.49067),(0.2,5.10693,0.45339,0.48505)]),
]

# ---- 统一方法配色: FGW 红 / LROT 绿 ----
FGW, LROT = "#E74C3C", "#2ECC71"
D1C, D2C = "#4E79A7", "#F28E2B"   # donor1 蓝实线 / donor2 橙虚线(面板b-d)
MASS, ARI = "#4E79A7", "#8E44AD"   # Δ同层质量 / Δ软ARI(面板e-f)
INK = "#222222"

fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.4))
xlabs = ["D1-P1\n151507-508", "D1-P2\n151508-509", "D2-P1\n151669-670", "D2-P2\n151670-671"]

def donor_style(i):
    """按 i -> (donor dd, pair pp): 颜色随 donor、线型随 donor、marker随 pair"""
    dd = 0 if i < 2 else 1
    pp = i % 2
    color = D1C if dd == 0 else D2C
    ls = "-" if dd == 0 else "--"
    mark = "o" if pp == 0 else "s"
    return color, ls, mark

# (A) entropy at default: FGW vs LROT
ax = axes[0, 0]
x = np.arange(4)
fg = [d[2] for d in D]; lr = [d[3] for d in D]
ax.bar(x - 0.19, fg, width=0.38, color=FGW, label="FGW (γ=0)", edgecolor="white", linewidth=0.5)
ax.bar(x + 0.19, lr, width=0.38, color=LROT, label="LROT (γ=0.1)", edgecolor="white", linewidth=0.5)
for i in range(4):
    top = max(fg[i], lr[i])
    ax.text(i - 0.19, fg[i] + 0.12, "%.2f" % fg[i], ha="center", fontsize=7.8, fontweight="bold")
    ax.text(i + 0.19, lr[i] + 0.12, "%.2f" % lr[i], ha="center", fontsize=7.8, fontweight="bold")
    ax.text(i, top + 0.55, "-%.1f%%" % D[i][4], ha="center", fontsize=8.6, color=INK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(xlabs, fontsize=8.2)
ax.set_ylabel("Transport entropy (nats)")
ax.set_ylim(0, 7.95)
ax.set_title("A: Entropy at default γ", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, frameon=False)
for s in ["top", "right"]: ax.spines[s].set_visible(False)

# (B) normalized entropy gamma sweeps
ax = axes[0, 1]
for i, d in enumerate(D):
    g = [v[0] for v in d[9]]; e = [100 * v[1] / d[9][0][1] for v in d[9]]
    color, ls, mark = donor_style(i)
    ax.plot(g, e, marker=mark, color=color, ls=ls, lw=1.8, ms=4.5,
            label="%s %s" % (d[1][:8], d[0]))
ax.axhline(100, color="#BBBBBB", lw=0.8, zorder=0)
ax.set_ylim(60, 102)
ax.set_xlabel("γ")
ax.set_ylabel("Entropy (% of γ=0)")
ax.set_title("B: γ-sweep: entropy (normalised)", fontsize=11, fontweight="bold")
ax.legend(fontsize=7.2, frameon=False, loc="lower left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)

# (C) same-layer mass gamma sweeps
ax = axes[0, 2]
for i, d in enumerate(D):
    g = [v[0] for v in d[9]]; m = [v[3] for v in d[9]]
    color, ls, mark = donor_style(i)
    ax.plot(g, m, marker=mark, color=color, ls=ls, lw=1.8, ms=4.5,
            label="%s %s" % (d[1][:8], d[0]))
ax.set_xlabel("γ")
ax.set_ylabel("Same-layer mass")
ax.set_title("C: γ-sweep: same-layer mass", fontsize=11, fontweight="bold")
ax.legend(fontsize=7.2, frameon=False)
for s in ["top", "right"]: ax.spines[s].set_visible(False)

# (D) soft ARI gamma sweeps
ax = axes[1, 0]
for i, d in enumerate(D):
    g = [v[0] for v in d[9]]; a = [v[2] for v in d[9]]
    color, ls, mark = donor_style(i)
    ax.plot(g, a, marker=mark, color=color, ls=ls, lw=1.8, ms=4.5,
            label="%s %s" % (d[1][:8], d[0]))
ax.set_xlabel("γ")
ax.set_ylabel("Soft ARI")
ax.set_title("D: γ-sweep: soft ARI", fontsize=11, fontweight="bold")
ax.legend(fontsize=7.2, frameon=False)
for s in ["top", "right"]: ax.spines[s].set_visible(False)

# (E) deltas at default gamma=0.1
ax = axes[1, 1]
dm = [d[5] - d[6] for d in D]  # FGW same-mass - LROT same-mass
da = [d[7] - d[8] for d in D]  # FGW ari - LROT ari
ax.axhline(0, color="#999999", lw=1.0)
ax.bar(x - 0.19, dm, width=0.38, color=MASS, edgecolor="white", linewidth=0.5)
ax.bar(x + 0.19, da, width=0.38, color=ARI, edgecolor="white", linewidth=0.5)
for i in range(4):
    ax.text(i - 0.19, dm[i] + (0.004 if dm[i] >= 0 else -0.008), "%+.4f" % dm[i],
            ha="center", fontsize=7.2, fontweight="bold")
    ax.text(i + 0.19, da[i] + (0.006 if da[i] >= 0 else -0.010), "%+.3f" % da[i],
            ha="center", fontsize=7.2, fontweight="bold")
ylo = min(0, min(dm), min(da)) - 0.015
ylo -= 0.012   # 给负向标注额外空间
yhi = max(0, max(dm), max(da)) + 0.015
yhi += 0.012
ax.set_ylim(ylo, yhi)
ax.set_xticks(x); ax.set_xticklabels(xlabs, fontsize=8.2)
ax.set_ylabel("Δ at γ=0.1 (vs γ=0)")
ax.set_title("E: Changes at default (γ=0.1)", fontsize=11, fontweight="bold")
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=MASS, label="Δ same-layer mass"),
                   mpatches.Patch(color=ARI, label="Δ soft ARI")],
          fontsize=8.0, frameon=False, loc="lower left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)

# (F) deltas at gamma=0.05 (sweet spot)
ax = axes[1, 2]
dm5, da5 = [], []
for d in D:
    s0 = d[9][0]; s5 = d[9][2]
    dm5.append(s0[3] - s5[3]); da5.append(s0[2] - s5[2])
ax.axhline(0, color="#999999", lw=1.0)
ax.bar(x - 0.19, dm5, width=0.38, color=MASS, edgecolor="white", linewidth=0.5)
ax.bar(x + 0.19, da5, width=0.38, color=ARI, edgecolor="white", linewidth=0.5)
for i in range(4):
    ax.text(i - 0.19, dm5[i] + (0.004 if dm5[i] >= 0 else -0.008), "%+.4f" % dm5[i],
            ha="center", fontsize=7.2, fontweight="bold")
    ax.text(i + 0.19, da5[i] + (0.006 if da5[i] >= 0 else -0.010), "%+.3f" % da5[i],
            ha="center", fontsize=7.2, fontweight="bold")
ylo = min(0, min(dm5), min(da5)) - 0.027
yhi = max(0, max(dm5), max(da5)) + 0.027
ax.set_ylim(ylo, yhi)
ax.set_xticks(x); ax.set_xticklabels(xlabs, fontsize=8.2)
ax.set_ylabel("Δ at γ=0.05 (vs γ=0)")
ax.set_title("F: Changes at sweet spot (γ=0.05)", fontsize=11, fontweight="bold")
ax.legend(handles=[mpatches.Patch(color=MASS, label="Δ same-layer mass"),
                   mpatches.Patch(color=ARI, label="Δ soft ARI")],
          fontsize=8.0, frameon=False, loc="lower left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)

fig.tight_layout()

# 全面板统一: 浅灰网格 + 细灰边框(对齐校准图标准)
for a in axes.flat:
    for s in ["top", "right", "left", "bottom"]:
        a.spines[s].set_visible(True)
        a.spines[s].set_color("#AAAAAA")
        a.spines[s].set_linewidth(0.9)
    a.grid(True, color="#DADADA", lw=0.7, alpha=0.9)
    a.set_axisbelow(True)
fig.tight_layout()
path = OUT / "lrot_supp_S18_donor_replication.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
print("saved", path)
