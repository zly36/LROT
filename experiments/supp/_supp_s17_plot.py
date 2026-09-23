# -*- coding: utf-8 -*-
"""Plot the support checks behind Supplementary Fig. S17. (restyled to match main-figure look)"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 自定位仓库根：向上找到含 lrot_core.py 的目录（代码包解压到任意路径均可运行）
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

OUT = Path(_R, 'lrot_output')
sys.path.insert(0, _R)
from lrot_paths import find_input                                    # noqa: E402
# 三个 json 位于 lrot_output/_中间产物/S17_support/，用多候选查找而非写死路径
sweep = json.load(open(find_input("baseline_ceiling_sweep.json"), encoding="utf-8"))
eps = json.load(open(find_input("eps_sensitivity.json"), encoding="utf-8"))
ctl = json.load(open(find_input("lr_program_control.json"), encoding="utf-8"))

syn = sweep["synthetic"]["rows"]
rea = sweep["realistic"]["rows"]

def default(r, gamma, a=0.5, b=0.3):
    return next(x for x in r if abs(x["gamma"] - gamma) < 1e-9
                and abs(x["alpha"] - a) < 1e-9 and abs(x["beta"] - b) < 1e-9)

def best(r, gamma):
    return max([x for x in r if abs(x["gamma"] - gamma) < 1e-9], key=lambda x: x["acc"])

fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))

# ---- 统一方法配色: FGW 红 / LROT 绿 ----
FGW, LROT = "#E74C3C", "#2ECC71"
LRP, NOLR = "#4E79A7", "#8C8C8C"   # 面板F: +LR程序(蓝) / 无LR(灰)
INK = "#333333"

# ---------- (A,B) baseline-ceiling bars ----------
for ax, rows, title, note, L in [
    (axes[0, 0], syn, "Synthetic 4-region (seed 42/123)", "acc@default: FGW 0.363, LROT 0.631", "A"),
    (axes[0, 1], rea, "Realistic 6-region (seed 42)", "acc@default: FGW 0.711, LROT 0.712", "B"),
]:
    dF = default(rows, 0.0); bF = best(rows, 0.0)
    dL = default(rows, 0.1); bL = best(rows, 0.1)
    vals = [dF["acc"], bF["acc"], dL["acc"], bL["acc"]]
    ents = [dF["entropy"], bF["entropy"], dL["entropy"], bL["entropy"]]
    labels = ["FGW default\n(γ=0)", "FGW best\ngrid (γ=0)", "LROT default\n(γ=0.1)", "LROT best\ngrid (γ=0.1)"]
    # default=方法色, best=方法色加深(不用浅色/斜纹); 白边分隔相邻柱
    FGW_B, LROT_B = "#C0392B", "#1E8449"
    cols = [FGW, FGW_B, LROT, LROT_B]
    bars = ax.bar(range(4), vals, color=cols, width=0.62, edgecolor="white",
                  linewidth=0.8, zorder=3)
    for i, (v, e) in enumerate(zip(vals, ents)):
        ax.text(i, v + 0.02, "%.3f" % v, ha="center", fontsize=8.6, fontweight="bold", zorder=4)
        ax.text(i, 0.12, "ent=%.2f" % e, ha="center", fontsize=7.2, color="white", zorder=4)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylim(0, 1.06); ax.set_ylabel("Region mapping accuracy")
    ax.set_title("%s: %s" % (L, title), fontsize=11, fontweight="bold")
    ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=7.8,
            va="top", color=INK)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

# ---------- (C) eps sensitivity ----------
ax = axes[0, 2]
mults = sorted({r["eps_mult"] for r in eps["rows"]})
for gamma, color, lab in [(0.0, FGW, "FGW (γ=0)"), (0.1, LROT, "LROT (γ=0.1)")]:
    sub = sorted([r for r in eps["rows"] if abs(r["gamma"] - gamma) < 1e-9],
                 key=lambda r: r["eps_mult"])
    ax.plot([r["eps_mult"] for r in sub], [r["entropy"] for r in sub],
            marker="o", color=color, label=lab, lw=1.8, zorder=3)
    for k, r in enumerate(sub):
        dy = 10 if k % 2 == 0 else -15          # 交替上/下, 避免标注重叠
        ax.annotate("acc=%.3f" % r["acc"], (r["eps_mult"], r["entropy"]),
                    textcoords="offset points", xytext=(4, dy), fontsize=6.8)
drop = []
for r in sorted(eps["rows"], key=lambda r: r["eps_mult"]):
    if abs(r["gamma"] - 0.1) < 1e-9:
        fg = next(x for x in eps["rows"] if x["eps_mult"] == r["eps_mult"] and x["gamma"] == 0.0)
        drop.append(100 * (fg["entropy"] - r["entropy"]) / fg["entropy"])
ax.text(0.5, 0.02, "entropy reduction: 3.8% / 2.7% / 1.1% for ε×0.5 / ×1 / ×2",
        transform=ax.transAxes, fontsize=7.8, color=INK)
ax.set_xlabel("Entropic regularisation multiplier (ε×)")
ax.set_ylabel("Mean transport entropy")
ax.set_title("C: ε-sensitivity (realistic, seed 42)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

# ---------- (D,E) LR-program control: accuracy ----------
for col, ds, title, L in [(0, "synthetic", "Synthetic control (3 seeds)", "D"),
                          (1, "realistic", "Realistic control (3 seeds)", "E")]:
    ax = axes[1, col]
    data = ctl[ds]
    xpos = np.arange(4)
    means, errs = [], []
    for prog in [True, False]:
        for gamma in [0.0, 0.1]:
            sub = [r["acc"] for r in data if r["lr_program"] == prog and abs(r["gamma"] - gamma) < 1e-9]
            means.append(float(np.mean(sub))); errs.append(float(np.std(sub)))
    # 0,2 = FGW(红) ; 1,3 = LROT(绿)
    cols = [FGW, LROT, FGW, LROT]
    bars = ax.bar(xpos, means, yerr=errs, color=cols, width=0.6,
                  edgecolor="white", linewidth=0.6,
                  error_kw=dict(lw=0.9, capsize=3, color="#555555"))
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(i, m + e + 0.03, "%.3f" % m, ha="center", fontsize=8.4, fontweight="bold")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["LR program\nFGW", "LR program\nLROT", "no LR program\nFGW", "no LR program\nLROT"],
                       fontsize=8)
    ytop = max([mm + ee for mm, ee in zip(means, errs)]) + 0.18
    ax.set_ylim(0, max(ytop, 1.05))
    ax.set_ylabel("Region mapping accuracy")
    ax.set_title("%s: %s" % (L, title), fontsize=11, fontweight="bold")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

# ---------- (F) entropy reduction across settings ----------
ax = axes[1, 2]
settings = ["Synthetic\n+LR program", "Synthetic\nno LR program",
            "Realistic\n+LR program", "Realistic\nno LR program"]
reductions, errs = [], []
for ds in ["synthetic", "realistic"]:
    for prog in [True, False]:
        drops = []
        for seed in [42, 43, 44]:
            fg = next(r for r in ctl[ds] if r["seed"] == seed and r["lr_program"] == prog and abs(r["gamma"] - 0.0) < 1e-9)
            lr = next(r for r in ctl[ds] if r["seed"] == seed and r["lr_program"] == prog and abs(r["gamma"] - 0.1) < 1e-9)
            drops.append(100 * (fg["entropy"] - lr["entropy"]) / fg["entropy"])
        reductions.append(float(np.mean(drops))); errs.append(float(np.std(drops)))
# +LR程序=蓝, 无LR=灰(实心, 不用浅色); synthetic vs realistic 由横轴标签区分
cols = [LRP, NOLR, LRP, NOLR]
bars = ax.bar(range(4), reductions, yerr=errs, color=cols,
              width=0.6, edgecolor="white", linewidth=0.6,
              error_kw=dict(lw=0.9, capsize=3, color="#555555"))
for i, (r, e) in enumerate(zip(reductions, errs)):
    ax.text(i, r + e + 0.45, "%.1f%%" % r, ha="center", fontsize=8.6, fontweight="bold")
ytF = max([x + e for x, e in zip(reductions, errs)]) + 1.3
ax.set_ylim(0, ytF)
ax.set_xticks(range(4))
ax.set_xticklabels(settings, fontsize=7.8, rotation=15, ha="right")
ax.set_ylabel("Entropy reduction of LROT vs FGW (%)")
ax.set_title("F: LR-program control: entropy reduction", fontsize=11, fontweight="bold")
import matplotlib.patches as mpatches
ax.legend(handles=[mpatches.Patch(color=LRP, label="+ LR program"),
                   mpatches.Patch(color=NOLR, label="no LR program")],
          fontsize=8, frameon=False, loc="upper left")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

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
path = OUT / "lrot_supp_S17_boundary_checks.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
print("saved", path)
