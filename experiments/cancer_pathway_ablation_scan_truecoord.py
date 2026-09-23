# -*- coding: utf-8 -*-
"""
cancer_pathway_ablation_scan_truecoord.py — 乳腺癌通路级系统消融扫描（leave-one-out）
动机: ERBB2 单家族消融后熵降不降反增, 因此对全部家族做 leave-one-out 扫描, 看熵降与通路归属的关系。

做法: 对所有有效表达的对/信号家族做 leave-one-out 系统扫描,
      完整报告消融谱: 哪些移除后熵降减弱(正贡献), 哪些增强(负贡献/干扰)。

指标: 熵降%(相对FGW基线)。full 为完整库; 逐个移除后对比。
输出: lrot_output/lrot_cancer_ablation_scan_truecoord_results.txt, lrot_cancer_ablation_scan_truecoord.png
"""
import os
import sys
import warnings
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

from run_cancer_visium_truecoord import (load_breast_slice as download_and_prepare, apply_misalignment,
                               compute_lr_strength_human, HUMAN_LR_DB, HUMAN_ALL_LR)
from lrot_core import fgw_lr_solver, compute_ot_cost

OUT_DIR = os.path.join(_R, 'lrot_output')
N = 3000
GAMMA = 0.1
SEED_A, SEED_B = 42, 99

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

buf = []
def log(s):
    print(s, flush=True)
    buf.append(s)

def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()

def compute_ent0(A, B):
    """FGW 基线熵（只算一次, 与移除哪个对无关）。

    基线口径：按主实验规则用**全部基因**做 C_expr（不传 lr_gene_list）。
    γ=0 时无 C_lr 项 ⇒ 无重复计数问题，且为更保守的强基线。
    """
    S0 = np.zeros((A['n_spots'], B['n_spots']))
    P0, _ = fgw_lr_solver(A, B, S0, gamma=0.0, beta=0.3, alpha=0.5,
                          max_iter=200, verbose=False)
    return mean_entropy(P0)

def run_lrot(A, B, db):
    S, top = compute_lr_strength_human(A, B, db)
    ligs = sorted(set(k[0] for k in db) | set(k[1] for k in db))
    Pl, _ = fgw_lr_solver(A, B, S, gamma=GAMMA, beta=0.3, alpha=0.5,
                          max_iter=200, verbose=False, lr_gene_list=ligs)
    return mean_entropy(Pl), top

def run_imp(ent0, entl):
    return (ent0 - entl) / ent0 * 100

# 互斥的信号家族定义（基于 HUMAN_LR_DB 的注释分组；EGF-EGFR/TGFA-EGFR 归入 ERBB2/HER 家族）
FAMILIES = {
    'ERBB2/HER': {("ERBB2","ERBB2"), ("ERBB2","ERBB3"), ("EREG","EGFR"), ("BTC","EGFR"),
                  ("HBEGF","EGFR"), ("NRG1","ERBB3"), ("NRG1","ERBB4"), ("EGF","EGFR"), ("TGFA","EGFR")},
    'Neurotrophin/Growth': {("NTNG1","NTRK2"), ("BDNF","NTRK2"), ("NTF3","NTRK3"), ("NGF","NTRK1"),
                            ("GDNF","GFRA1"), ("ARTN","GFRA3"), ("NRTN","GFRA2"), ("FGF8","FGFR1"),
                            ("FGF9","FGFR2"), ("FGF10","FGFR2"), ("FGF2","FGFR1"), ("HGF","MET"),
                            ("VEGFA","KDR"), ("VEGFB","FLT1"), ("PDGFA","PDGFRA"), ("PDGFB","PDGFRB"),
                            ("IGF1","IGF1R"), ("IGF2","IGF1R")},
    'Wnt': {("WNT3A","FZD1"), ("WNT5A","FZD5"), ("WNT7A","FZD10"), ("WNT1","FZD1")},
    'Notch': {("DLL1","NOTCH1"), ("DLL4","NOTCH4"), ("JAG1","NOTCH1"), ("JAG2","NOTCH2")},
    'Ephrin': {("EFNB1","EPHB2"), ("EFNA1","EPHA4"), ("EFNB2","EPHB4"), ("EFNA5","EPHA3")},
    'Semaphorin': {("SEMA3A","NRP1"), ("SEMA3F","NRP2"), ("SEMA4D","PLXNB1"), ("SEMA6A","PLXNA2")},
    'Slit/Robo': {("SLIT1","ROBO1"), ("SLIT2","ROBO2"), ("SLIT3","ROBO2")},
    'Chemokine/Cytokine': {("CXCL12","CXCR4"), ("CXCL13","CXCR5"), ("CCL2","CCR2"), ("CCL5","CCR5"),
                           ("IL1B","IL1R1"), ("TNF","TNFRSF1A"), ("TGFB1","TGFBR1"), ("TGFB2","TGFBR2")},
    'BMP': {("BMP4","BMPR1A"), ("BMP7","BMPR2"), ("BMP2","ACVR1")},
    'Cell Adhesion': {("CDH1","CDH1"), ("CDH2","CDH2"), ("NCAM1","NCAM1"), ("L1CAM","L1CAM"),
                      ("COL1A1","ITGA2"), ("VTN","ITGAV")},
    'Hedgehog': {("SHH","PTCH1"), ("IHH","PTCH2"), ("DHH","PTCH1")},
    'Reelin': {("RELN","LRP8"), ("RELN","VLDLR")},
}

def main():
    log("=" * 72)
    log("  乳腺癌通路级系统消融扫描（leave-one-out）")
    log("=" * 72)
    A = download_and_prepare("V1_Breast_Cancer_Block_A_Section_1", N, SEED_A)
    B = download_and_prepare("V1_Breast_Cancer_Block_A_Section_2", N, SEED_B)
    B['coords'] = apply_misalignment(B['coords'], rotation_deg=2.0, translation=0.5)

    # FGW 基线熵 (只算一次)
    ent0 = compute_ent0(A, B)
    log(f"  FGW 基线熵: {ent0:.4f}")
    # 完整库 baseline
    entl_full, full_top = run_lrot(A, B, HUMAN_LR_DB)
    full_imp = run_imp(ent0, entl_full)
    log(f"  完整LR库({len(HUMAN_LR_DB)}对): 熵降 {full_imp:.3f}%")
    log(f"  Top LR 信号: {[f'{l}-{r}' for l,r,w in full_top[:5]]}")

    # 家族级消融: 移除每个信号家族
    log("\n  家族级消融: 移除每个信号家族后的熵降% (Δ=熵降-完整库)")
    log(f"  {'移除的家族':<24} {'对数':<6} {'熵降%':<10} {'Δ':<10} {'判定':<10}")
    log(f"  {'-'*62}")
    rows = []
    for fname, pairs in FAMILIES.items():
        db = {k: v for k, v in HUMAN_LR_DB.items() if k not in pairs}
        entl, _ = run_lrot(A, B, db)
        imp = run_imp(ent0, entl)
        d = imp - full_imp
        verdict = '正贡献' if d < -0.1 else ('负贡献/干扰' if d > 0.1 else '中性')
        rows.append((fname, len(pairs), imp, d, verdict))
        log(f"  {fname:<24} {len(pairs):<6} {imp:<10.3f} {d:<+10.3f} {verdict}")

    # 排序
    rows_sorted = sorted(rows, key=lambda x: x[3])
    log("\n" + "=" * 72)
    log("  消融谱（按Δ升序：最左=移除后熵降最弱=正贡献最大）")
    log("=" * 72)
    for fname, npair, imp, d, v in rows_sorted:
        log(f"  {fname:<24} 熵降{imp:.3f}% Δ={d:+.3f}% {v}")
    pos = [r for r in rows_sorted if r[3] < -0.1]
    neg = [r for r in rows_sorted if r[3] > 0.1]
    log(f"\n  正贡献家族(移除后熵降减弱): {len(pos)} 个 → {[r[0] for r in pos] if pos else '无'}")
    log(f"  负贡献家族(移除后熵降增强): {len(neg)} 个 → {[r[0] for r in neg] if neg else '无'}")
    log(f"  中性家族: {len(rows)-len(pos)-len(neg)} 个")
    log("\n  结论: 若存在正贡献家族, 可报告其为乳腺癌 LR 引导熵降的主要来源之一;")
    log("  消融谱(含负贡献/中性)全部随文报告。若所有家族都是中性/负贡献,")
    log("  则乳腺癌熵降过弱(噪声级)无法归因到单一家族, 这一结果同样如实报告。")

    # 保存
    with open(os.path.join(OUT_DIR, 'lrot_cancer_ablation_scan_truecoord_results.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(buf) + "\n")

    # 图
    fig, ax = plt.subplots(figsize=(9, 4.6))
    labels = [r[0] for r in rows_sorted]
    ds = [r[3] for r in rows_sorted]
    colors = ['#4DAF4A' if d < -0.1 else ('#D6604D' if d > 0.1 else '#888888') for d in ds]
    bars = ax.bar(labels, ds, color=colors, alpha=0.85, width=0.6)
    for b, d in zip(bars, ds):
        ax.text(b.get_x() + b.get_width()/2, d + (0.03 if d >= 0 else -0.06), f'{d:+.2f}',
                ha='center', fontsize=8)
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(full_imp, color='gray', ls='--', lw=1)
    ax.text(len(labels)-0.5, full_imp + 0.03, f'Full-library reduction {full_imp:.2f}%', fontsize=8, color='gray', ha='right')
    ax.set_xlabel('Removed signaling family (leave-one-family-out)')
    ax.set_ylabel('\u0394 entropy reduction (vs full library, %)')
    ax.set_title('(B) Breast-cancer pathway ablation spectrum: green=reduction weakens (positive), red=interference, gray=neutral')
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'lrot_cancer_ablation_scan_truecoord.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    log(f"\n  图已保存: {os.path.join(OUT_DIR, 'lrot_cancer_ablation_scan_truecoord.png')}")

if __name__ == '__main__':
    main()
