# -*- coding: utf-8 -*-
"""
cancer_erbb2_ablation_truecoord.py — 乳腺癌 ERBB2/HER2 通路消融实验
动机: 检验乳腺癌上的熵降是否经由 ERBB2/HER2 通路产生——移除该家族后熵降是否减弱。
设计: 以乳腺癌特异性的 LR 通路(ERBB2/HER2)作消融对照, 判断熵降是否依赖该通路。

设计(与 run_cancer_visium.py 一致): 乳腺癌 Section1 vs Section2, 1,000 spots, 施加旋转2°/平移0.5。
三种 LR 库:
  1. full        : 完整 HUMAN_LR_DB
  2. no_erbb2    : 移除 ERBB2/HER2 家族对 (ERBB2-ERBB2, ERBB2-ERBB3, NRG1-ERBB3/4,
                    EGF/EREG/BTC/HBEGF-EGFR, TGFA-EGFR)
  3. erbb2_only  : 仅保留 ERBB2/HER2 家族对
比较: LROT(γ=0.1) vs FGW(γ=0) 的熵降, 判断熵降是否依赖 ERBB2 特异性信号。
输出: lrot_output/lrot_cancer_erbb2_ablation_truecoord_results.txt, lrot_cancer_erbb2_ablation_truecoord.png
"""
import os
import sys
import time
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
from lrot_core import fgw_lr_solver, compute_ot_cost, compute_mapping_consistency

OUT_DIR = os.path.join(_R, 'lrot_output')
N = 3000
GAMMA = 0.1
SEED_A, SEED_B = 42, 99

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ERBB2/HER2 家族对（消融组）
ERBB2_PAIRS = {("ERBB2", "ERBB2"), ("ERBB2", "ERBB3"), ("NRG1", "ERBB3"),
               ("NRG1", "ERBB4"), ("EGF", "EGFR"), ("EREG", "EGFR"),
               ("BTC", "EGFR"), ("HBEGF", "EGFR"), ("TGFA", "EGFR")}

buf = []
def log(s):
    print(s, flush=True)
    buf.append(s)

def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()

def build_db(variant):
    if variant == 'full':
        return HUMAN_LR_DB
    if variant == 'no_erbb2':
        return {k: v for k, v in HUMAN_LR_DB.items() if k not in ERBB2_PAIRS}
    if variant == 'erbb2_only':
        return {k: v for k, v in HUMAN_LR_DB.items() if k in ERBB2_PAIRS}
    raise ValueError(variant)

def main():
    log("=" * 72)
    log("  乳腺癌 ERBB2/HER2 通路消融实验")
    log("=" * 72)
    # 加载（与 run_cancer_visium 完全一致）
    A = download_and_prepare("V1_Breast_Cancer_Block_A_Section_1", N, SEED_A)
    B = download_and_prepare("V1_Breast_Cancer_Block_A_Section_2", N, SEED_B)
    B['coords'] = apply_misalignment(B['coords'], rotation_deg=2.0, translation=0.5)
    labels_A, labels_B = A['region_labels'], B['region_labels']
    log(f"  乳腺癌: {A['n_spots']} spots × {A['n_genes']} genes (Section1 vs Section2, 旋转2°/平移0.5)")

    variants = [('full', '完整LR库'), ('no_erbb2', '移除ERBB2/HER2'), ('erbb2_only', '仅ERBB2/HER2')]
    results = {}
    log(f"\n  {'版本':<16} {'对数':<6} {'熵':<10} {'熵降%':<10} {'成本':<10} {'精度':<8}")
    log(f"  {'-'*66}")
    for variant, desc in variants:
        db = build_db(variant)
        S, top = compute_lr_strength_human(A, B, db)
        n_eff = len(top)
        ligs = sorted(set(k[0] for k in db) | set(k[1] for k in db))

        # FGW 基线 (γ=0)：按基线口径用**全部基因**做 C_expr（不传 lr_gene_list）。
        # 理由同主实验：γ=0 无 C_lr 项 ⇒ 无重复计数问题，且为更保守的强基线。
        P0, _ = fgw_lr_solver(A, B, np.zeros_like(S), gamma=0.0, beta=0.3, alpha=0.5,
                              max_iter=200, verbose=False)
        ent0 = mean_entropy(P0); cost0 = compute_ot_cost(P0, A['expr'], B['expr'])
        acc0 = compute_mapping_consistency(P0, labels_A, labels_B)

        # LROT
        Pl, _ = fgw_lr_solver(A, B, S, gamma=GAMMA, beta=0.3, alpha=0.5,
                              max_iter=200, verbose=False, lr_gene_list=ligs)
        entl = mean_entropy(Pl); costl = compute_ot_cost(Pl, A['expr'], B['expr'])
        accl = compute_mapping_consistency(Pl, labels_A, labels_B)
        imp = (ent0 - entl) / ent0 * 100
        results[variant] = {'ent0': ent0, 'entl': entl, 'imp': imp,
                            'cost0': cost0, 'costl': costl, 'acc0': acc0, 'accl': accl,
                            'n_eff': n_eff, 'top': top[:5]}
        log(f"  {desc:<16} {n_eff:<6} {entl:<10.4f} {imp:<10.3f} {costl:<10.4f} {accl:<8.4f}")

    log("\n" + "=" * 72)
    log("  熵降对比（相对 FGW 基线）")
    log("=" * 72)
    full_imp = results['full']['imp']
    no_imp = results['no_erbb2']['imp']
    only_imp = results['erbb2_only']['imp']
    log(f"  完整LR库:   熵降 {full_imp:.3f}%")
    log(f"  移除ERBB2:  熵降 {no_imp:.3f}%")
    log(f"  仅ERBB2:    熵降 {only_imp:.3f}%")
    log(f"  Top LR 信号: 完整库={[f'{l}-{r}' for l,r,w in results['full']['top']]}")
    log(f"               仅ERBB2={[f'{l}-{r}' for l,r,w in results['erbb2_only']['top']]}")

    log("\n" + "=" * 72)
    log("  结论")
    log("=" * 72)
    if no_imp < full_imp - 0.1:
        log(f"  移除 ERBB2/HER2 通路后熵降由 {full_imp:.2f}% 降至 {no_imp:.2f}%——"
            f"熵降显著依赖乳腺癌特异的 ERBB2 信号, 表明 LR 引导在乳腺癌中确通过 ERBB2/HER2 通路起作用")
    else:
        log(f"  移除 ERBB2/HER2 通路后熵降基本不变 ({full_imp:.2f}% → {no_imp:.2f}%)——"
            f"说明该小幅熵降不依赖 ERBB2 特异性信号")
    log(f"  仅 ERBB2 通路单独引导熵降 {only_imp:.2f}%{' (接近完整库)' if abs(only_imp-full_imp)<0.3 else ' (弱于完整库)'}——"
        f"验证 ERBB2 是否为乳腺癌 LR 引导的关键信号")

    # 保存
    with open(os.path.join(OUT_DIR, 'lrot_cancer_erbb2_ablation_truecoord_results.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(buf) + "\n")

    # 图
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    labels = ['Full LR library', 'Remove ERBB2/HER2', 'ERBB2/HER2 only']
    imps = [full_imp, no_imp, only_imp]
    colors = ['#2166AC', '#D6604D', '#4DAF4A']
    bars = ax.bar(labels, imps, color=colors, alpha=0.85, width=0.6)
    for b, v in zip(bars, imps):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}%', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Transport-entropy reduction (vs FGW, %)')
    ax.set_title('(A) Breast-cancer ERBB2/HER2 pathway ablation: source of entropy reduction')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylim(min(imps)-0.4, max(imps)+0.5)
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'lrot_cancer_erbb2_ablation_truecoord.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    log(f"\n  图已保存: {os.path.join(OUT_DIR, 'lrot_cancer_erbb2_ablation_truecoord.png')}")

if __name__ == '__main__':
    main()
