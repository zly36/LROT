# -*- coding: utf-8 -*-
"""
pathway_activity_dlpfc.py — 在 DLPFC（真实层标注）上重复"通路活性空间相关性"γ扫描
目标: 判断是否存在"过拟合阈值"——γ>0.1 后非 LR 相关性显著下降 且 层一致性开始下降？

方法（与图S4一致, 但真实数据 + 人类LR库 + 层一致性监控）:
  通路活性        = 通路内基因表达均值 (DLPFC, 人类 HUGO 大写符号)
  通路活性空间相关 = pearson(A 通路活性, Pn @ B 通路活性)   (对齐保真度)
  层一致性        = eval_plan: 同层质量 / 软ARI / 软准确率 / 传输熵
γ 扫描 {0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5} × 两对连续切片 (151507→151508, 151508→151509)
输出: lrot_output/lrot_pathway_dlpfc_results.txt, lrot_output/lrot_pathway_dlpfc.png
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
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_dlpfc_visium import load_dlpfc_slice, eval_plan
from run_cancer_visium import (HUMAN_LR_DB, HUMAN_ALL_LR, apply_misalignment,
                               compute_lr_strength_human)
from lrot_core import fgw_lr_solver

OUT_DIR = os.path.join(_R, 'lrot_output')
N = 1000
PAIRS = [("151507", "151508"), ("151508", "151509")]
GAMMAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
LIGS = sorted(set(k[0] for k in HUMAN_LR_DB) | set(k[1] for k in HUMAN_LR_DB))

# 人类通路基因（HUGO 大写，与 HUMAN_LR_DB 对应）
PATHWAYS = {
    'Neurotrophin / Growth Factor': [
        'NTNG1', 'NTRK2', 'BDNF', 'NTF3', 'NTRK3', 'NGF', 'NTRK1', 'GDNF', 'GFRA1',
        'FGF8', 'FGFR1', 'FGF9', 'FGFR2', 'FGF10', 'FGF2', 'EGF', 'EGFR', 'HGF', 'MET',
        'VEGFA', 'KDR', 'VEGFB', 'FLT1', 'PDGFA', 'PDGFRA', 'PDGFB', 'PDGFRB',
        'IGF1', 'IGF1R', 'IGF2'],
    'Ephrin / Semaphorin': [
        'EFNB1', 'EPHB2', 'EFNA1', 'EPHA4', 'EFNB2', 'EPHB4', 'EFNA5', 'EPHA3',
        'SEMA3A', 'NRP1', 'SEMA3F', 'NRP2', 'SEMA4D', 'PLXNB1', 'SEMA6A', 'PLXNA2'],
    'Chemokine / Cytokine': [
        'CXCL12', 'CXCR4', 'CXCL13', 'CXCR5', 'CCL2', 'CCR2', 'CCL5', 'CCR5',
        'IL1B', 'IL1R1', 'TNF', 'TNFRSF1A', 'TGFB1', 'TGFBR1', 'TGFB2', 'TGFBR2'],
}

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

buf = []
def log(s):
    print(s, flush=True)
    buf.append(s)

def compute_pathway_activity(slice_data, pathway_genes):
    gene_to_idx = {g.upper(): i for i, g in enumerate(slice_data['gene_names'])}
    idxs = [gene_to_idx[g] for g in pathway_genes if g in gene_to_idx]
    if not idxs:
        return None, 0
    return slice_data['expr'][:, idxs].mean(axis=1), len(idxs)

def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()

def analyze_pair(sample_A, sample_B):
    log("\n" + "=" * 72)
    log(f"  对: {sample_A} → {sample_B} (Br8325, 各 {N} spots)")
    log("=" * 72)
    A = load_dlpfc_slice(sample_A, N, 42)
    B = load_dlpfc_slice(sample_B, N, 99)
    B['coords'] = apply_misalignment(B['coords'], rotation_deg=2.0, translation=0.5)
    S, _ = compute_lr_strength_human(A, B, HUMAN_LR_DB)

    # 通路活性 (A参考 与 B)
    activity = {}
    for pname, genes in PATHWAYS.items():
        act_ref, n_g = compute_pathway_activity(A, genes)
        act_B, _ = compute_pathway_activity(B, genes)
        if act_ref is not None and n_g >= 3:
            activity[pname] = {'ref': act_ref, 'B': act_B, 'n_genes': n_g}
            log(f"  通路 {pname}: 命中 {n_g} 基因")
    # 非 LR 对照 (20 个非 LR 基因)
    gene_set = set(g.upper() for g in A['gene_names'])
    non_lr = sorted(gene_set - set(HUMAN_ALL_LR))
    rng = np.random.RandomState(123)
    control_genes = list(rng.choice(non_lr, min(20, len(non_lr)), replace=False))
    act_ref_c, _ = compute_pathway_activity(A, control_genes)
    act_B_c, _ = compute_pathway_activity(B, control_genes)
    activity['Non-LR Control'] = {'ref': act_ref_c, 'B': act_B_c, 'n_genes': len(control_genes)}
    log(f"  非LR对照: 从 {len(non_lr)} 个非LR基因中选 {len(control_genes)} 个")

    rows = []
    for g in GAMMAS:
        t0 = time.time()
        S_in = S if g > 0 else np.zeros_like(S)
        P, _ = fgw_lr_solver(A, B, S_in, gamma=g, beta=0.3, alpha=0.5,
                             max_iter=200, verbose=False, lr_gene_list=LIGS)
        dt = time.time() - t0
        Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
        ent = mean_entropy(P)
        mm = eval_plan(P, A['region_labels'], B['region_labels'],
                       A['expr'], B['expr'], A['coords'], B['coords'])
        row = {'gamma': g, 'entropy': ent,
               'same_mass': mm['same_mass'], 'soft_ari': mm['soft_ari'], 'soft_acc': mm['soft_acc']}
        for name, acts in activity.items():
            r, _ = pearsonr(acts['ref'], Pn @ acts['B'])
            row[name] = r
        rows.append(row)
        log(f"  γ={g:<5} 熵={ent:.4f} 同层质量={mm['same_mass']:.4f} 软ARI={mm['soft_ari']:.3f} "
            f"软准确率={mm['soft_acc']:.3f} | " +
            " | ".join(f"{n.split('/')[0]}={row[n]:.3f}" for n in activity) +
            f" (耗时{dt:.0f}s)")
    return rows

def main():
    log("=" * 72)
    log("  DLPFC 通路活性空间相关性 γ 扫描（过拟合阈值判断）")
    log("=" * 72)
    all_rows = {}
    for a, b in PAIRS:
        all_rows[f"{a}→{b}"] = analyze_pair(a, b)

    # 汇总（两对平均）
    log("\n" + "=" * 72)
    log("  两对平均")
    log("=" * 72)
    merged = []
    for i, g in enumerate(GAMMAS):
        row = {'gamma': g}
        keys = ['entropy', 'same_mass', 'soft_ari', 'soft_acc'] + list(PATHWAYS) + ['Non-LR Control']
        for k in keys:
            row[k] = np.mean([pr[i][k] for pr in all_rows.values()])
        merged.append(row)
        log(f"  γ={g:<5} 熵={row['entropy']:.4f} 同层={row['same_mass']:.4f} 软ARI={row['soft_ari']:.3f} "
            f"软准确率={row['soft_acc']:.3f} | Neuro={row['Neurotrophin / Growth Factor']:.3f} "
            f"Ephrin={row['Ephrin / Semaphorin']:.3f} Chemo={row['Chemokine / Cytokine']:.3f} "
            f"Ctrl={row['Non-LR Control']:.3f}")

    # 过拟合阈值判断: 非LR相关在γ>0.1后是否显著下降 + 层一致性是否开始下降
    g0 = merged[0]
    g01 = next(r for r in merged if abs(r['gamma'] - 0.1) < 1e-9)
    g02 = next(r for r in merged if abs(r['gamma'] - 0.2) < 1e-9)
    g05 = next(r for r in merged if abs(r['gamma'] - 0.5) < 1e-9)
    log("\n" + "=" * 72)
    log("  过拟合阈值分析")
    log("=" * 72)
    ctrl0, ctrl01, ctrl02, ctrl05 = (g0['Non-LR Control'], g01['Non-LR Control'],
                                     g02['Non-LR Control'], g05['Non-LR Control'])
    sm0, sm01, sm02, sm05 = (g0['same_mass'], g01['same_mass'], g02['same_mass'], g05['same_mass'])
    acc0, acc01, acc02, acc05 = (g0['soft_acc'], g01['soft_acc'], g02['soft_acc'], g05['soft_acc'])
    log(f"  非LR对照相关: γ=0: {ctrl0:.3f} → γ=0.1: {ctrl01:.3f} → γ=0.2: {ctrl02:.3f} → γ=0.5: {ctrl05:.3f}")
    log(f"  同层质量:     γ=0: {sm0:.4f} → γ=0.1: {sm01:.4f} → γ=0.2: {sm02:.4f} → γ=0.5: {sm05:.4f}")
    log(f"  软准确率:     γ=0: {acc0:.3f} → γ=0.1: {acc01:.3f} → γ=0.2: {acc02:.3f} → γ=0.5: {acc05:.3f}")
    # 判断
    ctrl_drop_after_01 = (ctrl02 - ctrl01) / max(abs(ctrl01), 1e-6)
    sm_drop_after_01 = sm02 - sm01
    ctrl_drop_strong = (ctrl0 - ctrl05) / max(abs(ctrl0), 1e-6)
    verdict = []
    if ctrl_drop_after_01 < -0.1:
        verdict.append("非LR对照相关在γ>0.1后显著下降")
    if sm_drop_after_01 < -0.005:
        verdict.append("同层质量在γ>0.1后开始下降")
    if acc02 < acc01 - 0.01:
        verdict.append("软准确率在γ>0.1后开始下降")
    if not verdict:
        verdict.append("未观察到γ>0.1后的显著过拟合")
    log(f"  判断: {'; '.join(verdict)}")
    log("  结论: 若层一致性在较高γ下仍保持或微降, 则『过拟合阈值』不明显, "
        "LR引导在默认γ=0.1下不牺牲非LR保真度; 但随着γ继续增大, 存在保真度-熵权衡, "
        "应强调γ需在真实数据上校准而非无脑增大。")

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'lrot_pathway_dlpfc_results.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(buf) + "\n")

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    gammas = [r['gamma'] for r in merged]
    colors = {'Neurotrophin / Growth Factor': '#2166AC', 'Ephrin / Semaphorin': '#D6604D',
              'Chemokine / Cytokine': '#4DAF4A', 'Non-LR Control': '#984EA3'}
    ax = axes[0]
    for name in list(PATHWAYS) + ['Non-LR Control']:
        ax.plot(gammas, [r[name] for r in merged], 'o-', color=colors[name], lw=1.8, ms=4,
                label=name.split('/')[0] if name != 'Non-LR Control' else 'Non-LR Control')
    ax.axvline(0.1, color='gray', ls='--', lw=1)
    ax.text(0.105, ax.get_ylim()[0] + 0.02, 'γ=0.1', fontsize=8, color='gray')
    ax.set_xlabel('LR Guidance Weight γ')
    ax.set_ylabel('Pathway Activity Correlation (fidelity)')
    ax.set_title('(B) DLPFC: pathway-activity fidelity vs \u03b3')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(gammas, [r['same_mass'] for r in merged], 'o-', color='#2C3E50', lw=1.8, ms=4, label='same-layer mass')
    ax.plot(gammas, [r['soft_acc'] for r in merged], 's--', color='#C0392B', lw=1.8, ms=4, label='soft accuracy')
    ax2 = ax.twinx()
    ax2.plot(gammas, [r['entropy'] for r in merged], '*-', color='#B2182B', lw=2.2, ms=6, label='transport entropy')
    ax.axvline(0.1, color='gray', ls='--', lw=1)
    ax.set_xlabel('LR Guidance Weight γ')
    ax.set_ylabel('Layer concordance')
    ax2.set_ylabel('Transport entropy', color='#B2182B')
    ax2.tick_params(axis='y', labelcolor='#B2182B')
    ax.set_title('(C) DLPFC: layer concordance & entropy vs \u03b3')
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'lrot_pathway_dlpfc.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    log(f"\n  图已保存: {os.path.join(OUT_DIR, 'lrot_pathway_dlpfc.png')}")

if __name__ == '__main__':
    main()
