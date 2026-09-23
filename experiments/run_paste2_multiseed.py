# -*- coding: utf-8 -*-
"""
run_paste2_multiseed.py — PASTE2 多seed稳健性检验
在 20 个独立种子 (1..20) 下重复真实感模拟数据实验，比较:
FGW基线(γ=0) / LROT(γ=0.1) / 官方PASTE2(KL, α=0.1) 的精度与熵稳定性。
数据配置与主实验一致: 1000 spots, 150 genes, rotation=5.0, batch_effect=0.15。
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import numpy as np
import anndata

from paste2.PASTE2 import partial_pairwise_align
from lrot_core import (
    fgw_lr_solver, compute_lr_strength_matrix, LR_DB,
    compute_mapping_consistency, compute_ot_cost,
)
from real_st_loader import RealisticSTGenerator

OUT_DIR = os.path.join(os.getcwd(), "lrot_output")
SEEDS = list(range(1, 21))


def evaluate(P, A, B):
    acc = compute_mapping_consistency(P, A['region_labels'], B['region_labels'])
    cost = compute_ot_cost(P, A['expr'], B['expr'])
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    ent = -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()
    return acc, ent, cost


def to_anndata(sd):
    ad = anndata.AnnData(X=sd['expr'].astype(np.float64),
                         obs={'region_labels': sd['region_labels'].astype(str)},
                         var={'gene_names': sd['gene_names']})
    ad.var_names = [str(g) for g in sd['gene_names']]
    ad.obs_names = [f"spot{i}" for i in range(sd['n_spots'])]
    ad.obsm['spatial'] = sd['coords'].astype(np.float64)
    return ad


def main():
    rows = []
    print(f"运行 {len(SEEDS)} 个种子 (seed=1..20) ...")
    for seed in SEEDS:
        gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000,
                                   n_genes=150, seed=seed)
        paired = gen.generate_paired_slices(rotation=5.0, batch_effect=0.15)
        A = paired.slice_A.to_dict()
        B = paired.slice_B.to_dict()

        # FGW (γ=0)
        S0 = np.zeros((A['n_spots'], B['n_spots']))
        t0 = time.time()
        P_fgw, _ = fgw_lr_solver(A, B, S0, gamma=0.0, beta=0.3, alpha=0.5,
                                 max_iter=200, verbose=False)
        t_fgw = time.time() - t0
        acc_fgw, ent_fgw, _ = evaluate(P_fgw, A, B)

        # LROT (γ=0.1)
        S_lr, _ = compute_lr_strength_matrix(A, B, LR_DB)
        t0 = time.time()
        P_lrot, _ = fgw_lr_solver(A, B, S_lr, gamma=0.1, beta=0.3, alpha=0.5,
                                  max_iter=200, verbose=False)
        t_lrot = time.time() - t0
        acc_lrot, ent_lrot, _ = evaluate(P_lrot, A, B)

        # Official PASTE2 (KL, α=0.1)
        adA, adB = to_anndata(A), to_anndata(B)
        p = np.ones(adA.shape[0]) / adA.shape[0]
        q = np.ones(adB.shape[0]) / adB.shape[0]
        s = float(min(p.sum(), q.sum()))
        t0 = time.time()
        pi = partial_pairwise_align(adA, adB, s=s, alpha=0.1, dissimilarity='kl',
                                    a_distribution=p, b_distribution=q, verbose=False)
        t_p2 = time.time() - t0
        acc_p2, ent_p2, _ = evaluate(pi, A, B)

        rows.append(dict(seed=seed, fgw=acc_fgw, fgw_ent=ent_fgw, lrot=acc_lrot,
                         lrot_ent=ent_lrot, p2=acc_p2, p2_ent=ent_p2,
                         t_fgw=t_fgw, t_lrot=t_lrot, t_p2=t_p2))
        print(f"  seed={seed:3d}  FGW={acc_fgw:.4f}  LROT={acc_lrot:.4f}  PASTE2={acc_p2:.4f}  (P2 {t_p2:.1f}s)")

    accs = {k: np.array([r[k] for r in rows]) for k in ['fgw', 'lrot', 'p2']}
    ents = {k: np.array([r[k + '_ent'] for r in rows]) for k in ['fgw', 'lrot', 'p2']}
    n_win = int(np.sum(accs['lrot'] > accs['p2']))
    n_tie = int(np.sum(accs['lrot'] == accs['p2']))

    lines = []
    lines.append("=" * 100)
    lines.append("PASTE2 多seed稳健性检验 (真实感模拟数据, 1000 spots, 150 genes)")
    lines.append("配置: FGW(γ=0) / LROT(γ=0.1) / 官方PASTE2(KL, α=0.1, s=1.0) ; 20 seeds (1..20)")
    lines.append("=" * 100)
    lines.append(f"{'seed':>4} {'FGW.acc':>8} {'FGW.ent':>8} {'LROT.acc':>8} {'LROT.ent':>8} "
                 f"{'P2.acc':>8} {'P2.ent':>8} {'tFGW(s)':>8} {'tLROT(s)':>8} {'tP2(s)':>8}")
    for r in rows:
        lines.append(f"{r['seed']:>4} {r['fgw']:>8.4f} {r['fgw_ent']:>8.4f} {r['lrot']:>8.4f} "
                     f"{r['lrot_ent']:>8.4f} {r['p2']:>8.4f} {r['p2_ent']:>8.4f} "
                     f"{r['t_fgw']:>8.2f} {r['t_lrot']:>8.2f} {r['t_p2']:>8.2f}")
    lines.append("-" * 100)
    for k, name in [('fgw', 'FGW (γ=0)'), ('lrot', 'LROT (γ=0.1)'), ('p2', 'Official PASTE2 (KL, α=0.1)')]:
        lines.append(f"{name:<28} acc 均值±标准差 = {accs[k].mean():.4f} ± {accs[k].std(ddof=1):.4f}  "
                     f"范围 [{accs[k].min():.4f}, {accs[k].max():.4f}]  "
                     f"熵均值 = {ents[k].mean():.4f}")
    lines.append("-" * 100)
    lines.append(f"LROT 优于 PASTE2 的种子数: {n_win}/{len(SEEDS)} (持平 {n_tie})")
    lines.append(f"LROT - PASTE2 平均精度差 = {(accs['lrot'] - accs['p2']).mean():.4f} "
                 f"(±{(accs['lrot'] - accs['p2']).std(ddof=1):.4f})")
    text = "\n".join(lines)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "paste2_multiseed_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n已保存: {out_path}")


if __name__ == '__main__':
    main()