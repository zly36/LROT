# -*- coding: utf-8 -*-
"""
run_official_paste2.py — 官方 PASTE2 (paste2 包) 合成数据基线评估
与其它基线脚本使用相同的数据配置与评估口径。
PASTE2 为部分对齐方法，其 partial_fused_gromov_wasserstein 要求 s 严格 <= min(|p|_1, |q|_1)，
本实验数据为完全重叠，取 s = min(sum(p), sum(q))（即完整传输）。
注意：PASTE2 输出为无熵正则的精确部分 FGW 硬匹配计划，其传输熵恒为 0，与 LROT 的软计划熵不可直接比较。
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
    compute_mapping_consistency, compute_mapping_consistency_hard,
    compute_aligned_nn_accuracy, compute_ot_cost,
)
from real_st_loader import RealisticSTGenerator

OUT_DIR = os.path.join(os.getcwd(), "lrot_output")


def evaluate(P, slice_A, slice_B):
    """与其它基线脚本完全相同的评估口径"""
    acc = compute_mapping_consistency(P, slice_A['region_labels'], slice_B['region_labels'])
    cost = compute_ot_cost(P, slice_A['expr'], slice_B['expr'])
    P_norm = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    entropy = -np.sum(P_norm * np.log(np.maximum(P_norm, 1e-10)), axis=1).mean()
    sparsity = np.sum(P > 1e-10) / P.size
    acc_hard = compute_mapping_consistency_hard(P, slice_A['region_labels'], slice_B['region_labels'])
    acc_nn = compute_aligned_nn_accuracy(P, slice_A['coords'], slice_B['coords'],
                                         slice_A['region_labels'], slice_B['region_labels'])
    return {'accuracy': acc, 'accuracy_hard': acc_hard, 'accuracy_nn': acc_nn,
            'cost': cost, 'entropy': entropy, 'sparsity': sparsity}


def to_anndata(slice_dict):
    """把 lrot_core 的 slice dict 转成 PASTE2 需要的 AnnData"""
    ad = anndata.AnnData(
        X=slice_dict['expr'].astype(np.float64),
        obs={'region_labels': slice_dict['region_labels'].astype(str)},
        var={'gene_names': slice_dict['gene_names']},
    )
    ad.var_names = [str(g) for g in slice_dict['gene_names']]
    ad.obs_names = [f"spot{i}" for i in range(slice_dict['n_spots'])]
    ad.obsm['spatial'] = slice_dict['coords'].astype(np.float64)
    return ad


def run_official_paste2(adataA, adataB, alpha, diss, s, p, q):
    """运行官方 PASTE2 部分对齐，返回 (pi, info)"""
    t0 = time.time()
    pi = partial_pairwise_align(
        adataA, adataB,
        s=s, alpha=alpha, dissimilarity=diss,
        a_distribution=p, b_distribution=q,
        verbose=False,
    )
    elapsed = time.time() - t0
    return pi, {'method': f'Official PASTE2 (α={alpha}, diss={diss})',
                'alpha': alpha, 'dissimilarity': diss, 'time': elapsed}


def main():
    print("生成真实感数据（与基线脚本相同配置）...")
    gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000,
                               n_genes=150, seed=42)
    paired = gen.generate_paired_slices(rotation=5.0, batch_effect=0.15)
    slice_A = paired.slice_A.to_dict()
    slice_B = paired.slice_B.to_dict()
    adataA = to_anndata(slice_A)
    adataB = to_anndata(slice_B)
    print(f"  Slice A: {slice_A['n_spots']} spots x {slice_A['n_genes']} genes")
    print(f"  Slice B: {slice_B['n_spots']} spots x {slice_B['n_genes']} genes")

    # PASTE2 要求 s <= min(|p|_1, |q|_1)；完全重叠数据取最大可行值
    p = np.ones(adataA.shape[0]) / adataA.shape[0]
    q = np.ones(adataB.shape[0]) / adataB.shape[0]
    s = float(min(p.sum(), q.sum()))
    print(f"  完全重叠: s = {s:.6f}")

    results = []
    print("\n" + "=" * 78)
    print("  官方 PASTE2 (paste2 包) 基线评估")
    print("=" * 78)
    print(f"  {'方法':<38} {'精度(软)':<8} {'精度(硬)':<8} {'精度(NN)':<8} {'成本':<8} {'熵':<8} {'时间(s)':<8}")
    print(f"  {'-' * 78}")
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for diss in ['kl', 'euclidean']:
            pi, info = run_official_paste2(adataA, adataB, alpha, diss, s, p, q)
            metrics = evaluate(pi, slice_A, slice_B)
            results.append({**info, **metrics})
            print(f"  {info['method']:<38} {metrics['accuracy']:<8.4f} {metrics['accuracy_hard']:<8.4f} "
                  f"{metrics['accuracy_nn']:<8.4f} {metrics['cost']:<8.4f} {metrics['entropy']:<8.4f} "
                  f"{info['time']:<8.2f}")
    print("=" * 78)

    lines = []
    lines.append("=" * 78)
    lines.append("官方 PASTE2 (paste2 包) 合成数据基线评估结果")
    lines.append("=" * 78)
    lines.append("数据: 1000x1000 spots, 150 genes, seed=42, rotation=5.0, batch_effect=0.15")
    lines.append("PASTE2 配置: s=1.0(完全重叠,最大可行质量), 均匀分布, norm=True, 精确部分FGW(CG,无熵正则)")
    for r in results:
        lines.append(
            f"{r['method']:<38} 精度={r['accuracy']:.4f} 精度硬={r['accuracy_hard']:.4f} "
            f"精度NN={r['accuracy_nn']:.4f} 成本={r['cost']:.4f} 熵={r['entropy']:.4f} "
            f"时间={r['time']:.2f}s 稀疏度={r['sparsity']:.3f}")
    lines.append("")
    lines.append("参照(现有实现, 同口径):")
    lines.append("Expr OT only         0.708 / 熵1.752")
    lines.append("PASTE (FGW, γ=0)     0.711 / 熵4.035")
    lines.append("LROT (γ=0.05)        0.716 / 熵3.994")
    lines.append("LROT (γ=0.1)         0.712 / 熵3.928")
    lines.append("LROT (γ=0.2)         0.709 / 熵3.821")
    lines.append("STAligner(深度)       0.799 / 熵—(口径不同)")
    text = "\n".join(lines)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "official_paste2_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n已保存: {out_path}")


if __name__ == '__main__':
    main()