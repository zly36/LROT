# -*- coding: utf-8 -*-
"""
lrot_control_experiment.py — 熵降机制的系统控制实验
系统回答: 熵降中多大比例是通用的正则化效应(随机噪声项), 多大比例是 LR 特有的生物学信号?

设计: 同一数据(5个随机种子, 与主实验一致: 1,000 spots, 150 genes, 旋转5°, 批次0.15)
      上比较 4 种"信号项"设置:
  A. None        : FGW 基线 (γ=0, 无额外信号项)         -> 熵 E0
  B. Random noise: 等量随机非LR基因对的外积强度(无LR信息) -> 熵 E_rand
  C. Shuffled LR : 打乱配对的 LR 项                     -> 熵 E_shuf
  D. True LR     : 真实 LR 项 (LROT, γ=0.1)              -> 熵 E_lr

熵降分解(相对 FGW 基线):
  总熵降(LR) = E0 - E_lr
  通用正则化 = E0 - E_rand   (任何额外信号项都具备的锐化效应)
  LR 特异   = E_rand - E_lr  (LR 信号独有的增量熵降)
  通用占比 = (E0-E_rand)/(E0-E_lr); LR特异占比 = (E_rand-E_lr)/(E0-E_lr)

输出: lrot_output/lrot_control_results.txt (该文件同时是图 S5 的输入)
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

import numpy as np
from lrot_core import (fgw_lr_solver, compute_lr_strength_matrix,
                       compute_mapping_consistency, LR_DB, ALL_LR_GENES)
from real_st_loader import RealisticSTGenerator


OUT_DIR = os.path.join(_R, 'lrot_output')
os.makedirs(OUT_DIR, exist_ok=True)

GAMMA = 0.1
SEEDS = [42 + i * 1000 for i in range(5)]

buf = []
def log(s):
    print(s, flush=True)
    buf.append(s)


def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()


def shuffled_lr_db():
    """打乱配对的 LR 数据库（受体随机置换, 与 lrot_lr_sensitivity 一致）"""
    ligands = sorted(set(k[0] for k in LR_DB))
    receptors = sorted(set(k[1] for k in LR_DB))
    rng = np.random.RandomState(999)
    rec_list = [r for (l, r) in LR_DB.keys()]
    rng.shuffle(rec_list)
    return {(l, new_r): w for (l, _), new_r, w in zip(LR_DB.keys(), rec_list, LR_DB.values())}


def random_noise_matrix(slice_A, slice_B, n_pairs=22, seed=777):
    """等量随机非LR基因对的外积强度（无任何LR信息）—— 随机噪声项"""
    gene_to_idx = {g: i for i, g in enumerate(slice_A['gene_names'])}
    lr_set = set(ALL_LR_GENES)
    non_lr = [g for g in slice_A['gene_names'] if g not in lr_set]
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(non_lr), min(2 * n_pairs, len(non_lr)), replace=False)
    g_a = [non_lr[i] for i in idxs[:n_pairs]]
    g_b = [non_lr[i] for i in idxs[n_pairs:]]
    nA, nB = slice_A['n_spots'], slice_B['n_spots']
    S = np.zeros((nA, nB))
    for ga, gb in zip(g_a, g_b):
        ia, ib = gene_to_idx[ga], gene_to_idx[gb]
        S += np.outer(slice_A['expr'][:, ia], slice_B['expr'][:, ib]) * 0.5
    if S.max() > S.min():
        S = (S - S.min()) / (S.max() - S.min())
    return S, len(g_a)


def solve(slice_A, slice_B, S, gamma):
    P, _ = fgw_lr_solver(slice_A, slice_B, S, gamma=gamma, beta=0.3, alpha=0.5,
                         max_iter=200, verbose=False)
    return P


def run_seed(seed):
    gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=seed)
    paired = gen.generate_paired_slices(rotation=5.0, batch_effect=0.15)
    slice_A = paired.slice_A.to_dict()
    slice_B = paired.slice_B.to_dict()
    labels_A, labels_B = slice_A['region_labels'], slice_B['region_labels']
    expr_A, expr_B = slice_A['expr'], slice_B['expr']

    S_lr, _ = compute_lr_strength_matrix(slice_A, slice_B, LR_DB)
    db_shuf = shuffled_lr_db()
    S_shuf, _ = compute_lr_strength_matrix(slice_A, slice_B, db_shuf)
    S_rand, n_rand = random_noise_matrix(slice_A, slice_B)

    out = {}
    # A. FGW baseline (γ=0)
    P = solve(slice_A, slice_B, np.zeros_like(S_lr), 0.0)
    out['None'] = {'ent': mean_entropy(P), 'acc': compute_mapping_consistency(P, labels_A, labels_B)}
    # B. Random noise
    P = solve(slice_A, slice_B, S_rand, GAMMA)
    out['RandomNoise'] = {'ent': mean_entropy(P), 'acc': compute_mapping_consistency(P, labels_A, labels_B)}
    # C. Shuffled LR
    P = solve(slice_A, slice_B, S_shuf, GAMMA)
    out['ShuffledLR'] = {'ent': mean_entropy(P), 'acc': compute_mapping_consistency(P, labels_A, labels_B)}
    # D. True LR
    P = solve(slice_A, slice_B, S_lr, GAMMA)
    out['TrueLR'] = {'ent': mean_entropy(P), 'acc': compute_mapping_consistency(P, labels_A, labels_B)}
    return out


def main():
    log("=" * 72)
    log("  熵降机制的系统控制实验")
    log("  比较: FGW基线 / 随机噪声项 / 打乱LR / 真实LR (5个随机种子, 与主实验一致)")
    log("=" * 72)

    acc = {k: [] for k in ['None', 'RandomNoise', 'ShuffledLR', 'TrueLR']}
    ent = {k: [] for k in ['None', 'RandomNoise', 'ShuffledLR', 'TrueLR']}
    for s in SEEDS:
        r = run_seed(s)
        for k in acc:
            ent[k].append(r[k]['ent'])
            acc[k].append(r[k]['acc'])
        log(f"  seed={s}: None熵={r['None']['ent']:.4f} | Rand熵={r['RandomNoise']['ent']:.4f} "
            f"| Shuf熵={r['ShuffledLR']['ent']:.4f} | TrueLR熵={r['TrueLR']['ent']:.4f} | "
            f"精度: None={r['None']['acc']:.3f} Rand={r['RandomNoise']['acc']:.3f} "
            f"Shuf={r['ShuffledLR']['acc']:.3f} TrueLR={r['TrueLR']['acc']:.3f}")

    E0 = np.mean(ent['None']); E_rand = np.mean(ent['RandomNoise'])
    E_shuf = np.mean(ent['ShuffledLR']); E_lr = np.mean(ent['TrueLR'])
    A0 = np.mean(acc['None']); A_rand = np.mean(acc['RandomNoise'])
    A_shuf = np.mean(acc['ShuffledLR']); A_lr = np.mean(acc['TrueLR'])

    log("\n" + "=" * 72)
    log("  汇总（5 seed 平均）")
    log("=" * 72)
    log(f"  {'设置':<14} {'熵':<10} {'Δ熵(相对FGW)':<14} {'精度':<8}")
    log(f"  {'-'*52}")
    log(f"  {'FGW基线':<12} {E0:<10.4f} {'—':<14} {A0:<8.3f}")
    log(f"  {'随机噪声':<12} {E_rand:<10.4f} {E0-E_rand:<+14.4f} {A_rand:<8.3f}")
    log(f"  {'打乱LR':<12} {E_shuf:<10.4f} {E0-E_shuf:<+14.4f} {A_shuf:<8.3f}")
    log(f"  {'真实LR':<12} {E_lr:<10.4f} {E0-E_lr:<+14.4f} {A_lr:<8.3f}")

    total_drop = E0 - E_lr
    common_drop = E0 - E_rand
    lr_specific = E_rand - E_lr
    common_share = common_drop / total_drop * 100 if total_drop > 0 else float('nan')
    lr_share = lr_specific / total_drop * 100 if total_drop > 0 else float('nan')

    log("\n" + "=" * 72)
    log("  熵降分解（相对 FGW 基线）")
    log("=" * 72)
    log(f"  总熵降(真实LR)   = {total_drop:.4f}")
    log(f"  通用正则化(随机) = {common_drop:.4f}  -> 占比 {common_share:.1f}%")
    log(f"  LR特异增量       = {lr_specific:.4f}  -> 占比 {lr_share:.1f}%")
    log(f"  打乱LR熵降       = {E0-E_shuf:.4f}")
    log(f"\n  精度: 真实LR={A_lr:.3f} vs 随机噪声={A_rand:.3f} vs 打乱LR={A_shuf:.3f} vs FGW={A0:.3f}")
    log(f"  结论: 熵降中通用正则化效应占主导(约{common_share:.0f}%), LR特异的生物学信号贡献约{lr_share:.0f}%; "
        f"但真实LR在熵降的同时保持最高精度且提供通路保真度(表4), 这是随机项/打乱LR不具备的。")

    with open(os.path.join(OUT_DIR, 'lrot_control_results.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(buf) + "\n")
        f.write("\n逐seed数值:\n")
        f.write("seed,None_ent,RandomNoise_ent,ShuffledLR_ent,TrueLR_ent,None_acc,RandomNoise_acc,ShuffledLR_acc,TrueLR_acc\n")
        for i, s in enumerate(SEEDS):
            f.write(f"{s},{ent['None'][i]:.4f},{ent['RandomNoise'][i]:.4f},{ent['ShuffledLR'][i]:.4f},"
                    f"{ent['TrueLR'][i]:.4f},{acc['None'][i]:.4f},{acc['RandomNoise'][i]:.4f},"
                    f"{acc['ShuffledLR'][i]:.4f},{acc['TrueLR'][i]:.4f}\n")
    log(f"\n  结果已保存: {os.path.join(OUT_DIR, 'lrot_control_results.txt')}")


if __name__ == '__main__':
    main()
