# -*- coding: utf-8 -*-
"""
batch_effect_scan.py — 真实批次效应模型（逐基因系统性偏移）下的鲁棒性扫描
背景: 早期实现 _add_batch_effects 采用逐 spot 乘性缩放, 该模型会被余弦距离(标度不变)
  完全吸收, 因此需要改用更接近真实数据的批次效应模型重新评测鲁棒性。
本脚本改用真实批次效应模型(逐基因系统性偏移, 与scRNA/ST领域一致):
  model='gene':   expr_B[:, g] *= f_g,  f_g ~ Lognormal(0, sigma)   (基因特异倍率变化)
  model='geneadd':expr_B[:, g] += a_g,  a_g ~ N(0, sigma)           (log1p空间加性, 零截断保持非负)
  model='spot':   原模型(逐spot缩放, 作为对照)
强度 sigma 取 0.1~0.5 (原 batch 参数口径保持 0.1~0.5 以便直接对比)。
输出: lrot_output/lrot_batch_effect_results.txt, lrot_output/lrot_batch_effect.png
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

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lrot_core import (fgw_lr_solver, compute_lr_strength_matrix,
                       compute_mapping_consistency, LR_DB, ALL_LR_GENES)
from real_st_loader import RealisticSTGenerator

OUT_DIR = os.path.join(_R, 'lrot_output')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42 + i * 1000 for i in range(5)]
GAMMA = 0.1
STRENGTHS = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

buf = []
def log(s):
    print(s, flush=True)
    buf.append(s)

def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()

def apply_real_batch(expr, mode, sigma, seed):
    """对 slice B 施加真实批次效应。
    mode='gene':  逐基因乘性 Lognormal(0, sigma)   — 真实: 批次特异基因倍率
    mode='geneadd': log1p 空间加性 N(0, sigma) (log1p(expr)+N(0,sigma) 后零截断, 保持非负)
    mode='spot':  原模型逐spot乘性 uniform(1±sigma) — 对照(应无效)
    """
    rng = np.random.RandomState(seed + 31)
    n_spots, n_genes = expr.shape
    if mode == 'gene':
        factors = np.exp(rng.normal(0, sigma, n_genes))
        return expr * factors[None, :]
    elif mode == 'geneadd':
        # log1p 空间加性: log1p(expr)+a_g 后 expm1 还原, 零截断保证非负
        shifts = rng.normal(0, sigma, n_genes)
        return np.maximum(np.expm1(np.log1p(expr) + shifts[None, :]), 0.0)
    else:  # spot
        scales = rng.uniform(1 - sigma, 1 + sigma, n_spots)
        return expr * scales[:, None]

def run_config(seed, mode, sigma):
    gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=seed)
    paired = gen.generate_paired_slices(rotation=5.0, scale=0.95, batch_effect=0.0)
    slice_A = paired.slice_A.to_dict()
    slice_B = paired.slice_B.to_dict()
    labels_A, labels_B = slice_A['region_labels'], slice_B['region_labels']
    slice_B['expr'] = apply_real_batch(slice_B['expr'], mode, sigma, seed)
    expr_A, expr_B = slice_A['expr'], slice_B['expr']

    S_lr, _ = compute_lr_strength_matrix(slice_A, slice_B, LR_DB)
    out = {}
    for gamma, tag in [(GAMMA, 'LROT'), (0.0, 'FGW')]:
        S_in = S_lr if gamma > 0 else np.zeros_like(S_lr)
        P, _ = fgw_lr_solver(slice_A, slice_B, S_in, gamma=gamma, beta=0.3, alpha=0.5,
                             max_iter=200, verbose=False)
        out[tag] = {'acc': compute_mapping_consistency(P, labels_A, labels_B),
                    'ent': mean_entropy(P)}
    return out

def main():
    log("=" * 72)
    log("  真实批次效应模型下的鲁棒性扫描")
    log("  gene=逐基因乘性Lognormal | geneadd=log1p尺度加性(零截断) | spot=原逐spot(对照)")
    log("  固定参数: α=0.5, β=0.3, γ=0.1; 5 seeds; batch_effect=0(基线)")
    log("=" * 72)

    models = [('gene', '逐基因乘性 Lognormal(0,σ)'),
              ('geneadd', 'log1p尺度加性 N(0,σ), 零截断'),
              ('spot', '原模型 逐spot乘性(对照)')]

    all_results = {}
    for mode, desc in models:
        log(f"\n--- 模型: {desc} ---")
        rows = []
        for sigma in STRENGTHS:
            accL, accF, entL, entF = [], [], [], []
            for seed in SEEDS:
                r = run_config(seed, mode, sigma)
                accL.append(r['LROT']['acc']); accF.append(r['FGW']['acc'])
                entL.append(r['LROT']['ent']); entF.append(r['FGW']['ent'])
            rows.append((sigma, np.mean(accL), np.std(accL), np.mean(accF),
                         np.std(accF), np.mean(entL), np.std(entL),
                         np.mean(entF), np.std(entF)))
            log(f"  σ={sigma:<4} LROT精度={np.mean(accL):.3f}±{np.std(accL):.3f} | "
                f"FGW精度={np.mean(accF):.3f}±{np.std(accF):.3f} | "
                f"LROT熵={np.mean(entL):.4f} | FGW熵={np.mean(entF):.4f}")
            # 增量保存, 防止长任务被终端超时中断而丢失
            with open(os.path.join(OUT_DIR, 'lrot_batch_effect_results.txt'), 'w', encoding='utf-8') as f:
                f.write("\n".join(buf) + f"\n[进度: {mode} σ={sigma} 完成]\n")
        all_results[mode] = rows

    # 基线（无批次效应）
    baseL, baseF = [], []
    for seed in SEEDS:
        r = run_config(seed, 'spot', 0.0)
        baseL.append(r['LROT']['acc']); baseF.append(r['FGW']['acc'])
    base_accL = np.mean(baseL); base_accF = np.mean(baseF)
    log(f"\n基线(无批次效应): LROT精度={base_accL:.3f} | FGW精度={base_accF:.3f}")

    # 汇总
    log("\n" + "=" * 72)
    log("  汇总（5 seed 平均，相对基线的精度变化）")
    log("=" * 72)
    for mode, desc in models:
        log(f"\n  [{desc}]")
        log(f"    {'σ':<6} {'LROT精度':<10} {'ΔLROT':<9} {'FGW精度':<10} {'ΔFGW':<9} {'熵降%':<8}")
        for sigma, al, sl, af, sf, el, sel, ef, sef in all_results[mode]:
            dl = (al - base_accL) / base_accL * 100
            df = (af - base_accF) / base_accF * 100
            ent_imp = (ef - el) / ef * 100 if ef > 0 else 0
            log(f"    {sigma:<6} {al:<10.3f} {dl:<+9.1f} {af:<10.3f} {df:<+9.1f} {ent_imp:<8.1f}")

    # 保存
    with open(os.path.join(OUT_DIR, 'lrot_batch_effect_results.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(buf) + "\n")

    # 图（图例用英文，避免中文无字体渲染乱码）
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = {'gene': '#C0392B', 'geneadd': '#2C3E50', 'spot': '#7F8C8D'}
    legend_en = {'gene': 'Per-gene Lognormal(0,σ)',
                 'geneadd': 'Log1p-scale additive N(0,σ)',
                 'spot': 'Original per-spot (control)'}
    for ax, key, ylab, title in [
            (axes[0], 'accL', 'LROT accuracy', '(A) LROT accuracy vs batch strength'),
            (axes[1], 'accF', 'FGW accuracy', '(B) FGW accuracy vs batch strength'),
            (axes[2], 'entL', 'LROT entropy', '(C) LROT entropy vs batch strength')]:
        for mode, desc in models:
            idx = {'accL': 1, 'accF': 3, 'entL': 5}[key]
            std_idx = {'accL': 2, 'accF': 4, 'entL': 6}[key]
            xs = [r[0] for r in all_results[mode]]
            ys = [r[idx] for r in all_results[mode]]
            errs = [r[std_idx] for r in all_results[mode]]
            ax.errorbar(xs, ys, yerr=errs, fmt='o-', lw=1.8, ms=4,
                        label=legend_en[mode], color=colors[mode], capsize=2)
        ax.set_xlabel('Batch effect strength σ')
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'lrot_batch_effect.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    log(f"\n  已保存: {os.path.join(OUT_DIR, 'lrot_batch_effect_results.txt')}")
    log(f"           {os.path.join(OUT_DIR, 'lrot_batch_effect.png')}")

if __name__ == '__main__':
    main()
