# -*- coding: utf-8 -*-
"""
structural_batch_scan.py — 结构性批次效应扫描（关键验证）
乘性全局缩放(batch_effect)会被成本归一化吸收, 不改变三成本相对量级; 因此需另行考察更细粒度的结构性扰动。
关键场景: 不同基因子集受到差异化缩放, 使 C_expr(非LR主导) 与 C_lr(LR主导) 相对量级剧烈变化。

设计(5 seeds, 与主实验一致): 固定默认参数(α=0.5, β=0.3, γ=0.1), 对 B 切片:
  (a) LR 基因强缩放 s (uniform(1-s,1+s)), 非 LR 基因保持默认 0.15
  (b) 非 LR 基因强缩放 s, LR 基因保持默认 0.15
  (c) 全部基因强缩放 s (乘性, 作为对照)
  s ∈ {0.1, 0.3, 0.5}
实现说明: 采用逐基因独立乘性缩放(每个基因乘独立标量, 与 batch_effect_scan 的
  gene 模型一致)。逐spot的整体缩放会被 C_expr 的余弦距离(标度不变)吸收, 不进入求解器, 故不采用。
观察: 固定参数精度在结构性批次效应下是否仍保持; 若下降, 揭示真正的鲁棒性极限。
输出: lrot_output/lrot_structural_batch_results.txt
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
from lrot_core import (fgw_lr_solver, compute_lr_strength_matrix,
                       compute_mapping_consistency, LR_DB, ALL_LR_GENES)
from real_st_loader import RealisticSTGenerator


OUT_DIR = os.path.join(_R, 'lrot_output')
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42 + i * 1000 for i in range(5)]
GAMMA = 0.1
S_STRONG = [0.1, 0.3, 0.5]

buf = []
def log(s):
    print(s, flush=True)
    buf.append(s)


def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()


def apply_structural(expr, gene_names, lr_set, mode, s, seed, default=0.15):
    """mode: 'lr'=LR基因强缩放s(非LR保持默认default);
             'nlr'=非LR基因强缩放s(LR保持默认default);
             'all'=全部基因强缩放s(乘性对照, 无弱缩放子集)
    逐基因独立乘性缩放(每个选中基因乘独立标量 uniform(1±s)),
    与 batch_effect_scan 的 gene 模型一致; 避免被余弦距离吸收。"""
    rng = np.random.RandomState(seed + 7)
    out = expr.astype(float).copy()
    mask = np.array([g in lr_set for g in gene_names])
    if mode == 'lr':
        strong, weak = mask, ~mask
    elif mode == 'nlr':
        strong, weak = ~mask, mask
    else:  # 'all'
        strong = np.ones(expr.shape[1], dtype=bool)
        weak = np.zeros(expr.shape[1], dtype=bool)
    # 逐基因独立乘性缩放: 每个选中基因乘独立标量
    out[:, strong] *= rng.uniform(1 - s, 1 + s, strong.sum())[None, :]
    if weak.sum() > 0:
        out[:, weak] *= rng.uniform(1 - default, 1 + default, weak.sum())[None, :]
    return out


def run_config(seed, s, mode):
    gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=seed)
    paired = gen.generate_paired_slices(rotation=5.0, scale=0.95, batch_effect=0.0)
    slice_A = paired.slice_A.to_dict()
    slice_B = paired.slice_B.to_dict()
    labels_A, labels_B = slice_A['region_labels'], slice_B['region_labels']
    lr_set = set(ALL_LR_GENES) & set(slice_B['gene_names'])
    slice_B['expr'] = apply_structural(slice_B['expr'], list(slice_B['gene_names']),
                                       lr_set, mode, s, seed).astype(np.float32)
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
    log("  结构性批次效应扫描（关键验证）")
    log("  模式: lr=LR基因强缩放(改变C_expr/C_lr相对量级) | nlr=非LR基因强缩放 | all=全部(对照)")
    log("  缩放: 逐基因独立乘性(避免被余弦距离的标度不变性吸收); 固定参数 α=0.5, β=0.3, γ=0.1; 5 seeds")
    log("=" * 72)
    for mode, desc in [('lr', 'LR基因差异化缩放 (C_expr/C_lr 相对量级剧变)'),
                       ('nlr', '非LR基因差异化缩放 (C_expr相对量级剧变)'),
                       ('all', '全部基因差异化缩放 (乘性对照)')]:
        log(f"\n--- 模式: {desc} ---")
        for s in S_STRONG:
            accL, accF, entL = [], [], []
            for seed in SEEDS:
                r = run_config(seed, s, mode)
                accL.append(r['LROT']['acc']); accF.append(r['FGW']['acc']); entL.append(r['LROT']['ent'])
            log(f"  s={s:<4} LROT精度={np.mean(accL):.3f}±{np.std(accL):.3f} | FGW精度={np.mean(accF):.3f}±{np.std(accF):.3f} | "
                f"LROT熵={np.mean(entL):.4f} | Δ(LROT-FGW)={np.mean(accL)-np.mean(accF):+.3f}")

    # 参考: 默认(乘性0.15)精度
    base = []
    for seed in SEEDS:
        r = run_config(seed, 0.15, 'all')
        base.append(r['LROT']['acc'])
    log(f"\n参考: 默认乘性批次0.15 LROT精度={np.mean(base):.3f}")

    with open(os.path.join(OUT_DIR, 'lrot_structural_batch_results.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(buf) + "\n")
    log(f"\n  已保存: {os.path.join(OUT_DIR, 'lrot_structural_batch_results.txt')}")


if __name__ == '__main__':
    main()
