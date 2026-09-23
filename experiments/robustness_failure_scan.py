# -*- coding: utf-8 -*-
"""
robustness_failure_scan.py -- 固定参数失效域扫描
目的: 展示固定默认参数 (alpha=0.5, beta=0.3, gamma=0.1) 鲁棒性的适用边界。
做法: 采用真正能进入成本矩阵的 log1p 加性批次模型 (geneadd, 零截断),
      扫描 alpha x sigma 网格:
        sigma in {0.0, 0.3, 0.5, 1.0, 1.5}   (0.3-0.5 为批次效应量程, 1.0-1.5 为压测延伸)
        alpha in {0.3, 0.5, 0.7, 0.9}
      另在默认 alpha=0.5 处计算 FGW (gamma=0) 作为对照, 报告 LROT 相对 FGW 的精度差与熵降。
      beta 固定 0.3, gamma(LROT)=0.1。
输出: lrot_output/lrot_failure_domain_results.txt (+ .png)
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
                       compute_mapping_consistency, LR_DB)
from real_st_loader import RealisticSTGenerator

OUT_DIR = os.path.join(_R, 'lrot_output')
os.makedirs(OUT_DIR, exist_ok=True)
SEEDS = [42 + i * 1000 for i in range(5)]
BETA, GAMMA = 0.3, 0.1
SIGMAS = [0.0, 0.3, 0.5, 1.0, 1.5]
ALPHAS = [0.3, 0.5, 0.7, 0.9]
buf = []

def log(s):
    print(s, flush=True)
    buf.append(s)

def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()

def add_geneadd(expr, sigma, seed):
    rng = np.random.RandomState(seed + 31)
    sh = rng.normal(0, sigma, expr.shape[1])
    return np.maximum(np.expm1(np.log1p(expr) + sh[None, :]), 0.0)

def solve(seed, sigma, alpha, gamma=GAMMA):
    gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=seed)
    paired = gen.generate_paired_slices(rotation=5.0, scale=0.95, batch_effect=0.0)
    A = paired.slice_A.to_dict()
    B = paired.slice_B.to_dict()
    if sigma > 0:
        B['expr'] = add_geneadd(B['expr'], sigma, seed).astype(np.float32)
    S, _ = compute_lr_strength_matrix(A, B, LR_DB)
    S_in = S if gamma > 0 else np.zeros_like(S)
    P, _ = fgw_lr_solver(A, B, S_in, gamma=gamma, beta=BETA, alpha=alpha,
                         max_iter=200, verbose=False)
    return (compute_mapping_consistency(P, A['region_labels'], B['region_labels']),
            mean_entropy(P))

def main():
    log('=' * 78)
    log('Failure-domain scan: robustness limit of fixed default weights (alpha=0.5,beta=0.3,gamma=0.1)')
    log('log1p-additive batch on slice B, sigma in {0,0.3,0.5,1.0,1.5}; 5 seeds; rotation=5deg')
    log('Grid: LROT gamma=0.1, alpha in {0.3,0.5,0.7,0.9}; FGW(gamma=0) at alpha=0.5 as reference')
    log('=' * 78)

    # LROT x FGW at default alpha=0.5 across sigma
    ref = {}
    log('\n--- Default alpha=0.5 : LROT vs FGW vs batch strength ---')
    log('%-6s %-16s %-16s %-8s %-14s %-14s %-8s' %
        ('sigma', 'LROT acc', 'FGW acc', 'dAcc', 'LROT entropy', 'FGW entropy', 'EntDrop%'))
    for sig in SIGMAS:
        la, fa, le, fe = [], [], [], []
        for sd in SEEDS:
            a1, e1 = solve(sd, sig, 0.5, GAMMA)
            a0, e0 = solve(sd, sig, 0.5, 0.0)
            la.append(a1); fa.append(a0); le.append(e1); fe.append(e0)
        ref[sig] = (np.mean(la), np.std(la), np.mean(fa), np.std(fa),
                    np.mean(le), np.std(le), np.mean(fe), np.std(fe))
        mla, sla, mfa, sfa, mle, sle, mfe, sfe = ref[sig]
        drop = (mfe - mle) / mfe * 100 if mfe > 0 else 0.0
        log('%-6.1f %-16.3f %-16.3f %-8.3f %-14.4f %-14.4f %-8.1f' %
            (sig, mla, mfa, mla - mfa, mle, mfe, drop))
        _save()
    base_acc = ref[0.0][0]
    for sig in [0.3, 0.5, 1.0, 1.5]:
        log('  limit check: sigma=%.1f default acc=%.3f (rel change %.1f%% vs sigma=0)'
            % (sig, ref[sig][0], (ref[sig][0] - base_acc) / base_acc * 100))

    # alpha x sigma grid
    grid = {}
    log('\n--- alpha x sigma grid (LROT, gamma=0.1) : accuracy / entropy ---')
    log('%-6s %s' % ('sigma', ' '.join('a=%.1f acc/ent' % a for a in ALPHAS)))
    for sig in SIGMAS:
        row = {}
        for al in ALPHAS:
            acc, ent = [], []
            for sd in SEEDS:
                a, e = solve(sd, sig, al, GAMMA)
                acc.append(a); ent.append(e)
            row[al] = (np.mean(acc), np.std(acc), np.mean(ent), np.std(ent))
            log('  sigma=%.1f alpha=%.1f -> acc=%.3f +/- %.3f | ent=%.4f +/- %.4f'
                % (sig, al, row[al][0], row[al][1], row[al][2], row[al][3]))
        grid[sig] = row
        _save()

    # best alpha per sigma by mean accuracy
    log('\n--- Best alpha per sigma (mean accuracy) ---')
    log('%-6s %-10s %-14s %-10s' % ('sigma', 'best alpha', 'best acc', 'default acc'))
    for sig in SIGMAS:
        best_al = max(ALPHAS, key=lambda al: grid[sig][al][0])
        log('%-6.1f %-10.1f %-14.3f %-10.3f' % (sig, best_al, grid[sig][best_al][0], grid[sig][0.5][0]))

    _save()

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    cols = ['#2C3E50', '#C0392B', '#16A085', '#8E44AD']
    ax = axes[0]
    for al, c in zip(ALPHAS, cols):
        ys = [grid[s][al][0] for s in SIGMAS]
        err = [grid[s][al][1] for s in SIGMAS]
        ax.errorbar(SIGMAS, ys, yerr=err, fmt='o-', lw=1.6, ms=4, capsize=2.5, color=c,
                    label='alpha=%.1f (LR)' % al)
    mfa = [ref[s][2] for s in SIGMAS]
    sfa = [ref[s][3] for s in SIGMAS]
    ax.errorbar(SIGMAS, mfa, yerr=sfa, fmt='s--', lw=1.5, ms=4, capsize=2.5,
                color='#7F8C8D', label='FGW alpha=0.5')
    ax.axvspan(0.0, 0.5, color='green', alpha=0.06)
    ax.text(0.22, ax.get_ylim()[0], 'fixed-weight scan range', fontsize=7.5, color='green')
    ax.set_xlabel('Batch strength sigma (log1p additive)')
    ax.set_ylabel('Region-mapping accuracy')
    ax.set_title('(C) Accuracy: where fixed weights fail')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    for al, c in zip(ALPHAS, cols):
        ys = [grid[s][al][2] for s in SIGMAS]
        ax.plot(SIGMAS, ys, 'o-', lw=1.6, ms=4, color=c, label='alpha=%.1f (LR)' % al)
    mfe = [ref[s][6] for s in SIGMAS]
    ax.plot(SIGMAS, mfe, 's--', lw=1.5, ms=4, color='#7F8C8D', label='FGW alpha=0.5')
    ax.set_xlabel('Batch strength sigma (log1p additive)')
    ax.set_ylabel('Transport entropy')
    ax.set_title('(D) Mapping sharpness under batch stress')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'lrot_failure_domain.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    log('Saved ' + os.path.join(OUT_DIR, 'lrot_failure_domain_results.txt'))
    log('      ' + os.path.join(OUT_DIR, 'lrot_failure_domain.png'))

def _save():
    with open(os.path.join(OUT_DIR, 'lrot_failure_domain_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(buf) + '\n')

if __name__ == '__main__':
    main()