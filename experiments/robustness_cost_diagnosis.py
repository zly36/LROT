# -*- coding: utf-8 -*-
"""
robustness_cost_diagnosis.py -- 成本平衡诊断
目的: 检验 "固定 alpha/beta/gamma 在批次效应下鲁棒" 到底是真鲁棒还是扰动太弱。
做法: 在标准合成切片上 (1,000 spots x 150 genes, rotation=5deg, scale=0.95, batch=0)
      对 slice B 施加两类逐基因批次效应:
        gene    : 每基因乘性 exp(N(0,sigma))      (对数正态)
        geneadd : log1p 空间加性 N(0,sigma), 零截断 (与 batch_effect_scan 一致)
      在最优传输计划 P* 下评估真实目标中的三项贡献:
        expr = alpha * <P, C_expr>      (C_expr: 非LR基因余弦距离, 逐项除以自身max)
        lr   = gamma  * <P, C_lr>
        gw   = (1-alpha)*beta * <P, C_gw>
      同时报告 C_expr / C_lr / C_gw 矩阵相对 sigma=0 的平均绝对变动 (mean |dC|),
      用于判断扰动确实进入成本矩阵(非平凡), 而计划级指标却保持稳定(结构性鲁棒)。
输出: lrot_output/lrot_cost_diagnosis_results.txt (+ .png)
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
                       compute_mapping_consistency, cosine_distance,
                       euclidean_distance_matrix, gromov_cost_matrix,
                       LR_DB, ALL_LR_GENES)
from real_st_loader import RealisticSTGenerator

OUT_DIR = os.path.join(_R, 'lrot_output')
os.makedirs(OUT_DIR, exist_ok=True)
SEEDS = [42 + i * 1000 for i in range(5)]
ALPHA, BETA, GAMMA = 0.5, 0.3, 0.1
SIGMAS = [0.3, 0.5]
MODES = [('gene', 'Per-gene Lognormal'), ('geneadd', 'Log1p additive')]
LBL = {'gene': 'Lognormal', 'geneadd': 'Log1p additive'}
buf = []

def log(s):
    print(s, flush=True)
    buf.append(s)

def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()

def apply_batch(expr, mode, sigma, seed):
    rng = np.random.RandomState(seed + 31)
    if mode == 'gene':
        f = np.exp(rng.normal(0, sigma, expr.shape[1]))
        return expr * f[None, :]
    s = rng.normal(0, sigma, expr.shape[1])
    return np.maximum(np.expm1(np.log1p(expr) + s[None, :]), 0.0)

def cost_matrices(A, B, S_lr):
    names = A['gene_names']
    mask = np.array([g not in set(ALL_LR_GENES) for g in names])
    if mask.sum() > 0:
        Ce = cosine_distance(A['expr'][:, mask], B['expr'][:, mask])
    else:
        Ce = cosine_distance(A['expr'], B['expr'])
    Ce = Ce / np.maximum(Ce.max(), 1e-8)
    Cl = -np.log(S_lr + 1e-4)
    Cl = Cl / np.maximum(Cl.max(), 1e-8)
    DA = euclidean_distance_matrix(A['coords'], A['coords'])
    DB = euclidean_distance_matrix(B['coords'], B['coords'])
    DA = DA / np.maximum(DA.max(), 1e-8)
    DB = DB / np.maximum(DB.max(), 1e-8)
    Cg = gromov_cost_matrix(DA, DB)  # const part placeholder, P-dependent added later
    return Ce, Cl, DA, DB

def eval_config(A, B, S_lr, seed=None):
    P, _ = fgw_lr_solver(A, B, S_lr, gamma=GAMMA, beta=BETA, alpha=ALPHA,
                         max_iter=200, verbose=False)
    acc = compute_mapping_consistency(P, A['region_labels'], B['region_labels'])
    ent = mean_entropy(P)
    Ce, Cl, DA, DB = cost_matrices(A, B, S_lr)
    Cg = gromov_cost_matrix(DA, DB) - 2.0 * (DA @ P @ DB.T)
    Cg = Cg / np.maximum(Cg.max(), 1e-8)
    e_term = ALPHA * float(np.sum(P * Ce))
    l_term = GAMMA * float(np.sum(P * Cl))
    g_term = (1.0 - ALPHA) * BETA * float(np.sum(P * Cg))
    tot = e_term + l_term + g_term
    return {'acc': acc, 'ent': ent, 'P': P, 'Ce': Ce, 'Cl': Cl, 'Cg': Cg,
            'e': e_term, 'l': l_term, 'g': g_term, 'tot': tot,
            'e_sh': e_term / tot, 'l_sh': l_term / tot, 'g_sh': g_term / tot}

def meanabs_delta(M1, M2):
    return float(np.mean(np.abs(M1 - M2)))

def main():
    log('=' * 76)
    log('Cost-balance diagnosis: is "robustness" structural or trivial?')
    log('Fixed alpha=0.5, beta=0.3, gamma=0.1; 5 seeds; rotation=5deg; batch=0 base')
    log('Per-gene batch models on slice B: Lognormal(0,sigma) / Log1p-additive N(0,sigma)')
    log('=' * 76)

    base_rows = []
    rows = {mode: {s: [] for s in SIGMAS} for mode, _ in MODES}
    for seed in SEEDS:
        gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=seed)
        paired = gen.generate_paired_slices(rotation=5.0, scale=0.95, batch_effect=0.0)
        A = paired.slice_A.to_dict()
        B0 = paired.slice_B.to_dict()
        S0, _ = compute_lr_strength_matrix(A, B0, LR_DB)
        r0 = eval_config(A, B0, S0)
        base_rows.append(r0)
        log('seed=%d base: acc=%.3f ent=%.4f shares expr/lr/gw=%.3f/%.3f/%.3f'
            % (seed, r0['acc'], r0['ent'], r0['e_sh'], r0['l_sh'], r0['g_sh']))
        for mode, _ in MODES:
            for sig in SIGMAS:
                B = dict(B0)
                B['expr'] = apply_batch(B0['expr'], mode, sig, seed).astype(np.float32)
                S, _ = compute_lr_strength_matrix(A, B, LR_DB)
                r = eval_config(A, B, S)
                dCe = meanabs_delta(r['Ce'], r0['Ce'])
                dCl = meanabs_delta(r['Cl'], r0['Cl'])
                dCg = meanabs_delta(r['Cg'], r0['Cg'])
                rows[mode][sig].append((seed, r, dCe, dCl, dCg))
                log('  %s sigma=%.1f seed=%d: acc=%.3f ent=%.4f | dCe=%.4f dCl=%.4f dCg=%.4f'
                    % (mode, sig, seed, r['acc'], r['ent'], dCe, dCl, dCg))
        # incremental save
        _save('lrot_cost_diagnosis_results.txt')

    # aggregation
    base_acc = np.mean([r['acc'] for r in base_rows]); base_ent = np.mean([r['ent'] for r in base_rows])
    base_sh = (np.mean([r['e_sh'] for r in base_rows]), np.mean([r['l_sh'] for r in base_rows]),
               np.mean([r['g_sh'] for r in base_rows]))
    log('\n' + '=' * 76)
    log('AGGREGATE (5-seed means)  base acc=%.3f  base entropy=%.4f' % (base_acc, base_ent))
    log('base contribution shares: expr=%.1f%%  lr=%.1f%%  gw=%.1f%%' % (100 * base_sh[0], 100 * base_sh[1], 100 * base_sh[2]))
    log('%-12s %-6s %-10s %-10s %-9s %-9s %-9s %-9s %-9s' % ('mode', 'sigma', 'acc', 'entropy', 'E-share', 'LR-share', 'GW-share', 'dCe', 'dCl'))
    for mode, _ in MODES:
        for sig in SIGMAS:
            dat = rows[mode][sig]
            acc = np.mean([x[1]['acc'] for x in dat]); ent = np.mean([x[1]['ent'] for x in dat])
            es = np.mean([x[1]['e_sh'] for x in dat]); ls = np.mean([x[1]['l_sh'] for x in dat])
            gs = np.mean([x[1]['g_sh'] for x in dat])
            dCe = np.mean([x[2] for x in dat]); dCl = np.mean([x[3] for x in dat])
            log('%-12s %-6.1f %-10.3f %-10.4f %-9.3f %-9.3f %-9.3f %-9.4f %-9.4f'
                % (mode, sig, acc, ent, es, ls, gs, dCe, dCl))

    _save('lrot_cost_diagnosis_results.txt')
    _fig(rows, base_rows)
    log('Saved ' + os.path.join(OUT_DIR, 'lrot_cost_diagnosis_results.txt'))

def _save(fname):
    with open(os.path.join(OUT_DIR, fname), 'w', encoding='utf-8') as f:
        f.write('\n'.join(buf) + '\n')

def _fig(rows, base_rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    xs = [0.0] + SIGMAS
    ax = axes[0]
    for comp, key, col in [('Expr', 'e_sh', '#C0392B'), ('LR', 'l_sh', '#2C3E50'), ('GW', 'g_sh', '#7F8C8D')]:
        for mode, ls in [('gene', '--'), ('geneadd', '-')]:
            ys = [np.mean([r[key] for r in base_rows])]
            errs = [np.std([r[key] for r in base_rows])]
            for sig in SIGMAS:
                dat = rows[mode][sig]
                ys.append(np.mean([x[1][key] for x in dat]))
                errs.append(np.std([x[1][key] for x in dat]))
            ax.errorbar(xs, ys, yerr=errs, fmt='o' + ls, lw=1.6, ms=3.5, capsize=2,
                        color=col, label='%s (%s)' % (comp, LBL[mode]))
    ax.set_xlabel('Batch strength sigma')
    ax.set_ylabel('Contribution share of total objective')
    ax.set_title('(A) Cost contribution shares at optimal plan')
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    for comp, matkey, col in [('C_expr', 'Ce', '#C0392B'), ('C_lr', 'Cl', '#2C3E50'), ('C_gw', 'Cg', '#7F8C8D')]:
        for mode, ls in [('gene', '--'), ('geneadd', '-')]:
            ys = []
            for sig in SIGMAS:
                dat = rows[mode][sig]
                ys.append(np.mean([x[{'Ce': 2, 'Cl': 3, 'Cg': 4}[matkey]] for x in dat]))
            ax.plot(SIGMAS, ys, 'o' + ls, lw=1.6, ms=3.5, color=col, label='%s (%s)' % (comp, LBL[mode]))
    ax.set_xlabel('Batch strength sigma')
    ax.set_ylabel('Mean |C(sigma) - C(0)|')
    ax.set_title('(B) Cost-matrix perturbation actually entering solver')
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'lrot_cost_diagnosis.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()