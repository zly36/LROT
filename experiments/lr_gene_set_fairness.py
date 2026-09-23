# -*- coding: utf-8 -*-
"""
lr_gene_set_fairness.py -- LR 基因集公平性对照实验
回应: C_expr 仅用非LR基因(约2/3) 是否造成 LROT 与 FGW 基线"信息不对等"。
设计: 与主实验一致的数据 (RealisticSTGenerator, 1,000 spots x 150 genes,
      rotation=5deg, scale=0.95, batch_effect=0.15 默认), 5 个随机种子;
      对同一对切片在两种基因集策略下各跑 LROT(gamma=0.1) 与 FGW(gamma=0):
        (a) exclude-LR : C_expr 排除 LR 基因 (当前默认, lr_gene_list=ALL_LR_GENES)
        (b) all-genes  : C_expr 使用全部基因 (lr_gene_list=[] 即不排除)
      检验: (1) LROT(exclude) 相对 FGW(exclude) 与 FGW(all-genes) 的熵降/精度差是否成立;
            (2) 同一 gamma 下排除 vs 全基因策略结果是否一致 (冗余担忧是否改变结论)。
输出: lrot_output/lrot_lr_gene_fairness_results.txt (+ .png)
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
ALPHA, BETA = 0.5, 0.3
CONFIGS = [
    ('FGW-exclude', 0.0, None),
    ('LROT-exclude', 0.1, None),
    ('FGW-allgenes', 0.0, []),
    ('LROT-allgenes', 0.1, []),
]
buf = []
def log(s):
    print(s, flush=True)
    buf.append(s)

def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()

def run_one(seed):
    gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=seed)
    paired = gen.generate_paired_slices(rotation=5.0, scale=0.95, batch_effect=0.15)
    A = paired.slice_A.to_dict()
    B = paired.slice_B.to_dict()
    S, _ = compute_lr_strength_matrix(A, B, LR_DB)
    n_lr = sum(1 for g in A['gene_names'] if g in set(ALL_LR_GENES))
    out = {}
    for tag, gamma, lr_list in CONFIGS:
        P, _ = fgw_lr_solver(A, B, S, gamma=gamma, beta=BETA, alpha=ALPHA,
                             lr_gene_list=lr_list, max_iter=200, verbose=False)
        out[tag] = {'acc': compute_mapping_consistency(P, A['region_labels'], B['region_labels']),
                    'ent': mean_entropy(P)}
    return out, n_lr

def main():
    log('=' * 78)
    log('Item-3 control: LR genes excluded from C_expr vs all-genes C_expr')
    log('Data: 1,000 spots x 150 genes, rotation=5deg, scale=0.95, batch_effect=0.15 (main default)')
    log('5 seeds; alpha=0.5, beta=0.3; LROT gamma=0.1 vs FGW gamma=0')
    log('=' * 78)

    res = {tag: [] for tag, _, _ in CONFIGS}
    n_lr_vals = []
    for seed in SEEDS:
        r, n_lr = run_one(seed)
        n_lr_vals.append(n_lr)
        for tag in res:
            res[tag].append(r[tag])
        line = 'seed=%d | ' % seed + ' '.join('%s acc=%.3f ent=%.4f' % (t, r[t]['acc'], r[t]['ent']) for t, _, _ in CONFIGS)
        log(line)
        _save()

    log('\nC_expr gene usage: n_total=%d, n_LR=%d (excluded in "exclude" policy), n_nonLR=%d'
        % (len(range(150)), n_lr_vals[0], 150 - n_lr_vals[0]))
    log('\n' + '=' * 78)
    log('AGGREGATE (5-seed means +/- std)')
    log('%-18s %-14s %-12s %-10s' % ('config', 'accuracy', 'entropy', 'n'))
    for tag, _, _ in CONFIGS:
        acc = np.mean([x['acc'] for x in res[tag]])
        ent = np.mean([x['ent'] for x in res[tag]])
        log('%-18s %-14.3f %-12.4f' % (tag, acc, ent))

    def stat(tag):
        return (np.mean([x['acc'] for x in res[tag]]), np.std([x['acc'] for x in res[tag]]),
                np.mean([x['ent'] for x in res[tag]]), np.std([x['ent'] for x in res[tag]]))

    ae, af, be, bf = stat('FGW-exclude'), stat('LROT-exclude'), stat('FGW-allgenes'), stat('LROT-allgenes')
    log('\nFairness checks (mean over 5 seeds):')
    log('  1) LROT(exclude) vs FGW(exclude) : dAcc=%+.3f | entropy drop=%.2f%%' % (af[0] - ae[0], (ae[2] - af[2]) / ae[2] * 100))
    log('  2) LROT(exclude) vs FGW(all-genes full info): dAcc=%+.3f | entropy drop=%.2f%%' % (af[0] - be[0], (be[2] - af[2]) / be[2] * 100))
    log('  3) gene-set effect under gamma=0 : dAcc=%+.3f | dEntropy=%+.4f  (FGW-allgen vs FGW-excl)' % (be[0] - ae[0], be[2] - ae[2]))
    log('  4) gene-set effect under gamma=.1: dAcc=%+.3f | dEntropy=%+.4f  (LROT-allgen vs LROT-excl)' % (bf[0] - af[0], bf[2] - af[2]))
    log('\nInterpretation: if (2) remains positive/comparable and (3)-(4) are small,')
    log('the LR entropy benefit and accuracy parity do not come from an information-poor FGW baseline.')

    _save()
    _fig(res)
    log('Saved ' + os.path.join(OUT_DIR, 'lrot_lr_gene_fairness_results.txt'))
    log('      ' + os.path.join(OUT_DIR, 'lrot_lr_gene_fairness.png'))

def _save():
    with open(os.path.join(OUT_DIR, 'lrot_lr_gene_fairness_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(buf) + '\n')

def _fig(res):
    tags = [t for t, _, _ in CONFIGS]
    acc = [np.mean([x['acc'] for x in res[t]]) for t in tags]
    acc_sd = [np.std([x['acc'] for x in res[t]]) for t in tags]
    ent = [np.mean([x['ent'] for x in res[t]]) for t in tags]
    ent_sd = [np.std([x['ent'] for x in res[t]]) for t in tags]
    x = np.arange(4)
    colors = ['#7F8C8D', '#C0392B', '#95A5A6', '#E67E22']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    bars = ax.bar(x, acc, yerr=acc_sd, width=0.62, color=colors, capsize=3, alpha=0.9)
    for i, v in enumerate(acc):
        ax.text(i, v + 0.006, '%.3f' % v, ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=12, fontsize=8)
    ax.set_ylabel('Region-mapping accuracy')
    ax.set_ylim(0.55, 0.8)
    ax.set_title('(A) Accuracy by gene-set policy')
    ax.grid(True, axis='y', alpha=0.3)
    ax = axes[1]
    bars = ax.bar(x, ent, yerr=ent_sd, width=0.62, color=colors, capsize=3, alpha=0.9)
    for i, v in enumerate(ent):
        ax.text(i, v + 0.03, '%.3f' % v, ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=12, fontsize=8)
    ax.set_ylabel('Transport entropy')
    ax.set_ylim(0, 6.5)
    ax.set_title('(B) Entropy by gene-set policy')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'lrot_lr_gene_fairness.png'), dpi=360, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()