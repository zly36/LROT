# -*- coding: utf-8 -*-
"""Figure S17 support checks.

1) FGW (gamma=0) baseline ceiling over an alpha x beta grid
   (synthetic 4-region data and realistic 6-region data);
2) entropic-regularisation (eps) sensitivity on the realistic dataset;
3) LR-spatial-program-absent control (synthetic and realistic).
Metrics mirror the manuscript: row-normalised soft-plan vote accuracy and
row-mean transport entropy (natural log).
"""
import os
import sys, json, time
from pathlib import Path

# 自定位仓库根：向上找到含 lrot_core.py 的目录（代码包解压到任意路径均可运行）
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

WS = Path(_R)
OUT = WS / "lrot_output" / "S17_support"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WS))

import numpy as np

import lrot_core as core
from real_st_loader import (RealisticSTGenerator, LR_DB,
                            lr_gene_list, ALL_LR_GENES as RL_LR)

LIGANDS, RECEPTORS = lr_gene_list(LR_DB)


# ------------------------------------------------------------ metrics
def mean_row_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    ent = -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1)
    return float(ent.mean())


def soft_vote_accuracy(P, labels_A, labels_B):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    n_regions = int(max(labels_A.max(), labels_B.max())) + 1
    pred = Pn @ np.eye(n_regions)[labels_B]
    pred = pred.argmax(1)
    return float(np.mean(pred == labels_A))


# ------------------------------------------- synthetic 4-region data
def simulate_slice_4reg(n_spots=1000, n_genes=200, seed=42, lr_program=True,
                        x_range=(0, 20), y_range=(0, 20)):
    """Copy of lrot_prototype.simulate_slice (N_REGIONS=4) plus an
    lr_program switch: False removes LR region enrichment, so LR genes
    behave as uninformative background genes."""
    N_REGIONS = 4
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(n_spots)))
    xs = np.linspace(x_range[0], x_range[1], side)
    ys = np.linspace(y_range[0], y_range[1], side)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    coords = coords[:n_spots] + rng.randn(n_spots, 2) * 0.3

    center = np.array([(x_range[0] + x_range[1]) / 2,
                       (y_range[0] + y_range[1]) / 2])
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    region_labels = np.floor((angles + np.pi) / (2 * np.pi / N_REGIONS)).astype(int)
    region_labels = np.clip(region_labels, 0, N_REGIONS - 1)

    n_lr0 = len(RL_LR)
    gene_names = RL_LR[:] + [f"Gene_{i}" for i in range(max(0, n_genes - n_lr0))]
    gene_names = gene_names[:n_genes]

    expr = rng.gamma(1, 1, (n_spots, n_genes)).astype(np.float32) * 0.5

    if lr_program:
        region_weights = {}
        for reg in range(N_REGIONS):
            region_weights[reg] = {}
            pathways = [LIGANDS[:8], LIGANDS[8:16], LIGANDS[16:24], LIGANDS[24:32]]
            for lig in pathways[reg]:
                if lig in RL_LR and lig in gene_names:
                    idx = gene_names.index(lig)
                    region_weights[reg][idx] = 3.0
            rec_pathways = [RECEPTORS[:8], RECEPTORS[8:16], RECEPTORS[16:24], RECEPTORS[24:32]]
            for rec in rec_pathways[reg]:
                if rec in RL_LR and rec in gene_names:
                    idx = gene_names.index(rec)
                    region_weights[reg][idx] = 2.5
        for spot in range(n_spots):
            reg = region_labels[spot]
            if reg in region_weights:
                for gidx, w in region_weights[reg].items():
                    expr[spot, gidx] *= w

    expr += rng.gamma(0.5, 0.5, expr.shape) * 0.3
    return {'coords': coords, 'expr': expr,
            'region_labels': region_labels, 'gene_names': gene_names,
            'n_spots': n_spots, 'n_genes': n_genes}


def synthetic_pair(seed_A=42, seed_B=123, lr_program=True):
    A = simulate_slice_4reg(seed=seed_A, lr_program=lr_program)
    B = simulate_slice_4reg(seed=seed_B, lr_program=lr_program)
    theta = 0.15
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    center = B['coords'].mean(0)
    B['coords'] = (B['coords'] - center) @ R.T + center + np.array([0.5, 0.3])
    return A, B


# -------------------------------------------- realistic (6-region) data
class ControlSTGenerator(RealisticSTGenerator):
    """lr_program=False: LR genes are never picked as region marker genes and
    receive no LR hotspot enhancement (uninformative background)."""

    def __init__(self, *args, lr_program=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_program = lr_program

    def _generate_region_profiles(self, n_regions, n_genes):
        if self.lr_program:
            return super()._generate_region_profiles(n_regions, n_genes)
        profiles = np.zeros((n_regions, n_genes))
        n_marker = max(n_genes // (n_regions * 2), 5)
        n_lr = min(len(RL_LR), n_genes // 3)
        bg_pool = np.arange(n_lr, n_genes)
        if len(bg_pool) == 0:
            bg_pool = np.arange(n_genes)
        for r in range(n_regions):
            n_mark = min(n_marker, len(bg_pool))
            marker_genes = self.rng.choice(bg_pool, n_mark, replace=False)
            profiles[r, marker_genes] = self.rng.uniform(2.0, 5.0, n_mark)
            bg = np.setdiff1d(np.arange(n_genes), marker_genes)
            profiles[r, bg] = self.rng.uniform(0.1, 0.5, len(bg))
        for r in range(n_regions - 1):
            overlap = self.rng.choice(np.arange(n_genes), n_marker // 3, replace=False)
            profiles[r + 1, overlap] += profiles[r, overlap] * 0.3
        return profiles

    def _add_lr_hotspots(self, coords, expr, region_labels, lr_gene_indices,
                         hotspot_fraction=0.15):
        if not self.lr_program:
            return expr
        return super()._add_lr_hotspots(coords, expr, region_labels,
                                       lr_gene_indices, hotspot_fraction)


def realistic_pair(seed=42, lr_program=True, n_spots=1000, n_genes=150):
    gen = ControlSTGenerator(n_spots_A=n_spots, n_spots_B=n_spots,
                             n_genes=n_genes, seed=seed,
                             lr_program=lr_program)
    paired = gen.generate_paired_slices(rotation=5.0, shear=0.02,
                                        scale=0.95, batch_effect=0.15)
    A = paired.slice_A.to_dict()
    B = paired.slice_B.to_dict()
    return A, B


def solve(A, B, S_lr, gamma=0.1, alpha=0.5, beta=0.3,
          reg_ot=0.01, reg_gw=0.01, max_iter=200):
    P, _ = core.fgw_lr_solver(A, B, S_lr, gamma=gamma, beta=beta,
                              reg_gw=reg_gw, reg_ot=reg_ot, alpha=alpha,
                              max_iter=max_iter, tol=1e-5, verbose=False)
    return P


def metrics_for(P, A, B):
    return {'acc': soft_vote_accuracy(P, A['region_labels'], B['region_labels']),
            'entropy': mean_row_entropy(P)}


# ---------------------------------------------------------------- I/O
def dump(name, obj):
    path = OUT / (name + '.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    print('  [save]', path)
    return path


def write_summary(name, text):
    path = OUT / (name + '.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print('  [save]', path)


def run_sanity():
    print('=== sanity: synthetic 4-region defaults ===')
    A, B = synthetic_pair()
    S_lr, _ = core.compute_lr_strength_matrix(A, B, LR_DB, scale=10.0)
    for g in [0.0, 0.1]:
        t0 = time.time()
        P = solve(A, B, S_lr, gamma=g, max_iter=120)
        m = metrics_for(P, A, B)
        print('  gamma=%s: acc=%.4f entropy=%.4f (%.1fs)' % (g, m['acc'], m['entropy'], time.time() - t0))

    print('=== sanity: realistic 6-region defaults ===')
    A, B = realistic_pair(seed=42)
    S_lr, _ = core.compute_lr_strength_matrix(A, B, LR_DB, scale=10.0)
    for g in [0.0, 0.1]:
        t0 = time.time()
        P = solve(A, B, S_lr, gamma=g, max_iter=200)
        m = metrics_for(P, A, B)
        print('  gamma=%s: acc=%.4f entropy=%.4f (%.1fs)' % (g, m['acc'], m['entropy'], time.time() - t0))


def run_sweep():
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    betas = [0.1, 0.3, 0.6]
    res = {}
    for ds in ['synthetic', 'realistic']:
        A, B = (synthetic_pair(seed_A=42, seed_B=123) if ds == 'synthetic'
                else realistic_pair(seed=42))
        S_lr, _ = core.compute_lr_strength_matrix(A, B, LR_DB, scale=10.0)
        rows = []
        for alpha in alphas:
            for beta in betas:
                for gamma in [0.0, 0.1]:
                    t0 = time.time()
                    P = solve(A, B, S_lr, gamma=gamma, alpha=alpha, beta=beta,
                              max_iter=120 if ds == 'synthetic' else 200)
                    m = metrics_for(P, A, B)
                    rows.append({'alpha': alpha, 'beta': beta, 'gamma': gamma, **m})
                    print('%s a=%s b=%s g=%s: acc=%.4f ent=%.4f (%.1fs)'
                          % (ds, alpha, beta, gamma, m['acc'], m['entropy'], time.time() - t0))
        res[ds] = {'grid': {'alphas': alphas, 'betas': betas}, 'rows': rows}
    dump('baseline_ceiling_sweep', res)

    # short human-readable summary
    lines = []
    for ds, d in res.items():
        rows = d['rows']
        fgw = [r for r in rows if r['gamma'] == 0.0]
        lrot = [r for r in rows if r['gamma'] == 0.1]
        fgw_best = max(fgw, key=lambda r: r['acc'])
        lrot_def = next(r for r in lrot if abs(r['alpha'] - 0.5) < 1e-9 and abs(r['beta'] - 0.3) < 1e-9)
        lrot_best = max(lrot, key=lambda r: r['acc'])
        lines.append('== %s ==' % ds)
        lines.append('default LROT(a=.5,b=.3,g=.1): acc=%.4f ent=%.4f' % (lrot_def['acc'], lrot_def['entropy']))
        lines.append('best FGW over grid (g=0):      acc=%.4f ent=%.4f at a=%s b=%s'
                     % (fgw_best['acc'], fgw_best['entropy'], fgw_best['alpha'], fgw_best['beta']))
        lines.append('best LROT over grid (g=.1):    acc=%.4f ent=%.4f at a=%s b=%s'
                     % (lrot_best['acc'], lrot_best['entropy'], lrot_best['alpha'], lrot_best['beta']))
        fgw_def = next(r for r in fgw if abs(r['alpha'] - 0.5) < 1e-9 and abs(r['beta'] - 0.3) < 1e-9)
        lines.append('default FGW(a=.5,b=.3,g=0):    acc=%.4f ent=%.4f' % (fgw_def['acc'], fgw_def['entropy']))
    write_summary('baseline_ceiling_summary', '\n'.join(lines) + '\n')


def run_eps():
    A, B = realistic_pair(seed=42)
    S_lr, _ = core.compute_lr_strength_matrix(A, B, LR_DB, scale=10.0)
    rows = []
    for mult in [0.5, 1.0, 2.0]:
        for gamma in [0.0, 0.1]:
            P = solve(A, B, S_lr, gamma=gamma, reg_ot=0.01 * mult,
                      reg_gw=0.01 * mult, max_iter=200)
            m = metrics_for(P, A, B)
            rows.append({'eps_mult': mult, 'gamma': gamma, **m})
            print('mult=%s gamma=%s: acc=%.4f ent=%.4f' % (mult, gamma, m['acc'], m['entropy']))
    dump('eps_sensitivity', {'dataset': 'realistic', 'seed': 42, 'rows': rows})
    lines = ['eps-sensitivity (realistic 6-region, seed 42)']
    lines.append('mult  gamma  acc     entropy')
    for r in rows:
        lines.append('%.2f  %.1f  %.4f  %.4f' % (r['eps_mult'], r['gamma'], r['acc'], r['entropy']))
    write_summary('eps_sensitivity_summary', '\n'.join(lines) + '\n')


def run_control():
    seeds = (42, 43, 44)
    out = {'synthetic': [], 'realistic': []}
    for ds in ['synthetic', 'realistic']:
        for seed in seeds:
            for lr_program in [True, False]:
                if ds == 'synthetic':
                    A, B = synthetic_pair(seed_A=seed, seed_B=seed + 81,
                                          lr_program=lr_program)
                    max_iter = 120
                else:
                    A, B = realistic_pair(seed=seed, lr_program=lr_program)
                    max_iter = 200
                S_lr, _ = core.compute_lr_strength_matrix(A, B, LR_DB, scale=10.0)
                for gamma in [0.0, 0.1, 0.2]:
                    P = solve(A, B, S_lr, gamma=gamma, alpha=0.5, beta=0.3,
                              max_iter=max_iter)
                    m = metrics_for(P, A, B)
                    out[ds].append({'seed': seed, 'lr_program': lr_program,
                                    'gamma': gamma, **m})
                    print('%s seed=%s lr_program=%s gamma=%s: acc=%.4f ent=%.4f'
                          % (ds, seed, lr_program, gamma, m['acc'], m['entropy']))
    dump('lr_program_control', out)

    lines = []
    for ds in ['synthetic', 'realistic']:
        lines.append('== %s control (3 seeds) ==' % ds)
        for lr_program in [True, False]:
            sub = [r for r in out[ds] if r['lr_program'] == lr_program]
            for gamma in [0.0, 0.1, 0.2]:
                g = [r for r in sub if r['gamma'] == gamma]
                acc = float(np.mean([r['acc'] for r in g]))
                ent = float(np.mean([r['entropy'] for r in g]))
                lines.append('lr_program=%-5s gamma=%.1f  acc=%.4f+-%.4f ent=%.4f+-%.4f'
                             % (lr_program, gamma, acc,
                                float(np.std([r['acc'] for r in g])),
                                ent, float(np.std([r['entropy'] for r in g]))))
    write_summary('lr_program_control_summary', '\n'.join(lines) + '\n')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'sanity'
    t0 = time.time()
    if stage == 'sanity':
        run_sanity()
    elif stage == 'sweep':
        run_sweep()
    elif stage == 'eps':
        run_eps()
    elif stage == 'control':
        run_control()
    else:
        print('unknown stage:', stage)
    print('[done] stage=%s elapsed=%.1fs' % (stage, time.time() - t0))