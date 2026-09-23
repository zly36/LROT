# -*- coding: utf-8 -*-
"""
run_dlpfc_baselines.py — DLPFC 真实数据上的外部基线对比
数据/口径与 run_dlpfc_visium.py 完全一致 (同一供体 Br8325, 151507/151508/151509,
每张 1,000 spots, 7 层层标注; B 施加旋转2°/平移0.5 错位)。
基线: FGW(γ=0) / LROT(γ=0.1) [本文实现] + 官方PASTE(paste_math, KL) / PASTE2(paste2, KL) / STAligner(官方实现)
预处理说明:
  - OT 方法 (FGW/LROT/PASTE/PASTE2): 原始 count。PASTE/PASTE2 的 KL 相异性内部加 0.01 伪计数 (官方实现自带)。
  - STAligner: 官方推荐预处理 (normalize_total + log1p + PCA-50), 潜空间最近邻评估, 其熵口径与传输熵不同。
输出: lrot_output/dlpfc_baselines_results.txt
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'experiments'))

import numpy as np

OUT_DIR = os.path.join(_R, 'lrot_output')
N_SAMPLE = 1000
GAMMA = 0.1
SEED_A, SEED_B = 42, 99
PAIRS = [("151507", "151508"), ("151508", "151509")]
LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]

from run_dlpfc_visium import load_dlpfc_slice, eval_plan
from run_cancer_visium import HUMAN_LR_DB, apply_misalignment, compute_lr_strength_human
from lrot_core import fgw_lr_solver

def to_anndata(sd):
    import anndata
    ad = anndata.AnnData(X=sd['expr'].astype(np.float64))
    ad.var_names = [str(g) for g in sd['gene_names']]
    ad.obs_names = [f"spot{i}" for i in range(sd['n_spots'])]
    ad.obsm['spatial'] = sd['coords'].astype(np.float64)
    return ad

def run_paste(slice_A, slice_B, alpha):
    from paste_math import pairwise_align
    adA, adB = to_anndata(slice_A), to_anndata(slice_B)
    t0 = time.time()
    pi = pairwise_align(adA, adB, alpha=alpha, dissimilarity='kl',
                        numItermax=200, verbose=False)
    dt = time.time() - t0
    m = eval_plan(pi, slice_A['region_labels'], slice_B['region_labels'],
                  slice_A['expr'], slice_B['expr'],
                  slice_A['coords'], slice_B['coords'])
    m['time'] = dt
    return m

def run_paste2(slice_A, slice_B, alpha=0.1):
    from paste2.PASTE2 import partial_pairwise_align
    adA, adB = to_anndata(slice_A), to_anndata(slice_B)
    p = np.ones(adA.shape[0]) / adA.shape[0]
    q = np.ones(adB.shape[0]) / adB.shape[0]
    s = float(min(p.sum(), q.sum()))
    t0 = time.time()
    pi = partial_pairwise_align(adA, adB, s=s, alpha=alpha, dissimilarity='kl',
                                a_distribution=p, b_distribution=q, verbose=False)
    dt = time.time() - t0
    m = eval_plan(pi, slice_A['region_labels'], slice_B['region_labels'],
                  slice_A['expr'], slice_B['expr'],
                  slice_A['coords'], slice_B['coords'])
    m['time'] = dt
    return m

def run_staligner(slice_A, slice_B, n_epochs=200):
    import scanpy as sc
    import scipy.sparse as sp
    import STAligner as sta
    from scipy.spatial.distance import cdist

    def mk_adata(sd, batch):
        ad = sc.AnnData(X=sd['expr'].astype(np.float32))
        ad.obsm['spatial'] = sd['coords'].astype(np.float32)
        ad.obs['batch'] = batch
        ad.obs['region'] = sd['region_labels'].astype(str)
        return ad

    adata1, adata2 = mk_adata(slice_A, 'A'), mk_adata(slice_B, 'B')
    adata = adata1.concatenate(adata2, batch_key='batch', index_unique='_')
    adata.obs['batch_name'] = adata.obs['batch'].values
    # STAligner 预处理 (与本论文补充S2描述一致): normalize_total + log1p + PCA-50;
    # DLPFC 全转录组(33,538基因)在CPU上直接以全基因表达训练不可行(官方训练固定500轮STAGATE预训练,
    # 输入为 data.X.todense()), 故以PCA-50潜变量作为图自编码器输入(与合成数据基线 n_epochs=200 同配置);
    # anndata 的 X 形状必须等于 (n_obs, n_var), 因此训练在单独的 PCA-X AnnData 上进行。
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.X = sp.csr_matrix(adata.X)
    sc.tl.pca(adata, n_comps=50)
    sta.Cal_Spatial_Net(adata, rad_cutoff=150, verbose=False)
    mnns = sta.create_dictionary_mnn(adata, use_rep='X_pca', batch_name='batch_name',
                                     k=50, save_on_disk=False, verbose=0)
    name2idx = {n: i for i, n in enumerate(adata.obs_names)}
    edge_list = [[], []]
    for bpair in mnns:
        for anchor, neighs in mnns[bpair].items():
            ai = name2idx.get(str(anchor))
            if ai is None:
                continue
            for n in neighs:
                ni = name2idx.get(str(n))
                if ni is None:
                    continue
                edge_list[0].append(ai)
                edge_list[1].append(ni)
    adata.uns['edgeList'] = np.array(edge_list)
    ad_tr = sc.AnnData(X=sp.csr_matrix(adata.obsm['X_pca'].astype(np.float32)))
    ad_tr.obs = adata.obs.copy()
    ad_tr.obsm = adata.obsm.copy()
    ad_tr.uns = adata.uns.copy()
    t0 = time.time()
    sta.train_STAligner(ad_tr, n_epochs=n_epochs, lr=0.001, verbose=False,
                        device='cpu', random_seed=666)
    dt = time.time() - t0
    lat = ad_tr.obsm['STAligner']
    b = adata.obs['batch'].values
    lat_A, lat_B = lat[b == '0'], lat[b == '1']
    D = cdist(lat_A, lat_B)
    idx = D.argmin(axis=1)
    pred = slice_B['region_labels'][idx]
    acc = float(np.mean(pred == slice_A['region_labels']))
    from sklearn.metrics import adjusted_rand_score
    ari = float(adjusted_rand_score(slice_A['region_labels'], pred))
    sim = 1.0 / (D + 1e-8)
    sim_n = sim / sim.sum(1, keepdims=True)
    ent = float(-np.sum(sim_n * np.log(np.maximum(sim_n, 1e-10)), axis=1).mean())
    return {'entropy': ent, 'soft_ari': ari, 'same_mass': float('nan'),
            'soft_acc': acc, 'time': dt, 'latent_nn': True}

def run_pair(sample_A, sample_B):
    slice_A = load_dlpfc_slice(sample_A, N_SAMPLE, SEED_A)
    slice_B = load_dlpfc_slice(sample_B, N_SAMPLE, SEED_B)
    slice_B['coords'] = apply_misalignment(slice_B['coords'],
                                           rotation_deg=2.0, translation=0.5)
    ligs = sorted(set(k[0] for k in HUMAN_LR_DB) | set(k[1] for k in HUMAN_LR_DB))
    S_lr, _ = compute_lr_strength_human(slice_A, slice_B, HUMAN_LR_DB)
    S_zero = np.zeros_like(S_lr)

    def solve(S, gamma):
        return fgw_lr_solver(slice_A, slice_B, S, gamma=gamma, beta=0.3, alpha=0.5,
                             max_iter=200, verbose=False, lr_gene_list=ligs)

    print(f"\n{'='*70}\n  对 {sample_A} → {sample_B}\n{'='*70}", flush=True)
    out = {'pair': f"{sample_A}-{sample_B}"}

    t0 = time.time(); P_fgw, _ = solve(S_zero, 0.0); dt = time.time() - t0
    out['fgw'] = eval_plan(P_fgw, slice_A['region_labels'], slice_B['region_labels'],
                           slice_A['expr'], slice_B['expr'], slice_A['coords'], slice_B['coords'])
    out['fgw']['time'] = dt
    print(f"  FGW(γ=0):       熵={out['fgw']['entropy']:.4f} 软ARI={out['fgw']['soft_ari']:.3f} "
          f"同层={out['fgw']['same_mass']:.4f} 时间={dt:.1f}s", flush=True)

    t0 = time.time(); P_lrot, _ = solve(S_lr, GAMMA); dt = time.time() - t0
    out['lrot'] = eval_plan(P_lrot, slice_A['region_labels'], slice_B['region_labels'],
                            slice_A['expr'], slice_B['expr'], slice_A['coords'], slice_B['coords'])
    out['lrot']['time'] = dt
    print(f"  LROT(γ=0.1):    熵={out['lrot']['entropy']:.4f} 软ARI={out['lrot']['soft_ari']:.3f} "
          f"同层={out['lrot']['same_mass']:.4f} 时间={dt:.1f}s", flush=True)

    out['paste01'] = run_paste(slice_A, slice_B, 0.1)
    print(f"  PASTE(α=0.1):   熵={out['paste01']['entropy']:.4f} 软ARI={out['paste01']['soft_ari']:.3f} "
          f"同层={out['paste01']['same_mass']:.4f} 时间={out['paste01']['time']:.1f}s", flush=True)
    out['paste05'] = run_paste(slice_A, slice_B, 0.5)
    print(f"  PASTE(α=0.5):   熵={out['paste05']['entropy']:.4f} 软ARI={out['paste05']['soft_ari']:.3f} "
          f"同层={out['paste05']['same_mass']:.4f} 时间={out['paste05']['time']:.1f}s", flush=True)

    out['paste2'] = run_paste2(slice_A, slice_B, 0.1)
    print(f"  PASTE2(α=0.1):  熵={out['paste2']['entropy']:.4f} 软ARI={out['paste2']['soft_ari']:.3f} "
          f"同层={out['paste2']['same_mass']:.4f} 时间={out['paste2']['time']:.1f}s", flush=True)

    out['stal'] = run_staligner(slice_A, slice_B)
    print(f"  STAligner:      熵(latent)={out['stal']['entropy']:.4f} ARI(NN)={out['stal']['soft_ari']:.3f} "
          f"准确率(NN)={out['stal']['soft_acc']:.3f} 时间={out['stal']['time']:.1f}s", flush=True)
    return out

def main():
    print("=" * 70)
    print("  DLPFC 外部基线对比 (官方PASTE / PASTE2 / STAligner vs LROT/FGW)")
    print(f"  数据: Br8325, {PAIRS}, {N_SAMPLE} spots, {len(LAYER_ORDER)} 层标注")
    print("=" * 70, flush=True)
    results = [run_pair(a, b) for a, b in PAIRS]

    lines = []
    lines.append("=" * 100)
    lines.append("DLPFC 外部基线对比 (同一供体 Br8325, 1,000 spots, 7层层标注, B施加旋转2°/平移0.5)")
    lines.append("OT方法(FGW/LROT/PASTE/PASTE2): 原始count, 传输熵口径; STAligner: log1p+PCA50, 以PCA-50为图自编码器输入(33k基因CPU不可行), 潜空间最近邻口径(熵/ARI/准确率与其不同)")
    lines.append("=" * 100)
    for r in results:
        lines.append(f"\nPair {r['pair']}")
        hdr = f"  {'方法':<16}{'熵':>9}{'软ARI':>9}{'同层质量':>10}{'准确率':>9}{'时间(s)':>9}"
        lines.append(hdr)
        rows = [
            ('FGW (γ=0)', r['fgw']), ('LROT (γ=0.1)', r['lrot']),
            ('PASTE (α=0.1)', r['paste01']), ('PASTE (α=0.5)', r['paste05']),
            ('PASTE2 (α=0.1)', r['paste2']), ('STAligner', r['stal']),
        ]
        for name, m in rows:
            ent_s = f"{m['entropy']:.4f}*" if m.get('latent_nn') else f"{m['entropy']:.4f}"
            ari_s = f"{m['soft_ari']:.3f}*" if m.get('latent_nn') else f"{m['soft_ari']:.3f}"
            mass_s = "—" if m['same_mass'] != m['same_mass'] else f"{m['same_mass']:.4f}"
            lines.append(f"  {name:<16}{ent_s:>9}{ari_s:>9}{mass_s:>10}{m['soft_acc']:>9.3f}{m['time']:>9.1f}")
        lines.append("  注: * 表示STAligner为潜空间最近邻口径，与传输熵/软ARI不可直接比较。")
        ent_fgw, ent_lrot = r['fgw']['entropy'], r['lrot']['entropy']
        lines.append(f"  熵降: LROT vs FGW = {(ent_fgw-ent_lrot)/ent_fgw*100:.1f}%")
    text = "\n".join(lines)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "dlpfc_baselines_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\n已保存: {out_path}")

if __name__ == '__main__':
    main()