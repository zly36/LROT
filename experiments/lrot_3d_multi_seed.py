"""
lrot_3d_multi_seed.py — 3D 重建的多 seed 统计验证（含 FGW 基线）
在多个随机 seed 下重复 3D 重建，比较 LROT (γ=0.1) 与 PASTE 风格 FGW (γ=0)
"""
import os
import numpy as np
import time
import sys
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)
from lrot_core import (
    fgw_lr_solver, compute_lr_strength_matrix,
    compute_alignment, compute_mapping_consistency, compute_ot_cost,
    LR_DB, LIGANDS, RECEPTORS, ALL_LR_GENES
)
from real_st_loader import RealisticSTGenerator

import warnings

warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(_R, 'lrot_output')

N_SLICES = 5
N_SPOTS = 500
N_GENES = 100
BASE_ROT = 3.0
BASE_BATCH = 0.08
N_SEEDS = 10   # 10 个 seed

def generate_slices(seed):
    """生成一次 3D 连续切片序列，返回 slices 列表"""
    rng = np.random.RandomState(seed)
    gen = RealisticSTGenerator(
        n_spots_A=N_SPOTS, n_spots_B=N_SPOTS,
        n_genes=N_GENES, seed=seed
    )
    base_coords = gen._generate_hex_grid(N_SPOTS, jitter=0.05)
    region_profiles = gen._generate_region_profiles(gen.n_regions, N_GENES)
    region_labels_0, centroids = gen._assign_spots_to_regions(base_coords)

    ligs, recs = LIGANDS, RECEPTORS
    all_lr = sorted(set(ligs + recs))  # sorted: 避免 PYTHONHASHSEED 导致跨进程 set 顺序不同
    n_lr = min(len(all_lr), N_GENES // 3)
    other_genes = [f"Gene_{i}" for i in range(N_GENES - n_lr)]
    gene_names = all_lr[:n_lr] + other_genes

    def make_expr(coords, region_labels):
        expr = np.zeros((len(coords), N_GENES))
        for r in range(gen.n_regions):
            mask = region_labels == r
            n_in = mask.sum()
            if n_in == 0:
                continue
            base = region_profiles[r]
            shape = 5.0
            scale = base / shape
            gamma_sample = rng.gamma(shape, scale, (n_in, N_GENES))
            expr[mask] = np.log1p(rng.poisson(gamma_sample + 1e-8))
        return expr

    base_expr = make_expr(base_coords, region_labels_0)

    slices = []
    coords_prev = base_coords.copy()
    for z_idx in range(N_SLICES):
        if z_idx == 0:
            coords_z = base_coords.copy()
            expr_z = base_expr.copy()
            labels_z = region_labels_0
        else:
            cum_rot = BASE_ROT * z_idx
            cum_batch = BASE_BATCH * z_idx
            theta = np.deg2rad(cum_rot)
            c, s = np.cos(theta), np.sin(theta)
            M = np.array([[c, -s], [s, c]]) @ np.diag([0.98, 0.98])
            center = coords_prev.mean(0)
            coords_z = (coords_prev - center) @ M.T + center
            coords_z += rng.randn(N_SPOTS, 2) * 0.05
            labels_z, _ = gen._assign_spots_to_regions(coords_z)
            expr_z = make_expr(coords_z, labels_z)
            lr_idx = [i for i, g in enumerate(gene_names) if g in all_lr]
            n_hot = max(int(N_SPOTS * 0.12), 5)
            hot_idx = rng.choice(N_SPOTS, n_hot, replace=False)
            for hi in hot_idx[:3]:
                dists = np.linalg.norm(coords_z - coords_z[hi:hi+1], axis=1)
                local = dists < np.percentile(dists, 15)
                for gi in lr_idx:
                    expr_z[local, gi] *= rng.uniform(1.5, 2.0)
            expr_z *= rng.uniform(1 - cum_batch, 1 + cum_batch, N_SPOTS)[:, None]
            expr_z = expr_z.astype(np.float32)
            coords_prev = coords_z.copy()

        sl = {
            'coords': coords_z.astype(np.float64),
            'expr': expr_z.astype(np.float32),
            'gene_names': gene_names[:],
            'region_labels': labels_z,
            'n_spots': N_SPOTS,
            'n_genes': N_GENES,
        }
        slices.append(sl)
    return slices

def sequential_align(slices, gamma, use_lr):
    """对同一批切片做顺序两两对齐，返回相邻精度列表"""
    adjacent_accs = []
    for i in range(N_SLICES - 1):
        if use_lr:
            S_lr, _ = compute_lr_strength_matrix(slices[i], slices[i+1], LR_DB)
        else:
            S_lr = np.zeros((slices[i]['n_spots'], slices[i+1]['n_spots']))
        P, _ = fgw_lr_solver(
            slices[i], slices[i+1], S_lr, gamma=gamma,
            beta=0.3, alpha=0.5, max_iter=150, verbose=False
        )
        acc = compute_mapping_consistency(
            P, slices[i]['region_labels'], slices[i+1]['region_labels'])
        adjacent_accs.append(acc)
    return adjacent_accs

def main():
    print("=" * 70)
    print(f"  3D Reconstruction Multi-seed Validation (LROT vs FGW)")
    print(f"  {N_SEEDS} seeds, {N_SLICES} slices, {N_SPOTS} spots each")
    print("=" * 70)

    lrot_all = []   # [seed][adjacent_idx]
    fgw_all = []    # [seed][adjacent_idx]
    t0_all = time.time()
    for s in range(N_SEEDS):
        seed = 42 + s * 100
        slices = generate_slices(seed)
        lrot_accs = sequential_align(slices, 0.1, True)
        fgw_accs = sequential_align(slices, 0.0, False)
        lrot_all.append(lrot_accs)
        fgw_all.append(fgw_accs)
        print(f"  seed {s+1}/{N_SEEDS} (seed={seed}): "
              f"LROT={np.mean(lrot_accs):.4f}  FGW={np.mean(fgw_accs):.4f}")

    lrot_all = np.array(lrot_all)  # (N_SEEDS, N_SLICES-1)
    fgw_all = np.array(fgw_all)

    lrot_mean, lrot_std = lrot_all.mean(), lrot_all.std()
    fgw_mean, fgw_std = fgw_all.mean(), fgw_all.std()
    delta_mean = (lrot_all - fgw_all).mean()

    print("\n" + "=" * 70)
    print("  结果统计")
    print("=" * 70)
    print(f"  LROT 总体: {lrot_mean:.4f} ± {lrot_std:.4f}")
    print(f"  FGW  总体: {fgw_mean:.4f} ± {fgw_std:.4f}")
    print(f"  Δ(LROT-FGW): {delta_mean:+.4f}")

    # 保存 LROT 单独统计（向后兼容）
    result_path = f"{OUT_DIR}/lrot_3d_multi_seed_results.txt"
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"  3D Reconstruction Multi-seed Results\n")
        f.write(f"  {N_SEEDS} seeds, {N_SLICES} slices, {N_SPOTS} spots\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"总体平均对齐精度: {lrot_mean:.4f} ± {lrot_std:.4f}\n\n")
        f.write(f"相邻切片对:\n")
        for j in range(N_SLICES - 1):
            f.write(f"  切片{j}↔{j+1}: {lrot_all[:, j].mean():.4f} ± {lrot_all[:, j].std():.4f}\n")
        f.write(f"\n总耗时: {time.time()-t0_all:.1f}s\n")

    # 保存 LROT vs FGW 基线对比
    base_path = f"{OUT_DIR}/lrot_3d_multi_seed_baseline_results.txt"
    with open(base_path, 'w', encoding='utf-8') as f:
        f.write("=" * 76 + "\n")
        f.write("  3D Multi-seed Baseline Comparison: LROT vs PASTE-style FGW\n")
        f.write(f"  {N_SEEDS} seeds, {N_SLICES} slices, {N_SPOTS} spots, same slices per seed\n")
        f.write("=" * 76 + "\n\n")
        f.write("  {:<12} {:<14} {:<14} {:<14}\n".format("Adjacent", "LROT acc", "FGW acc", "delta"))
        f.write("  " + "-" * 54 + "\n")
        for j in range(N_SLICES - 1):
            f.write("  {:<12} {:<14.4f} {:<14.4f} {:<+14.4f}\n".format(
                f"slice {j}↔{j+1}",
                lrot_all[:, j].mean(), fgw_all[:, j].mean(),
                lrot_all[:, j].mean() - fgw_all[:, j].mean()))
        f.write("  " + "-" * 54 + "\n")
        f.write("  {:<12} {:<14.4f} {:<14.4f} {:<+14.4f}\n".format(
            "Mean", lrot_mean, fgw_mean, delta_mean))
        f.write(f"\n  LROT overall: {lrot_mean:.4f} ± {lrot_std:.4f}\n")
        f.write(f"  FGW  overall: {fgw_mean:.4f} ± {fgw_std:.4f}\n")
        f.write(f"  配对 t 检验（LROT vs FGW，每 seed 均值）: p=")
        from scipy.stats import ttest_rel
        p = ttest_rel(lrot_all.mean(axis=1), fgw_all.mean(axis=1)).pvalue
        f.write(f"{p:.4f}\n")
        f.write(f"\n总耗时: {time.time()-t0_all:.1f}s\n")

    print(f"\n  结果已保存: {result_path}")
    print(f"  基线对比已保存: {base_path}")
    print(f"  总耗时: {time.time()-t0_all:.1f}s")

if __name__ == '__main__':
    main()