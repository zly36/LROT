"""
pathway_significance_test.py — 多 seed 配对显著性检验

正当性验证: LROT(γ=0.2) 是否真的显著优于 FGW(γ=0)?
  在多个随机 seed 下重复实验，对配对结果做统计检验
  - 配对 t 检验 (参数)
  - Wilcoxon 符号秩检验 (非参数, 更稳健)

只有统计显著的指标才能声称 "LROT 显著优于 FGW"。
否则，如实报告 "无显著差异"。

输出:
  lrot_output/lrot_pathway_significance.txt
"""
import numpy as np
import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


sys.path.insert(0, _R)
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ttest_rel, wilcoxon
from lrot_core import (
    fgw_lr_solver, compute_lr_strength_matrix, compute_mapping_consistency,
    LR_DB, ALL_LR_GENES, SEED

)

OUT_DIR = os.path.join(_R, 'lrot_output')
os.makedirs(OUT_DIR, exist_ok=True)

N_SEEDS = 20          # 20 个随机 seed
N_SPOTS = 1000
GAMMA_LROT = 0.2      # 待检验的 LROT 权重

PATHWAYS = {
    'Neurotrophin / Growth Factor': [
        'Ntng1', 'Ntrk2', 'Bdnf', 'Ntf3', 'Ntrk3', 'Ngf', 'Ntrk1',
        'Gdnf', 'Gfra1', 'Fgf8', 'Fgfr1', 'Fgf15', 'Fgfr2', 'Fgf10', 'Fgf2',
        'Egf', 'Egfr', 'Hgf', 'Met', 'Vegfa', 'Kdr', 'Vegfb', 'Flt1',
        'Pdgfa', 'Pdgfra', 'Pdgfb', 'Pdgfrb', 'Igf1', 'Igf1r', 'Igf2'],
    'Ephrin / Semaphorin': [
        'Efnb1', 'Ephb2', 'Efna1', 'Epha4', 'Efnb2', 'Ephb4', 'Efna5', 'Epha3',
        'Sema3a', 'Nrp1', 'Sema3f', 'Nrp2', 'Sema4d', 'Plxnb1', 'Sema6a', 'Plxna2'],
    'Chemokine / Cytokine': [
        'Cxcl12', 'Cxcr4', 'Cxcl13', 'Cxcr5', 'Ccl2', 'Ccr2', 'Ccl5', 'Ccr5',
        'Il1b', 'Il1r1', 'Tnf', 'Tnfrsf1a', 'Tgfb1', 'Tgfbr1', 'Tgfb2', 'Tgfbr2'],
}


def generate_slices(seed, n_spots=1000, n_genes=150, n_regions=6):
    """生成配对切片（LR基因作为区域marker）"""
    rng = np.random.RandomState(seed)
    all_lr = list(ALL_LR_GENES)
    n_lr = min(len(all_lr), n_genes // 3)
    selected_lr = all_lr[:n_lr]
    n_other = n_genes - n_lr
    other_genes = [f"Gene_{i}" for i in range(n_other)]
    gene_names = selected_lr + other_genes
    lr_indices = list(range(n_lr))
    non_lr_indices = list(range(n_lr, n_genes))

    def hex_grid(n, jitter=0.05):
        side = int(np.ceil(np.sqrt(n)))
        xs, ys = [], []
        for i in range(side):
            for j in range(side):
                x = i * np.sqrt(3)
                y = j * 1.5 + (0.75 if i % 2 == 1 else 0.0)
                xs.append(x); ys.append(y)
        xs = np.array(xs); ys = np.array(ys)
        if len(xs) > n:
            idx = rng.choice(len(xs), n, replace=False)
        else:
            idx = np.arange(len(xs))
        coords = np.column_stack([xs[idx], ys[idx]])
        coords += rng.randn(n, 2) * jitter
        return coords

    coords_A = hex_grid(n_spots)
    x_min, x_max = coords_A[:, 0].min(), coords_A[:, 0].max()
    y_min, y_max = coords_A[:, 1].min(), coords_A[:, 1].max()
    mx, my = (x_max - x_min) * 0.15, (y_max - y_min) * 0.15
    centroids = np.column_stack([
        rng.uniform(x_min + mx, x_max - mx, n_regions),
        rng.uniform(y_min + my, y_max - my, n_regions),
    ])
    dists = cdist(coords_A, centroids)
    labels_A = dists.argmin(axis=1)

    n_marker = max(n_genes // (n_regions * 2), 5)
    region_profiles = np.zeros((n_regions, n_genes))
    lr_markers_per_region = max(n_lr // n_regions, 2)
    for r in range(n_regions):
        lr_markers = rng.choice(lr_indices, lr_markers_per_region, replace=False)
        region_profiles[r, lr_markers] = rng.uniform(3.0, 6.0, len(lr_markers))
        n_mk = max(n_marker - lr_markers_per_region, 3)
        other_markers = rng.choice(non_lr_indices, n_mk, replace=False)
        region_profiles[r, other_markers] = rng.uniform(3.0, 6.0, len(other_markers))
        bg = np.setdiff1d(np.arange(n_genes), np.concatenate([lr_markers, other_markers]))
        region_profiles[r, bg] = rng.uniform(0.1, 0.5, len(bg))

    def generate_expr(coords, labels):
        n = len(coords)
        expr = np.zeros((n, n_genes))
        for r in range(n_regions):
            mask = labels == r
            n_in = mask.sum()
            if n_in == 0:
                continue
            base = region_profiles[r]
            for s in np.where(mask)[0]:
                shape = 5.0
                scale = base / shape
                expr[s] = rng.poisson(rng.gamma(shape, scale) + 1e-8)
        return np.log1p(expr)

    expr_A = generate_expr(coords_A, labels_A)
    center = coords_A.mean(axis=0)
    coords_c = coords_A - center
    theta = np.deg2rad(5.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    M = np.array([[cos_t, -sin_t], [sin_t, cos_t]]) @ np.diag([0.95, 0.95])
    coords_B = coords_c @ M.T + center
    dists_B = cdist(coords_B, centroids)
    labels_B = dists_B.argmin(axis=1)
    expr_B = generate_expr(coords_B, labels_B)
    batch = rng.uniform(0.85, 1.15, n_spots)
    expr_B = expr_B * batch[:, np.newaxis]

    slice_A = {'coords': coords_A.astype(float), 'expr': expr_A.astype(np.float32),
               'gene_names': gene_names, 'region_labels': labels_A,
               'n_spots': n_spots, 'n_genes': n_genes}
    slice_B = {'coords': coords_B.astype(float), 'expr': expr_B.astype(np.float32),
               'gene_names': gene_names, 'region_labels': labels_B,
               'n_spots': n_spots, 'n_genes': n_genes}
    return slice_A, slice_B


def compute_pathway_activity(slice_data, pathway_genes):
    gene_to_idx = {g: i for i, g in enumerate(slice_data['gene_names'])}
    idxs = [gene_to_idx[g] for g in pathway_genes if g in gene_to_idx]
    if not idxs:
        return None, 0
    return slice_data['expr'][:, idxs].mean(axis=1), len(idxs)


def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()


def main():
    print("=" * 70)
    print(f"  Multi-seed Significance Test: LROT(γ={GAMMA_LROT}) vs FGW(γ=0)")
    print(f"  {N_SEEDS} seeds, {N_SPOTS} spots each")
    print("=" * 70)

    # 存储每个 seed 的指标
    # 每项: list over seeds
    metrics = {
        'entropy': {'fgw': [], 'lrot': []},
        'accuracy': {'fgw': [], 'lrot': []},
    }
    for pname in PATHWAYS:
        metrics[pname] = {'fgw': [], 'lrot': []}

    non_lr_control = None  # 用第一个 seed 的基因名
    control_genes = None

    t0_all = time.time()
    from real_st_loader import RealisticSTGenerator
    for s in range(N_SEEDS):
        seed = SEED + s * 1000
        _gen = RealisticSTGenerator(n_spots_A=N_SPOTS, n_spots_B=N_SPOTS,
                                    n_genes=150, seed=seed)
        _paired = _gen.generate_paired_slices(rotation=5.0, batch_effect=0.15)
        slice_A, slice_B = _paired.slice_A.to_dict(), _paired.slice_B.to_dict()
        S_lr, _ = compute_lr_strength_matrix(slice_A, slice_B, LR_DB)

        # 非LR对照基因（固定20个，用seed0的基因名）
        if control_genes is None:
            non_lr = [g for g in slice_A['gene_names'] if g not in ALL_LR_GENES]
            rngc = np.random.RandomState(123)
            control_genes = list(rngc.choice(non_lr, 20, replace=False))

        # FGW (γ=0)
        P_fgw, _ = fgw_lr_solver(slice_A, slice_B, np.zeros_like(S_lr), gamma=0.0,
                                 beta=0.3, alpha=0.5, max_iter=200, verbose=False)
        Pn_fgw = P_fgw / np.maximum(P_fgw.sum(1, keepdims=True), 1e-10)

        # LROT (γ=0.2)
        P_lrot, _ = fgw_lr_solver(slice_A, slice_B, S_lr, gamma=GAMMA_LROT,
                                  beta=0.3, alpha=0.5, max_iter=200, verbose=False)
        Pn_lrot = P_lrot / np.maximum(P_lrot.sum(1, keepdims=True), 1e-10)

        metrics['entropy']['fgw'].append(mean_entropy(P_fgw))
        metrics['entropy']['lrot'].append(mean_entropy(P_lrot))
        metrics['accuracy']['fgw'].append(
            compute_mapping_consistency(P_fgw, slice_A['region_labels'], slice_B['region_labels']))
        metrics['accuracy']['lrot'].append(
            compute_mapping_consistency(P_lrot, slice_A['region_labels'], slice_B['region_labels']))

        # 通路相关
        for pname in PATHWAYS:
            act_ref, _ = compute_pathway_activity(slice_A, PATHWAYS[pname])
            act_B, _ = compute_pathway_activity(slice_B, PATHWAYS[pname])
            r_fgw, _ = pearsonr(act_ref, Pn_fgw @ act_B)
            r_lrot, _ = pearsonr(act_ref, Pn_lrot @ act_B)
            metrics[pname]['fgw'].append(r_fgw)
            metrics[pname]['lrot'].append(r_lrot)

        # 非LR对照
        act_ref_c, _ = compute_pathway_activity(slice_A, control_genes)
        act_B_c, _ = compute_pathway_activity(slice_B, control_genes)
        r_fgw_c, _ = pearsonr(act_ref_c, Pn_fgw @ act_B_c)
        r_lrot_c, _ = pearsonr(act_ref_c, Pn_lrot @ act_B_c)
        if 'Non-LR Control' not in metrics:
            metrics['Non-LR Control'] = {'fgw': [], 'lrot': []}
        metrics['Non-LR Control']['fgw'].append(r_fgw_c)
        metrics['Non-LR Control']['lrot'].append(r_lrot_c)

        if (s + 1) % 5 == 0:
            print(f"  seed {s+1}/{N_SEEDS} done ({time.time()-t0_all:.1f}s)")

    # ========== 统计检验 ==========
    print("\n" + "=" * 70)
    print("  Paired Significance Tests (LROT vs FGW)")
    print("=" * 70)
    print(f"  {'指标':<32} {'FGW均值':<10} {'LROT均值':<10} {'Δ':<10} "
          f"{'t-test p':<10} {'Wilcoxon p':<12} {'显著?'}")
    print(f"  {'-'*92}")

    output_lines = []
    for name in ['entropy', 'accuracy'] + list(PATHWAYS.keys()) + ['Non-LR Control']:
        fgw_vals = np.array(metrics[name]['fgw'])
        lrot_vals = np.array(metrics[name]['lrot'])
        diff = lrot_vals - fgw_vals
        mean_fgw, mean_lrot = fgw_vals.mean(), lrot_vals.mean()
        mean_diff = diff.mean()

        # 配对 t 检验
        if diff.std() > 1e-12:
            t_stat, p_ttest = ttest_rel(lrot_vals, fgw_vals)
            # Wilcoxon 符号秩检验
            try:
                w_stat, p_wilc = wilcoxon(lrot_vals, fgw_vals)
            except ValueError:
                p_wilc = np.nan
        else:
            p_ttest, p_wilc = np.nan, np.nan

        sig = "YES" if (p_ttest < 0.05 and p_wilc < 0.05) else "no"
        direction = ""
        if mean_diff > 0:
            direction = "↑"
        elif mean_diff < 0:
            direction = "↓"

        line = (f"{name:<32} {mean_fgw:<10.4f} {mean_lrot:<10.4f} "
                f"{mean_diff:<+10.4f} {p_ttest:<10.4f} {p_wilc:<12.4f} {sig}{direction}")
        print("  " + line)
        output_lines.append(line)

    # 保存
    result_path = os.path.join(OUT_DIR, 'lrot_pathway_significance.txt')
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"  Multi-seed Significance Test: LROT(γ={GAMMA_LROT}) vs FGW(γ=0)\n")
        f.write(f"  {N_SEEDS} seeds, {N_SPOTS} spots, 配对检验\n")
        f.write("=" * 70 + "\n\n")
        f.write("说明: 只有配对t检验和Wilcoxon检验均p<0.05才标记YES\n")
        f.write("      (Δ>0表示LROT提升该指标, Δ<0表示LROT降低)\n\n")
        f.write(f"{'指标':<32} {'FGW均值':<10} {'LROT均值':<10} {'Δ':<10} "
                f"{'t-test p':<10} {'Wilcoxon p':<12} {'显著?'}\n")
        f.write("-" * 92 + "\n")
        for l in output_lines:
            f.write(l + "\n")
        f.write("\n诚实结论:\n")
        f.write("  - 熵: 若显著下降(p<0.05), 则可声称 LROT 显著降低不确定性\n")
        f.write("  - 精度: 若无显著差异, 则支持 'LROT 不以牺牲精度为代价' 的论断\n")
        f.write("  - 通路相关: 仅当显著时才可声称LROT提升保真度, 否则如实报告无显著差异\n")
    print(f"\n  结果已保存: {result_path}")
    print(f"  总耗时: {time.time()-t0_all:.1f}s")


if __name__ == '__main__':
    main()
