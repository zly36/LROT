# -*- coding: utf-8 -*-
"""
run_dlpfc_visium.py — 在人类 DLPFC 10x Visium 数据上验证 LROT
真实连续切片: 151507 / 151508 / 151509 (同一供体 Br8325, Maynard et al. 2021)
数据: spatialLIBD 官方 AWS; 层标注: LieberInstitute/HumanPilot barcode_level_layer_map.tsv
新增卖点: 真实皮层分层标注 (L1-L6, WM) → ARI / 输运加权同层质量 等正确性指标

评估: LROT(γ=0.1) vs FGW(γ=0): 熵 + 成本 + 软/硬 ARI + 准确率 + NN准确率 + 时间
      + γ-正确性权衡扫描 (熵 vs 软ARI vs 同层质量)
输出: lrot_output/lrot_dlpfc_results.txt, lrot_dlpfc.png
"""
import numpy as np
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
from lrot_core import (
    fgw_lr_solver, compute_ot_cost, compute_alignment,
    compute_mapping_consistency_hard, compute_aligned_nn_accuracy, save_transport_data
)
from run_cancer_visium import HUMAN_LR_DB, apply_misalignment, compute_lr_strength_human
from sklearn.metrics import adjusted_rand_score

# ========== 配置 ==========
DATA_DIR = os.path.join(_R, 'dlpfc_data')
OUT_DIR = os.path.join(_R, 'lrot_output')
N_SAMPLE = 1000
GAMMA = 0.1
SEED_A, SEED_B = 42, 99
PAIRS = [("151507", "151508"), ("151508", "151509")]
LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]
GAMMA_SWEEP = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]

def load_dlpfc_slice(sample_id, target_spots=1000, seed=42):
    """读取 DLPFC h5 + 真实坐标 + 层标注, 保留有标注 spot 并采样"""
    import scanpy as sc
    h5_path = os.path.join(DATA_DIR, f"{sample_id}_filtered_feature_bc_matrix.h5")
    pos_path = os.path.join(DATA_DIR, f"{sample_id}_tissue_positions_list.txt")
    lm_path = os.path.join(DATA_DIR, "barcode_level_layer_map.tsv")

    adata = sc.read_10x_h5(h5_path)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()

    pos = {}
    with open(pos_path, encoding='utf-8') as f:
        for ln in f:
            p = ln.rstrip('\n').split(',')
            if len(p) == 6:
                pos[p[0]] = (float(p[4]), float(p[5]))

    layers = {}
    with open(lm_path, encoding='utf-8') as f:
        for ln in f:
            p = ln.rstrip('\n').split('\t')
            if len(p) == 3 and p[1] == sample_id:
                layers[p[0]] = p[2]

    barcodes = [bc for bc in adata.obs_names if bc in pos and bc in layers]
    print(f"  {sample_id}: h5={adata.n_obs}, 有坐标+标注的 spot={len(barcodes)}", flush=True)

    expr = adata[barcodes].X.toarray().astype(np.float32)
    gene_names = list(adata.var_names)
    coords = np.array([pos[bc] for bc in barcodes], dtype=np.float64)
    coords = coords - coords.mean(axis=0)

    label_map = {name: i for i, name in enumerate(LAYER_ORDER)}
    labels = np.array([label_map[layers[bc]] for bc in barcodes], dtype=int)

    rng = np.random.RandomState(seed)
    n = min(target_spots, len(barcodes))
    idx = rng.choice(len(barcodes), n, replace=False)

    return {
        'coords': coords[idx], 'expr': expr[idx], 'gene_names': gene_names,
        'region_labels': labels[idx], 'n_spots': n, 'n_genes': expr.shape[1],
        'sample_id': sample_id,
    }

def eval_plan(P, labels_A, labels_B, expr_A, expr_B, coords_A, coords_B, n_layers=7):
    """对齐计划完整评估"""
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    ent = float(-np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean())
    cost = float(compute_ot_cost(P, expr_A, expr_B))
    same_mass = float(np.sum(P * (labels_A[:, None] == labels_B[None, :])) / max(P.sum(), 1e-10))
    onehot = np.eye(n_layers)[labels_B]
    pred_soft = np.argmax(Pn @ onehot, axis=1)
    soft_ari = float(adjusted_rand_score(labels_A, pred_soft))
    soft_acc = float(np.mean(pred_soft == labels_A))
    idx_hard = np.argmax(P, axis=1)
    hard_ari = float(adjusted_rand_score(labels_A, labels_B[idx_hard]))
    hard_acc = compute_mapping_consistency_hard(P, labels_A, labels_B)
    nn_acc = compute_aligned_nn_accuracy(P, coords_A, coords_B, labels_A, labels_B)
    return {'entropy': ent, 'cost': cost, 'same_mass': same_mass,
            'soft_ari': soft_ari, 'hard_ari': hard_ari,
            'soft_acc': soft_acc, 'hard_acc': hard_acc, 'nn_acc': nn_acc}

def analyze_pair(sample_A, sample_B, target_spots=1000):
    """一对连续切片: FGW vs LROT + γ 权衡扫描"""
    print("\n" + "=" * 70)
    print(f"  对齐对: {sample_A} → {sample_B}")
    print("=" * 70, flush=True)

    slice_A = load_dlpfc_slice(sample_A, target_spots, SEED_A)
    slice_B = load_dlpfc_slice(sample_B, target_spots, SEED_B)
    slice_B['coords'] = apply_misalignment(slice_B['coords'],
                                           rotation_deg=2.0, translation=0.5)
    ligs = sorted(set(k[0] for k in HUMAN_LR_DB) | set(k[1] for k in HUMAN_LR_DB))

    S_lr, top_pairs = compute_lr_strength_human(slice_A, slice_B, HUMAN_LR_DB)
    S_zero = np.zeros_like(S_lr)

    def solve(S, gamma):
        return fgw_lr_solver(slice_A, slice_B, S, gamma=gamma, beta=0.3, alpha=0.5,
                             max_iter=200, verbose=False, lr_gene_list=ligs if gamma > 0 else ligs)

    t0 = time.time()
    P_fgw, _ = solve(S_zero, 0.0)
    fgw_time = time.time() - t0
    m_fgw = eval_plan(P_fgw, slice_A['region_labels'], slice_B['region_labels'],
                      slice_A['expr'], slice_B['expr'], slice_A['coords'], slice_B['coords'])
    m_fgw['time'] = fgw_time

    t0 = time.time()
    P_lrot, loss_hist = solve(S_lr, GAMMA)
    lrot_time = time.time() - t0
    m_lrot = eval_plan(P_lrot, slice_A['region_labels'], slice_B['region_labels'],
                       slice_A['expr'], slice_B['expr'], slice_A['coords'], slice_B['coords'])
    m_lrot['time'] = lrot_time

    # γ 权衡扫描 (熵 / 软ARI / 同层质量 vs γ)
    sweep = []
    for g in GAMMA_SWEEP:
        P, _ = solve(S_lr if g > 0 else S_zero, g)
        mm = eval_plan(P, slice_A['region_labels'], slice_B['region_labels'],
                       slice_A['expr'], slice_B['expr'], slice_A['coords'], slice_B['coords'])
        sweep.append({'gamma': g, 'entropy': mm['entropy'], 'soft_ari': mm['soft_ari'],
                      'same_mass': mm['same_mass']})

    return {
        'pair': f"{sample_A}-{sample_B}", 'slice_A': sample_A, 'slice_B': sample_B,
        'n_spots': slice_A['n_spots'], 'n_genes': slice_A['n_genes'],
        'lr_matched': len(top_pairs), 'lr_total': len(HUMAN_LR_DB),
        'top_pairs': [(l, r, float(w)) for l, r, w in top_pairs[:8]],
        'loss_hist': loss_hist, 'sweep': sweep,
        'lrot': m_lrot, 'fgw': m_fgw,
        'P_lrot': P_lrot, 'P_fgw': P_fgw,
        'coords_A': slice_A['coords'], 'coords_B': slice_B['coords'],
        'labels_A': slice_A['region_labels'], 'labels_B': slice_B['region_labels'],
    }

def fmt_metrics(d):
    return (f"熵={d['entropy']:.4f}  成本={d['cost']:.4f}  软ARI={d['soft_ari']:.3f}  "
            f"硬ARI={d['hard_ari']:.3f}  同层质量={d['same_mass']:.4f}  "
            f"软准确率={d['soft_acc']:.3f}  硬准确率={d['hard_acc']:.3f}  NN准确率={d['nn_acc']:.3f}  "
            f"时间={d['time']:.1f}s")

def main():
    print("=" * 70)
    print("  LROT on Human DLPFC Visium (Maynard et al. 2021)")
    print("  Real continuous slices: 151507/151508/151509 (Br8325)")
    print(f"  层标注: {len(LAYER_ORDER)} 层 (L1-L6, WM), 每切片 {N_SAMPLE} spots")
    print("=" * 70, flush=True)

    results = [analyze_pair(a, b) for a, b in PAIRS]

    # ---- 控制台汇总 ----
    print("\n" + "=" * 70)
    print("  RESULTS: Human DLPFC Visium (Real Continuous Slices + Layer Labels)")
    print("=" * 70)
    for r in results:
        imp_ent = (r['fgw']['entropy'] - r['lrot']['entropy']) / r['fgw']['entropy'] * 100
        print(f"\n  对 {r['pair']}: LR匹配 {r['lr_matched']}/{r['lr_total']}")
        print(f"    FGW(γ=0):  {fmt_metrics(r['fgw'])}")
        print(f"    LROT(γ=0.1): {fmt_metrics(r['lrot'])}")
        print(f"    熵降 {imp_ent:.2f}% | 软ARI {r['lrot']['soft_ari']-r['fgw']['soft_ari']:+.3f} | "
              f"硬ARI {r['lrot']['hard_ari']-r['fgw']['hard_ari']:+.3f} | "
              f"同层质量 {r['lrot']['same_mass']-r['fgw']['same_mass']:+.4f}")

    avg_ent_imp = np.mean([(r['fgw']['entropy'] - r['lrot']['entropy']) / r['fgw']['entropy'] * 100 for r in results])
    avg_ari = np.mean([r['lrot']['soft_ari'] - r['fgw']['soft_ari'] for r in results])
    avg_mass = np.mean([r['lrot']['same_mass'] - r['fgw']['same_mass'] for r in results])
    print(f"\n  平均: 熵降 {avg_ent_imp:.2f}% | 软ARI {avg_ari:+.3f} | 同层质量 {avg_mass:+.4f}")

    # ---- 落盘 ----
    os.makedirs(OUT_DIR, exist_ok=True)
    txt_path = os.path.join(OUT_DIR, "lrot_dlpfc_results.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 72 + "\n")
        f.write("  LROT on Human DLPFC Visium - RESULTS\n")
        f.write("  Real continuous slices: 151507/151508/151509 (Br8325, Maynard et al. 2021)\n")
        f.write("  Layer labels: L1-L6, WM (spatialLIBD / LieberInstitute HumanPilot)\n")
        f.write("=" * 72 + "\n\n")
        for r in results:
            f.write(f"Pair {r['pair']}  ({r['n_spots']} spots, {r['n_genes']} genes)\n")
            f.write(f"  LR pairs matched: {r['lr_matched']}/{r['lr_total']}\n")
            f.write(f"  Top LR pairs: " + ", ".join(f"{l}-{r}" for l, r, _ in r['top_pairs'][:6]) + "\n")
            f.write(f"  FGW(γ=0):   熵={r['fgw']['entropy']:.4f} 成本={r['fgw']['cost']:.4f} "
                    f"软ARI={r['fgw']['soft_ari']:.3f} 硬ARI={r['fgw']['hard_ari']:.3f} "
                    f"同层质量={r['fgw']['same_mass']:.4f} 软准确率={r['fgw']['soft_acc']:.3f} "
                    f"硬准确率={r['fgw']['hard_acc']:.3f} NN准确率={r['fgw']['nn_acc']:.3f} 时间={r['fgw']['time']:.1f}s\n")
            f.write(f"  LROT(γ=0.1): 熵={r['lrot']['entropy']:.4f} 成本={r['lrot']['cost']:.4f} "
                    f"软ARI={r['lrot']['soft_ari']:.3f} 硬ARI={r['lrot']['hard_ari']:.3f} "
                    f"同层质量={r['lrot']['same_mass']:.4f} 软准确率={r['lrot']['soft_acc']:.3f} "
                    f"硬准确率={r['lrot']['hard_acc']:.3f} NN准确率={r['lrot']['nn_acc']:.3f} 时间={r['lrot']['time']:.1f}s\n")
            imp_ent = (r['fgw']['entropy'] - r['lrot']['entropy']) / r['fgw']['entropy'] * 100
            f.write(f"  熵降 {imp_ent:.2f}% | 软ARI变化 {r['lrot']['soft_ari']-r['fgw']['soft_ari']:+.3f} | "
                    f"硬ARI变化 {r['lrot']['hard_ari']-r['fgw']['hard_ari']:+.3f} | "
                    f"同层质量变化 {r['lrot']['same_mass']-r['fgw']['same_mass']:+.4f}\n")
            f.write(f"  γ权衡扫描 (熵 / 软ARI / 同层质量):\n")
            for s in r['sweep']:
                f.write(f"    γ={s['gamma']:<5} 熵={s['entropy']:.4f} 软ARI={s['soft_ari']:.3f} "
                        f"同层质量={s['same_mass']:.4f}\n")
            f.write("\n")
        f.write(f"Average (2 pairs): 熵降 {avg_ent_imp:.2f}% | 软ARI变化 {avg_ari:+.3f} | "
                f"同层质量变化 {avg_mass:+.4f}\n")
    print(f"\n  结果已保存: {txt_path}")

    make_figure(results)
    r0 = results[0]
    coords_aligned = compute_alignment(r0['P_lrot'], r0['coords_A'], r0['coords_B'])
    save_transport_data('dlpfc', r0['P_lrot'],
                        {'coords': r0['coords_A'], 'expr': None, 'gene_names': None,
                         'region_labels': r0['labels_A'], 'n_spots': r0['n_spots']},
                        {'coords': r0['coords_B'], 'expr': None, 'gene_names': None,
                         'region_labels': r0['labels_B'], 'n_spots': r0['n_spots']},
                        loss_hist=r0['loss_hist'], aligned_coords=coords_aligned,
                        extra={'P_fgw': r0['P_fgw']})
    print("\n  Done! DLPFC validation complete.")

def make_figure(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    layer_colors = matplotlib.colormaps['tab10'](np.linspace(0, 1, len(LAYER_ORDER)))
    r0 = results[0]
    coords_aligned = compute_alignment(r0['P_lrot'], r0['coords_A'], r0['coords_B'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # A: 增强 Before —— 切片A(实心按层) 与 切片B原始(描边按层) 叠加, 层边界错位可见
    coords_al = coords_aligned  # P@B: B 的传输质量投影(2°下贴合 A 帧)
    layer_al = r0['labels_A']   # 投影到 A 的 spot 用 A 层着色作层一致性检查

    def layer_draw(ax, X, labels, s, edge=None, lw=0.0, alpha=0.8, legend=False):
        for l in range(len(LAYER_ORDER)):
            mk = labels == l
            if mk.sum() == 0:
                continue
            ax.scatter(X[mk, 0], X[mk, 1], c=[layer_colors[l]], s=s, alpha=alpha,
                       linewidths=lw, edgecolors=edge if edge else 'none',
                       label=LAYER_ORDER[l] if legend else None)

    ax = axes[0, 0]
    layer_draw(ax, r0['coords_A'], r0['labels_A'], s=6, alpha=0.5, legend=True)
    layer_draw(ax, r0['coords_B'], r0['labels_B'], s=3.5, edge='black', lw=0.3, alpha=0.95)
    ax.set_title('A: Before LROT (solid=151507, outlined=151508, color=layer)',
                 fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.9)

    # B: 对齐后 —— B 的传输质量按匹配到的 A 层层色投影到 A 帧 → 层色重合检查
    ax = axes[0, 1]
    layer_draw(ax, r0['coords_A'], r0['labels_A'], s=6, alpha=0.5)
    layer_draw(ax, coords_al, layer_al, s=3.5, edge='black', lw=0.3, alpha=0.95)
    ax.set_title('B: After LROT Alignment (outlined = 151508 mass on 151507, color=matched layer)',
                 fontsize=11)
    ax.set_aspect('equal')

    # C: loss
    axes[0, 2].plot(r0['loss_hist'], 'b-', linewidth=1.5)
    axes[0, 2].set_xlabel('Iteration', fontsize=10); axes[0, 2].set_ylabel('Loss', fontsize=10)
    axes[0, 2].set_title(f"C: LROT Loss ({len(r0['loss_hist'])} iter)", fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)

    # D: 熵对比
    ent = [r0['fgw']['entropy'], r0['lrot']['entropy']]
    bars = axes[1, 0].bar(['FGW (γ=0)', 'LROT (γ=0.1)'], ent, color=['#E74C3C', '#2ECC71'], edgecolor='gray', alpha=0.9)
    axes[1, 0].set_ylim(min(ent) * 0.98, max(ent) * 1.02)
    for b, v in zip(bars, ent):
        axes[1, 0].text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    axes[1, 0].set_ylabel('Entropy', fontsize=11)
    imp_ent0 = (r0['fgw']['entropy'] - r0['lrot']['entropy']) / r0['fgw']['entropy'] * 100
    axes[1, 0].set_title(f"D: Alignment Entropy (↓{imp_ent0:.1f}%)", fontsize=11)

    # E: 正确性指标对比 (软ARI / 同层质量)
    x = np.arange(2); w = 0.35
    ari_vals = [r0['fgw']['soft_ari'], r0['lrot']['soft_ari']]
    bars = axes[1, 1].bar(x, ari_vals, w, color=['#E74C3C', '#2ECC71'], edgecolor='gray', alpha=0.9)
    for b, v in zip(bars, ari_vals):
        axes[1, 1].text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(['FGW (γ=0)', 'LROT (γ=0.1)'], fontsize=10)
    axes[1, 1].set_ylabel('Soft ARI', fontsize=11)
    axes[1, 1].set_title('E: Layer Concordance (Soft ARI)', fontsize=11)

    # F: γ 权衡曲线 (熵 与 同层质量)
    gs = [s['gamma'] for s in r0['sweep']]
    ents = [s['entropy'] for s in r0['sweep']]
    masses = [s['same_mass'] for s in r0['sweep']]
    ax1 = axes[1, 2]
    ax1.plot(gs, ents, 'o-', color='#2980B9', label='Entropy', linewidth=1.8)
    ax1.set_xlabel('γ (LR guidance weight)', fontsize=10)
    ax1.set_ylabel('Entropy', fontsize=10, color='#2980B9')
    ax1.tick_params(axis='y', labelcolor='#2980B9')
    ax1.set_xticks(gs)
    ax2 = ax1.twinx()
    ax2.plot(gs, masses, 's--', color='#C0392B', label='Same-layer mass', linewidth=1.8)
    ax2.set_ylabel('Same-layer mass', fontsize=10, color='#C0392B')
    ax2.tick_params(axis='y', labelcolor='#C0392B')
    ax1.set_title('F: Entropy–Accuracy Trade-off vs γ', fontsize=11)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=9, loc='center right')

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    # 论文图9(lrot_dlpfc.png)的正式渲染见 experiments/_render_dlpfc_fig9.py
    # (软映射 + 共享坐标窗)；此处保留稀疏渲染，仅作对照，避免覆盖正式图。
    out_png = os.path.join(OUT_DIR, 'lrot_dlpfc_sparse_render.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  图已保存: {out_png}")

if __name__ == '__main__':
    main()
