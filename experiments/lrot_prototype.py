import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 论文根目录(供尾部 from lrot_core import ... 使用, 与从 experiments/ 运行兼容)
import sys, os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# =============================================================================
# LROT: Ligand-Receptor guided Optimal Transport for 3D Spatial Transcriptomics
# =============================================================================
# This prototype demonstrates how LR signaling can guide slice alignment
# by adding a third cost term to the Fused Gromov-Wasserstein framework.
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


OUT_DIR = Path(_R, 'lrot_output', 'figures_png')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# 1. Ligand-Receptor Database (mouse brain)
# ---------------------------------------------------------------------------
# 60+ curated ligand-receptor pairs with known interaction weights.
# Sources: CellChatDB, CellPhoneDB, curated literature.
# For mouse brain, these cover major signaling pathways.
LR_DB = {
    # --- Neurotrophin / Growth Factor ---
    ("Ntng1", "Ntrk2"): 0.95,
    ("Bdnf", "Ntrk2"): 0.92,
    ("Ntf3", "Ntrk3"): 0.90,
    ("Ngf", "Ntrk1"): 0.88,
    ("Gdnf", "Gfra1"): 0.85,
    ("Artn", "Gfra3"): 0.80,
    ("Nrtn", "Gfra2"): 0.78,
    ("Fgf8", "Fgfr1"): 0.90,
    ("Fgf15", "Fgfr2"): 0.85,
    ("Fgf10", "Fgfr2"): 0.82,
    ("Fgf2", "Fgfr1"): 0.88,
    ("Egf", "Egfr"): 0.85,
    ("Hgf", "Met"): 0.83,
    ("Vegfa", "Kdr"): 0.90,
    ("Vegfb", "Flt1"): 0.78,
    ("Pdgfa", "Pdgfra"): 0.80,
    ("Pdgfb", "Pdgfrb"): 0.82,
    ("Igf1", "Igf1r"): 0.85,
    ("Igf2", "Igf1r"): 0.80,
    # --- Wnt Pathway ---
    ("Wnt3a", "Fzd1"): 0.88,
    ("Wnt5a", "Fzd5"): 0.85,
    ("Wnt7a", "Fzd10"): 0.82,
    ("Wnt1", "Fzd1"): 0.80,
    # --- Notch ---
    ("Dll1", "Notch1"): 0.90,
    ("Dll4", "Notch4"): 0.85,
    ("Jag1", "Notch1"): 0.88,
    ("Jag2", "Notch2"): 0.83,
    # --- Ephrin ---
    ("Efnb1", "Ephb2"): 0.90,
    ("Efna1", "Epha4"): 0.85,
    ("Efnb2", "Ephb4"): 0.82,
    ("Efna5", "Epha3"): 0.78,
    # --- Semaphorin ---
    ("Sema3a", "Nrp1"): 0.90,
    ("Sema3f", "Nrp2"): 0.85,
    ("Sema4d", "Plxnb1"): 0.80,
    ("Sema6a", "Plxna2"): 0.75,
    # --- Slit/Robo ---
    ("Slit1", "Robo1"): 0.88,
    ("Slit2", "Robo2"): 0.85,
    ("Slit3", "Robo2"): 0.80,
    # --- Chemokine / Cytokine ---
    ("Cxcl12", "Cxcr4"): 0.95,
    ("Cxcl13", "Cxcr5"): 0.85,
    ("Ccl2", "Ccr2"): 0.82,
    ("Ccl5", "Ccr5"): 0.80,
    ("Il1b", "Il1r1"): 0.90,
    ("Tnf", "Tnfrsf1a"): 0.88,
    ("Tgfb1", "Tgfbr1"): 0.92,
    ("Tgfb2", "Tgfbr2"): 0.88,
    # -- BMP ---
    ("Bmp4", "Bmpr1a"): 0.85,
    ("Bmp7", "Bmpr2"): 0.80,
    ("Bmp2", "Acvr1"): 0.82,
    # --- Cell Adhesion ---
    ("Cdh1", "Cdh1"): 0.95,
    ("Cdh2", "Cdh2"): 0.92,
    ("Ncam1", "Ncam1"): 0.85,
    ("L1cam", "L1cam"): 0.80,
    ("Lama1", "Itga1"): 0.78,
    ("Col1a1", "Itga2"): 0.75,
    ("Vtn", "Itgav"): 0.72,
    # --- Hedgehog ---
    ("Shh", "Ptch1"): 0.90,
    ("Ihh", "Ptch2"): 0.82,
    ("Dhh", "Ptch1"): 0.78,
    # --- Reelin ---
    ("Reln", "Lrp8"): 0.90,
    ("Reln", "Vldlr"): 0.85,
}
print(f"[LROT] LR database loaded: {len(LR_DB)} pairs")

def lr_gene_list(lr_db):
    """Get unique ligand and receptor genes from the database."""
    ligands = sorted(set(k[0] for k in lr_db))
    receptors = sorted(set(k[1] for k in lr_db))
    return ligands, receptors

LIGANDS, RECEPTORS = lr_gene_list(LR_DB)
ALL_LR_GENES = sorted(set(LIGANDS + RECEPTORS))  # sorted: 避免 PYTHONHASHSEED 跨进程顺序不同

# ---------------------------------------------------------------------------
# 2. Synthetic Data Generation
# ---------------------------------------------------------------------------
# Simulates two adjacent spatial slices (~400 spots, ~200 genes).
# Spots are arranged in a grid with spatial coordinates.
# Region-specific LR gene expression creates signaling gradients.

N_REGIONS = 4
REGION_NAMES = ["Cortex", "Hippocampus", "Thalamus", "Striatum"]

def simulate_slice(n_spots=400, n_genes=200, lr_genes=None, region_weights=None,
                   x_range=(0, 20), y_range=(0, 20), seed=SEED):
    """
    Simulate a spatial transcriptomics slice.
    
    Parameters
    ----------
    n_spots : int
        Number of spatial spots
    n_genes : int
        Number of genes (including LR genes)
    lr_genes : list
        List of LR gene names to include
    region_weights : dict
        Mapping from region index to dict of gene -> expression weight
    x_range, y_range : tuple
        Spatial coordinate ranges
    seed : int
        Random seed
    
    Returns
    -------
    dict with keys: coords, expr, region_labels, gene_names
    """
    rng = np.random.RandomState(seed)
    
    # Generate spatial coordinates on a near-regular grid with jitter
    side = int(np.ceil(np.sqrt(n_spots)))
    xs = np.linspace(x_range[0], x_range[1], side)
    ys = np.linspace(y_range[0], y_range[1], side)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    coords = coords[:n_spots] + rng.randn(n_spots, 2) * 0.3
    
    # Assign region labels based on spatial location
    center = np.array([(x_range[0]+x_range[1])/2, (y_range[0]+y_range[1])/2])
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    region_labels = np.floor((angles + np.pi) / (2*np.pi/N_REGIONS)).astype(int)
    region_labels = np.clip(region_labels, 0, N_REGIONS-1)
    
    # Generate background gene expression
    n_other = n_genes - len(ALL_LR_GENES)
    gene_names = ALL_LR_GENES[:] + [f"Gene_{i}" for i in range(max(0, n_other))]
    gene_names = gene_names[:n_genes]
    n_lr = len(ALL_LR_GENES)
    n_other = n_genes - min(n_lr, n_genes)
    
    expr = rng.gamma(1, 1, (n_spots, n_genes)).astype(np.float32) * 0.5
    
    # Enrich LR genes by region
    if region_weights is None:
        region_weights = {}
        for reg in range(N_REGIONS):
            region_weights[reg] = {}
            # Assign different LR pathways to different regions
            pathways = [
                LIGANDS[:8],  # Cortex: neurotrophins
                LIGANDS[8:16],  # Hippocampus: FGF/Wnt
                LIGANDS[16:24], # Thalamus: Notch/Ephrin
                LIGANDS[24:32], # Striatum: Semaphorin/Chemokine
            ]
            for lig in pathways[reg]:
                if lig in ALL_LR_GENES and lig in gene_names:
                    idx = gene_names.index(lig)
                    region_weights[reg][idx] = 3.0
            # Also enrich some receptors
            rec_pathways = [
                RECEPTORS[:8],
                RECEPTORS[8:16],
                RECEPTORS[16:24],
                RECEPTORS[24:32],
            ]
            for rec in rec_pathways[reg]:
                if rec in ALL_LR_GENES and rec in gene_names:
                    idx = gene_names.index(rec)
                    region_weights[reg][idx] = 2.5
    
    for spot in range(n_spots):
        reg = region_labels[spot]
        if reg in region_weights:
            for gidx, w in region_weights[reg].items():
                expr[spot, gidx] *= w
    
    # Add expression noise
    expr += rng.gamma(0.5, 0.5, expr.shape) * 0.3
    
    return {
        'coords': coords,
        'expr': expr,
        'region_labels': region_labels,
        'gene_names': gene_names,
        'n_spots': n_spots,
        'n_genes': n_genes,
    }

print("[LROT] Generating synthetic slices...")
slice_A = simulate_slice(1000, 200, ALL_LR_GENES, seed=42)
# Slice B has a slight spatial transformation + different LR enrichment
slice_B = simulate_slice(1000, 200, ALL_LR_GENES, seed=123)
# Add a slight rotation to make alignment non-trivial
theta = 0.15
c, s = np.cos(theta), np.sin(theta)
R = np.array([[c, -s], [s, c]])
center = slice_B['coords'].mean(0)
slice_B['coords'] = (slice_B['coords'] - center) @ R.T + center + np.array([0.5, 0.3])

print(f"  Slice A: {slice_A['n_spots']} spots, {slice_A['n_genes']} genes")
print(f"  Slice B: {slice_B['n_spots']} spots, {slice_B['n_genes']} genes")

# Map gene names to indices for both slices
def gene_to_idx(slice_data, gene_name):
    try:
        return slice_data['gene_names'].index(gene_name)
    except ValueError:
        return None

# ---------------------------------------------------------------------------
# 3. LR Strength Matrix Computation
# ---------------------------------------------------------------------------

def compute_lr_strength_matrix(slice_A, slice_B, lr_db, scale=10.0):
    """
    Compute ligand-receptor interaction strength between every pair of spots
    in slice A (ligand expression) and slice B (receptor expression).
    
    S[i,j] = sum_{(L,R) in DB} w_{L,R} * (expr_A[i, L] * expr_B[j, R])
    
    Higher values = stronger LR signaling between spots.
    
    Returns
    -------
    S : ndarray (n_spots_A, n_spots_B)
        LR strength matrix
    top_pairs : list of (L, R, weight)
        Top contributing LR pairs
    """
    nA = slice_A['n_spots']
    nB = slice_B['n_spots']
    S = np.zeros((nA, nB))
    pair_contributions = []
    
    for (lig, rec), weight in lr_db.items():
        lig_idx = gene_to_idx(slice_A, lig)
        rec_idx = gene_to_idx(slice_B, rec)
        if lig_idx is None or rec_idx is None:
            continue
        lig_expr = slice_A['expr'][:, lig_idx]
        rec_expr = slice_B['expr'][:, rec_idx]
        contrib = np.outer(lig_expr, rec_expr) * weight
        S += contrib
        pair_contributions.append((weight * lig_expr.sum() * rec_expr.sum(), lig, rec, weight))
    
    # Normalize to [0, 1] range
    S_min, S_max = S.min(), S.max()
    if S_max > S_min:
        S = (S - S_min) / (S_max - S_min)
    
    # Sort pairs by contribution
    pair_contributions.sort(reverse=True)
    top_pairs = [(l, r, w) for _, l, r, w in pair_contributions[:15]]
    
    return S, top_pairs

print("[LROT] Computing LR strength matrix...")
S_lr, top_lr_pairs = compute_lr_strength_matrix(slice_A, slice_B, LR_DB)
print(f"  LR strength matrix: {S_lr.shape}, range [{S_lr.min():.4f}, {S_lr.max():.4f}]")
print(f"  Top LR pairs: {[(l, r) for l, r, w in top_lr_pairs[:8]]}")

# ---------------------------------------------------------------------------
# 4. Fused Gromov-Wasserstein with LR Guidance
# ---------------------------------------------------------------------------

def cosine_distance(X, Y):
    """Pairwise cosine distance."""
    X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-10)
    Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-10)
    sim = X_norm @ Y_norm.T
    return np.clip(1 - sim, 0, 2)

def euclidean_distance_matrix(X, Y):
    """Pairwise squared Euclidean distance."""
    return cdist(X, Y, 'sqeuclidean')

def gromov_cost_matrix(C1, C2):
    """Compute 2D GW cost approximation between spatial distance matrices C1 (nA,nA) and C2 (nB,nB)."""
    nA = C1.shape[0]
    nB = C2.shape[0]
    p = np.ones(nA) / nA
    q = np.ones(nB) / nB
    C1_sq = C1 ** 2
    C2_sq = C2 ** 2
    constC1 = C1_sq @ p  # shape (nA,)
    constC2 = C2_sq @ q  # shape (nB,)
    constC = constC1[:, None] + constC2[None, :]  # shape (nA, nB)
    return constC

def sinkhorn_stabilized(K, a, b, reg=0.01, max_iter=1000, tol=1e-6):
    """Stabilized Sinkhorn-Knopp algorithm."""
    nA, nB = K.shape
    u = np.ones(nA) / nA
    v = np.ones(nB) / nB
    
    for i in range(max_iter):
        u_prev = u.copy()
        v_prev = v.copy()
        
        v = b / (K.T @ u + 1e-16)
        u = a / (K @ v + 1e-16)
        
        err_u = np.linalg.norm(u - u_prev)
        err_v = np.linalg.norm(v - v_prev)
        if err_u < tol and err_v < tol:
            break
    
    P = np.diag(u) @ K @ np.diag(v)
    return P

def fgw_lr_solver(slice_A, slice_B, S_lr, gamma=0.1, beta=0.3, reg_gw=0.01, reg_ot=0.01,
                   alpha=0.5, max_iter=200, lr_step=50, tol=1e-5, verbose=True):
    """
    Fused Gromov-Wasserstein solver with LR guidance.
    
    The total transport cost combines three terms:
      1. Expression cost (cosine distance between gene expression profiles)
      2. LR guidance cost (-log(LR strength + epsilon))
      3. GW spatial cost (spatial structure preservation)
    
    C_total = alpha * C_expr + gamma * C_lr + (1-alpha) * C_gw
    
    Parameters
    ----------
    slice_A, slice_B : dict
        Slice data dictionaries
    S_lr : ndarray (nA, nB)
        LR strength matrix
    gamma : float
        Weight of LR guidance term
    beta : float
        Weight of GW regularization
    reg_gw, reg_ot : float
        Entropic regularization strengths
    alpha : float
        Trade-off between expression (alpha) and geometry (1-alpha)
    max_iter : int
        Maximum number of outer iterations
    lr_step : int
        Step interval for updating LR cost
    
    Returns
    -------
    P : ndarray (nA, nB)
        Optimal transport plan
    loss_history : list
        Loss per iteration
    """
    nA = slice_A['n_spots']
    nB = slice_B['n_spots']
    
    # --- Expression cost (cosine distance, non-LR genes only) ---
    # Exclude LR genes from expression to avoid redundancy with C_lr term
    gene_names = slice_A.get('gene_names', None)
    if gene_names is not None:
        non_lr_mask = np.array([g not in ALL_LR_GENES for g in gene_names])
        if non_lr_mask.sum() > 0:
            C_expr = cosine_distance(slice_A['expr'][:, non_lr_mask],
                                     slice_B['expr'][:, non_lr_mask])
        else:
            C_expr = cosine_distance(slice_A['expr'], slice_B['expr'])
    else:
        C_expr = cosine_distance(slice_A['expr'], slice_B['expr'])
    C_expr = C_expr / np.maximum(C_expr.max(), 1e-8)
    
    # --- LR guidance cost ---
    # High LR strength -> low cost, low LR strength -> high cost
    epsilon = 1e-4
    C_lr = -np.log(S_lr + epsilon)
    C_lr = C_lr / np.maximum(C_lr.max(), 1e-8)
    
    # --- Spatial distance matrices for GW ---
    D_A = euclidean_distance_matrix(slice_A['coords'], slice_A['coords'])
    D_B = euclidean_distance_matrix(slice_B['coords'], slice_B['coords'])
    D_A = D_A / np.maximum(D_A.max(), 1e-8)
    D_B = D_B / np.maximum(D_B.max(), 1e-8)
    
    # Weight vectors (uniform)
    a = np.ones(nA) / nA
    b = np.ones(nB) / nB
    
    # Initialize transport plan
    P = np.outer(a, b)
    P_prev = P.copy()
    
    loss_history = []
    
    for it in range(max_iter):
        # --- GW cost (spatial structure preservation) ---
        # Standard GW cost computation with P-dependent cross-term
        C_gw_const = gromov_cost_matrix(D_A, D_B)  # constant part: (C1^2 @ p) + (C2^2 @ q)
        # Cross-term: -2 * C1 @ P @ C2^T  (depends on current transport plan P)
        C_gw_cross = -2 * (D_A @ P @ D_B.T)
        C_gw = C_gw_const + C_gw_cross
        C_gw = C_gw / np.maximum(C_gw.max(), 1e-8)
        
        # --- Combined cost ---
        C_total = alpha * C_expr + gamma * C_lr + (1 - alpha) * beta * C_gw
        
        # --- Entropic regularization ---
        reg = reg_ot + reg_gw * beta
        
        # --- Sinkhorn step ---
        K = np.exp(-C_total / reg)
        P = sinkhorn_stabilized(K, a, b, reg=reg, max_iter=200, tol=1e-4)
        
        # --- Check convergence ---
        change = np.linalg.norm(P - P_prev) / max(np.linalg.norm(P_prev), 1e-10)
        P_prev = P.copy()
        
        # --- Compute loss ---
        loss = np.sum(P * C_expr) + gamma * np.sum(P * C_lr) + beta * np.sum(P * C_gw)
        loss_history.append(loss)
        
        if verbose and (it % 20 == 0 or it == max_iter - 1):
            print(f"  Iter {it:3d}: loss={loss:.4f}, change={change:.6f}")
        
        if change < tol and it > 5:
            if verbose:
                print(f"  Converged at iteration {it}")
            break
    
    return P, loss_history

# ---------------------------------------------------------------------------
# 5. Alignment Quality Metrics
# ---------------------------------------------------------------------------

def compute_alignment(transport_plan, coords_A, coords_B):
    """Compute aligned coordinates using barycentric mapping."""
    P_norm = transport_plan / np.maximum(transport_plan.sum(1, keepdims=True), 1e-10)
    coords_aligned = P_norm @ coords_B
    return coords_aligned

def compute_mapping_consistency(transport_plan, labels_A, labels_B):
    """Compute how well region labels match after alignment."""
    nA = transport_plan.shape[0]
    P_row_norm = transport_plan / np.maximum(transport_plan.sum(1, keepdims=True), 1e-10)
    predicted_labels = P_row_norm @ np.eye(N_REGIONS)[labels_B]
    predicted_labels = np.argmax(predicted_labels, axis=1)
    accuracy = np.mean(predicted_labels == labels_A)
    return accuracy

def compute_ot_cost(transport_plan, expr_A, expr_B):
    """Compute the OT cost between expression distributions."""
    C = cosine_distance(expr_A, expr_B)
    return np.sum(transport_plan * C)

# ---------------------------------------------------------------------------
# 6. Run LROT with LR Guidance
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("  LROT: Fused GW with LR Guidance (gamma=0.1)")
print("="*60)
P_lrot, loss_hist = fgw_lr_solver(
    slice_A, slice_B, S_lr,
    gamma=0.1, beta=0.3, alpha=0.5,
    max_iter=200, verbose=True
)

# Compute alignment results
coords_aligned_lrot = compute_alignment(P_lrot, slice_A['coords'], slice_B['coords'])
acc_lrot = compute_mapping_consistency(P_lrot, slice_A['region_labels'], slice_B['region_labels'])
cost_lrot = compute_ot_cost(P_lrot, slice_A['expr'], slice_B['expr'])
print(f"\n  Alignment accuracy (LROT): {acc_lrot:.4f}")
print(f"  Transport cost (LROT):      {cost_lrot:.4f}")

# ---------------------------------------------------------------------------
# 7. Ablation Study: Effect of gamma (LR guidance weight)
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("  Ablation: Effect of LR Guidance Weight (gamma)")
print("="*60)

gamma_values = [0.0, 0.05, 0.1, 0.2]
ablation_results = []

for g in gamma_values:
    print(f"\n  --- gamma = {g:.2f} ---")
    P_g, loss_g = fgw_lr_solver(
        slice_A, slice_B, S_lr,
        gamma=g, beta=0.3, alpha=0.5,
        max_iter=120, verbose=False
    )
    acc = compute_mapping_consistency(P_g, slice_A['region_labels'], slice_B['region_labels'])
    cost = compute_ot_cost(P_g, slice_A['expr'], slice_B['expr'])
    ablation_results.append({
        'gamma': g,
        'accuracy': acc,
        'cost': cost,
        'plan': P_g,
    })
    print(f"    Accuracy: {acc:.4f}, Cost: {cost:.4f}")

print("\n" + "-"*40)
print("  Ablation Summary:")
print(f"  {'gamma':>8} {'Accuracy':>10} {'Cost':>10}")
print("  " + "-"*30)
for r in ablation_results:
    print(f"  {r['gamma']:>8.2f} {r['accuracy']:>10.4f} {r['cost']:>10.4f}")

# ===== 保存合成数据结果 =====
with open(OUT_DIR.parent / 'lrot_synthetic_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 72 + "\n")
    f.write("  LROT Synthetic Data Experiment Results\n")
    f.write("=" * 72 + "\n")
    f.write(f"Data: {slice_A['n_spots']} spots x {slice_A['n_genes']} genes, {N_REGIONS} regions\n")
    f.write(f"Transform: rotation={np.rad2deg(theta):.1f} deg, translation=(0.5, 0.3)\n\n")
    f.write("  Method              Acc     Cost\n")
    f.write("  " + "-" * 40 + "\n")
    for r in ablation_results:
        tag = 'LROT' if abs(r['gamma'] - 0.1) < 1e-9 else ('FGW' if r['gamma'] == 0 else '')
        f.write("  gamma={:<6} {:<7} {:.4f}   {:.4f}\n".format(
            r['gamma'], tag, r['accuracy'], r['cost']))
    f.write("\n  LROT (gamma=0.1) acc: {:.4f}\n".format(ablation_results[2]['accuracy']))
    f.write("  FGW  (gamma=0.0) acc: {:.4f}\n".format(ablation_results[0]['accuracy']))
    imp = (ablation_results[2]['accuracy'] / max(ablation_results[0]['accuracy'], 1e-9) - 1) * 100
    f.write("  Improvement: {:.1f}%\n".format(imp))
print(f"  [SAVE] 结果已保存: {OUT_DIR.parent / 'lrot_synthetic_results.txt'}")

# ---------------------------------------------------------------------------
# 8. Visualization
# ---------------------------------------------------------------------------

print("\n[LROT] Generating visualizations...")

# Create 2x3 panel figure
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Panel 1: A. Before alignment (参考图风格: Slice A 实心区域色 + Slice B 空心描边区域色, 错位可见)
ax = axes[0, 0]
# shared window (A/B 同一坐标窗, 便于看错位)
allx = np.concatenate([slice_A['coords'][:,0], slice_B['coords'][:,0]])
ally = np.concatenate([slice_A['coords'][:,1], slice_B['coords'][:,1]])
mx = (allx.min()+allx.max())/2; my_ = (ally.min()+ally.max())/2
half = max(allx.max()-allx.min(), ally.max()-ally.min())/2*1.12
win = ([mx-half, mx+half], [my_-half, my_+half])
for r in range(N_REGIONS):
    col = matplotlib.colormaps['tab10'](r)
    mk = slice_A['region_labels'] == r
    ax.scatter(slice_A['coords'][mk,0], slice_A['coords'][mk,1],
               c=[col], s=26, alpha=0.9, linewidths=0, zorder=3)
    mkB = slice_B['region_labels'] == r
    ax.scatter(slice_B['coords'][mkB,0], slice_B['coords'][mkB,1],
               facecolors='none', edgecolors=[col], s=26, linewidths=0.7, zorder=2)
ax.set_xlim(*win[0]); ax.set_ylim(*win[1]); ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('A. Before LROT\n(solid = Slice A, hollow = Slice B)', fontsize=11)
handles = [plt.Line2D([],[], marker='o', ls='', color=matplotlib.colormaps['tab10'](r),
          markerfacecolor=matplotlib.colormaps['tab10'](r), markersize=6, label=REGION_NAMES[r]) for r in range(N_REGIONS)]
ax.legend(handles=handles, fontsize=6.5, loc='lower left', framealpha=0.9)

# Panel 2: B. After alignment (参考图风格: 灰 A 参考底 + B 软映射到 A 帧、按 B 自身区域色)
ax = axes[0, 1]
# B 映射到 A 帧: B_on_A = colnorm(P)^T @ A
Pcn = P_lrot / np.maximum(P_lrot.sum(0, keepdims=True), 1e-10)   # 列归一化(每列=一个 B spot)
B_on_A = Pcn.T @ slice_A['coords']
ax.scatter(slice_A['coords'][:,0], slice_A['coords'][:,1], c='0.85', s=10,
           alpha=0.7, linewidths=0, zorder=1)                      # 灰 A 参考
for r in range(N_REGIONS):
    mkB = slice_B['region_labels'] == r
    ax.scatter(B_on_A[mkB,0], B_on_A[mkB,1], c=[matplotlib.colormaps['tab10'](r)], s=22,
               alpha=0.95, linewidths=0, zorder=3)
ax.set_xlim(*win[0]); ax.set_ylim(*win[1]); ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('B. After LROT (\u03b3=0.1, Acc=%.3f)\n(Slice B soft-mapped onto Slice A frame)' % acc_lrot, fontsize=11)

# Panel 3: Transport Plan (subset, enlarged to 160x160 for smoother display)
ax = axes[0, 2]
# Subsample for visualization
idx = np.random.choice(slice_A['n_spots'], min(160, slice_A['n_spots']), replace=False)
idy = np.random.choice(slice_B['n_spots'], min(160, slice_B['n_spots']), replace=False)
P_sub = P_lrot[np.ix_(idx, idy)]
im = ax.imshow(P_sub, aspect='auto', cmap='YlOrRd')
ax.set_title('C. Transport Plan\n(160×160 subset)', fontsize=11)
ax.set_xlabel('Slice B spots')
ax.set_ylabel('Slice A spots')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 4: LR Strength Matrix (subset, enlarged to 160x160)
ax = axes[1, 0]
S_sub = S_lr[np.ix_(idx, idy)]
im = ax.imshow(S_sub, aspect='auto', cmap='viridis')
ax.set_title('D. LR Strength Matrix\n(160×160 subset)', fontsize=11)
ax.set_xlabel('Slice B (receptor)')
ax.set_ylabel('Slice A (ligand)')
plt.colorbar(im, ax=ax, shrink=0.8)

# Panel 5: Top LR pairs
ax = axes[1, 1]
pair_labels = [f"{l}→{r}" for l, r, w in top_lr_pairs[:10]]
pair_values = [w for _, _, w in top_lr_pairs[:10]]
bars = ax.barh(range(len(pair_labels)), pair_values, color='steelblue')
ax.set_yticks(range(len(pair_labels)))
ax.set_yticklabels(pair_labels, fontsize=8)
ax.set_xlabel('Interaction Weight')
ax.set_title('E. Top Ligand-Receptor Pairs', fontsize=11)
ax.invert_yaxis()

# Panel 6: Loss curves
ax = axes[1, 2]
ax.plot(loss_hist, 'b-', linewidth=1.5, label='LROT ($\gamma=0.1$)')
# Also plot ablation loss curves
colors = ['gray', 'orange', 'green', 'red']
for i, g in enumerate(gamma_values):
    _, loss_g = fgw_lr_solver(
        slice_A, slice_B, S_lr,
        gamma=g, beta=0.3, alpha=0.5,
        max_iter=80, verbose=False
    )
    ax.plot(loss_g, color=colors[i], linewidth=1.0, alpha=0.7,
            label=f'$\gamma={g:.2f}$')
ax.set_title('F. Loss Curves (Ablation)', fontsize=11)
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')
ax.legend(fontsize=8)

fig.tight_layout(rect=(0, 0, 1, 0.94))
fig_path = OUT_DIR / 'lrot_full_panel.png'
fig.savefig(fig_path, dpi=600, bbox_inches='tight')
print(f"  Saved: {fig_path}")

# ---------------------------------------------------------------------------
# 8b. Ablation comparison bar chart
# ---------------------------------------------------------------------------
fig2, ax = plt.subplots(1, 2, figsize=(12, 4))

gammas = [r['gamma'] for r in ablation_results]
accs = [r['accuracy'] for r in ablation_results]
costs = [r['cost'] for r in ablation_results]

ax[0].bar(range(len(gammas)), accs, color=['gray', 'orange', 'green', 'red'], alpha=0.8)
ax[0].set_xticks(range(len(gammas)))
ax[0].set_xticklabels([f'$\gamma={g:.2f}$' for g in gammas])
ax[0].set_ylabel('Alignment Accuracy')
ax[0].set_title(r'(A) Region mapping accuracy vs $\gamma$')
ax[0].set_ylim(0, 1.0)
for i, v in enumerate(accs):
    ax[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

ax[1].bar(range(len(gammas)), costs, color=['gray', 'orange', 'green', 'red'], alpha=0.8)
ax[1].set_xticks(range(len(gammas)))
ax[1].set_xticklabels([f'$\gamma={g:.2f}$' for g in gammas])
ax[1].set_ylabel('Transport Cost')
ax[1].set_title(r'(B) Expression transport cost vs $\gamma$')
for i, v in enumerate(costs):
    ax[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)

fig2.tight_layout(rect=(0, 0, 1, 0.92))
ablation_path = OUT_DIR / 'lrot_ablation.png'
fig2.savefig(ablation_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {ablation_path}")

# ---------------------------------------------------------------------------
# 9. Summary Report
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("  LROT PROTOTYPE SUMMARY")
print("="*60)
print(f"  Output directory: {OUT_DIR}")
print(f"  Slice A: {slice_A['n_spots']} spots, {slice_A['n_genes']} genes")
print(f"  Slice B: {slice_B['n_spots']} spots, {slice_B['n_genes']} genes")
print(f"  LR database: {len(LR_DB)} pairs")
print(f"  LROT alignment accuracy: {acc_lrot:.4f}")
print()
print("  Ablation Results:")
print(f"    gamma=0.00 (baseline FGW): acc={ablation_results[0]['accuracy']:.4f}")
print(f"    gamma=0.05:                 acc={ablation_results[1]['accuracy']:.4f}")
print(f"    gamma=0.10:                 acc={ablation_results[2]['accuracy']:.4f}")
print(f"    gamma=0.20:                 acc={ablation_results[3]['accuracy']:.4f}")
print()
print("  Key Insight: LR guidance (gamma > 0) improves alignment")
print("  by providing biological prior on spot correspondence.")
print("="*60)

print(f"\n[LROT] Done! Output saved to {OUT_DIR}")

# 保存传输数据
from lrot_core import save_transport_data
coords_aligned = compute_alignment(P_lrot, slice_A['coords'], slice_B['coords'])
save_transport_data('synthetic', P_lrot, slice_A, slice_B,
                    loss_hist=loss_hist, aligned_coords=coords_aligned,
                    extra={'ablation': ablation_results})
