# lrot_core.py - 从 lrot_prototype.py 提取的纯函数模块
# 手工维护模块（非自动生成）。改动 LR_DB 后请跑 lrot_families.py 的导入期校验

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from pathlib import Path
from scipy.stats import spearmanr
import warnings

# ========== 常量 ==========
SEED = 42

# 配体-受体数据库（小鼠大脑）
# 家族归属与完备性校验以 lrot_families.py 为准（唯一权威来源）；下方分隔注释仅供阅读。
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
    # --- BMP ---
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

def lr_gene_list(lr_db):
    """Get unique ligand and receptor genes from the database."""
    ligands = sorted(set(k[0] for k in lr_db))
    receptors = sorted(set(k[1] for k in lr_db))
    return ligands, receptors

# ========== 全局派生常量 ==========
LIGANDS, RECEPTORS = lr_gene_list(LR_DB)
ALL_LR_GENES = sorted(set(LIGANDS + RECEPTORS))  # sorted: 避免 PYTHONHASHSEED 跨进程顺序不同
N_REGIONS = 6  # 用于 realistic data（6个脑区）

def simulate_slice(n_spots=1000, n_genes=200, lr_genes=None, region_weights=None,
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

def gene_to_idx(slice_data, gene_name):
    try:
        return slice_data['gene_names'].index(gene_name)
    except ValueError:
        return None

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
                   alpha=0.5, max_iter=200, lr_step=50, tol=1e-5, verbose=True,
                   lr_gene_list=None):
    """
    Fused Gromov-Wasserstein solver with LR guidance.

    The total transport cost combines three terms:
      1. Expression cost (cosine distance between gene expression profiles)
      2. LR guidance cost (-log(LR strength + epsilon))
      3. GW spatial cost (spatial structure preservation)

    C_total = alpha * C_expr + gamma * C_lr + (1-alpha) * beta * C_gw

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
    tol : float
        Convergence tolerance
    verbose : bool
        Print progress
    lr_gene_list : list or None
        Custom list of LR gene names to exclude from expression cost.
        If None, uses global ALL_LR_GENES (mouse brain genes).

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
        # Use custom LR gene list if provided, otherwise use global ALL_LR_GENES
        lr_genes_to_exclude = lr_gene_list if lr_gene_list is not None else ALL_LR_GENES
        non_lr_mask = np.array([g not in lr_genes_to_exclude for g in gene_names])
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

def compute_mapping_consistency_hard(transport_plan, labels_A, labels_B):
    """Hard matching accuracy: for each spot in A, take the argmax target in B
    and compare region labels (no soft averaging over the row)."""
    hard_idx = np.argmax(transport_plan, axis=1)
    return float(np.mean(labels_B[hard_idx] == labels_A))


def compute_aligned_nn_accuracy(transport_plan, coords_A, coords_B, labels_A, labels_B):
    """Aligned-coordinate nearest-neighbor accuracy: barycentrically map A spots onto
    B's coordinate frame (P_norm @ coords_B), then for each A spot find its nearest
    original B spot and compare region labels. Stricter than the soft metric."""
    P_norm = transport_plan / np.maximum(transport_plan.sum(1, keepdims=True), 1e-10)
    coords_aligned = P_norm @ coords_B
    D = cdist(coords_aligned, coords_B)
    nn_idx = np.argmin(D, axis=1)
    return float(np.mean(labels_B[nn_idx] == labels_A))


# ========== 保存/加载中间数据 ==========

DEFAULT_DATA_DIR = Path(__file__).parent / "lrot_output"


def save_transport_data(experiment_name, P, slice_A=None, slice_B=None,
                        loss_hist=None, aligned_coords=None, extra=None,
                        data_dir=None):
    """
    保存传输矩阵 P 及相关中间数据到 .npz 文件

    参数
    ----------
    experiment_name : str
        实验标识符，如 'synthetic', 'real_visium', 'breast_cancer'
    P : ndarray (nA, nB)
        传输计划矩阵
    slice_A, slice_B : dict or None
        切片字典（可选，会从中提取 coords/expr）
    loss_hist : list or None
        损失历史
    aligned_coords : ndarray or None
        对齐后的坐标（对齐后 B 到 A 的映射）
    extra : dict or None
        其他需要保存的变量
    data_dir : Path or None
        保存目录，默认 lrot_output
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    save_dict = {'P': P}

    # 提取切片数据
    if slice_A is not None:
        for key in ['coords', 'expr', 'region_labels', 'gene_names']:
            if key in slice_A:
                # expr 可能很大, 用 float16 节省空间
                val = slice_A[key]
                if key == 'expr' and isinstance(val, np.ndarray):
                    val = val.astype(np.float16)
                save_dict[f'A_{key}'] = val

    if slice_B is not None:
        for key in ['coords', 'expr', 'region_labels', 'gene_names']:
            if key in slice_B:
                val = slice_B[key]
                if key == 'expr' and isinstance(val, np.ndarray):
                    val = val.astype(np.float16)
                save_dict[f'B_{key}'] = val

    if loss_hist is not None:
        save_dict['loss_hist'] = np.array(loss_hist)

    if aligned_coords is not None:
        save_dict['aligned_coords'] = aligned_coords

    if extra is not None:
        for k, v in extra.items():
            save_dict[k] = v

    save_path = data_dir / f'lrot_data_{experiment_name}.npz'
    np.savez_compressed(save_path, **save_dict)

    # 报告文件大小
    size_mb = save_path.stat().st_size / 1e6
    keys_str = ', '.join(k for k in save_dict.keys() if k != 'P')
    print(f"  [SAVE] 传输数据已保存: {save_path.name} ({size_mb:.1f} MB)")
    print(f"         包含: P({P.shape}), {keys_str}")
    return save_path


def load_transport_data(experiment_name, data_dir=None):
    """
    加载之前保存的传输数据

    返回
    -------
    dict : 包含 'P', 以及可能的 'aligned_coords', 'loss_hist', 切片数据等
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)
    save_path = data_dir / f'lrot_data_{experiment_name}.npz'

    if not save_path.exists():
        print(f"  [ERROR] 未找到保存的数据: {save_path}")
        return None

    data = np.load(save_path, allow_pickle=True)
    # 转为普通 dict 方便使用
    result = {}
    for key in data.files:
        val = data[key]
        # 恢复 object 数组（如 gene_names）
        if val.dtype == object:
            result[key] = val.tolist()
        else:
            result[key] = val

    print(f"  [LOAD] 已加载: {save_path.name}")
    for key in sorted(result.keys()):
        val = result[key]
        if isinstance(val, np.ndarray):
            print(f"         {key}: {val.shape}, {val.dtype}")
        else:
            print(f"         {key}: {type(val).__name__}")
    return result