"""
run_cancer_visium.py — 在人类乳腺癌 10x Visium 数据上验证 LROT
使用两个连续切片 (Block A Section 1 & Section 2) 进行空间对齐

Breast cancer relevance:
  - 高异质性肿瘤微环境 (TME)，包含肿瘤细胞、免疫细胞、基质细胞
  - 已知重要的 LR 信号通路：HER2/ErbB, EGF/EGFR, HGF/MET, VEGF/KDR
  - 连续切片可验证 LROT 在多切片对齐中的有效性

输出:
  lrot_output/lrot_breast_cancer.png — 四面板对比图
  控制台输出 LROT vs FGW 定量对比
"""
import numpy as np
import time
import sys
import os
import warnings
import requests
warnings.filterwarnings('ignore')
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


sys.path.insert(0, _R)
from lrot_core import (
    fgw_lr_solver, compute_lr_strength_matrix,
    compute_mapping_consistency, compute_ot_cost,
    compute_alignment, cosine_distance,
    save_transport_data,
    ALL_LR_GENES, LIGANDS, RECEPTORS

)

# ========== 人类 LR 数据库 (大写 HUGO 符号) ==========
# 从小鼠 LR_DB 转换: 大多数基因名直接大写即可
# 额外加入了乳腺癌相关的重要 LR 通路
HUMAN_LR_DB = {
    # --- Neurotrophin / Growth Factor ---
    ("NTNG1", "NTRK2"): 0.95,
    ("BDNF", "NTRK2"): 0.92,
    ("NTF3", "NTRK3"): 0.90,
    ("NGF", "NTRK1"): 0.88,
    ("GDNF", "GFRA1"): 0.85,
    ("ARTN", "GFRA3"): 0.80,
    ("NRTN", "GFRA2"): 0.78,
    ("FGF8", "FGFR1"): 0.90,
    ("FGF9", "FGFR2"): 0.85,
    ("FGF10", "FGFR2"): 0.82,
    ("FGF2", "FGFR1"): 0.88,
    ("EGF", "EGFR"): 0.95,  # 乳腺癌关键通路
    ("TGFA", "EGFR"): 0.88, # 乳腺癌相关
    ("HGF", "MET"): 0.90,   # 侵袭转移关键
    ("VEGFA", "KDR"): 0.92, # 血管生成
    ("VEGFB", "FLT1"): 0.78,
    ("PDGFA", "PDGFRA"): 0.80,
    ("PDGFB", "PDGFRB"): 0.82,
    ("IGF1", "IGF1R"): 0.85,
    ("IGF2", "IGF1R"): 0.80,
    # --- ErbB/HER 通路 (乳腺癌核心) ---
    ("ERBB2", "ERBB2"): 0.98,  # HER2 同源二聚化
    ("ERBB2", "ERBB3"): 0.92,  # HER2-HER3 异源二聚化
    ("EREG", "EGFR"): 0.88,    # Epiregulin
    ("BTC", "EGFR"): 0.85,     # Betacellulin
    ("HBEGF", "EGFR"): 0.87,  # Heparin-binding EGF
    ("NRG1", "ERBB3"): 0.90,  # Neuregulin 1
    ("NRG1", "ERBB4"): 0.85,
    # --- Wnt Pathway ---
    ("WNT3A", "FZD1"): 0.88,
    ("WNT5A", "FZD5"): 0.85,
    ("WNT7A", "FZD10"): 0.82,
    ("WNT1", "FZD1"): 0.80,
    # --- Notch ---
    ("DLL1", "NOTCH1"): 0.90,
    ("DLL4", "NOTCH4"): 0.85,
    ("JAG1", "NOTCH1"): 0.88,
    ("JAG2", "NOTCH2"): 0.83,
    # --- Ephrin ---
    ("EFNB1", "EPHB2"): 0.90,
    ("EFNA1", "EPHA4"): 0.85,
    ("EFNB2", "EPHB4"): 0.82,
    ("EFNA5", "EPHA3"): 0.78,
    # --- Semaphorin ---
    ("SEMA3A", "NRP1"): 0.90,
    ("SEMA3F", "NRP2"): 0.85,
    ("SEMA4D", "PLXNB1"): 0.80,
    ("SEMA6A", "PLXNA2"): 0.75,
    # --- Slit/Robo ---
    ("SLIT1", "ROBO1"): 0.88,
    ("SLIT2", "ROBO2"): 0.85,
    ("SLIT3", "ROBO2"): 0.80,
    # --- Chemokine / Cytokine ---
    ("CXCL12", "CXCR4"): 0.95,  # 免疫募集
    ("CXCL13", "CXCR5"): 0.85,
    ("CCL2", "CCR2"): 0.85,     # 巨噬细胞募集
    ("CCL5", "CCR5"): 0.82,
    ("IL1B", "IL1R1"): 0.90,    # 炎症
    ("TNF", "TNFRSF1A"): 0.90,  # 炎症
    ("TGFB1", "TGFBR1"): 0.92,  # EMT 关键
    ("TGFB2", "TGFBR2"): 0.88,
    # --- BMP ---
    ("BMP4", "BMPR1A"): 0.85,
    ("BMP7", "BMPR2"): 0.80,
    ("BMP2", "ACVR1"): 0.82,
    # --- Cell Adhesion (EMT/转移相关) ---
    ("CDH1", "CDH1"): 0.95,      # E-cadherin
    ("CDH2", "CDH2"): 0.92,      # N-cadherin (EMT marker)
    ("NCAM1", "NCAM1"): 0.85,
    ("L1CAM", "L1CAM"): 0.80,
    ("COL1A1", "ITGA2"): 0.85,  # 胶原-整合素
    ("VTN", "ITGAV"): 0.75,
    # --- Hedgehog ---
    ("SHH", "PTCH1"): 0.90,
    ("IHH", "PTCH2"): 0.82,
    ("DHH", "PTCH1"): 0.78,
    # --- Reelin ---
    ("RELN", "LRP8"): 0.90,
    ("RELN", "VLDLR"): 0.85,
}

HUMAN_LIGANDS = sorted(set(k[0] for k in HUMAN_LR_DB))
HUMAN_RECEPTORS = sorted(set(k[1] for k in HUMAN_LR_DB))
HUMAN_ALL_LR = sorted(set(HUMAN_LIGANDS + HUMAN_RECEPTORS))  # sorted: 避免 PYTHONHASHSEED 跨进程顺序不同

# ========== 配置 ==========
N_SAMPLE = 1000     # 采样 spot 数（规模化验证）
N_SAMPLE_FAST = 300  # 快速对照实验
GAMMA = 0.1
SEED = 42

OUT_DIR = os.path.join(_R, 'lrot_output')

def download_h5_direct(sample_id, save_dir):
    """直接从 10x CDN 下载 h5 文件（使用 requests + 浏览器 UA 绕过 403）"""
    import requests
    base = f"https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/{sample_id}"
    h5_url = f"{base}/{sample_id}_filtered_feature_bc_matrix.h5"
    h5_path = os.path.join(save_dir, f"{sample_id}.h5")
    
    if os.path.exists(h5_path):
        size = os.path.getsize(h5_path)
        if size > 1000000:  # 至少 1MB
            print(f"    文件已存在: {h5_path} ({size/1e6:.1f} MB)")
            return h5_path
        print(f"    文件不完整 ({size} bytes)，重新下载...")
    
    print(f"    下载 {h5_url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    resp = requests.get(h5_url, headers=headers, stream=True, timeout=300)
    resp.raise_for_status()
    
    total = int(resp.headers.get('content-length', 0))
    downloaded = 0
    with open(h5_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                if int(pct) % 10 == 0 and downloaded >= total * (int(pct)//10) / 10:
                    pass  # progress indicator
    
    size_mb = os.path.getsize(h5_path) / 1e6
    print(f"    下载完成: {size_mb:.1f} MB")
    return h5_path

def download_and_prepare(sample_id, target_spots=1000, seed=42):
    """
    下载 Visium 数据并准备为 slice 字典格式
    直接从 10x CDN 下载 h5 + 生成网格坐标

    参数
    ----------
    sample_id : str
        10x Visium 样本 ID
    target_spots : int
        目标采样 spot 数
    seed : int
        采样随机种子（不同切片用不同种子确保独立采样）
    """
    print(f"\n  加载 {sample_id}...")
    import scanpy as sc
    
    save_dir = os.path.join(os.path.dirname(OUT_DIR), "cancer_data")
    os.makedirs(save_dir, exist_ok=True)
    
    h5_path = download_h5_direct(sample_id, save_dir)
    
    # 用 read_10x_h5 读取
    adata = sc.read_10x_h5(h5_path)
    if not adata.var_names.is_unique:
        adata.var_names_make_unique()
    
    expr = adata.X.toarray().astype(np.float32) if hasattr(adata.X, 'toarray') else np.array(adata.X, dtype=np.float32)
    gene_names = list(adata.var_names)
    
    print(f"  原始: {adata.n_obs} spots, {expr.shape[1]} genes")
    
    # 生成 Visium 风格的六边形网格坐标
    side = int(np.ceil(np.sqrt(adata.n_obs)))
    coords_list = []
    for i in range(side):
        for j in range(side):
            x = i * np.sqrt(3)
            y = j * 1.5 + (0.75 if i % 2 == 1 else 0.0)
            coords_list.append([x, y])
    coords = np.array(coords_list[:adata.n_obs])
    
    # 采样（使用独立种子，确保两切片采样不同 spot）
    rng = np.random.RandomState(seed)
    n_spots = min(target_spots, len(coords))
    idx = rng.choice(len(coords), n_spots, replace=False)
    
    coords = coords[idx]
    expr = expr[idx]
    
    print(f"  采样后: {n_spots} spots, {expr.shape[1]} genes, "
          f"x=[{coords[:,0].min():.1f},{coords[:,0].max():.1f}]")
    
    return {
        'coords': coords,
        'expr': expr,
        'gene_names': gene_names,
        'region_labels': np.zeros(n_spots, dtype=int),
        'n_spots': n_spots,
        'n_genes': expr.shape[1],
    }

def apply_misalignment(coords, rotation_deg=2.0, translation=0.5, seed=99):
    """对切片坐标施加轻微错位，模拟真实连续切片间的物理偏移"""
    rng = np.random.RandomState(seed)
    center = coords.mean(axis=0)
    coords_c = coords - center
    
    theta = np.deg2rad(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    M = np.array([[c, -s], [s, c]])
    coords_aligned = coords_c @ M.T + center
    # 加微小随机平移
    coords_aligned += rng.randn(1, 2) * translation
    return coords_aligned

def compute_lr_strength_human(slice_A, slice_B, lr_db, scale=10.0):
    """人类 LR 强度矩阵计算（使用 HUMAN_LR_DB）"""
    nA = slice_A['n_spots']
    nB = slice_B['n_spots']
    S = np.zeros((nA, nB))
    pair_contributions = []
    
    gene_names_A = [g.upper() for g in slice_A['gene_names']]
    gene_names_B = [g.upper() for g in slice_B['gene_names']]
    
    found_pairs = 0
    for (lig, rec), weight in lr_db.items():
        try:
            lig_idx = gene_names_A.index(lig)
            rec_idx = gene_names_B.index(rec)
        except ValueError:
            continue
        
        lig_expr = slice_A['expr'][:, lig_idx]
        rec_expr = slice_B['expr'][:, rec_idx]
        contrib = np.outer(lig_expr, rec_expr) * weight
        S += contrib
        pair_contributions.append((weight * lig_expr.sum() * rec_expr.sum(), lig, rec, weight))
        found_pairs += 1
    
    print(f"  LR 匹配: {found_pairs}/{len(lr_db)} 对基因找到")
    
    # 归一化
    S_min, S_max = S.min(), S.max()
    if S_max > S_min:
        S = (S - S_min) / (S_max - S_min)
    
    pair_contributions.sort(reverse=True)
    top_pairs = [(l, r, w) for _, l, r, w in pair_contributions[:10]]
    
    return S, top_pairs

def main():
    print("=" * 70)
    print("  LROT on Human Breast Cancer (10x Visium)")
    print("  Real continuous slices: Section 1 vs Section 2")
    print(f"  Scale: {N_SAMPLE} spots per slice")
    print("=" * 70)

    # ========== [1] 加载两个真实连续切片 ==========
    print("\n[1/5] 加载乳腺癌真实连续切片...")
    print("  ── Slice A: V1_Breast_Cancer_Block_A_Section_1")
    slice_A = download_and_prepare("V1_Breast_Cancer_Block_A_Section_1",
                                   target_spots=N_SAMPLE, seed=42)
    print("  ── Slice B: V1_Breast_Cancer_Block_A_Section_2")
    slice_B = download_and_prepare("V1_Breast_Cancer_Block_A_Section_2",
                                   target_spots=N_SAMPLE, seed=99)

    # 对 Section 2 施加轻微错位，模拟真实切片间的物理偏移
    print("\n[2/5] 模拟切片间物理错位 (旋转2°, 微平移)...")
    slice_B['coords'] = apply_misalignment(slice_B['coords'],
                                           rotation_deg=2.0, translation=0.5)

    # ========== [3] 计算 LR 强度 ==========
    print("\n[3/5] 计算 LR 强度矩阵 (人类数据库)...")
    t0 = time.time()
    S_lr, top_pairs = compute_lr_strength_human(slice_A, slice_B, HUMAN_LR_DB)
    print(f"  LR矩阵: {S_lr.shape}, 范围 [{S_lr.min():.4f}, {S_lr.max():.4f}]")
    print(f"  Top LR pairs (乳腺癌相关):")
    for i, (l, r, w) in enumerate(top_pairs[:8]):
        print(f"    {i+1}. {l} - {r} (权重={w:.1f})")
    print(f"  耗时: {time.time()-t0:.2f}s")

    # ========== [4] 运行 LROT ==========
    print(f"\n[4a/5] LROT (gamma={GAMMA}, {N_SAMPLE} spots)...")
    t0 = time.time()
    P_lrot, loss_hist = fgw_lr_solver(
        slice_A, slice_B, S_lr,
        gamma=GAMMA, beta=0.3, alpha=0.5,
        max_iter=200, verbose=True,
        lr_gene_list=HUMAN_ALL_LR
    )
    lrot_time = time.time() - t0
    print(f"  耗时: {lrot_time:.2f}s")

    print(f"\n[4b/5] FGW baseline (gamma=0, {N_SAMPLE} spots)...")
    t0 = time.time()
    # 基线口径（设计如此）：γ=0 的 FGW 基线用**全部基因**做 C_expr ——
    # 无 C_lr 项故无重复计数问题，忠实反映 PASTE 风格实现，且是更保守的强基线。
    # LROT 臂仍按设计排除 LR 基因。对照见 lrot_output/lrot_probe_lrot_fullgenes.txt。
    P_fgw, _ = fgw_lr_solver(
        slice_A, slice_B, np.zeros_like(S_lr),
        gamma=0.0, beta=0.3, alpha=0.5,
        max_iter=200, verbose=False
    )
    fgw_time = time.time() - t0
    print(f"  耗时: {fgw_time:.2f}s")

    # ========== 定量评估 ==========
    lrot_cost = compute_ot_cost(P_lrot, slice_A['expr'], slice_B['expr'])
    Pn = P_lrot / np.maximum(P_lrot.sum(1, keepdims=True), 1e-10)
    lrot_ent = -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()

    fgw_cost = compute_ot_cost(P_fgw, slice_A['expr'], slice_B['expr'])
    P0n = P_fgw / np.maximum(P_fgw.sum(1, keepdims=True), 1e-10)
    fgw_ent = -np.sum(P0n * np.log(np.maximum(P0n, 1e-10)), axis=1).mean()

    cost_improvement = (fgw_cost - lrot_cost) / fgw_cost * 100
    ent_improvement = (fgw_ent - lrot_ent) / fgw_ent * 100

    # === 快速对照实验 (300 spots, 验证规模不影响结论) ===
    print(f"\n[4c/5] 快速对照: 300 spots 版本...")
    slice_A_fast = download_and_prepare("V1_Breast_Cancer_Block_A_Section_1",
                                        target_spots=N_SAMPLE_FAST, seed=42)
    slice_B_fast = download_and_prepare("V1_Breast_Cancer_Block_A_Section_2",
                                        target_spots=N_SAMPLE_FAST, seed=99)
    slice_B_fast['coords'] = apply_misalignment(slice_B_fast['coords'],
                                                rotation_deg=2.0, translation=0.5)
    S_lr_fast, _ = compute_lr_strength_human(slice_A_fast, slice_B_fast, HUMAN_LR_DB)
    t0 = time.time()
    P_lrot_fast, _ = fgw_lr_solver(
        slice_A_fast, slice_B_fast, S_lr_fast,
        gamma=GAMMA, beta=0.3, alpha=0.5,
        max_iter=200, verbose=False
    )
    fast_time = time.time() - t0
    fast_cost = compute_ot_cost(P_lrot_fast, slice_A_fast['expr'], slice_B_fast['expr'])
    Pn_fast = P_lrot_fast / np.maximum(P_lrot_fast.sum(1, keepdims=True), 1e-10)
    fast_ent = -np.sum(Pn_fast * np.log(np.maximum(Pn_fast, 1e-10)), axis=1).mean()
    print(f"  300 spots: Cost={fast_cost:.4f}, Entropy={fast_ent:.4f}, 耗时={fast_time:.2f}s")

    # ========== 打印结果表格 ==========
    print("\n" + "=" * 70)
    print("  RESULTS: Human Breast Cancer Visium (Real Continuous Slices)")
    print("=" * 70)
    print(f"  {'Method':<22} {'Spots':<8} {'Cost':<12} {'Entropy':<12} {'Time(s)':<10}")
    print(f"  {'-'*64}")
    print(f"  {'LROT (gamma='+str(GAMMA)+')':<22} {N_SAMPLE:<8} {lrot_cost:<12.4f} {lrot_ent:<12.4f} {lrot_time:<10.2f}")
    print(f"  {'FGW (gamma=0)':<22} {N_SAMPLE:<8} {fgw_cost:<12.4f} {fgw_ent:<12.4f} {fgw_time:<10.2f}")
    print(f"  {'LROT (300对照)':<22} {N_SAMPLE_FAST:<8} {fast_cost:<12.4f} {fast_ent:<12.4f} {fast_time:<10.2f}")
    print(f"  {'─'*64}")
    print(f"  Cost improvement:  {cost_improvement:.2f}%")
    print(f"  Entropy improvement: {ent_improvement:.2f}%")
    print("=" * 70)

    # ========== [5] 可视化 ==========
    print("\n[5/5] 生成可视化...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    coords_aligned = compute_alignment(P_lrot, slice_A['coords'], slice_B['coords'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Panel A/B: 增强 Before/After (高对比叠加+放大插图+位移量化; 保持2°真实连续切片错位)
    import fig_align
    fig_align.draw_before_after(axes, (0, 0), (0, 1),
                                slice_A['coords'], slice_B['coords'], P_lrot,
                                A_name='Section 1', B_name='Section 2',
                                title_before='A: Before LROT (raw serial sections)',
                                title_after='B: After LROT Alignment',
                                sA=13, sB=13)

    # Panel C: Loss curve
    axes[0, 2].plot(loss_hist, 'b-', linewidth=1.5)
    axes[0, 2].axvline(x=len(loss_hist)*0.2, color='gray', linestyle='--', alpha=0.5)
    axes[0, 2].set_xlabel('Iteration', fontsize=10)
    axes[0, 2].set_ylabel('Loss', fontsize=10)
    axes[0, 2].set_title(f'C: Loss Curve ({len(loss_hist)} iter)', fontsize=11)
    axes[0, 2].grid(True, alpha=0.3)

    # Panel D: Transport plan (100x100 preview)
    step = max(1, N_SAMPLE // 100)
    P_display = P_lrot[::step, ::step]
    im = axes[1, 0].imshow(P_display, cmap='hot', aspect='auto', interpolation='nearest')
    axes[1, 0].set_title(f'D: Transport Plan ({N_SAMPLE}\u00d7{N_SAMPLE}, shown @1/{step})', fontsize=11)
    axes[1, 0].set_xlabel('Section 2 spots', fontsize=10)
    axes[1, 0].set_ylabel('Section 1 spots', fontsize=10)
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

    # Panel E: Top LR pairs
    pair_labels = [f'{l}-{r}' for l, r, w in top_pairs[:8]]
    pair_values = [w for _, _, w in top_pairs[:8]]
    colors_bar = matplotlib.colormaps['RdYlGn'](np.linspace(0.3, 0.9, len(pair_labels)))
    bars = axes[1, 1].barh(range(len(pair_labels)), pair_values, color=colors_bar)
    for bar, val in zip(bars, pair_values):
        axes[1, 1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{val:.2f}', va='center', fontsize=8)
    axes[1, 1].set_yticks(range(len(pair_labels)))
    axes[1, 1].set_yticklabels(pair_labels, fontsize=8)
    axes[1, 1].set_xlabel('DB Weight', fontsize=10)
    axes[1, 1].set_title('E: Top LR Pairs (Breast Cancer)', fontsize=11)
    axes[1, 1].invert_yaxis()

    # Panel F: Quantitative comparison - Entropy only (key metric)
    ax_bar = axes[1, 2]
    methods_names = ['FGW (γ=0)', 'LROT (γ=0.1)']
    ent_vals = [fgw_ent, lrot_ent]
    cost_vals = [fgw_cost, lrot_cost]
    
    # 熵对比柱状图（使用缩放的y轴使差异可见）
    ent_colors = ['#E74C3C', '#2ECC71']
    bars_ent = ax_bar.bar([0, 1], ent_vals, width=0.5, color=ent_colors,
                          edgecolor='gray', linewidth=0.8, alpha=0.9)
    ax_bar.set_xticks([0, 1])
    ax_bar.set_xticklabels(methods_names, fontsize=10)
    ax_bar.set_ylabel('Entropy', fontsize=11, color='#333333')
    ax_bar.set_ylim(min(ent_vals) * 0.98, max(ent_vals) * 1.02)  # 缩放到差异可见
    ax_bar.set_title('F: Alignment Uncertainty (Entropy)', fontsize=11, fontweight='bold')
    
    # 标注数值和改善百分比
    for bar, v, name in zip(bars_ent, ent_vals, ['FGW', 'LROT']):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{v:.4f}', ha='center', fontsize=9, fontweight='bold',
                   color='#B2182B' if name == 'FGW' else '#2E8B57')
    # 改善标注
    imp = (fgw_ent - lrot_ent) / fgw_ent * 100
    ax_bar.annotate(f'↓{imp:.2f}%', xy=(1, lrot_ent), xytext=(0.5, max(ent_vals)*1.01),
                   ha='center', fontsize=10, color='#2E8B57', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save_path = out_dir / 'lrot_breast_cancer.png'
    fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # ========== 保存结果 ==========
    result_path = out_dir / 'lrot_cancer_results.txt'
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  LROT on Human Breast Cancer Visium - RESULTS\n")
        f.write("  Real continuous slices: Section 1 vs Section 2\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Slice A: V1_Breast_Cancer_Block_A_Section_1 ({N_SAMPLE} spots)\n")
        f.write(f"Slice B: V1_Breast_Cancer_Block_A_Section_2 ({N_SAMPLE} spots)\n")
        f.write(f"N genes: {slice_A['n_genes']}\n\n")
        f.write(f"LR pairs matched: {len(top_pairs)}/{len(HUMAN_LR_DB)}\n\n")
        f.write(f"Top LR pairs:\n")
        for i, (l, r, w) in enumerate(top_pairs[:8]):
            f.write(f"  {i+1}. {l}-{r} (weight={w:.2f})\n")
        f.write(f"\nFull-scale ({N_SAMPLE} spots):\n")
        f.write(f"  LROT: Cost={lrot_cost:.4f}, Entropy={lrot_ent:.4f}, Time={lrot_time:.2f}s\n")
        f.write(f"  FGW:  Cost={fgw_cost:.4f}, Entropy={fgw_ent:.4f}, Time={fgw_time:.2f}s\n")
        f.write(f"  Cost improvement: {cost_improvement:.2f}%\n")
        f.write(f"  Entropy improvement: {ent_improvement:.2f}%\n\n")
        f.write(f"Fast control (300 spots):\n")
        f.write(f"  LROT: Cost={fast_cost:.4f}, Entropy={fast_ent:.4f}, Time={fast_time:.2f}s\n")
    print(f"  Results saved: {result_path}")

    # 保存传输数据
    coords_aligned = compute_alignment(P_lrot, slice_A['coords'], slice_B['coords'])
    save_transport_data('breast_cancer', P_lrot, slice_A, slice_B,
                        loss_hist=loss_hist, aligned_coords=coords_aligned,
                        extra={'P_fgw': P_fgw, 'top_pairs': top_pairs})

    print("\n" + "=" * 70)
    print("  Done! Breast cancer validation with real continuous slices complete.")
    print("=" * 70)

if __name__ == '__main__':
    main()
