"""
lrot_3d_reconstruction.py — 多切片 3D 重建
顺序对齐 5 张连续切片，生成 3D 体积可视化
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
    LR_DB
)
from real_st_loader import RealisticSTGenerator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(_R, 'lrot_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 参数 ==========
N_SLICES = 5          # 切片数量
N_SPOTS = 500         # 每切片 spot 数
N_GENES = 100         # 基因数
BASE_ROT = 3.0        # 切片间旋转增量（度）
BASE_BATCH = 0.08     # 切片间批次效应增量

print("=" * 60)
print("  LROT 多切片 3D 重建")
print(f"  切片: {N_SLICES}, 每切片 spots: {N_SPOTS}")
print("=" * 60)

# ========== 1. 生成多张连续切片 ==========
print("\n[1/4] 生成连续切片序列...")
slices = []
region_labels_list = []

# 第 0 张：基准切片
gen = RealisticSTGenerator(
    n_spots_A=N_SPOTS, n_spots_B=N_SPOTS,
    n_genes=N_GENES, seed=42
)
# 只用 generator 生成单张切片
rng = np.random.RandomState(42)

# 构建基准切片的坐标和表达
base_coords = gen._generate_hex_grid(N_SPOTS, jitter=0.05)
region_profiles = gen._generate_region_profiles(gen.n_regions, N_GENES)
region_labels_0, centroids = gen._assign_spots_to_regions(base_coords)

# 标准化基因名
ligs, recs = __import__('lrot_core', fromlist=['LIGANDS', 'RECEPTORS']).LIGANDS, \
             __import__('lrot_core', fromlist=['LIGANDS', 'RECEPTORS']).RECEPTORS
all_lr = sorted(set(ligs + recs))  # sorted: 避免 PYTHONHASHSEED 导致跨进程 set 顺序不同
n_lr = min(len(all_lr), N_GENES // 3)
other_genes = [f"Gene_{i}" for i in range(N_GENES - n_lr)]
gene_names = all_lr[:n_lr] + other_genes

def make_expr(coords, region_labels):
    """为给定坐标和区域生成表达矩阵"""
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

# 生成连续切片：每张在前一张基础上增加旋转 + 批次效应
coords_prev = base_coords.copy()
for z_idx in range(N_SLICES):
    if z_idx == 0:
        coords_z = base_coords.copy()
        expr_z = base_expr.copy()
    else:
        # 累积变形：每张切片比前一张多转 BASE_ROT 度
        cum_rot = BASE_ROT * z_idx
        cum_batch = BASE_BATCH * z_idx
        theta = np.deg2rad(cum_rot)
        c, s = np.cos(theta), np.sin(theta)
        M = np.array([[c, -s], [s, c]]) @ np.diag([0.98, 0.98])
        center = coords_prev.mean(0)
        coords_z = (coords_prev - center) @ M.T + center
        coords_z += rng.randn(N_SPOTS, 2) * 0.05  # 小随机扰动
        
        # 重新分配区域标签
        region_labels_z, _ = gen._assign_spots_to_regions(coords_z)
        
        # 生成表达 + LR 热点 + 批次效应
        expr_z = make_expr(coords_z, region_labels_z)
        lr_idx = [i for i, g in enumerate(gene_names) if g in all_lr]
        n_hot = max(int(N_SPOTS * 0.12), 5)
        hot_idx = rng.choice(N_SPOTS, n_hot, replace=False)
        for hi in hot_idx[:3]:
            dists = np.linalg.norm(coords_z - coords_z[hi:hi+1], axis=1)
            local = dists < np.percentile(dists, 15)
            for gi in lr_idx:
                expr_z[local, gi] *= rng.uniform(1.5, 2.0)
        # 批次效应
        expr_z *= rng.uniform(1 - cum_batch, 1 + cum_batch, N_SPOTS)[:, None]
        expr_z = expr_z.astype(np.float32)
        coords_prev = coords_z.copy()
    
    if z_idx == 0:
        region_labels_z, _ = gen._assign_spots_to_regions(coords_z)
    
    sl = {
        'coords': coords_z.astype(np.float64),
        'expr': expr_z.astype(np.float32),
        'gene_names': gene_names[:],
        'region_labels': region_labels_z if z_idx > 0 else region_labels_0,
        'n_spots': N_SPOTS,
        'n_genes': N_GENES,
    }
    slices.append(sl)
    region_labels_list.append(sl['region_labels'])
    print(f"  切片 {z_idx}: {N_SPOTS} spots, {N_GENES} genes")

# ========== 2. 顺序对齐 ==========
print("\n[2/4] 顺序对齐 (sequential pairwise alignment)...")
transport_plans = []
loss_histories = []
aligned_coords_list = [slices[0]['coords'].copy()]  # 第一张不动

for i in range(N_SLICES - 1):
    print(f"  对齐 切片 {i} ↔ 切片 {i+1}...")
    S_lr, top = compute_lr_strength_matrix(slices[i], slices[i+1], LR_DB)
    t0 = time.time()
    P, loss = fgw_lr_solver(
        slices[i], slices[i+1], S_lr, gamma=0.1,
        beta=0.3, alpha=0.5, max_iter=150, verbose=False
    )
    elapsed = time.time() - t0
    
    # 将对齐后的 i+1 坐标转换到第 0 张切片的参考系
    if i == 0:
        aligned = compute_alignment(P, slices[0]['coords'], slices[i+1]['coords'])
    else:
        # 累积：先用 P 对齐到第 i 张，再用之前的变换累积到第 0 张
        aligned_i = compute_alignment(P, aligned_coords_list[i], slices[i+1]['coords'])
        aligned = aligned_coords_list[i] + (aligned_i - aligned_coords_list[i])
    
    aligned_coords_list.append(aligned)
    transport_plans.append(P)
    loss_histories.append(loss)
    
    acc = compute_mapping_consistency(P, slices[i]['region_labels'],
                                       slices[i+1]['region_labels'])
    cost = compute_ot_cost(P, slices[i]['expr'], slices[i+1]['expr'])
    print(f"    精度={acc:.4f}, 成本={cost:.4f}, 时间={elapsed:.2f}s, "
          f"迭代={len(loss)}")

# ========== 3. 评估跨切片一致性 ==========
print("\n[3/4] 计算跨切片一致性...")
# 计算每对切片之间的区域映射一致性
pairwise_acc = np.zeros((N_SLICES, N_SLICES))
for i in range(N_SLICES):
    for j in range(N_SLICES):
        if i == j:
            pairwise_acc[i, j] = 1.0
        elif j > i:
            # 用传输计划计算 i→j
            S_lr_ij, _ = compute_lr_strength_matrix(slices[i], slices[j], LR_DB)
            P_ij, _ = fgw_lr_solver(
                slices[i], slices[j], S_lr_ij, gamma=0.1,
                max_iter=100, verbose=False
            )
            acc = compute_mapping_consistency(
                P_ij, slices[i]['region_labels'], slices[j]['region_labels'])
            pairwise_acc[i, j] = acc
            pairwise_acc[j, i] = acc

print(f"\n  跨切片映射精度矩阵 ({N_SLICES}x{N_SLICES}):")
print(f"  {''.join([f'{j:>8}' for j in range(N_SLICES)])}")
for i in range(N_SLICES):
    row = ''.join([f'{pairwise_acc[i,j]:>8.3f}' for j in range(N_SLICES)])
    print(f"  {i}{row}")

mean_acc = np.mean([pairwise_acc[i, i+1] for i in range(N_SLICES-1)])
print(f"\n  相邻切片平均对齐精度: {mean_acc:.4f}")

# ===== 保存 3D 重建结果 =====
with open(OUT_DIR / 'lrot_3d_reconstruction_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 72 + "\n")
    f.write("  LROT 3D Reconstruction Results (synthetic serial sections)\n")
    f.write("=" * 72 + "\n")
    f.write(f"Slices: {N_SLICES}, spots/slice: {N_SPOTS}, genes: {N_GENES}\n")
    f.write(f"Base rotation increment: {BASE_ROT} deg, batch increment: {BASE_BATCH}\n\n")
    f.write("  Adjacent pair accuracy:\n")
    for i in range(N_SLICES - 1):
        f.write(f"    slice {i}<->{i+1}: {pairwise_acc[i, i+1]:.4f}\n")
    f.write(f"\n  Mean adjacent accuracy: {mean_acc:.4f}\n")
    f.write("\n  Cross-slice consistency matrix (row=sliceA, col=sliceB):\n")
    for i in range(N_SLICES):
        f.write("    " + " ".join(f"{pairwise_acc[i, j]:.3f}" for j in range(N_SLICES)) + "\n")
print(f"  [SAVE] 结果已保存: {OUT_DIR / 'lrot_3d_reconstruction_results.txt'}")

# ========== 4. 3D 可视化 ==========
print("\n[4/4] 生成 3D 可视化...")

# 堆叠成 3D 坐标
all_coords_3d = []
all_labels = []
all_z = []
for z in range(N_SLICES):
    coords_3d = np.column_stack([
        aligned_coords_list[z],
        np.full(N_SPOTS, z * 0.5)  # z 方向间隔
    ])
    all_coords_3d.append(coords_3d)
    all_labels.append(slices[z]['region_labels'])
    all_z.append(np.full(N_SPOTS, z))

all_coords_3d = np.vstack(all_coords_3d)
all_labels = np.concatenate(all_labels)
region_colors = matplotlib.colormaps['tab10'](np.linspace(0, 1, gen.n_regions))

# ---- 图 A: 3D 堆叠散点图 ----
fig_a = plt.figure(figsize=(10, 8))
ax = fig_a.add_subplot(111, projection='3d')
for r in range(gen.n_regions):
    mask = all_labels == r
    ax.scatter(all_coords_3d[mask, 0], all_coords_3d[mask, 1],
               all_coords_3d[mask, 2], c=[region_colors[r]],
               s=4, alpha=0.6, label=f'Region {r}')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (slice)')
ax.set_title(f'A: LROT 3D reconstruction ({N_SLICES} slices)', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
fig_a.tight_layout()
fig_a.savefig(OUT_DIR / 'lrot_3d_reconstruction.png', dpi=300, bbox_inches='tight')
plt.close(fig_a)
print(f"  已保存: {OUT_DIR / 'lrot_3d_reconstruction.png'}")

# ---- 图 B: 热图矩阵 ----
fig_b, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(pairwise_acc, cmap='YlOrRd', vmin=0, vmax=1)
for i in range(N_SLICES):
    for j in range(N_SLICES):
        ax.text(j, i, f'{pairwise_acc[i,j]:.2f}', ha='center', va='center',
                fontsize=10, color='black' if pairwise_acc[i,j] > 0.5 else 'white')
ax.set_xticks(range(N_SLICES))
ax.set_yticks(range(N_SLICES))
ax.set_xlabel('Slice B')
ax.set_ylabel('Slice A')
ax.set_title('B: Cross-slice mapping consistency', fontsize=12)
plt.colorbar(im, ax=ax, shrink=0.8, label='Accuracy')
fig_b.tight_layout()
fig_b.savefig(OUT_DIR / 'lrot_cross_slice_consistency.png', dpi=300, bbox_inches='tight')
plt.close(fig_b)
print(f"  已保存: {OUT_DIR / 'lrot_cross_slice_consistency.png'}")

# ---- 图 C: 单基因 3D 表达分布 ----
# 选择一个 LR 基因展示
target_gene = 'Cdh2'
if target_gene in gene_names:
    gene_idx = gene_names.index(target_gene)
    fig_c = plt.figure(figsize=(10, 8))
    ax = fig_c.add_subplot(111, projection='3d')
    # 取所有切片中该基因的表达
    gene_expr_all = []
    for sl in slices:
        gene_expr_all.append(sl['expr'][:, gene_idx])
    gene_expr_all = np.concatenate(gene_expr_all)
    sc = ax.scatter(all_coords_3d[:, 0], all_coords_3d[:, 1],
                    all_coords_3d[:, 2], c=gene_expr_all,
                    cmap='viridis', s=5, alpha=0.7)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (slice)')
    ax.set_title(f'3D Expression Distribution: {target_gene}', fontsize=12)
    plt.colorbar(sc, ax=ax, shrink=0.6, label='Expression')
    fig_c.tight_layout()
    fig_c.savefig(OUT_DIR / 'lrot_3d_gene_expression.png', dpi=300, bbox_inches='tight')
    plt.close(fig_c)
    print(f"  已保存: {OUT_DIR / 'lrot_3d_gene_expression.png'}")

# ========== 汇总 ==========
print("\n" + "=" * 60)
print("  3D 重建完成")
print("=" * 60)
print(f"  切片数: {N_SLICES}")
print(f"  每切片 spots: {N_SPOTS}")
print(f"  总 spots: {N_SLICES * N_SPOTS}")
print(f"  平均对齐精度: {mean_acc:.4f}")
print(f"  输出文件:")
print(f"    1. lrot_3d_reconstruction.png — 3D 堆叠重建")
print(f"    2. lrot_cross_slice_consistency.png — 跨切片一致性矩阵")
print(f"    3. lrot_3d_gene_expression.png — 单基因 3D 表达分布")
print("=" * 60)

# 保存传输数据
from lrot_core import save_transport_data
# 分别保存每对切片的传输矩阵
for i, P in enumerate(transport_plans):
    np.save(str(OUT_DIR / f'lrot_data_3d_P_{i}_{i+1}.npy'), P)
print(f"  [SAVE] 传输矩阵已保存: {len(transport_plans)} 个 .npy 文件")
save_transport_data('3d_reconstruction', transport_plans[0],
                    extra={'n_pairs': len(transport_plans),
                           'mean_acc': mean_acc,
                           'N_SLICES': N_SLICES, 'N_SPOTS': N_SPOTS})
