# lrot_real_data.py — 在真实感数据上运行 LROT 对齐
# 包含：数据生成、LROT对齐、消融实验、可视化

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

import sys
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)

from real_st_loader import RealisticSTGenerator, LR_DB, lr_gene_list, N_REGIONS, REGION_NAMES
from lrot_core import (
    compute_lr_strength_matrix, fgw_lr_solver,
    compute_alignment, compute_mapping_consistency, compute_ot_cost,
    save_transport_data,
    LIGANDS, RECEPTORS, ALL_LR_GENES

)

# ========== 输出目录 ==========
OUT_DIR = Path(_R, 'lrot_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 参数设置 ==========
N_SPOTS = 1000
N_GENES = 150
SEED = 42
ROTATION = 5.0
SHEAR = 0.02
SCALE = 0.95
BATCH_EFFECT = 0.15

# LROT 参数
GAMMA_LROT = 0.1      # LR guidance strength
GAMMA_BASELINE = 0.0  # baseline FGW (no LR guidance, gamma=0)

print("=" * 60)
print("  LROT 真实数据对齐实验")
print("=" * 60)

# ========== Step 1: 生成数据 ==========
print("\n[1/5] 生成真实感配对切片数据...")
gen = RealisticSTGenerator(
    n_spots_A=N_SPOTS, n_spots_B=N_SPOTS,
    n_genes=N_GENES, seed=SEED
)
paired = gen.generate_paired_slices(
    rotation=ROTATION, shear=SHEAR,
    scale=SCALE, batch_effect=BATCH_EFFECT
)
gen.print_info(paired)

# 转换为字典格式（兼容lrot_core接口）
slice_A = paired.slice_A.to_dict()
slice_B = paired.slice_B.to_dict()

# ========== Step 2: 计算LR强度矩阵 ==========
print("\n[2/5] 计算配体-受体信号强度矩阵...")
t0 = time.time()
S_lr, top_lr_pairs = compute_lr_strength_matrix(slice_A, slice_B, LR_DB, scale=10.0)
t_lr = time.time() - t0
print(f"  LR矩阵形状: {S_lr.shape}")
print(f"  LR矩阵范围: [{S_lr.min():.4f}, {S_lr.max():.4f}]")
print(f"  耗时: {t_lr:.2f}s")

# ========== Step 3: 运行 LROT ==========
print("\n[3/5] 运行 LROT 对齐 (LR-guided FGW)...")
t0 = time.time()
P_lrot, log_lrot = fgw_lr_solver(
    slice_A, slice_B, S_lr,
    gamma=GAMMA_LROT, beta=0.3,
    reg_gw=0.01, reg_ot=0.01, alpha=0.5,
    max_iter=200, lr_step=50, tol=1e-5, verbose=True
)
t_lrot = time.time() - t0
print(f"  LROT 传输计划形状: {P_lrot.shape}")
print(f"  耗时: {t_lrot:.2f}s")

# ========== Step 4: 运行 Baseline FGW ==========
print("\n[4/5] 运行 Baseline FGW (gamma=0)...")
t0 = time.time()
P_fgw, log_fgw = fgw_lr_solver(
    slice_A, slice_B, S_lr,
    gamma=GAMMA_BASELINE, beta=0.3,
    reg_gw=0.01, reg_ot=0.01, alpha=0.5,
    max_iter=200, lr_step=50, tol=1e-5, verbose=False
)
t_fgw = time.time() - t0
print(f"  FGW 传输计划形状: {P_fgw.shape}")
print(f"  耗时: {t_fgw:.2f}s")

# ========== 评估指标 ==========
print("\n[5/5] 计算评估指标...")

# 获取坐标和标签
coords_A = slice_A['coords']
coords_B = slice_B['coords']
labels_A = slice_A['region_labels']
labels_B = slice_B['region_labels']
expr_A = slice_A['expr']
expr_B = slice_B['expr']

# LROT 指标
coords_aligned_lrot = compute_alignment(P_lrot, coords_A, coords_B)
acc_lrot = compute_mapping_consistency(P_lrot, labels_A, labels_B)
cost_lrot = compute_ot_cost(P_lrot, expr_A, expr_B)
# 传输计划熵（度量不确定性）
P_lrot_norm = P_lrot / np.maximum(P_lrot.sum(1, keepdims=True), 1e-10)
entropy_lrot = -np.sum(P_lrot_norm * np.log(np.maximum(P_lrot_norm, 1e-10)), axis=1)

# FGW 指标
coords_aligned_fgw = compute_alignment(P_fgw, coords_A, coords_B)
acc_fgw = compute_mapping_consistency(P_fgw, labels_A, labels_B)
cost_fgw = compute_ot_cost(P_fgw, expr_A, expr_B)
P_fgw_norm = P_fgw / np.maximum(P_fgw.sum(1, keepdims=True), 1e-10)
entropy_fgw = -np.sum(P_fgw_norm * np.log(np.maximum(P_fgw_norm, 1e-10)), axis=1)

print("\n" + "=" * 60)
print("  评估结果对比")
print("=" * 60)
print(f"  {'指标':<25} {'LROT (γ=0.1)':<15} {'FGW (γ=0)':<15}")
print(f"  {'-'*55}")
print(f"  {'映射精度 (Mapping Acc)':<25} {acc_lrot:<15.4f} {acc_fgw:<15.4f}")
print(f"  {'OT 成本 (OT Cost)':<25} {cost_lrot:<15.4f} {cost_fgw:<15.4f}")
print(f"  {'传输熵均值 (Mean Entropy)':<25} {entropy_lrot.mean():<15.4f} {entropy_fgw.mean():<15.4f}")
print(f"  {'运行时间 (s)':<25} {t_lrot:<15.2f} {t_fgw:<15.2f}")
print("=" * 60)

# ========== 可视化 ==========
print("\n生成可视化...")

# --- 颜色映射 ---
region_colors = matplotlib.colormaps['tab10'](np.linspace(0, 1, N_REGIONS))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 面板1: 对齐前（Slice A + B 按区域着色）
ax = axes[0, 0]
for r in range(N_REGIONS):
    mask = labels_A == r
    ax.scatter(coords_A[mask, 0], coords_A[mask, 1],
               c=[region_colors[r]], s=8, alpha=0.7,
               label=REGION_NAMES[r] if r < len(REGION_NAMES) else f'Region{r}')
for r in range(N_REGIONS):
    mask = labels_B == r
    ax.scatter(coords_B[mask, 0], coords_B[mask, 1],
               c=[region_colors[r]], s=4, alpha=0.3, marker='x')
ax.set_title('A: Before alignment\n(dots = slice A, crosses = slice B)', fontsize=12)
ax.set_aspect('equal')
ax.legend(fontsize=6, loc='upper right', ncol=2)

# 面板2: LROT 对齐后
ax = axes[0, 1]
for r in range(N_REGIONS):
    mask = labels_A == r
    ax.scatter(coords_aligned_lrot[mask, 0], coords_aligned_lrot[mask, 1],
               c=[region_colors[r]], s=8, alpha=0.7,
               label=REGION_NAMES[r] if r < len(REGION_NAMES) else f'Region{r}')
for r in range(N_REGIONS):
    mask = labels_B == r
    ax.scatter(coords_B[mask, 0], coords_B[mask, 1],
               c=[region_colors[r]], s=4, alpha=0.3, marker='x')
ax.set_title(f'B: After LROT (γ={GAMMA_LROT})\nMapping Acc={acc_lrot:.3f}',
             fontsize=12)
ax.set_aspect('equal')
ax.legend(fontsize=6, loc='upper right', ncol=2)

# 面板3: 传输计划热图
ax = axes[0, 2]
n_vis = min(50, P_lrot.shape[0], P_lrot.shape[1])
P_vis = P_lrot[:n_vis, :n_vis]
im = ax.imshow(P_vis, cmap='viridis', aspect='auto',
               norm=LogNorm(vmin=max(P_vis.min(), 1e-10), vmax=P_vis.max()))
ax.set_title(f'C: Transport plan ({n_vis}×{n_vis})', fontsize=12)
ax.set_xlabel('Slice B spots')
ax.set_ylabel('Slice A spots')
plt.colorbar(im, ax=ax, shrink=0.8)

# 面板4: LR信号强度图
ax = axes[1, 0]
# 找LR信号最强的top对
lr_mean = S_lr.mean(axis=1)
top_lr_idx = np.argsort(lr_mean)[-30:]
sc = ax.scatter(coords_A[top_lr_idx, 0], coords_A[top_lr_idx, 1],
                c=lr_mean[top_lr_idx], s=40, cmap='hot',
                norm=Normalize(vmin=lr_mean[top_lr_idx].min(),
                               vmax=lr_mean[top_lr_idx].max()))
# 显示全部位置作为背景
ax.scatter(coords_A[:, 0], coords_A[:, 1], c='gray', s=2, alpha=0.3)
ax.set_title('D: Top-30 LR signalling hotspots\n(bright = stronger signal)', fontsize=12)
ax.set_aspect('equal')
plt.colorbar(sc, ax=ax, shrink=0.8, label='LR Strength')

# 面板5: 传输计划熵对比
ax = axes[1, 1]
bins = np.linspace(0, max(entropy_lrot.max(), entropy_fgw.max()) + 0.5, 30)
ax.hist(entropy_lrot, bins=bins, alpha=0.7, color='orange',
        label=f'LROT (γ={GAMMA_LROT})')
ax.hist(entropy_fgw, bins=bins, alpha=0.5, color='steelblue',
        label=f'FGW (γ={GAMMA_BASELINE})')
ax.axvline(entropy_lrot.mean(), color='orange', linestyle='--', alpha=0.8)
ax.axvline(entropy_fgw.mean(), color='steelblue', linestyle='--', alpha=0.8)
ax.set_xlabel('Transport Entropy')
ax.set_ylabel('Count')
ax.set_title('E: Transport entropy distribution\n(lower = more certain)', fontsize=12)
ax.legend(fontsize=10)

# 面板6: 3D堆叠重建
fig.delaxes(axes[1, 2])
ax = fig.add_subplot(2, 3, 6, projection='3d')
# 绘制A在z=0，对齐后的A在z=1
for r in range(N_REGIONS):
    mask = labels_A == r
    z0 = np.zeros(mask.sum())
    ax.scatter(coords_A[mask, 0], coords_A[mask, 1], z0,
               color=region_colors[r][:3], s=6, alpha=0.6)
    z1 = np.ones(mask.sum())
    ax.scatter(coords_aligned_lrot[mask, 0], coords_aligned_lrot[mask, 1], z1,
               color=region_colors[r][:3], s=6, alpha=0.6)
# 连接线（显示对应关系）
n_lines = min(30, labels_A.shape[0])
line_idx = np.random.RandomState(SEED).choice(
    labels_A.shape[0], n_lines, replace=False)
for idx_val in line_idx:
    ax.plot([coords_A[idx_val, 0], coords_aligned_lrot[idx_val, 0]],
            [coords_A[idx_val, 1], coords_aligned_lrot[idx_val, 1]],
            [0, 1], 'gray', alpha=0.15, linewidth=0.5)
ax.set_title('F: 3D stacked reconstruction\n(z=0 raw, z=1 aligned)', fontsize=12)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fig.tight_layout(rect=(0, 0, 1, 0.94))
panel_path = OUT_DIR / 'lrot_real_panel.png'
fig.savefig(panel_path, dpi=300)
plt.close()
print(f"  面板图已保存: {panel_path}")

# ========== 消融实验 ==========
print("\n运行消融实验 (gamma扫描)...")
gammas = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
ablation_results = []

for gamma in gammas:
    t0 = time.time()
    P, log_info = fgw_lr_solver(
        slice_A, slice_B, S_lr,
        gamma=gamma, beta=0.3,
        reg_gw=0.01, reg_ot=0.01, alpha=0.5,
        max_iter=200, lr_step=50, tol=1e-5, verbose=False
    )
    elapsed = time.time() - t0
    acc = compute_mapping_consistency(P, labels_A, labels_B)
    cost = compute_ot_cost(P, expr_A, expr_B)
    P_norm = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    ent = -np.sum(P_norm * np.log(np.maximum(P_norm, 1e-10)), axis=1)
    mean_ent = ent.mean()
    nz = np.sum(P > 1e-10) / P.size  # 非零比例
    ablation_results.append((gamma, acc, cost, mean_ent, elapsed, nz))
    print(f"  gamma={gamma:.2f} | Acc={acc:.4f} | Cost={cost:.4f} | "
          f"Entropy={mean_ent:.4f} | Time={elapsed:.2f}s | "
          f"Sparsity={nz:.4f}")

# 绘制消融图
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 7))

# 映射精度
ax = axes2[0, 0]
acc_vals = [r[1] for r in ablation_results]
bars = ax.bar(range(len(gammas)), acc_vals, color='steelblue', alpha=0.8)
best_idx = np.argmax(acc_vals)
bars[best_idx].set_color('orange')
for i, (g, v) in enumerate(zip(gammas, acc_vals)):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)
ax.set_xticks(range(len(gammas)))
ax.set_xticklabels([f'γ={g}' for g in gammas])
ax.set_ylabel('Mapping Accuracy')
ax.set_title('A: Mapping accuracy vs γ')
ax.set_ylim(0, 1.0)

# OT成本
ax = axes2[0, 1]
cost_vals = [r[2] for r in ablation_results]
ax.plot(range(len(gammas)), cost_vals, 'o-', color='crimson', linewidth=2)
for i, (g, v) in enumerate(zip(gammas, cost_vals)):
    ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)
ax.set_xticks(range(len(gammas)))
ax.set_xticklabels([f'γ={g}' for g in gammas])
ax.set_ylabel('OT Cost')
ax.set_title('B: Transport cost vs γ')

# 传输熵
ax = axes2[1, 0]
ent_vals = [r[3] for r in ablation_results]
ax.plot(range(len(gammas)), ent_vals, 's-', color='green', linewidth=2)
for i, (g, v) in enumerate(zip(gammas, ent_vals)):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
ax.set_xticks(range(len(gammas)))
ax.set_xticklabels([f'γ={g}' for g in gammas])
ax.set_ylabel('Mean Entropy')
ax.set_title('C: Alignment certainty vs γ\n(lower = more certain)')

# 运行时间
ax = axes2[1, 1]
time_vals = [r[4] for r in ablation_results]
ax.bar(range(len(gammas)), time_vals, color='purple', alpha=0.7)
for i, (g, v) in enumerate(zip(gammas, time_vals)):
    ax.text(i, v + 0.1, f'{v:.1f}s', ha='center', fontsize=9)
ax.set_xticks(range(len(gammas)))
ax.set_xticklabels([f'γ={g}' for g in gammas])
ax.set_ylabel('Runtime (s)')
ax.set_title('D: Runtime vs γ')

fig2.tight_layout(rect=(0, 0, 1, 0.92))
abl_path = OUT_DIR / 'lrot_real_ablation.png'
fig2.savefig(abl_path, dpi=380, bbox_inches='tight')
plt.close()
print(f"  消融图已保存: {abl_path}")

# ========== 保存结果摘要 ==========
summary_path = OUT_DIR / 'lrot_real_results.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("LROT Real Data Experiment Results\n")
    f.write("=" * 60 + "\n")
    f.write(f"Data: {N_SPOTS} spots x {N_GENES} genes\n")
    f.write(f"Transform: rot={ROTATION} deg, shear={SHEAR}, "
            f"scale={SCALE}, batch={BATCH_EFFECT}\n\n")
    f.write(f"{'Metric':<25} {'LROT':<15} {'FGW':<15}\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'Mapping Acc':<25} {acc_lrot:<15.4f} {acc_fgw:<15.4f}\n")
    f.write(f"{'OT Cost':<25} {cost_lrot:<15.4f} {cost_fgw:<15.4f}\n")
    f.write(f"{'Mean Entropy':<25} {entropy_lrot.mean():<15.4f} "
            f"{entropy_fgw.mean():<15.4f}\n")
    f.write(f"{'Runtime (s)':<25} {t_lrot:<15.2f} {t_fgw:<15.2f}\n\n")
    f.write("Ablation Results\n")
    f.write("-" * 55 + "\n")
    f.write(f"{'gamma':<10} {'Acc':<10} {'Cost':<10} {'Entropy':<10} "
            f"{'Time(s)':<10} {'Sparsity':<10}\n")
    for gamma, acc, cost, ent, elapsed, nz in ablation_results:
        f.write(f"{gamma:<10.2f} {acc:<10.4f} {cost:<10.2f} {ent:<10.4f} "
                f"{elapsed:<10.2f} {nz:<10.4f}\n")
    f.write("=" * 60 + "\n")

print(f"\n结果摘要已保存: {summary_path}")
print(f"\n所有输出文件位置: {OUT_DIR}")
print("  1. lrot_real_panel.png — 6面板可视化")
print("  2. lrot_real_ablation.png — 消融实验结果")
print("  3. lrot_real_results.txt — 数值结果")
print("=" * 60)

# 保存传输数据
coords_aligned = compute_alignment(P_lrot, slice_A['coords'], slice_B['coords'])
save_transport_data('realistic', P_lrot, slice_A, slice_B,
                    loss_hist=log_lrot, aligned_coords=coords_aligned)
