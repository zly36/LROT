"""
lr_parameter_sensitivity.py — LROT 参数敏感性分析（α, β）
固定 γ=0.1, 在真实感模拟数据(seed=42, 1000 spots)上联合扫描 α(表达/几何权衡) 与 β(GW正则化)。
评估 LROT 与 FGW(γ=0) 在精度、熵、成本上对参数的稳健性。
输出: lrot_alpha_beta_sensitivity.png + lrot_alpha_beta_results.txt
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

from lrot_core import fgw_lr_solver, compute_lr_strength_matrix, LR_DB
from real_st_loader import RealisticSTGenerator

OUT_DIR = os.path.join(_R, 'lrot_output')

# 生成数据（与主实验一致, sorted50）
gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=42)
paired = gen.generate_paired_slices(rotation=5.0, batch_effect=0.15)
slice_A = paired.slice_A.to_dict()
slice_B = paired.slice_B.to_dict()
S_lr, _ = compute_lr_strength_matrix(slice_A, slice_B, LR_DB)

def evaluate(P, slice_A, slice_B):
    from lrot_core import compute_mapping_consistency
    acc = compute_mapping_consistency(P, slice_A['region_labels'], slice_B['region_labels'])
    P_norm = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    entropy = -np.sum(P_norm * np.log(np.maximum(P_norm, 1e-10)), axis=1).mean()
    from lrot_core import compute_ot_cost
    cost = compute_ot_cost(P, slice_A['expr'], slice_B['expr'])
    return acc, entropy, cost

ALPHAS = [0.2, 0.35, 0.5, 0.65, 0.8]
BETAS = [0.1, 0.3, 0.5]
GAMMA = 0.1

print("=" * 78)
print("  LROT Parameter Sensitivity (α × β), γ=0.1, 1000 spots")
print("=" * 78)
print(f"  {'α':<6}{'β':<6}{'Acc':<8}{'Entropy':<10}{'Cost':<9}{'Time(s)':<8}")
print("  " + "-" * 56)

results = []  # (alpha, beta, acc_lrot, ent_lrot, acc_fgw, ent_fgw)
for alpha in ALPHAS:
    for beta in BETAS:
        # LROT (γ=0.1)
        t0 = time.time()
        P_lrot, _ = fgw_lr_solver(slice_A, slice_B, S_lr, gamma=GAMMA, beta=beta, alpha=alpha,
                                  max_iter=150, verbose=False)
        acc_l, ent_l, cost_l = evaluate(P_lrot, slice_A, slice_B)
        t_el = time.time() - t0
        # FGW (γ=0)
        S0 = np.zeros_like(S_lr)
        P_fgw, _ = fgw_lr_solver(slice_A, slice_B, S0, gamma=0.0, beta=beta, alpha=alpha,
                                 max_iter=150, verbose=False)
        acc_f, ent_f, _ = evaluate(P_fgw, slice_A, slice_B)
        results.append((alpha, beta, acc_l, ent_l, cost_l, acc_f, ent_f))
        print(f"  {alpha:<6.2f}{beta:<6.2f}{acc_l:<8.4f}{ent_l:<10.4f}{cost_l:<9.4f}{t_el:<8.2f}")

print("  " + "-" * 56)

# ---- 统计稳健性 ----
ent_lrot_arr = np.array([r[3] for r in results]).reshape(len(ALPHAS), len(BETAS))
ent_fgw_arr = np.array([r[6] for r in results]).reshape(len(ALPHAS), len(BETAS))
acc_lrot_arr = np.array([r[2] for r in results]).reshape(len(ALPHAS), len(BETAS))

print(f"\n  LROT 熵范围: {ent_lrot_arr.min():.4f} - {ent_lrot_arr.max():.4f} "
      f"(CV={ent_lrot_arr.std()/ent_lrot_arr.mean()*100:.1f}%)")
print(f"  LROT 精度范围: {acc_lrot_arr.min():.4f} - {acc_lrot_arr.max():.4f}")
print(f"  熵降范围 (vs FGW): "
      f"{(ent_fgw_arr-ent_lrot_arr).min():.4f} - {(ent_fgw_arr-ent_lrot_arr).max():.4f}")

# ---- 保存文本 ----
result_path = f"{OUT_DIR}/lrot_alpha_beta_results.txt"
with open(result_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("  LROT Parameter Sensitivity (α × β), γ=0.1, 1000 spots\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  {'α':<6}{'β':<6}{'LROT_Acc':<10}{'LROT_Ent':<10}{'FGW_Acc':<10}{'FGW_Ent':<10}\n")
    f.write("  " + "-" * 52 + "\n")
    for alpha, beta, acc_l, ent_l, cost_l, acc_f, ent_f in results:
        f.write(f"  {alpha:<6.2f}{beta:<6.2f}{acc_l:<10.4f}{ent_l:<10.4f}{acc_f:<10.4f}{ent_f:<10.4f}\n")
    f.write("\n")
    f.write(f"LROT 熵范围: {ent_lrot_arr.min():.4f} - {ent_lrot_arr.max():.4f} "
            f"(CV={ent_lrot_arr.std()/ent_lrot_arr.mean()*100:.1f}%)\n")
    f.write(f"LROT 精度范围: {acc_lrot_arr.min():.4f} - {acc_lrot_arr.max():.4f}\n")
    f.write(f"熵降范围 (vs FGW): {(ent_fgw_arr-ent_lrot_arr).min():.4f} - "
            f"{(ent_fgw_arr-ent_lrot_arr).max():.4f}\n")
print(f"\n  结果已保存: {result_path}")

# ---- 绘图: 双面板热图 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# 左: 熵热图
im1 = axes[0].imshow(ent_lrot_arr, cmap='YlGnBu_r', aspect='auto')
axes[0].set_xticks(range(len(BETAS))); axes[0].set_xticklabels([f'β={b}' for b in BETAS])
axes[0].set_yticks(range(len(ALPHAS))); axes[0].set_yticklabels([f'α={a}' for a in ALPHAS])
axes[0].set_title('(A) LROT entropy (lower = better)', fontsize=12)
for i in range(len(ALPHAS)):
    for j in range(len(BETAS)):
        axes[0].text(j, i, f'{ent_lrot_arr[i,j]:.3f}', ha='center', va='center', fontsize=9)
fig.colorbar(im1, ax=axes[0], fraction=0.046)

# 右: 精度热图
im2 = axes[1].imshow(acc_lrot_arr, cmap='YlGn', aspect='auto')
axes[1].set_xticks(range(len(BETAS))); axes[1].set_xticklabels([f'β={b}' for b in BETAS])
axes[1].set_yticks(range(len(ALPHAS))); axes[1].set_yticklabels([f'α={a}' for a in ALPHAS])
axes[1].set_title('(B) LROT accuracy', fontsize=12)
for i in range(len(ALPHAS)):
    for j in range(len(BETAS)):
        axes[1].text(j, i, f'{acc_lrot_arr[i,j]:.3f}', ha='center', va='center', fontsize=9)
fig.colorbar(im2, ax=axes[1], fraction=0.046)

fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f"{OUT_DIR}/lrot_alpha_beta_sensitivity.png", dpi=300)
plt.close(fig)
print(f"  图已保存: {OUT_DIR}/lrot_alpha_beta_sensitivity.png")
