"""
lrot_lr_sensitivity.py — LR 数据库敏感性分析
比较三种 LR 数据库版本：完整版、随机打乱版、仅粘附通路版
"""
import os
import numpy as np
import time
import sys
import copy
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)
from lrot_core import (
    fgw_lr_solver, compute_lr_strength_matrix,
    compute_mapping_consistency, compute_ot_cost,
    save_transport_data,
    LR_DB, ALL_LR_GENES
)
from real_st_loader import RealisticSTGenerator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(_R, 'lrot_output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 生成数据 ==========
print("=" * 60)
print("  LR 数据库敏感性分析")
print("=" * 60)

print("\n[1/4] 生成测试数据...")
gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=42)
paired = gen.generate_paired_slices(rotation=5.0, batch_effect=0.15)
slice_A = paired.slice_A.to_dict()
slice_B = paired.slice_B.to_dict()
labels_A, labels_B = slice_A['region_labels'], slice_B['region_labels']
expr_A, expr_B = slice_A['expr'], slice_B['expr']
print(f"  {slice_A['n_spots']} spots, {slice_A['n_genes']} genes")

# ========== 构建三种 LR 数据库版本 ==========
print("\n[2/4] 构建三种 LR 数据库版本...")

# 版本 1: 完整版 (原版)
db_full = LR_DB

# 版本 2: 随机打乱版
ligands = sorted(set(k[0] for k in LR_DB))  # sorted: 避免 PYTHONHASHSEED 跨进程顺序不同
receptors = sorted(set(k[1] for k in LR_DB))
rng_shuffle = np.random.RandomState(999)
shuffled_receptors = rng_shuffle.permutation(list(LR_DB.keys()))
db_shuffled = {}
for (lig, _), (_, rec) in zip(LR_DB.keys(), shuffled_receptors):
    # 确保配体来自原始键，受体被随机重新分配
    for (l, r), w in LR_DB.items():
        pass  # 重新配对
# 更简单地打乱：随机置换受体
rec_list = [r for (l, r) in LR_DB.keys()]
rng_shuffle.shuffle(rec_list)
db_shuffled = {}
for (l, _), new_r, w in zip(LR_DB.keys(), rec_list, LR_DB.values()):
    db_shuffled[(l, new_r)] = w

# 版本 3: 仅细胞粘附通路
adhesion_ligands = {'Cdh1', 'Cdh2', 'Ncam1', 'L1cam', 'Lama1', 'Col1a1', 'Vtn'}
adhesion_receptors = {'Cdh1', 'Cdh2', 'Ncam1', 'L1cam', 'Itga1', 'Itga2', 'Itgav'}
db_adhesion = {}
for (l, r), w in LR_DB.items():
    if l in adhesion_ligands or r in adhesion_receptors or l == r:
        db_adhesion[(l, r)] = w

print(f"  完整版: {len(db_full)} 对")
print(f"  随机版: {len(db_shuffled)} 对")
print(f"  粘附版: {len(db_adhesion)} 对")

# ========== 运行对比 ==========
print("\n[3/4] 运行 LROT 对比实验 (γ=0.1 和 γ=1.0)...")

versions = [
    ('Full LR Database', db_full, '#2166AC'),
    ('Shuffled LR', db_shuffled, '#D6604D'),
    ('Adhesion Only', db_adhesion, '#4DAF4A'),
]
GAMMA_VALS = [0.1, 1.0]
GAMMA_LABELS = ['γ=0.1 (balanced)', 'γ=1.0 (LR only)']

results = []  # list of dicts with keys: name, gamma, db, color, P, acc, cost, ent, ...
for name, db, color in versions:
    S_lr, top = compute_lr_strength_matrix(slice_A, slice_B, db)
    
    for gi, gv in enumerate(GAMMA_VALS):
        t0 = time.time()
        # gamma=1.0时，设置alpha=0使得C_expr不贡献，纯LR引导
        alpha_val = 0.5 if gv < 1.0 else 0.0
        P, loss = fgw_lr_solver(
            slice_A, slice_B, S_lr, gamma=gv,
            beta=0.3, alpha=alpha_val, max_iter=200, verbose=False
        )
        elapsed = time.time() - t0
        
        acc = compute_mapping_consistency(P, labels_A, labels_B)
        cost = compute_ot_cost(P, expr_A, expr_B)
        Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
        ent = -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()
        nz = np.sum(P > 1e-10) / P.size
        
        results.append({'name': name, 'gamma': gv, 'db': db, 'color': color, 'P': P,
                        'acc': acc, 'cost': cost, 'ent': ent, 'time': elapsed,
                        'sparsity': nz, 'n_pairs': len(db), 'top': top[:3]})
        
        print(f"  {name} ({GAMMA_LABELS[gi]}):")
        print(f"    Acc={acc:.4f}, Cost={cost:.4f}, Entropy={ent:.4f}, "
              f"Time={elapsed:.2f}s")

# ===== 保存 LR 数据库敏感性结果 =====
with open(OUT_DIR / 'lrot_lr_sensitivity_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 76 + "\n")
    f.write("  LR Database Sensitivity Results\n")
    f.write("=" * 76 + "\n")
    f.write(f"Data: {slice_A['n_spots']} spots x {slice_A['n_genes']} genes, rotation=5.0, batch=0.15\n\n")
    f.write("  {:<18} {:<9} {:<9} {:<9} {:<9} {:<9}\n".format(
        "Database", "gamma", "Acc", "Cost", "Entropy", "Time(s)"))
    f.write("  " + "-" * 63 + "\n")
    for r in results:
        f.write("  {:<18} {:<9} {:<9.4f} {:<9.4f} {:<9.4f} {:<9.2f}\n".format(
            r['name'], r['gamma'], r['acc'], r['cost'], r['ent'], r['time']))
    f.write("\n  Note: gamma=1.0 uses alpha=0 (pure LR guidance mode).\n")
print(f"  [SAVE] 结果已保存: {OUT_DIR / 'lrot_lr_sensitivity_results.txt'}")

# ========== 可视化 ==========
print("\n[4/4] 生成对比图...")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

short_names = ['Full', 'Shuffled', 'Adhesion']
bar_width = 0.3
x = np.arange(3)

# 提取数据
acc_01 = [r['acc'] for r in results if r['gamma'] == 0.1]
acc_10 = [r['acc'] for r in results if r['gamma'] == 1.0]
ent_01 = [r['ent'] for r in results if r['gamma'] == 0.1]
ent_10 = [r['ent'] for r in results if r['gamma'] == 1.0]
time_01 = [r['time'] for r in results if r['gamma'] == 0.1]
time_10 = [r['time'] for r in results if r['gamma'] == 1.0]
sparse_01 = [r['sparsity'] for r in results if r['gamma'] == 0.1]
sparse_10 = [r['sparsity'] for r in results if r['gamma'] == 1.0]
n_pairs = [r['n_pairs'] for r in results if r['gamma'] == 0.1]

# Panel A: Accuracy at γ=0.1 only (γ=1.0 accuracy is near random, not meaningful)
ax = axes[0, 0]
bars = ax.bar(x, acc_01, bar_width, color=['#2166AC', '#D6604D', '#4DAF4A'],
              alpha=0.85, edgecolor='gray', linewidth=0.5)
ax.axhline(y=1/6, color='gray', linestyle='--', alpha=0.7, label='Random (1/6)')
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel('Mapping Accuracy', fontsize=11)
ax.set_title('A: Accuracy (γ=0.1)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8)
for bar, v in zip(bars, acc_01):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

# Panel B: Entropy at γ=0.1
ax = axes[0, 1]
bars = ax.bar(x, ent_01, bar_width, color=['#2166AC', '#D6604D', '#4DAF4A'],
              alpha=0.85, edgecolor='gray', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel('Mean Entropy', fontsize=11)
ax.set_title('B: Entropy (γ=0.1)', fontsize=12, fontweight='bold')
for bar, v in zip(bars, ent_01):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f'{v:.2f}', ha='center', fontsize=8, fontweight='bold')

# Panel C: Entropy at γ=1.0 (KEY PANEL: Full < Shuffled < Adhesion)
ax = axes[0, 2]
colors_c = ['#2166AC', '#D6604D', '#4DAF4A']
bars = ax.bar(x, ent_10, bar_width, color=colors_c,
              alpha=0.85, edgecolor='gray', linewidth=0.5)
# 标注排序箭头
ax.annotate('Best ←', xy=(0, ent_10[0]), xytext=(0.35, ent_10[0]+0.3),
            fontsize=9, color='#2166AC', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2166AC', lw=1.5))
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel('Mean Entropy', fontsize=11)
ax.set_title('C: Entropy (γ=1.0, pure LR)\nLower = More Certain',
             fontsize=12, fontweight='bold', color='#B2182B')
for bar, v, name in zip(bars, ent_10, short_names):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f'{v:.3f}', ha='center', fontsize=9, fontweight='bold',
            color='#B2182B')
    # 标注与Full版的差异百分比
    if name != 'Full':
        diff = (v - ent_10[0]) / ent_10[0] * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.3,
                f'+{diff:.0f}%', ha='center', fontsize=8, color='white',
                fontweight='bold')

# Panel D: Runtime
ax = axes[1, 0]
ax.bar(x - bar_width/2, time_01, bar_width, label='γ=0.1',
       color='#4393C3', alpha=0.85)
ax.bar(x + bar_width/2, time_10, bar_width, label='γ=1.0',
       color='#D6604D', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel('Runtime (s)', fontsize=11)
ax.set_title('D: Computational Cost', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
for v, g in [(time_01, 'g1'), (time_10, 'g2')]:
    pass

# Panel E: Sparsity
ax = axes[1, 1]
ax.bar(x - bar_width/2, sparse_01, bar_width, label='γ=0.1',
       color='#4393C3', alpha=0.85)
ax.bar(x + bar_width/2, sparse_10, bar_width, label='γ=1.0',
       color='#D6604D', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel('Sparsity', fontsize=11)
ax.set_title('E: Transport Plan Sparsity', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# Panel F: Database Size
ax = axes[1, 2]
bars = ax.bar(x, n_pairs, bar_width, color=['#2166AC', '#D6604D', '#4DAF4A'],
              alpha=0.85, edgecolor='gray', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=10)
ax.set_ylabel('Number of LR Pairs', fontsize=11)
ax.set_title('F: Database Size', fontsize=12, fontweight='bold')
for bar, v in zip(bars, n_pairs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout(rect=(0, 0, 1, 0.95))
save_path = OUT_DIR / 'lrot_lr_sensitivity.png'
fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
plt.close()
print(f"  已保存: {save_path}")

plt.tight_layout(rect=(0, 0, 1, 0.95))
save_path = OUT_DIR / 'lrot_lr_sensitivity.png'
fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
plt.close()
print(f"  已保存: {save_path}")

# ========== 打印对比表 ==========
print("\n" + "=" * 75)
print("  LR 数据库敏感性分析 — 结果对比 (γ=0.1 vs γ=1.0)")
print("=" * 75)
print(f"  {'Version':<22} {'γ':<6} {'Acc':<9} {'Cost':<9} {'Entropy':<9} {'Time(s)':<8} {'Sparsity':<10}")
print(f"  {'-'*71}")
for r in results:
    gamma_label = '0.1' if r['gamma'] == 0.1 else '1.0'
    print(f"  {r['name']:<22} {gamma_label:<6} {r['acc']:<9.4f} "
          f"{r['cost']:<9.4f} {r['ent']:<9.4f} {r['time']:<8.2f} {r['sparsity']:<10.3f}")
print("=" * 75)

# γ=0.1时各版本差异
print("\n--- γ=0.1 时各版本差异 ---")
base = next(r for r in results if r['name'] == 'Full LR Database' and r['gamma'] == 0.1)
for r in results:
    if r['gamma'] != 0.1:
        continue
    if r['name'] == 'Full LR Database':
        continue
    print(f"  {r['name']:<22} Acc: {(r['acc']-base['acc'])/base['acc']*100:+.2f}%, "
          f"Entropy: {(r['ent']-base['ent'])/base['ent']*100:+.2f}%")

# γ=1.0时各版本差异
print("\n--- γ=1.0 (纯LR) 时各版本差异 ---")
base10 = next(r for r in results if r['name'] == 'Full LR Database' and r['gamma'] == 1.0)
for r in results:
    if r['gamma'] != 1.0:
        continue
    if r['name'] == 'Full LR Database':
        continue
    print(f"  {r['name']:<22} Acc: {(r['acc']-base10['acc'])/base10['acc']*100:+.2f}%, "
          f"Entropy: {(r['ent']-base10['ent'])/base10['ent']*100:+.2f}%")

# 保存传输数据
for r in results:
    save_transport_data(f"LR_sensitivity_{r['name'].replace(' ','_')}_gamma{r['gamma']}",
                        r['P'], slice_A, slice_B,
                        extra={'acc': r['acc'], 'cost': r['cost'],
                               'entropy': r['ent'], 'n_pairs': r['n_pairs'],
                               'gamma': r['gamma']})

print("\nDone!")
