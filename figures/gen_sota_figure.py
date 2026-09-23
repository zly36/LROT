"""
gen_sota_figure.py — 生成 SOTA 三层对比实验图
Expr OT only → PASTE (FGW) → LROT 的精度与熵对比
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


sys.path.insert(0, _R)

OUT_DIR = os.path.join(_R, 'lrot_output')

# ========== 数据 ==========
# 数据来源: 基线实测记录 (真实感模拟, 1000 spots) + run_official_paste2.py
# 精度图含 STAligner(深度); 熵图不含(STAligner熵口径为潜空间最近邻, 与传输熵不可比)
methods_acc = ['Expr OT\nonly', 'PASTE\n(FGW, γ=0)', 'LROT\n(γ=0.1)', 'LROT\n(γ=0.2)', 'PASTE2\n(α=0.1)', 'STAligner\n(deep)']
acc = [0.708, 0.711, 0.712, 0.709, 0.673, 0.799]
acc_colors = ['#AAAAAA', '#E68A8A', '#7FBF7F', '#2E8B57', '#B08BC9', '#D67236']
methods_ent = ['Expr OT\nonly', 'PASTE\n(FGW, γ=0)', 'LROT\n(γ=0.1)', 'LROT\n(γ=0.2)']
ent = [1.752, 4.035, 3.928, 3.821]
colors = ['#AAAAAA', '#E68A8A', '#7FBF7F', '#2E8B57']

# ========== 合成数据补充 ==========
# 合成数据 (lrot_prototype, 1000 spots)
syn_methods = ['FGW\n(γ=0)', 'LROT\n(γ=0.1)']
syn_acc = [0.363, 0.631]
syn_colors = ['#E68A8A', '#2E8B57']

# ========== 绘图 ==========
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

# ——— Panel A: Realistic simulated data — Accuracy ———
ax = axes[0]
bars = ax.bar(methods_acc, acc, color=acc_colors, edgecolor='gray',
              linewidth=0.8, width=0.6)
ax.set_ylabel('Mapping Accuracy', fontsize=12)
ax.set_title('A: Accuracy (Realistic Simulated)', fontsize=12, fontweight='bold')
ax.set_ylim(0.60, 0.85)
# Random基线(1/6≈0.167)远低于可见范围，不显示

# 标注数值
for bar, v in zip(bars, acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# ——— Panel B: Realistic simulated data — Entropy ———
ax = axes[1]
bars = ax.bar(methods_ent, ent, color=colors, edgecolor='gray',
              linewidth=0.8, width=0.6)
ax.set_ylabel('Mean Entropy ↓ (Lower = Better)', fontsize=12)
ax.set_title('B: Alignment Entropy (Realistic Simulated)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 5.0)

# 标注数值和箭头
for bar, v in zip(bars, ent):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

# 标注关键发现
# 红箭头：Expr OT(1.752) → PASTE(4.035) 熵上升，箭头朝上
ax.annotate('', xy=(0.3, 4.035), xytext=(0.3, 1.752),
            arrowprops=dict(arrowstyle='->', color='#CC3333', lw=2.5))
ax.text(0.3, 2.7, 'Entropy ↑130%', fontsize=9, color='#CC3333',
        fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 绿箭头：PASTE(4.035) → LROT(3.821) 熵下降，箭头朝下
ax.annotate('', xy=(2.3, 3.821), xytext=(2.3, 4.035),
            arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=2.5))
ax.text(2.3, 4.3, 'Entropy ↓5%', fontsize=9, color='#2E8B57',
        fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ——— Panel C: Synthetic data ———
ax = axes[2]
bars = ax.bar(syn_methods, syn_acc, color=syn_colors, edgecolor='gray',
              linewidth=0.8, width=0.5)
ax.set_ylabel('Mapping Accuracy', fontsize=12)
ax.set_title('C: Accuracy (Synthetic Data)', fontsize=12, fontweight='bold')
ax.set_ylim(0, 0.8)

for bar, v in zip(bars, syn_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

# 标注提升
ax.annotate('', xy=(0, 0.363), xytext=(1, 0.631),
            arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=2))
ax.text(0.5, 0.45, '+74%', ha='center', fontsize=13,
        color='#2E8B57', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=(0, 0, 1, 0.93))
save_path = f'{OUT_DIR}/lrot_sota_comparison.png'
fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {save_path}")

# ========== 也生成一个包含误差的详细版本 ==========
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# 左: 精度对比
ax = axes2[0]
bars = ax.bar(methods_acc, acc, color=acc_colors, edgecolor='gray', width=0.6)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
ax.set_ylim(0.60, 0.85)
# Random基线(1/6≈0.167)远低于当前y轴范围，移除误导性图例
for bar, v in zip(bars, acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
# x轴标签
ax.set_xticklabels(['Expr OT\n(no spatial)', 'PASTE\n(expression\n+ spatial)',
                     'LROT γ=0.1\n(+ LR)', 'LROT γ=0.2\n(+ stronger LR)', 'PASTE2\n(partial FGW)', 'STAligner\n(deep)'],
                    fontsize=8)

# 右: 熵对比
ax = axes2[1]
bars = ax.bar(methods_ent, ent, color=colors, edgecolor='gray', width=0.6)
ax.set_ylabel('Entropy (lower = better)', fontsize=11)
ax.set_title('Alignment Uncertainty', fontsize=12, fontweight='bold')
ax.set_ylim(0, 5.0)
for bar, v in zip(bars, ent):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.set_xticklabels(['Expr OT\n(no spatial)', 'PASTE\n(expression\n+ spatial)',
                     'LROT γ=0.1\n(+ LR)', 'LROT γ=0.2\n(+ stronger LR)'],
                    fontsize=8)

# 添加注释说明熵变化的含义
# 红色箭头：Expr OT(1.752) → PASTE(4.035) 熵升高，箭头朝上
ax.annotate('', xy=(0.4, 4.035), xytext=(0.4, 1.752),
            arrowprops=dict(arrowstyle='->', color='#CC3333', lw=2.5))
ax.text(0.4, 2.8, 'Spatial conflict', fontsize=8, color='#CC3333',
        fontweight='bold', ha='center')
ax.text(0.4, 2.3, '→ Entropy ↑', fontsize=8, color='#CC3333',
        fontweight='bold', ha='center')

# 绿色箭头：PASTE(4.035) → LROT(3.821) 熵降低，箭头朝下
ax.annotate('', xy=(2.5, 3.821), xytext=(2.5, 4.035),
            arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=2.5))
ax.text(2.5, 4.3, 'LR resolves', fontsize=8, color='#2E8B57',
        fontweight='bold', ha='center')
ax.text(2.5, 4.6, '→ Entropy ↓', fontsize=8, color='#2E8B57',
        fontweight='bold', ha='center')

plt.tight_layout(rect=(0, 0, 1, 0.93))
save_path2 = f'{OUT_DIR}/lrot_sota_comparison_detailed.png'
fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {save_path2}")
print("Done!")
