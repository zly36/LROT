"""
gen_runtime_figure.py — 运行时间对比图
LROT vs FGW复现 vs 官方PASTE(CG) vs PASTE2 vs STAligner
数据来源: run_runtime_benchmark.py (min of 3) + STAligner 训练记录 + run_official_paste2.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


OUT_DIR = os.path.join(_R, 'lrot_output')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

methods = ['LROT\n(γ=0.1)\n软FGW+LR', 'FGW\n(γ=0)\n软FGW', 'Official\nPASTE\nCG硬匹配', 'Official\nPASTE2\n部分FGW', 'STAligner\n(训练)\n深度训练']
times = [3.20, 3.96, 8.58, 2.91, 171.3]
colors = ['#2E8B57', '#E68A8A', '#D67236', '#B08BC9', '#8B1A1A']

fig, ax = plt.subplots(figsize=(8, 5.5))

bars = ax.bar(methods, times, color=colors, edgecolor='gray',
              linewidth=0.8, width=0.6, zorder=3)
ax.set_ylabel('Runtime (s, log scale) ↓', fontsize=12)
ax.set_yscale('log')
ax.set_ylim(1, 500)
ax.grid(axis='y', which='major', linestyle='--', alpha=0.4, zorder=0)

# 数值标注（说明文字已并入x轴标签，避免矮柱内裁剪）
for bar, v in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.12,
            f'{v:.1f}s', ha='center', fontsize=11, fontweight='bold')

ax.tick_params(axis='x', labelsize=8.5)

# 加速比标注
ax.annotate('2.7× faster', xy=(0, 3.20), xytext=(1.3, 14),
            arrowprops=dict(arrowstyle='->', color='#2E8B57', lw=2),
            fontsize=10, color='#2E8B57', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
ax.annotate('53× faster', xy=(0, 3.20), xytext=(3.6, 120),
            arrowprops=dict(arrowstyle='->', color='#8B1A1A', lw=2),
            fontsize=10, color='#8B1A1A', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout(rect=(0, 0, 1, 0.92))
save_path = os.path.join(OUT_DIR, 'figures_png', 'lrot_runtime_comparison.png')
fig.savefig(save_path, dpi=480)
plt.close()
print(f"Saved: {save_path}")
