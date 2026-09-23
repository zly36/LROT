# -*- coding: utf-8 -*-
"""从存档结果重绘图S7(扩展性图, image21), 提高分辨率而不重新测量。

为什么不用 `experiments/lrot_scaling.py` 重跑: 该图含**实测运行时间**, 而实时计时随机器
与负载变化、不属于可复现量。本脚本读取 `lrot_scaling_results.txt` 的存档数值,
用与 lrot_scaling.py **完全相同**的绘图代码重画, 只把 dpi 从 150 提到 360
(有效分辨率 284 → ~680 dpi), 数值与图注口径保持完全一致。

用法: python figures/gen_scaling_from_archive.py
输出: lrot_output/figures_png/lrot_scaling.png
"""
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)



ROOT = _R
SRC = os.path.join(ROOT, 'lrot_output', 'lrot_scaling_results.txt')
OUT = os.path.join(ROOT, 'lrot_output', 'figures_png', 'lrot_scaling.png')
DPI = 360
METHODS = ['Expr OT only', 'FGW (γ=0)', 'LROT (γ=0.1)']

# ---- 解析存档 ----
rows = []
with open(SRC, encoding='utf-8') as f:
    for line in f:
        m = re.match(r'\s*(\d+)\s+(Expr OT only|FGW \(γ=0\)|LROT \(γ=0\.1\))\s+'
                     r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
        if m:
            rows.append(dict(size=int(m.group(1)), method=m.group(2), acc=float(m.group(3)),
                             entropy=float(m.group(4)), cost=float(m.group(5)),
                             solve=float(m.group(6)), lr=float(m.group(7)),
                             peak=float(m.group(8))))
assert rows, '未从存档解析到数据: ' + SRC
print('从存档读到 %d 行 (%d 个规模 × %d 方法)'
      % (len(rows), len({r['size'] for r in rows}), len({r['method'] for r in rows})))

# ---- 与 lrot_scaling.py 完全一致的绘图 ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colors = {'LROT (γ=0.1)': '#d62728', 'FGW (γ=0)': '#1f77b4', 'Expr OT only': '#2ca02c'}
for m in METHODS:
    xs = [r['size'] for r in rows if r['method'] == m]
    ts = [r['solve'] for r in rows if r['method'] == m]
    ac = [r['acc'] for r in rows if r['method'] == m]
    axes[0].loglog(xs, ts, 'o-', color=colors[m], label=m, markersize=5)
    axes[1].semilogx(xs, ac, 'o-', color=colors[m], label=m, markersize=5)
axes[0].set_xlabel('Number of spots (n)')
axes[0].set_ylabel('Solve time (s)')
axes[0].set_title('(A) Runtime scaling (log-log)')
axes[0].grid(True, which='both', alpha=0.3)
axes[0].legend()
axes[1].set_xlabel('Number of spots (n)')
axes[1].set_ylabel('Mapping accuracy')
axes[1].set_title('(B) Accuracy vs. size')
axes[1].set_ylim(0.3, 1.0)
axes[1].grid(True, which='both', alpha=0.3)
axes[1].legend()
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=DPI)
plt.close(fig)
print('Saved:', OUT)
print('数值锚点(应与论文一致): 250 spot LROT %.2fs | 4,000 spot LROT %.2fs'
      % ([r['solve'] for r in rows if r['size'] == 250 and r['method'] == METHODS[2]][0],
         [r['solve'] for r in rows if r['size'] == 4000 and r['method'] == METHODS[2]][0]))
