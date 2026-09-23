"""
lr_cellchat_comparison.py — 真实 CellChatDB 大规模数据库对比
同一数据(seed=42, 1000 spots)下, 对比:
  - 现有 61 对 LR_DB
  - CellChatDB 派生的 ~2000 对大库（数量与有效对数由 data/CellChatDB_mouse_lr.pkl 读出）
验证“更大数据库是否提升对齐质量”, 并分析基因覆盖的约束作用。

输出：
  lrot_output/lrot_cellchat_comparison.png
  lrot_output/lrot_cellchat_results.txt
  lrot_output/figures_png/lrot_cellchat_comparison.png  （供 
      figures/rebuild_supp_composites.py 合成图S2 下半张使用）

注：本脚本生成的图已自带 (G)/(H) 面板字母，无需事后叠字。
"""
import os
import numpy as np
import time
import pickle
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

from lrot_core import (fgw_lr_solver, compute_lr_strength_matrix,
                       compute_mapping_consistency, compute_ot_cost, LR_DB)
from real_st_loader import RealisticSTGenerator

OUT_DIR = os.path.join(_R, 'lrot_output')

# ========== 数据（与主实验一致）==========
gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=42)
paired = gen.generate_paired_slices(rotation=5.0, batch_effect=0.15)
A, B = paired.slice_A, paired.slice_B
sa, sb = A.to_dict(), B.to_dict()
gene_set = set(list(A.gene_names))

# ========== 加载 CellChatDB 大库 ==========
with open(os.path.join(_R, 'data', 'CellChatDB_mouse_lr.pkl'), 'rb') as f:
    ccdb = pickle.load(f)
db_cellchat = ccdb['db_full']  # 由 tools/build_cellchat_db.py 生成（当前 2059 对）
print("=" * 70)
print("  CellChatDB 大规模数据库对比 (同一数据)")
print("=" * 70)
print(f"  现有 LR_DB: {len(LR_DB)} 对 | CellChatDB: {len(db_cellchat)} 对")
# 数据内有效对
valid61 = {k: v for k, v in LR_DB.items() if k[0] in gene_set and k[1] in gene_set}
validCC = {k: v for k, v in db_cellchat.items() if k[0] in gene_set and k[1] in gene_set}
print(f"  数据内有效: 61对→{len(valid61)} 对, CellChatDB→{len(validCC)} 对")

def run(db, gamma):
    S_lr, _ = compute_lr_strength_matrix(sa, sb, db)
    t0 = time.time()
    P, _ = fgw_lr_solver(sa, sb, S_lr, gamma=gamma, beta=0.3, alpha=0.5,
                         max_iter=150, verbose=False)
    el = time.time() - t0
    acc = compute_mapping_consistency(P, A.region_labels, B.region_labels)
    cost = compute_ot_cost(P, A.expr, B.expr)
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    ent = -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()
    return acc, ent, cost, el

print(f"\n  {'数据库':<22}{'γ':<6}{'Acc':<9}{'Entropy':<10}{'Cost':<9}{'Time':<7}")
print("  " + "-" * 62)
rows = []
CC_TAG = f'CellChatDB ({len(db_cellchat)}对)'
for db_name, db in [('LR_DB (61对)', LR_DB), (CC_TAG, db_cellchat)]:
    for gamma in [0.1, 0.2]:
        acc, ent, cost, el = run(db, gamma)
        rows.append((db_name, gamma, acc, ent, cost, el))
        print(f"  {db_name:<22}{gamma:<6.2f}{acc:<9.4f}{ent:<10.4f}{cost:<9.4f}{el:<7.2f}")

# 差异
base = [r for r in rows if '61' in r[0]]
cc = [r for r in rows if r[0] == CC_TAG]
print("  " + "-" * 62)
for (bn, bg, ba, be, bc, bt), (cn, cg, ca, ce, cc_, ct) in zip(base, cc):
    print(f"  γ={bg}: 熵 {be:.4f}→{ce:.4f} (Δ{ce-be:+.4f}), 精度 {ba:.4f}→{ca:.4f}")

# ========== 保存 ==========
with open(f"{OUT_DIR}/lrot_cellchat_results.txt", 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("  CellChatDB Large Database Comparison\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"现有 LR_DB: {len(LR_DB)} 对; CellChatDB: {len(db_cellchat)} 对\n")
    f.write(f"数据内有效: 61对→{len(valid61)}, CellChatDB→{len(validCC)}\n\n")
    f.write(f"  {'数据库':<22}{'γ':<6}{'Acc':<9}{'Entropy':<10}{'Cost':<9}\n")
    for db_name, gamma, acc, ent, cost, el in rows:
        f.write(f"  {db_name:<22}{gamma:<6.2f}{acc:<9.4f}{ent:<10.4f}{cost:<9.4f}\n")
print(f"\n  结果已保存: {OUT_DIR}/lrot_cellchat_results.txt")

# ========== 绘图 ==========
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
gs = [0.1, 0.2]
b_ent = [r[3] for r in base]; c_ent = [r[3] for r in cc]
b_acc = [r[2] for r in base]; c_acc = [r[2] for r in cc]

ax = axes[0]
ax.plot(gs, b_ent, 'o-', color='#2166AC', label=f'Current DB (61 pairs, {len(valid61)} effective)', lw=2, ms=8)
ax.plot(gs, c_ent, 's--', color='#D6604D', label=f'CellChatDB ({len(db_cellchat)} pairs, {len(validCC)} effective)', lw=2, ms=8)
ax.set_xlabel('LR weight γ'); ax.set_ylabel('Entropy (lower=better)')
ax.set_title('(G) Alignment Entropy', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(gs, b_acc, 'o-', color='#2166AC', label=f'Current DB (61 pairs, {len(valid61)} effective)', lw=2, ms=8)
ax.plot(gs, c_acc, 's--', color='#D6604D', label=f'CellChatDB ({len(db_cellchat)} pairs, {len(validCC)} effective)', lw=2, ms=8)
ax.set_xlabel('LR weight γ'); ax.set_ylabel('Accuracy')
ax.set_title('(H) Alignment Accuracy', fontsize=13, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(f"{OUT_DIR}/lrot_cellchat_comparison.png", dpi=300)
plt.close(fig)
print(f"  图已保存: {OUT_DIR}/lrot_cellchat_comparison.png")

# 同步到 figures_png/：合成脚本 figures/rebuild_supp_composites.py 从那里取组件。
# 图已自带 (G)/(H) 字母，同时写出 *_ABC.png 名字以免合成脚本找不到组件。
import shutil
FIG = os.path.join(_R, 'lrot_output', 'figures_png')
for name in ('lrot_cellchat_comparison.png', 'lrot_cellchat_comparison_ABC.png'):
    shutil.copy2(f"{OUT_DIR}/lrot_cellchat_comparison.png",
                 f"{FIG}/{name}")
print(f"  已同步到 {FIG}")
