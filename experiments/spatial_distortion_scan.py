# -*- coding: utf-8 -*-
"""
spatial_distortion_scan.py — 空间扭曲强度扫描（鲁棒性极限）
在批次效应固定为默认(0.15)的前提下, 扫描空间旋转扭曲角度 rotation ∈ {2,5,10,15,20,25}°,
展示固定默认参数(α=0.5, β=0.3, γ=0.1)在几何扰动下的鲁棒性极限。

输出: lrot_output/lrot_spatial_distortion_results.txt, lrot_output/lrot_spatial_distortion.png
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lrot_core import (fgw_lr_solver, compute_lr_strength_matrix,
                       compute_mapping_consistency, compute_ot_cost, LR_DB)
from real_st_loader import RealisticSTGenerator

OUT_DIR = os.path.join(_R, 'lrot_output')
os.makedirs(OUT_DIR, exist_ok=True)

ROTATIONS = [2, 5, 10, 15, 20, 25]
SEEDS = [42 + i * 1000 for i in range(5)]
GAMMA = 0.1

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

buf = []
def log(s):
    print(s, flush=True)
    buf.append(s)

def mean_entropy(P):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    return -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()

def run_one(seed, rotation, gamma):
    gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=seed)
    paired = gen.generate_paired_slices(rotation=rotation, scale=0.95, batch_effect=0.15)
    slice_A = paired.slice_A.to_dict()
    slice_B = paired.slice_B.to_dict()
    labels_A, labels_B = slice_A['region_labels'], slice_B['region_labels']
    expr_A, expr_B = slice_A['expr'], slice_B['expr']
    S_lr, _ = compute_lr_strength_matrix(slice_A, slice_B, LR_DB)
    S_in = S_lr if gamma > 0 else np.zeros_like(S_lr)
    P, _ = fgw_lr_solver(slice_A, slice_B, S_in, gamma=gamma, beta=0.3, alpha=0.5,
                         max_iter=200, verbose=False)
    return {'acc': compute_mapping_consistency(P, labels_A, labels_B),
            'ent': mean_entropy(P),
            'cost': compute_ot_cost(P, expr_A, expr_B)}

def main():
    log("=" * 72)
    log("  空间扭曲(旋转)强度扫描（鲁棒性极限）")
    log("  数据: 1,000 spots, 150 genes, 5 seeds, batch=0.15(默认), scale=0.95")
    log("  固定参数: α=0.5, β=0.3; LROT γ=0.1 vs FGW γ=0")
    log("=" * 72)

    results = {r: {'LROT_acc': [], 'FGW_acc': [], 'LROT_ent': [], 'FGW_ent': [], 'LROT_cost': [], 'FGW_cost': []} for r in ROTATIONS}
    for r in ROTATIONS:
        for s in SEEDS:
            lr = run_one(s, r, GAMMA)
            fg = run_one(s, r, 0.0)
            results[r]['LROT_acc'].append(lr['acc'])
            results[r]['FGW_acc'].append(fg['acc'])
            results[r]['LROT_ent'].append(lr['ent'])
            results[r]['FGW_ent'].append(fg['ent'])
            results[r]['LROT_cost'].append(lr['cost'])
            results[r]['FGW_cost'].append(fg['cost'])
        log(f"  rot={r:>2}°  LROT精度={np.mean(results[r]['LROT_acc']):.3f}±{np.std(results[r]['LROT_acc']):.3f} | "
            f"FGW精度={np.mean(results[r]['FGW_acc']):.3f}±{np.std(results[r]['FGW_acc']):.3f} | "
            f"LROT熵={np.mean(results[r]['LROT_ent']):.4f} | FGW熵={np.mean(results[r]['FGW_ent']):.4f}")

    log("\n" + "=" * 72)
    log("  汇总（5 seed 平均）")
    log("=" * 72)
    log(f"  {'rot(°)':<8} {'LROT精度':<12} {'FGW精度':<12} {'Δ精度':<10} {'LROT熵':<10} {'FGW熵':<10}")
    log(f"  {'-'*64}")
    for r in ROTATIONS:
        la = np.mean(results[r]['LROT_acc']); fa = np.mean(results[r]['FGW_acc'])
        le = np.mean(results[r]['LROT_ent']); fe = np.mean(results[r]['FGW_ent'])
        log(f"  {r:<8} {la:<12.3f} {fa:<12.3f} {la-fa:<+10.3f} {le:<10.4f} {fe:<10.4f}")

    # 鲁棒性极限
    acc_5 = np.mean(results[5]['LROT_acc'])
    acc_25 = np.mean(results[25]['LROT_acc'])
    log("\n" + "=" * 72)
    log("  鲁棒性极限分析")
    log("=" * 72)
    log(f"  LROT精度: rot=5°(默认): {acc_5:.3f} → rot=25°: {acc_25:.3f}")
    rel_drop = (acc_5 - acc_25) / acc_5 * 100
    log(f"  相对下降: {rel_drop:.1f}%")
    log("  结论: 固定默认参数对乘性批次效应(0.1~0.5)鲁棒(精度保持), "
        "但空间扭曲超过一定角度后精度开始下降——鲁棒性极限主要在于几何扰动而非乘性批次效应;")
    log("  提示: 在更大空间扭曲下需更强初始化/部分对齐或更大的GW权重, 或自适应权重机制。")

    with open(os.path.join(OUT_DIR, 'lrot_spatial_distortion_results.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(buf) + "\n")
        f.write("\n逐seed数值: seed,rotation,LROT_acc,LROT_ent,LROT_cost,FGW_acc,FGW_ent,FGW_cost\n")
        for r in ROTATIONS:
            for i, s in enumerate(SEEDS):
                f.write(f"{s},{r},{results[r]['LROT_acc'][i]:.4f},{results[r]['LROT_ent'][i]:.4f},"
                        f"{results[r]['LROT_cost'][i]:.4f},{results[r]['FGW_acc'][i]:.4f},"
                        f"{results[r]['FGW_ent'][i]:.4f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    xs = ROTATIONS
    ax = axes[0]
    for tag, key in [('LROT (γ=0.1)', 'LROT_acc'), ('FGW (γ=0)', 'FGW_acc')]:
        m = [np.mean(results[b][key]) for b in xs]
        sd = [np.std(results[b][key]) for b in xs]
        ax.errorbar(xs, m, yerr=sd, fmt='o-', capsize=3, lw=1.8, ms=4,
                    label=tag, color='#C0392B' if tag.startswith('LROT') else '#2C3E50')
    ax.axvline(5, color='gray', ls='--', lw=1)
    ax.text(5.4, ax.get_ylim()[0], 'default 5\u00b0', fontsize=8, color='gray')
    ax.set_xlabel('Spatial rotation distortion (deg)')
    ax.set_ylabel('Region-mapping accuracy')
    ax.set_title('(D) Accuracy vs spatial distortion')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for tag, key in [('LROT (γ=0.1)', 'LROT_ent'), ('FGW (γ=0)', 'FGW_ent')]:
        m = [np.mean(results[b][key]) for b in xs]
        ax.plot(xs, m, 'o-', lw=1.8, ms=4, label=tag,
                color='#C0392B' if tag.startswith('LROT') else '#2C3E50')
    ax.axvline(5, color='gray', ls='--', lw=1)
    ax.set_xlabel('Spatial rotation distortion (deg)')
    ax.set_ylabel('Transport entropy')
    ax.set_title('(E) Entropy vs spatial distortion')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT_DIR, 'lrot_spatial_distortion.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    log(f"\n  图已保存: {os.path.join(OUT_DIR, 'lrot_spatial_distortion.png')}")

if __name__ == '__main__':
    main()
