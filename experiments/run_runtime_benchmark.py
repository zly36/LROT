"""
run_runtime_benchmark.py — 统一运行时间基准
在同一真实感模拟数据(1000 spots, seed=42)上统一计时:
LROT / FGW复现 / 官方PASTE(CG)
STAligner 训练时间(171.3s) 取自实测记录
"""
import os
import sys
import time
import numpy as np
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


sys.path.insert(0, _R)
OUT_DIR = os.path.join(_R, 'lrot_output')

from lrot_core import fgw_lr_solver, compute_lr_strength_matrix
from real_st_loader import RealisticSTGenerator

import paste_math
from paste_math import pairwise_align
from paste2.PASTE2 import partial_pairwise_align

def to_anndata(sd):
    import anndata
    ad = anndata.AnnData(X=sd['expr'].astype(np.float64))
    ad.var_names = sd['gene_names']
    ad.obs_names = [f"spot{i}" for i in range(sd['n_spots'])]
    ad.obsm['spatial'] = sd['coords'].astype(np.float64)
    return ad

def main():
    print("生成真实感数据 (同基线脚本)...")
    gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=42)
    paired = gen.generate_paired_slices(rotation=5.0, batch_effect=0.15)
    sA, sB = paired.slice_A.to_dict(), paired.slice_B.to_dict()
    adA, adB = to_anndata(sA), to_anndata(sB)

    results = []
    N_RUN = 3  # 多次测量取最小值, 排除机器负载波动

    def best_of(fn, n=N_RUN):
        ts = []
        for _ in range(n):
            t0 = time.time()
            fn()
            ts.append(time.time() - t0)
        return min(ts)

    # 1) LROT (γ=0.1)
    from lrot_core import LR_DB
    S_lr, _ = compute_lr_strength_matrix(sA, sB, LR_DB)
    t_lrot = best_of(lambda: fgw_lr_solver(
        sA, sB, S_lr, gamma=0.1, beta=0.3, alpha=0.5, max_iter=200, verbose=False))
    results.append(('LROT (γ=0.1)', t_lrot))
    print(f"  LROT (γ=0.1): {t_lrot:.2f}s")

    # 2) FGW 复现 (γ=0)
    S0 = np.zeros_like(S_lr)
    t_fgw = best_of(lambda: fgw_lr_solver(
        sA, sB, S0, gamma=0.0, beta=0.3, alpha=0.5, max_iter=200, verbose=False))
    results.append(('FGW 复现 (γ=0)', t_fgw))
    print(f"  FGW 复现 (γ=0): {t_fgw:.2f}s")

    # 3) 官方 PASTE (α=0.5, CG)
    t_paste = best_of(lambda: pairwise_align(
        adA, adB, alpha=0.5, dissimilarity='kl', numItermax=200, verbose=False))
    results.append(('Official PASTE (α=0.5)', t_paste))
    print(f"  Official PASTE (α=0.5): {t_paste:.2f}s")

    # 4) 官方 PASTE2 (α=0.1, diss=kl, 部分FGW)
    p = np.ones(adA.shape[0]) / adA.shape[0]
    q = np.ones(adB.shape[0]) / adB.shape[0]
    s = float(min(p.sum(), q.sum()))
    t_paste2 = best_of(lambda: partial_pairwise_align(
        adA, adB, s=s, alpha=0.1, dissimilarity='kl',
        a_distribution=p, b_distribution=q, verbose=False))
    results.append(('Official PASTE2 (α=0.1)', t_paste2))
    print(f"  Official PASTE2 (α=0.1): {t_paste2:.2f}s")

    # 5) STAligner (训练, 取自实测记录)
    t_stal = 171.3
    results.append(('STAligner (训练)', t_stal))

    print("\n" + "=" * 60)
    print("  运行时间基准 (真实感模拟数据, 1000 spots)")
    print("=" * 60)
    lines = ["运行时间基准 (真实感模拟数据, 1000 spots, seed=42)",
             "=" * 60]
    for name, t in results:
        lines.append(f"{name:<24} {t:<8.2f}s")
    lines.append("=" * 60)
    lines.append("加速比: LROT vs Official PASTE = {:.1f}x; LROT vs PASTE2 = {:.1f}x; LROT vs STAligner = {:.1f}x".format(
        t_paste / t_lrot, t_paste2 / t_lrot, t_stal / t_lrot))
    text = "\n".join(lines)
    print(text)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "runtime_benchmark.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n已保存: {os.path.join(OUT_DIR, 'runtime_benchmark.txt')}")

if __name__ == '__main__':
    main()
