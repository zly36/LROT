"""
lrot_scaling.py — 扩展性实验：LROT vs FGW vs Expr OT 的规模-时间/内存曲线
在真实感模拟数据上扫描 spot 规模（250-4000），测量单次对齐求解时间、
峰值内存、映射精度与传输熵，输出结果表与 log-log 图。
"""
import numpy as np
import time
import os
import sys
import threading
import psutil
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


sys.path.insert(0, _R)
from lrot_core import (
    fgw_lr_solver, compute_lr_strength_matrix,
    compute_mapping_consistency, compute_ot_cost,
    LR_DB,
)
from real_st_loader import RealisticSTGenerator

import warnings

warnings.filterwarnings('ignore')

OUT_DIR = os.path.join(_R, 'lrot_output')
SIZES = [250, 500, 1000, 2000, 4000]
N_GENES = 150
SEED = 42
MAX_ITER = 200
RESULT_PATH = os.path.join(OUT_DIR, "lrot_scaling_results.txt")
FIG_PATH = os.path.join(OUT_DIR, "lrot_scaling.png")

METHODS = ['Expr OT only', 'FGW (γ=0)', 'LROT (γ=0.1)']

def make_slices(n_spots):
    """生成一对真实感配对切片（与主实验相同配置）"""
    gen = RealisticSTGenerator(n_spots_A=n_spots, n_spots_B=n_spots,
                               n_genes=N_GENES, seed=SEED)
    paired = gen.generate_paired_slices(rotation=5.0, batch_effect=0.15)
    return paired.slice_A.to_dict(), paired.slice_B.to_dict()

class PeakMemoryMonitor:
    """后台线程轮询进程峰值常驻内存（MB）"""

    def __init__(self):
        self._proc = psutil.Process(os.getpid())
        self._peak = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop:
            try:
                rss = self._proc.memory_info().rss / (1024.0 ** 2)
                if rss > self._peak:
                    self._peak = rss
            except Exception:
                pass
            time.sleep(0.02)

    def start(self):
        self._peak = self._proc.memory_info().rss / (1024.0 ** 2)
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)

    @property
    def peak_mb(self):
        return self._peak

def run_method(slice_A, slice_B, method):
    """运行单个方法，返回 (P, info)；info 含求解时间/LR矩阵时间/峰值内存"""
    nA, nB = slice_A['n_spots'], slice_B['n_spots']
    if method == 'LROT (γ=0.1)':
        t0 = time.time()
        S_lr, _ = compute_lr_strength_matrix(slice_A, slice_B, LR_DB)
        t_lr = time.time() - t0
        monitor = PeakMemoryMonitor()
        monitor.start()
        t0 = time.time()
        P, _ = fgw_lr_solver(slice_A, slice_B, S_lr, gamma=0.1,
                             beta=0.3, alpha=0.5, max_iter=MAX_ITER, verbose=False)
        t_solve = time.time() - t0
        monitor.stop()
        return P, {'lr_time': t_lr, 'solve_time': t_solve, 'peak_mb': monitor.peak_mb}
    elif method == 'FGW (γ=0)':
        S_lr = np.zeros((nA, nB))
        monitor = PeakMemoryMonitor()
        monitor.start()
        t0 = time.time()
        P, _ = fgw_lr_solver(slice_A, slice_B, S_lr, gamma=0.0,
                             beta=0.3, alpha=0.5, max_iter=MAX_ITER, verbose=False)
        t_solve = time.time() - t0
        monitor.stop()
        return P, {'lr_time': 0.0, 'solve_time': t_solve, 'peak_mb': monitor.peak_mb}
    else:  # Expr OT only (alpha=1, beta=0 -> 无 GW 项)
        S_lr = np.zeros((nA, nB))
        monitor = PeakMemoryMonitor()
        monitor.start()
        t0 = time.time()
        P, _ = fgw_lr_solver(slice_A, slice_B, S_lr, gamma=0.0,
                             beta=0.0, alpha=1.0, max_iter=MAX_ITER, verbose=False)
        t_solve = time.time() - t0
        monitor.stop()
        return P, {'lr_time': 0.0, 'solve_time': t_solve, 'peak_mb': monitor.peak_mb}

def evaluate(P, slice_A, slice_B):
    acc = compute_mapping_consistency(P, slice_A['region_labels'],
                                      slice_B['region_labels'])
    cost = compute_ot_cost(P, slice_A['expr'], slice_B['expr'])
    P_norm = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    entropy = -np.sum(P_norm * np.log(np.maximum(P_norm, 1e-10)), axis=1).mean()
    return acc, cost, entropy

def power_law_exponent(sizes, times):
    x = np.log(np.array(sizes, dtype=float))
    y = np.log(np.array(times, dtype=float))
    k, _ = np.polyfit(x, y, 1)
    return float(k)

def main():
    print("=" * 78)
    print("  LROT Scaling Experiment (realistic simulated data, seed=%d)" % SEED)
    print("  sizes = %s, n_genes = %d" % (SIZES, N_GENES))
    print("  data: rotation=5.0 deg, batch_effect=0.15")
    print("=" * 78)
    print("  {:<8} {:<14} {:>8} {:>9} {:>8} {:>10} {:>9} {:>9}".format(
        "spots", "method", "acc", "entropy", "cost", "solve(s)", "lr(s)", "peakMB"))
    print("  " + "-" * 76)

    all_rows = []
    t_all = time.time()
    for n in SIZES:
        print(f"\n  --- generating data with n_spots = {n} ...")
        sys.stdout.flush()
        slice_A, slice_B = make_slices(n)
        for m in METHODS:
            P, info = run_method(slice_A, slice_B, m)
            acc, cost, entropy = evaluate(P, slice_A, slice_B)
            row = dict(size=n, method=m, acc=acc, entropy=entropy, cost=cost,
                       solve_time=info['solve_time'], lr_time=info['lr_time'],
                       peak_mb=info['peak_mb'])
            all_rows.append(row)
            print("  {:<8} {:<14} {:>8.4f} {:>9.4f} {:>8.4f} {:>10.2f} {:>9.2f} {:>9.1f}".format(
                n, m, acc, entropy, cost, info['solve_time'],
                info['lr_time'], info['peak_mb']))
            sys.stdout.flush()

    elapsed = time.time() - t_all
    print("\n  总耗时: %.1fs" % elapsed)

    # ---- 幂律拟合：log(time) ~ k*log(n) ----
    k_lrot = power_law_exponent(
        [r['size'] for r in all_rows if r['method'] == 'LROT (γ=0.1)'],
        [r['solve_time'] for r in all_rows if r['method'] == 'LROT (γ=0.1)'])
    k_fgw = power_law_exponent(
        [r['size'] for r in all_rows if r['method'] == 'FGW (γ=0)'],
        [r['solve_time'] for r in all_rows if r['method'] == 'FGW (γ=0)'])
    print(f"  经验幂律指数 k (LROT): {k_lrot:.2f}")
    print(f"  经验幂律指数 k (FGW):  {k_fgw:.2f}")

    # ---- 保存结果 ----
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  LROT Scaling Experiment (realistic simulated data)\n")
        f.write("  seed=%d, n_genes=%d, rotation=5.0 deg, batch_effect=0.15\n" % (SEED, N_GENES))
        f.write("=" * 80 + "\n\n")
        f.write("  {:<8} {:<14} {:>8} {:>9} {:>8} {:>10} {:>9} {:>9}\n".format(
            "spots", "method", "acc", "entropy", "cost", "solve(s)", "lr(s)", "peakMB"))
        f.write("  " + "-" * 78 + "\n")
        for r in all_rows:
            f.write("  {:<8} {:<14} {:>8.4f} {:>9.4f} {:>8.4f} {:>10.2f} {:>9.2f} {:>9.1f}\n".format(
                r['size'], r['method'], r['acc'], r['entropy'], r['cost'],
                r['solve_time'], r['lr_time'], r['peak_mb']))
        f.write("\n  经验幂律指数 k (log time vs log n):\n")
        f.write("    LROT (γ=0.1): %.2f\n" % k_lrot)
        f.write("    FGW  (γ=0):   %.2f\n" % k_fgw)
        f.write("  （FGW 类方法每轮含 O(n^3) 的 GW 交叉项计算；实测经验指数 LROT %.2f、FGW %.2f，低于理论最坏情形，\n" % (k_lrot, k_fgw))
        f.write("    因收敛轮数随规模变化且 Sinkhorn 内层仅 O(n^2)）\n")
        f.write("\n  总耗时: %.1fs\n" % elapsed)
    print(f"\n  结果已保存: {RESULT_PATH}")

    # ---- 绘图 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        colors = {'LROT (γ=0.1)': '#d62728', 'FGW (γ=0)': '#1f77b4',
                  'Expr OT only': '#2ca02c'}
        for m in METHODS:
            xs = [r['size'] for r in all_rows if r['method'] == m]
            ts = [r['solve_time'] for r in all_rows if r['method'] == m]
            ac = [r['acc'] for r in all_rows if r['method'] == m]
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
        fig.savefig(FIG_PATH, dpi=150)
        print(f"  图已保存: {FIG_PATH}")
    except Exception as e:
        print("  [WARN] 绘图失败: %s" % e)

if __name__ == '__main__':
    main()
