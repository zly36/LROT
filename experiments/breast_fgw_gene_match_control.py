# -*- coding: utf-8 -*-
"""breast_fgw_gene_match_control.py — 量化"乳腺癌两臂基因表不一致"的影响（精简带进度版）。

背景：`ALL_LR_GENES`（小鼠写法）与人类大写基因名交集为 0 ⇒ 基线臂不传 lr_gene_list 时
C_expr 实际用了全部基因，而 LROT 臂排除了 LR 基因 ⇒ 两臂表达式不一致。

本脚本用与生产脚本相同的 npz 输入，只跑关键臂并**边跑边写报告**，便于中途查看：
    1) FGW_mouse : solve(S_zero, 0)                  ← 正文基线数字来源
    2) FGW_human : solve(S_zero, 0, HUMAN_ALL_LR)    ← 人源基因匹配基线
    3) LROT_human: solve(S_lr, 0.1, HUMAN_ALL_LR)    ← 健全性检查（正文报 5.8176）

输出：lrot_output/lrot_breast_fgw_genematch_results.txt
"""
import io
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


ROOT = _R
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from lrot_core import fgw_lr_solver, compute_ot_cost, ALL_LR_GENES  # noqa: E402
from run_cancer_visium import (HUMAN_LR_DB, HUMAN_ALL_LR,           # noqa: E402
                               compute_lr_strength_human)


NPZ = os.path.join(ROOT, "lrot_output", "数据_npz",
                   "lrot_data_breast_cancer_truecoord.npz")
OUT = os.path.join(ROOT, "lrot_output", "lrot_breast_fgw_genematch_results.txt")
GAMMA = 0.1
PUBLISHED_LROT_ENT = 5.8176          # 正文值，用于健全性检查
buf = []


def say(msg):
    """同时打屏（flush）与写入报告文件（便于中途查看）。"""
    print(msg, flush=True)
    buf.append(msg)
    io.open(OUT, 'w', encoding='utf-8').write('\n'.join(buf) + '\n')


def row_norm(P):
    return P / np.maximum(P.sum(1, keepdims=True), 1e-10)


def stats(P, expr_A, expr_B):
    Pn = row_norm(P)
    ent = -np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean()
    return dict(entropy=float(ent), top1=float(Pn.max(axis=1).mean()),
                cost=float(compute_ot_cost(P, expr_A, expr_B)))


t_all = time.time()
say('=' * 78)
say('  乳腺癌两臂基因表一致性对照（npz 输入，与生产脚本同源）')
say('=' * 78)

z = np.load(NPZ, allow_pickle=True)
say('npz 载入完成（%.1fs），keys=%s' % (time.time() - t_all, list(z.keys())))

# --- 关键：npz 里的表达矩阵是 float16（run_cancer_visium_truecoord.py L204 为省磁盘降精度存盘），
#     必须转回 float32 才能与生产口径一致。否则：
#       · float16 的 X_norm @ Y_norm.T 在 NumPy 中无 BLAS 路径 → 单线程通用循环，
#         单次 cosine 从 ~6 s 变成 ~20 min（约 200 倍）；
#       · float16 仅约 3 位十进制有效位，会污染 C_expr/loss。
#     其它数组（P/P_fgw/loss_hist/coords）本就是 float64，无需处理。
expr_A = np.asarray(z['A_expr'], dtype=np.float32)
expr_B = np.asarray(z['B_expr'], dtype=np.float32)
say('表达矩阵精度: npz=%s -> 求解器=%s（生产脚本 L69 亦为 float32）'
    % (z['A_expr'].dtype, expr_A.dtype))

A = dict(coords=z['A_coords'], expr=expr_A,
         gene_names=[str(g) for g in z['A_gene_names']],
         region_labels=z['A_region_labels'],
         n_spots=int(expr_A.shape[0]), n_genes=int(expr_A.shape[1]))
B = dict(coords=z['B_coords'], expr=expr_B,
         gene_names=[str(g) for g in z['B_gene_names']],
         region_labels=z['B_region_labels'],
         n_spots=int(expr_B.shape[0]), n_genes=int(expr_B.shape[1]))

g = A['gene_names']
n_mouse = sum(1 for x in g if x in ALL_LR_GENES)
n_human = sum(1 for x in g if x in HUMAN_ALL_LR)
say('')
say('数据: %d spots × %d genes | LR 对 %d 对（人类库）'
    % (A['n_spots'], len(g), len(HUMAN_LR_DB)))
say('  ALL_LR_GENES(小鼠写法, %d) 匹配到 %d  → 不传时 C_expr 实际排除 %d 个'
    % (len(ALL_LR_GENES), n_mouse, n_mouse))
say('  HUMAN_ALL_LR(人类写法, %d) 匹配到 %d  → 传时 C_expr 实际排除 %d 个'
    % (len(HUMAN_ALL_LR), n_human, n_human))

t0 = time.time()
S_lr, top = compute_lr_strength_human(A, B, HUMAN_LR_DB)
S_zero = np.zeros_like(S_lr)
say('')
say('S_lr 完成 (%.1fs)；匹配 %d/%d 对' % (time.time() - t0, len(top),
                                        len(HUMAN_LR_DB)))

ARMS = [('FGW_mouse  （正文基线）', S_zero, 0.0, None),
        ('FGW_human  （修好后的基线）', S_zero, 0.0, HUMAN_ALL_LR),
        ('LROT_human （健全性检查）', S_lr, GAMMA, HUMAN_ALL_LR)]

res = {}
say('')
for name, S, gam, ligs in ARMS:
    say('>>> 求解 %s ...' % name)
    t0 = time.time()
    P, loss = fgw_lr_solver(A, B, S, gamma=gam, beta=0.3, alpha=0.5,
                            max_iter=200, verbose=False, lr_gene_list=ligs)
    dt = time.time() - t0
    st = stats(P, A['expr'], B['expr'])
    st['iter'] = len(loss)
    res[name] = st
    say('    %-26s entropy=%.4f  cost=%.4f  top1=%.4f  外层迭代=%d  (%.1fs)'
        % (name, st['entropy'], st['cost'], st['top1'], st['iter'], dt))

eo, en = res['FGW_mouse  （正文基线）'], res['FGW_human  （人源基因匹配基线）']
lo = res['LROT_human （健全性检查）']
d_old = 100 * (eo['entropy'] - lo['entropy']) / eo['entropy']
d_new = 100 * (en['entropy'] - lo['entropy']) / en['entropy']

say('')
say('--- 结论 ---')
say('  FGW 熵   %.4f → %.4f   (Δ = %+.4f, %+.3f%%)'
    % (eo['entropy'], en['entropy'], en['entropy'] - eo['entropy'],
       100 * (en['entropy'] / eo['entropy'] - 1)))
say('  FGW 成本 %.4f → %.4f   (Δ = %+.4f)'
    % (eo['cost'], en['cost'], en['cost'] - eo['cost']))
say('  熵降     %.2f%% → %.2f%%      （正文报 0.49%%）' % (d_old, d_new))
say('  LROT 熵  %.4f（正文报 %.4f，差 %+.4f）'
    % (lo['entropy'], PUBLISHED_LROT_ENT, lo['entropy'] - PUBLISHED_LROT_ENT))
say('')
say('总耗时 %.1f 分钟' % ((time.time() - t_all) / 60))
sys.stderr.write('写出 %s\n' % OUT)
