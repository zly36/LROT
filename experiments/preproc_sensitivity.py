# -*- coding: utf-8 -*-
"""preproc_sensitivity.py — 表达预处理（文库大小归一化 / 对数变换）对结论的影响。

动机：真实数据（DLPFC、乳腺癌）直接使用 10x 过滤后的原始
UMI 计数，未作文库大小归一化，也未作对数变换。由于
  · C_expr（式4）与传输成本（式15）用**余弦距离** —— 对每个向量的正标量缩放不变，
    因此逐 spot 的文库大小归一化**不改变**这两项；
  · S_lr（式2）是**未归一化的加权外积和** Σ w·e_i^A(L)·e_j^B(R) —— 与 spot 的
    总计数成正比，故未归一化可能使 LR 项偏向高深度 spot。
本脚本量化这一点：在同一数据上比较三种预处理
  raw   = 原始 UMI 计数（正文口径）
  cpm   = 逐 spot 总量归一化到 10,000（只改尺度，余弦项应为不变）
  log1p = log1p(CPM)（标准 scanpy 归一化）
比较 LROT(γ=0.1) 与 FGW(γ=0) 的传输熵、熵降、同层质量、软 ARI/准确率（DLPFC）。

输出：lrot_output/lrot_preproc_sensitivity_results.txt
用法：python -X utf8 experiments/preproc_sensitivity.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


ROOT = Path(_R)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from lrot_core import (fgw_lr_solver, compute_ot_cost,          # noqa: E402
                       compute_mapping_consistency)
from run_dlpfc_visium import load_dlpfc_slice                   # noqa: E402
from run_cancer_visium_truecoord import load_breast_slice       # noqa: E402
from run_cancer_visium import (HUMAN_LR_DB, HUMAN_ALL_LR,     # noqa: E402
                               compute_lr_strength_human)
from sklearn.metrics import adjusted_rand_score                 # noqa: E402

OUT = ROOT / "lrot_output"
GAMMA, BETA, ALPHA = 0.1, 0.3, 0.5
N_LAYERS = 7

def preprocess(expr, mode):
    if mode == 'raw':
        return expr
    tot = np.maximum(expr.sum(1, keepdims=True), 1e-10)
    cpm = expr / tot * 1e4
    if mode == 'cpm':
        return cpm.astype(np.float32)
    if mode == 'log1p':
        return np.log1p(cpm).astype(np.float32)
    raise ValueError(mode)

def metrics(P, lab_A, lab_B, expr_A, expr_B):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    ent = float(-np.sum(Pn * np.log(np.maximum(Pn, 1e-10)), axis=1).mean())
    cost = float(compute_ot_cost(P, expr_A, expr_B))
    out = {'ent': ent, 'cost': cost}
    if lab_A is not None:
        out['same'] = float(np.sum(P * (lab_A[:, None] == lab_B[None, :]))
                            / max(P.sum(), 1e-10))
        pred = np.argmax(Pn @ np.eye(N_LAYERS)[lab_B], axis=1)
        out['ari'] = float(adjusted_rand_score(lab_A, pred))
        out['acc'] = float(np.mean(pred == lab_A))
    return out

def run_slice_pair(name, A, B, lab_A, lab_B, lr_genes_for_lrot=None):
    """lr_genes_for_lrot：**只给 LROT 臂**（与主实验口径一致）。

    主实验口径（设计如此）：
      · LROT 臂：C_expr 排除 LR 基因（§2.3/图1 的设计，避免与 C_lr 冗余）；
      · FGW 基线 (γ=0)：C_expr 用全部基因（无 C_lr 项故无重复计数问题，
        忠实反映 PASTE 风格实现，且为更保守的强基线）。
    ⇒ 乳腺癌主实验两臂 C_expr 集合不同（0.49% 含此口径差），
      无混淆对照为 0.54%（见 lrot_output/lrot_probe_lrot_fullgenes.txt）。结论方向不变。
    """
    print('\n' + '=' * 74)
    print('  %s' % name)
    print('=' * 74)
    rows = []
    for mode in ('raw', 'cpm', 'log1p'):
        a = dict(A)
        b = dict(B)
        a['expr'] = preprocess(A['expr'], mode)
        b['expr'] = preprocess(B['expr'], mode)
        S, _top = compute_lr_strength_human(a, b, HUMAN_LR_DB)
        # 深度依赖诊断：spot 总计数 vs 该 spot 的最大 LR 强度
        tot = A['expr'].sum(1)
        rho = float(np.corrcoef(tot, S.max(axis=1))[0, 1])
        for meth, g in (('FGW', 0.0), ('LROT', GAMMA)):
            t0 = time.time()
            # 只 LROT 臂排除 LR 基因（见 run_slice_pair docstring 的口径说明）
            P, _l = fgw_lr_solver(a, b, S, gamma=g, beta=BETA, alpha=ALPHA,
                                  max_iter=200, verbose=False,
                                  lr_gene_list=lr_genes_for_lrot if meth == 'LROT' else None)
            m = metrics(P, lab_A, lab_B, a['expr'], b['expr'])
            m.update(mode=mode, method=meth, secs=time.time() - t0, rho=rho)
            rows.append(m)
            print('  %-6s %-5s  Ent=%.4f  Cost=%.4f  %s  (%.1fs)'
                  % (mode, meth, m['ent'], m['cost'],
                     ('same=%.4f ari=%.3f acc=%.3f' % (m['same'], m['ari'], m['acc']))
                     if 'same' in m else '', m['secs']))
        print('     （总计数 vs max_j S_lr 的 Pearson r = %.3f）' % rho)
    return rows

all_rows = {}
A = load_dlpfc_slice('151507', 1000, 42)
B = load_dlpfc_slice('151508', 1000, 99)
all_rows['DLPFC 151507→151508'] = run_slice_pair(
    'DLPFC 供体1 对1（151507→151508，1,000 spots，7 层标注）',
    A, B, A['region_labels'], B['region_labels'])

A = load_breast_slice('V1_Breast_Cancer_Block_A_Section_1', 3000, 42)
B = load_breast_slice('V1_Breast_Cancer_Block_A_Section_2', 3000, 99)
all_rows['Breast S1→S2'] = run_slice_pair(
    '乳腺癌 Section1→Section2（3,000 spots，无逐 spot 层标注）',
    A, B, None, None, lr_genes_for_lrot=HUMAN_ALL_LR)

# ---------------- 汇总 ----------------
lines = []
lines.append('=' * 88)
lines.append('  表达预处理敏感性（文库大小归一化 / 对数变换）')
lines.append('=' * 88)
lines.append('设置：LROT γ=0.1, α=0.5, β=0.3；FGW 为 γ=0 的同源软 FGW 基线。')
lines.append('raw   = 原始 UMI 计数（正文口径）；cpm = 逐 spot 归一化到 1e4；log1p = log1p(CPM)')
lines.append('')
lines.append('说明：C_expr（式4）与传输成本（式15）用余弦距离，对逐向量正标量缩放不变，')
lines.append('      故 cpm 与 raw 的这两项在数值上相同；只有 S_lr（式2，未归一化外积和）')
lines.append('      随预处理改变。因此本对照主要检验 LR 项对测序深度的依赖是否会改变结论。')
lines.append('      口径：熵降百分数一律由本节表中列出的 4 位小数熵值反算')
lines.append('      （即 round(ent, 4)），以便读者直接用表中的数字复现；')
lines.append('      若用全精度熵值计算，个别档位会在小数点后第 2 位差 1（如 DLPFC log1p：')
lines.append('      表中 6.1123→4.8254 反算为 21.05%，全精度为 21.06%）。')

# 熵降统一走 R4（与表中显示精度一致），避免"表里数字算不出文中百分数"。
R4 = lambda x: round(float(x), 4)
for k, rows in all_rows.items():
    lines.append('')
    lines.append('【%s】' % k)
    hdr = '  %-7s %-6s %-9s %-9s %-9s %s' % (
        '预处理', '方法', '传输熵', '传输成本',
        '同层质量' if 'same' in rows[0] else '——',
        ('软ARI    软准确率' if 'ari' in rows[0] else ''))
    lines.append(hdr)
    for m in rows:
        lines.append('  %-7s %-6s %-9.4f %-9.4f %-9s %s'
                     % (m['mode'], m['method'], m['ent'], m['cost'],
                        ('%.4f' % m['same']) if 'same' in m else '——',
                        ('%.3f    %.3f' % (m['ari'], m['acc'])) if 'ari' in m else ''))
    # 熵降（用表中 4 位小数反算）
    for mode in ('raw', 'cpm', 'log1p'):
        f = [m for m in rows if m['mode'] == mode and m['method'] == 'FGW'][0]
        l = [m for m in rows if m['mode'] == mode and m['method'] == 'LROT'][0]
        d = (R4(f['ent']) - R4(l['ent'])) / R4(f['ent']) * 100
        extra = ''
        if 'same' in f:
            extra = ('；同层质量 %.4f→%.4f（Δ%+.4f），软ARI %.3f→%.3f'
                     % (f['same'], l['same'], l['same'] - f['same'],
                        f['ari'], l['ari']))
        lines.append('    %s：熵降 %.2f%%%s' % (mode, d, extra))
    lines.append('    S_lr 深度依赖：总计数 vs max_j S_lr 的 r = %.3f' % rows[0]['rho'])
lines.append('')
lines.append('【结论】见文末自动判定。')
# 自动判定
def ent_drop(rows, mode):
    f = [m for m in rows if m['mode'] == mode and m['method'] == 'FGW'][0]
    l = [m for m in rows if m['mode'] == mode and m['method'] == 'LROT'][0]
    return (R4(f['ent']) - R4(l['ent'])) / R4(f['ent']) * 100

for k, rows in all_rows.items():
    d = {m: ent_drop(rows, m) for m in ('raw', 'cpm', 'log1p')}
    spread = max(d.values()) - min(d.values())
    lines.append('  %s：熵降 raw %.2f%% / cpm %.2f%% / log1p %.2f%%，极差 %.2f 个百分点'
                 % (k, d['raw'], d['cpm'], d['log1p'], spread))
    if 'ari' in rows[0]:
        a = {m: [r for r in rows if r['mode'] == m and r['method'] == 'LROT'][0]['ari']
             for m in ('raw', 'cpm', 'log1p')}
        lines.append('       LROT 软 ARI raw %.3f / cpm %.3f / log1p %.3f'
                     % (a['raw'], a['cpm'], a['log1p']))

rep = OUT / 'lrot_preproc_sensitivity_results.txt'
rep.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('\n'.join(lines))
print('\n写出 %s' % rep)
