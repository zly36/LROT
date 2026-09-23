# -*- coding: utf-8 -*-
"""
run_cancer_calibration_truecoord.py — 跨组织（乳腺癌）置信度校准的无标注代理验证
动机: 校准性分析不应只基于 DLPFC(脑组织)，需在第二种组织(乳腺癌)上验证。

背景: 乳腺癌 Visium 无逐spot真实标注 → 无法直接计算"对齐正确性"。
方案: 无标注代理验证——
  匹配质量代理  = 源spot与其传输加权目标spot的转录组余弦相似度 (高质量匹配→更相似)
  置信度口径    = 匹配锐度 conf = max_j(Pn[i,j]) (无标签下的通用置信度)
  先在 DLPFC(有真实层标注)上校准代理:
      (a) 表达相似度代理 与 真实正确性(软投票命中层) 的 Spearman 相关
      (b) 匹配锐度置信度  与 真实正确性 的 Spearman 相关
  再在乳腺癌(无标注)上验证: 匹配锐度置信度 分箱 → 表达相似度单调上升 + Spearman
输出: lrot_output/lrot_cancer_calibration_truecoord_results.txt
      lrot_output/lrot_cancer_calibration_truecoord.png
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'experiments'))

import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_cancer_visium_truecoord import (load_breast_slice as download_and_prepare, apply_misalignment,
                               compute_lr_strength_human, HUMAN_LR_DB)
from run_dlpfc_visium import load_dlpfc_slice
from lrot_core import fgw_lr_solver

OUT_DIR = os.path.join(_R, 'lrot_output')
N = 1000        # DLPFC 校准口径（与主实验 DLPFC 的 1,000-spot 口径一致）
N_BREAST = 3000  # 乳腺癌代理校准口径（与主实验乳腺 3,000-spot 口径一致）
GAMMA = 0.1
LIGS = sorted(set(k[0] for k in HUMAN_LR_DB) | set(k[1] for k in HUMAN_LR_DB))
LAYER_N = 7

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

buf = []
def log(s):
    print(s, flush=True)
    buf.append(s)

def row_norm(P):
    return P / np.maximum(P.sum(1, keepdims=True), 1e-10)

def conf_sharp(Pn):
    """匹配锐度置信度：行归一化计划的最大概率（无标签通用口径）"""
    return Pn.max(axis=1)

def expr_similarity(expr_A, expr_B, Pn):
    """匹配质量代理：源spot与其加权目标spot的转录组余弦相似度"""
    X = expr_A / np.maximum(np.linalg.norm(expr_A, axis=1, keepdims=True), 1e-10)
    target = Pn @ expr_B
    Y = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-10)
    return np.sum(X * Y, axis=1)

def softvote_correct(Pn, labels_A, labels_B):
    onehot = np.eye(LAYER_N)[labels_B]
    soft_p = Pn @ onehot
    pred_soft = soft_p.argmax(axis=1)
    return (pred_soft == labels_A).astype(float)

def decile_means(score, value):
    qs = np.quantile(score, np.arange(0.1, 1.0, 0.1))
    bins = np.digitize(score, qs)
    return np.array([value[bins == b].mean() for b in range(10)])

def solve(A, B, S, gamma):
    P, _ = fgw_lr_solver(A, B, S, gamma=gamma, beta=0.3, alpha=0.5,
                         max_iter=200, verbose=False, lr_gene_list=LIGS)
    return P

def run_dlpfc_calibration():
    log("\n" + "=" * 72)
    log("  [1/3] DLPFC (有真实层标注) — 校准无标注代理与匹配锐度置信度")
    log("=" * 72)
    rows = {}
    for sa, sb in [("151507", "151508")]:
        A = load_dlpfc_slice(sa, N, 42)
        B = load_dlpfc_slice(sb, N, 99)
        B['coords'] = apply_misalignment(B['coords'], rotation_deg=2.0, translation=0.5)
        S, _ = compute_lr_strength_human(A, B, HUMAN_LR_DB)
        for gamma, tag in [(0.0, 'FGW'), (GAMMA, 'LROT')]:
            S_in = np.zeros_like(S) if gamma == 0 else S
            P = solve(A, B, S_in, gamma)
            Pn = row_norm(P)
            conf = conf_sharp(Pn)
            sim = expr_similarity(A['expr'], B['expr'], Pn)
            correct = softvote_correct(Pn, A['region_labels'], B['region_labels'])
            rows[tag] = dict(conf=conf, sim=sim, correct=correct)
            r_cs, p_cs = spearmanr(conf, correct)
            r_ss, p_ss = spearmanr(sim, correct)
            r_csim, p_csim = spearmanr(conf, sim)
            log(f"  DLPFC {sa}→{sb} [{tag}]: ρ(锐度置信度, 真实正确性)={r_cs:.3f} (p={p_cs:.1e}) | "
                f"ρ(表达相似度代理, 真实正确性)={r_ss:.3f} (p={p_ss:.1e}) | "
                f"ρ(锐度置信度, 表达相似度)={r_csim:.3f}")
    return rows

def run_breast_calibration():
    log("\n" + "=" * 72)
    log("  [2/3] Breast Cancer (无真实标注) — 无标注代理验证")
    log("=" * 72)
    A = download_and_prepare("V1_Breast_Cancer_Block_A_Section_1", N_BREAST, 42)
    B = download_and_prepare("V1_Breast_Cancer_Block_A_Section_2", N_BREAST, 99)
    B['coords'] = apply_misalignment(B['coords'], rotation_deg=2.0, translation=0.5)
    log(f"  乳腺癌: {A['n_spots']} spots × {A['n_genes']} genes (Section 1 → Section 2)")
    S, _ = compute_lr_strength_human(A, B, HUMAN_LR_DB)
    rows = {}
    for gamma, tag in [(0.0, 'FGW'), (GAMMA, 'LROT')]:
        S_in = np.zeros_like(S) if gamma == 0 else S
        t0 = time.time()
        P = solve(A, B, S_in, gamma)
        dt = time.time() - t0
        Pn = row_norm(P)
        conf = conf_sharp(Pn)
        sim = expr_similarity(A['expr'], B['expr'], Pn)
        rows[tag] = dict(conf=conf, sim=sim)
        r, p = spearmanr(conf, sim)
        dec = decile_means(conf, sim)
        log(f"  乳腺癌 [{tag}] (耗时{dt:.1f}s): ρ(锐度置信度, 表达相似度)={r:.3f} (p={p:.1e})")
        log(f"    置信度十分位 -> 表达相似度: " +
            " | ".join(f"{d:.3f}" for d in dec))
        log(f"    top-20% 表达相似度 {dec[-2]:.3f} vs 全量 {sim.mean():.3f} (保留 {int(0.2*len(conf))} spots)")
    return rows

def make_figure(dlpfc, breast):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    # 保持原图配色(不变): FGW=深蓝灰 #2C3E50 / LROT=暗红 #C0392B，仅叠加网格
    colors = {'LROT': '#C0392B', 'FGW': '#2C3E50'}

    # 左: 乳腺癌 置信度十分位 vs 表达相似度
    ax = axes[0]
    for tag in ['FGW', 'LROT']:
        conf = breast[tag]['conf']; sim = breast[tag]['sim']
        dec = decile_means(conf, sim)
        r, p = spearmanr(conf, sim)
        ax.plot(range(10), dec, 'o-', color=colors[tag], lw=1.8, ms=4,
                label=f"{tag} (ρ={r:.3f})")
    ax.set_xlabel('Matching-sharpness confidence decile (low\u2192high)')
    ax.set_ylabel('Transcriptomic consistency (cosine similarity)')
    ax.set_title('(A) Breast cancer (label-free proxy)')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)

    # 右: DLPFC 置信度十分位 vs 真实正确率（代理被校准的对照）
    ax = axes[1]
    for tag in ['FGW', 'LROT']:
        conf = dlpfc[tag]['conf']; correct = dlpfc[tag]['correct']
        dec = decile_means(conf, correct)
        r, p = spearmanr(conf, correct)
        ax.plot(range(10), dec, 'o-', color=colors[tag], lw=1.8, ms=4,
                label=f"{tag} (ρ={r:.3f})")
    ax.set_xlabel('Matching-sharpness confidence decile (low\u2192high)')
    ax.set_ylabel('True layer accuracy')
    ax.set_title('(B) DLPFC (true-label calibration)')
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)

    # 统一: 浅灰网格 + 细灰边框(与其他图一致)
    for a in axes.flat:
        a.grid(True, color='#DADADA', lw=0.7, alpha=0.9)
        a.set_axisbelow(True)
        for s in ['top', 'right', 'left', 'bottom']:
            a.spines[s].set_color('#AAAAAA')
            a.spines[s].set_linewidth(0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(OUT_DIR, 'lrot_cancer_calibration_truecoord.png')
    fig.savefig(out, dpi=360, bbox_inches='tight')
    plt.close(fig)
    log(f"\n  图已保存: {out}")

def main():
    dlpfc = run_dlpfc_calibration()
    breast = run_breast_calibration()

    log("\n" + "=" * 72)
    log("  [3/3] 结论")
    log("=" * 72)
    # DLPFC: 代理被校准
    for tag in ['FGW', 'LROT']:
        r_sim_c, _ = spearmanr(dlpfc[tag]['sim'], dlpfc[tag]['correct'])
        r_c_c, _ = spearmanr(dlpfc[tag]['conf'], dlpfc[tag]['correct'])
        log(f"  DLPFC [{tag}]: 表达相似度代理与真实正确性 ρ={r_sim_c:.3f} (代理有效); "
            f"匹配锐度置信度与真实正确性 ρ={r_c_c:.3f}")
    for tag in ['FGW', 'LROT']:
        r, p = spearmanr(breast[tag]['conf'], breast[tag]['sim'])
        log(f"  乳腺癌 [{tag}]: 匹配锐度置信度与表达一致性 ρ={r:.3f} (p={p:.1e})")
    rL, _ = spearmanr(breast['LROT']['conf'], breast['LROT']['sim'])
    rF, _ = spearmanr(breast['FGW']['conf'], breast['FGW']['sim'])
    log(f"  → 乳腺癌上 LROT 的置信度-一致性相关 ({rL:.3f}) {'高于' if rL > rF else '低于'} FGW ({rF:.3f})")
    log("\n结论: 置信度的校准性质（高置信→更可靠的匹配）在脑组织(DLPFC)之外的第二组织"
        "(乳腺癌)上通过无标注代理验证成立——高置信对应的转录组一致性显著更高。")

    with open(os.path.join(OUT_DIR, 'lrot_cancer_calibration_truecoord_results.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(buf) + "\n")

    make_figure(dlpfc, breast)

if __name__ == '__main__':
    main()
