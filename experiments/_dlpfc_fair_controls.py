# -*- coding: utf-8 -*-
"""
_dlpfc_fair_controls.py — 回应核心 so-what 的三个公平对照 (DLPFC Br8325, 1,000 spots, 真实层标注)
  A. Entropy-matched FGW: 把 FGW(无 LR) 熵正则调小, 使其行熵降到 LROT(γ=0.1) 同水平,
     比较 同层质量/软ARI —— 检验"LROT 熵降是否只是加项/变锐的效应"。
  B. Shuffled-LR (partner shuffle): 受体端打乱(基因集不变), 量化真实 LR 的净增量 (5 seed)。
  C. Calibration retention-accuracy AUC: LROT vs FGW 的"保留-精度"曲线 AUC (两对合并 n=2,000)。
输出: lrot_output/lrot_dlpfc_fair_controls_results.txt (+ 可选图)
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

from run_dlpfc_visium import load_dlpfc_slice, eval_plan
from run_cancer_visium import HUMAN_LR_DB, apply_misalignment, compute_lr_strength_human
from lrot_core import fgw_lr_solver


OUT_DIR = os.path.join(_R, 'lrot_output')
N = 1000
GAMMA = 0.1
SEED_A, SEED_B = 42, 99
SLICES = ["151507", "151508", "151509"]
LAYER_ORDER = ["L1", "L2", "L3", "L4", "L5", "L6", "WM"]
N_LAYERS = 7

buf = []
def log(s=""):
    print(s, flush=True)
    buf.append(str(s))


def lig_list():
    return sorted(set(k[0] for k in HUMAN_LR_DB) | set(k[1] for k in HUMAN_LR_DB))


def solve(A, B, S, gamma, reg_ot=0.01, reg_gw=0.01):
    t0 = time.time()
    P, _ = fgw_lr_solver(A, B, S, gamma=gamma, beta=0.3, alpha=0.5, reg_ot=reg_ot,
                         reg_gw=reg_gw, max_iter=200, verbose=False, lr_gene_list=lig_list())
    m = eval_plan(P, A['region_labels'], B['region_labels'],
                  A['expr'], B['expr'], A['coords'], B['coords'])
    m['time'] = time.time() - t0
    return P, m


def shuffle_db(rng):
    items = list(HUMAN_LR_DB.items())          # {(lig, rec): weight}
    ligs = [k[0] for k, w in items]
    recs = [k[1] for k, w in items]
    ws = [w for k, w in items]
    recs_sh = rng.permutation(recs).tolist()
    return {(lig, rec_sh): w for lig, rec_sh, w in zip(ligs, recs_sh, ws)}


def per_spot(P, labA, labB):
    Pn = P / np.maximum(P.sum(1, keepdims=True), 1e-10)
    onehot = np.eye(N_LAYERS)[labB]
    sp = Pn @ onehot
    conf = sp.max(axis=1)
    correct = (sp.argmax(axis=1) == labA).astype(float)
    return conf, correct


def retention_auc(conf, correct, n_steps=100):
    order = np.argsort(-conf, kind='stable')
    n = len(conf)
    cum = np.cumsum(correct[order]) / np.arange(1, n + 1)   # accuracy at each retained k
    fr = np.arange(1, n + 1) / n
    # AUC over retention fraction grid (uniform 0..1) via trapz on fine grid
    grid = np.linspace(0, 1, n_steps + 1)[1:]
    acc = np.interp(grid, fr, cum)
    return float(np.trapezoid(acc, grid)) if hasattr(np, 'trapezoid') else float(np.trapz(acc, grid)), acc, grid


def main():
    log("=" * 90)
    log("DLPFC 公平对照 A/B/C (Br8325, 1,000 spots/片, 7层真实标注, B旋转2°/平移0.5)")
    log("=" * 90)

    # ------- load 3 slices (main convention: 151507@42; 151508/151509@99) -------
    slices, raw = {}, {}
    for i, sid in enumerate(SLICES):
        seed = SEED_A if i == 0 else SEED_B
        slices[i] = load_dlpfc_slice(sid, N, seed)
        raw[i] = slices[i]['coords'].copy()
    mis = {i: raw[i].copy() for i in range(3)}
    for i in (1, 2):
        mis[i] = apply_misalignment(raw[i], rotation_deg=2.0, translation=0.5)

    def pair_slices(Ai, Bi):
        A = dict(slices[Ai]); B = dict(slices[Bi])
        A['coords'] = raw[Ai]; B['coords'] = mis[Bi]
        return A, B

    # ------- reference solves: pair1 & pair2, LROT & FGW -------
    S_lr = {}
    for (Ai, Bi) in [(0, 1), (1, 2)]:
        A, B = pair_slices(Ai, Bi)
        S_lr[(Ai, Bi)], _ = compute_lr_strength_human(A, B, HUMAN_LR_DB)
    S_ze = {k: np.zeros_like(v) for k, v in S_lr.items()}

    plans = {}   # (pair, method) -> (P, m)
    for (Ai, Bi) in [(0, 1), (1, 2)]:
        A, B = pair_slices(Ai, Bi)
        P_l, m_l = solve(A, B, S_lr[(Ai, Bi)], GAMMA)
        P_f, m_f = solve(A, B, S_ze[(Ai, Bi)], 0.0)
        plans[((Ai, Bi), 'LROT')] = (P_l, m_l)
        plans[((Ai, Bi), 'FGW')] = (P_f, m_f)
        log(f"  ref pair {SLICES[Ai]}→{SLICES[Bi]}: LROT ent={m_l['entropy']:.4f} same={m_l['same_mass']:.4f} sARI={m_l['soft_ari']:.3f} | "
            f"FGW ent={m_f['entropy']:.4f} same={m_f['same_mass']:.4f} sARI={m_f['soft_ari']:.3f}")

    # ================= A. Temperature Pareto: FGW vs LROT at matched softness =================
    log("\n" + "=" * 90)
    log("A. Fair control: for the SAME Sinkhorn temperature (reg scale s), compare")
    log("   FGW(γ=0) vs LROT(γ=0.05) vs LROT(γ=0.1) on (entropy, same-layer mass, soft ARI)")
    log("   (固定温度下 LR 是否在更低熵下保持/提升层一致性; 并看 FGW 能否靠单纯变锐达到同熵)")
    log("=" * 90)
    (Ai, Bi) = (0, 1)
    A, B = pair_slices(Ai, Bi)
    log(f"  {'s':>6} {'method':>7} {'熵':>9} {'同层质量':>9} {'软ARI':>7} {'软ACC':>7}")
    pareto = {'FGW': [], 'LROT05': [], 'LROT10': []}
    for s in [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12]:
        for tag, gamma, S in [('FGW', 0.0, S_ze[(Ai, Bi)]),
                              ('LROT05', 0.05, S_lr[(Ai, Bi)]),
                              ('LROT10', GAMMA, S_lr[(Ai, Bi)])]:
            P, m = solve(A, B, S, gamma, reg_ot=0.01 * s, reg_gw=0.01 * s)
            pareto[tag].append((s, m['entropy'], m['same_mass'], m['soft_ari'], m['soft_acc']))
            log(f"  {s:>6.2f} {tag:>7} {m['entropy']:>9.4f} {m['same_mass']:>9.4f} {m['soft_ari']:>7.3f} {m['soft_acc']:>7.3f}")
    log("")
    # matched-entropy summary: entropy at default LROT10 (s=1.0) vs FGW needing smaller s
    e_def = pareto['LROT10'][0][1]   # s=1.0 entropy of LROT γ=0.1
    fgw_row = min([r for r in pareto['FGW']], key=lambda r: abs(r[1] - e_def))
    log(f"  matched-entropy check: LROT(γ=0.1, default s=1.0) ent={e_def:.4f} "
        f"same={pareto['LROT10'][0][2]:.4f} sARI={pareto['LROT10'][0][3]:.3f}")
    log(f"    FGW closest to this entropy: s={fgw_row[0]:.2f} ent={fgw_row[1]:.4f} "
        f"same={fgw_row[2]:.4f} sARI={fgw_row[3]:.3f}")
    # matched-temperature deltas (default parameters, s=1.0)
    for tag in ('LROT05', 'LROT10'):
        r_f = pareto['FGW'][0]; r_l = pareto[tag][0]
        log(f"    at default temperature (s=1.0): {tag} vs FGW -> Δent={r_l[1]-r_f[1]:+.4f} "
            f"Δsame={r_l[2]-r_f[2]:+.4f} ΔsARI={r_l[3]-r_f[3]:+.3f}")

    # ================= B. Shuffled-LR partner control (pair 1) =================
    log("\n" + "=" * 90)
    log("B. Shuffled-LR partner control (pair 151507→151508): shuffle receptor partners (5 seeds)")
    log("    量化真实 LR 的净增量 (固定默认参数: 熵/同层质量/软ARI 相对 FGW 与 true-LR)")
    log("=" * 90)
    stats = {'ent': [], 'same': [], 'sari': [], 'ent_drop': [], 'soft_acc': []}
    fgw_ent = pareto['FGW'][0][1]
    for sd in range(5):
        rng = np.random.RandomState(100 + sd)
        db_s = shuffle_db(rng)
        A, B = pair_slices(Ai, Bi)
        S_sh, _ = compute_lr_strength_human(A, B, db_s)
        P_s, m_s = solve(A, B, S_sh, GAMMA)
        stats['ent'].append(m_s['entropy']); stats['same'].append(m_s['same_mass'])
        stats['sari'].append(m_s['soft_ari']); stats['soft_acc'].append(m_s['soft_acc'])
        stats['ent_drop'].append((fgw_ent - m_s['entropy']) / fgw_ent * 100)
        log(f"  shuffle seed {sd}: ent={m_s['entropy']:.4f} same={m_s['same_mass']:.4f} sARI={m_s['soft_ari']:.3f} "
            f"ent_drop={stats['ent_drop'][-1]:.2f}%")
    def ms(x): return f"{np.mean(x):.4f}±{np.std(x):.4f}"
    log(f"  shuffled-LR (5 seed) mean±sd: ent={ms(stats['ent'])} same={ms(stats['same'])} "
        f"sARI(mean)={np.mean(stats['sari']):.3f} ent_drop={np.mean(stats['ent_drop']):.2f}%")
    log(f"  true-LR (LROT γ=0.1, s=1.0):  ent={pareto['LROT10'][0][1]:.4f} same={pareto['LROT10'][0][2]:.4f} "
        f"sARI={pareto['LROT10'][0][3]:.3f} ent_drop={(fgw_ent-pareto['LROT10'][0][1])/fgw_ent*100:.2f}%")
    log(f"  FGW (no LR, s=1.0):            ent={fgw_ent:.4f} same={pareto['FGW'][0][2]:.4f} "
        f"sARI={pareto['FGW'][0][3]:.3f}")

    # ================= C. Calibration retention-accuracy AUC (merged n=2,000) =================
    log("\n" + "=" * 90)
    log("C. Calibration retention-accuracy AUC (merged two pairs n=2,000, soft-vote correctness)")
    log("=" * 90)
    # 显式标注为 dict：下面既要往里追加列表（conf/corr），又要存 float（_auc），
    # 不加标注时静态分析会把值类型推断成 list，于是在 `pooled[meth]['_auc'] = auc` 报
    # 「float 不能赋给 list」——运行时并无此限制。
    pooled: dict = {'LROT': {'conf': [], 'corr': []}, 'FGW': {'conf': [], 'corr': []}}
    for (Ai, Bi) in [(0, 1), (1, 2)]:
        for meth in ('LROT', 'FGW'):
            P, _ = plans[((Ai, Bi), meth)]
            conf, corr = per_spot(P, slices[Ai]['region_labels'], slices[Bi]['region_labels'])
            pooled[meth]['conf'].extend(conf.tolist())
            pooled[meth]['corr'].extend(corr.tolist())
    for meth in ('LROT', 'FGW'):
        conf = np.array(pooled[meth]['conf']); corr = np.array(pooled[meth]['corr'])
        auc, acc_grid, grid = retention_auc(conf, corr)
        rho, p = spearmanr(conf, corr)
        # top-20%
        k = int(round(0.2 * len(conf)))
        order = np.argsort(-conf, kind='stable')
        top20 = corr[order[:k]].mean()
        log(f"  {meth}: retention-accuracy AUC={auc:.4f} | top-20% acc={top20:.3f} | "
            f"full acc={corr.mean():.3f} | ρ(conf,correct)={rho:+.3f} (p={p:.1e})")
        pooled[meth]['_auc'] = auc

    with open(os.path.join(OUT_DIR, "lrot_dlpfc_fair_controls_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(buf) + "\n")
    print("\nresults ->", os.path.join(OUT_DIR, "lrot_dlpfc_fair_controls_results.txt"))


if __name__ == "__main__":
    main()
