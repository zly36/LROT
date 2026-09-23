# -*- coding: utf-8 -*-
"""Independent-donor replication on DLPFC (Supplementary Fig. S18).
Runs the exact same pipeline as run_dlpfc_visium.py on a second donor
(samples 151669/151670/151671) and saves JSON + human-readable results.
"""
import os
import sys, json, time
from pathlib import Path
# 自定位仓库根：向上找到含 lrot_core.py 的目录（代码包解压到任意路径均可运行）
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, 'experiments'))

from run_dlpfc_visium import analyze_pair, fmt_metrics, LAYER_ORDER

OUT = Path(_R, 'lrot_output')
DONOR2 = OUT / "dlpfc_donor2"
DONOR2.mkdir(parents=True, exist_ok=True)

PAIRS = [("151669", "151670"), ("151670", "151671")]
results = []
t0 = time.time()
for a, b in PAIRS:
    r = analyze_pair(a, b, target_spots=1000)
    results.append(r)
print("\n==== SUMMARY donor2 ====")
for r in results:
    imp = (r['fgw']['entropy'] - r['lrot']['entropy']) / r['fgw']['entropy'] * 100
    print(f"Pair {r['pair']}: LR matched {r['lr_matched']}/{r['lr_total']}")
    print("  FGW :", fmt_metrics(r['fgw']))
    print("  LROT:", fmt_metrics(r['lrot']))
    print(f"  entropy reduction {imp:.2f}% | soft-ARI {r['lrot']['soft_ari']-r['fgw']['soft_ari']:+.3f} "
          f"| same-layer mass {r['lrot']['same_mass']-r['fgw']['same_mass']:+.4f}")

summary = []
lines = ["LROT second-donor DLPFC replication (151669/151670/151671)",
         "Pipeline identical to run_dlpfc_visium.py (1000 spots, 2 deg + 0.5 misalignment)"]
for r in results:
    imp = (r['fgw']['entropy'] - r['lrot']['entropy']) / r['fgw']['entropy'] * 100
    d = {
        'pair': r['pair'], 'lr_matched': r['lr_matched'], 'lr_total': r['lr_total'],
        'fgw': {k: r['fgw'][k] for k in ['entropy', 'cost', 'soft_ari', 'hard_ari',
                                          'same_mass', 'soft_acc', 'hard_acc', 'nn_acc', 'time']},
        'lrot': {k: r['lrot'][k] for k in ['entropy', 'cost', 'soft_ari', 'hard_ari',
                                           'same_mass', 'soft_acc', 'hard_acc', 'nn_acc', 'time']},
        'entropy_reduction_pct': imp,
        'sweep': [{k: s[k] for k in ['gamma', 'entropy', 'soft_ari', 'same_mass']} for s in r['sweep']],
        'top_pairs': [(a1, b1, w) for a1, b1, w in r['top_pairs'][:6]],
    }
    summary.append(d)
    lines.append(f"Pair {r['pair']}: entropy reduction {imp:.2f}% | "
                 f"soft-ARI change {d['lrot']['soft_ari']-d['fgw']['soft_ari']:+.3f} | "
                 f"same-layer mass change {d['lrot']['same_mass']-d['fgw']['same_mass']:+.4f}")
    lines.append(f"  FGW : " + fmt_metrics(r['fgw']))
    lines.append(f"  LROT: " + fmt_metrics(r['lrot']))

means = {
    'entropy_reduction_pct': float(sum(d['entropy_reduction_pct'] for d in summary) / len(summary)),
    'soft_ari_change': float(sum(d['lrot']['soft_ari'] - d['fgw']['soft_ari'] for d in summary) / len(summary)),
    'same_mass_change': float(sum(d['lrot']['same_mass'] - d['fgw']['same_mass'] for d in summary) / len(summary)),
}
lines.append("Donor2 mean (2 pairs): entropy reduction %.2f%% | soft-ARI %+.3f | same-layer mass %+.4f"
             % (means['entropy_reduction_pct'], means['soft_ari_change'], means['same_mass_change']))
json.dump({'results': summary, 'mean': means}, open(DONOR2 / 'donor2_results.json', 'w', encoding='utf-8'),
          indent=1, ensure_ascii=False)
open(DONOR2 / 'donor2_results.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print("saved to", P1)
print("elapsed %.1fs" % (time.time() - t0))