# -*- coding: utf-8 -*-
"""go_enrichment.py — LR 基因的 GO 生物学过程富集分析

分析口径：
  · 注释来源：GO Consortium 官方 `go-basic.obo`（本体/术语名/父子关系） + `mgi.gaf.gz`（小鼠注释）
  · 只取本体域 biological_process；排除 NOT 限定符
  · **按基因去重**（每个基因 → 唯一 GO 项集合）
  · **真实背景**：小鼠全部有 BP 注释的基因（≈2 万）；每个术语的背景规模取真实注释数
  · **true-path rule**：注释沿 is_a / part_of 传播到祖先术语（标准做法）
  · Fisher 精确检验（单尾 greater）+ Benjamini–Hochberg 在**全部**被检验术语上校正
  · 报告 padj < 0.05 且前景命中基因数 ≥ 3 的术语

输出：
  lrot_output/lrot_go_enrichment_results.txt   完整结果表
  lrot_output/lrot_go_enrichment_data.pkl      供画图用的数据
用法：
  python -X utf8 experiments/go_enrichment.py
"""
import gzip
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy.stats import fisher_exact
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


sys.path.insert(0, _R)
from lrot_core import LIGANDS, RECEPTORS    # noqa: E402


ROOT = _R
DATA = os.path.join(ROOT, 'data', 'go')
OUT = os.path.join(ROOT, 'lrot_output')
OBO = os.path.join(DATA, 'go-basic.obo')
GAF = os.path.join(DATA, 'mgi.gaf.gz')

# 注：LR 库的基因符号已是标准小鼠符号（Artn / Nrtn 而非 Artemin / Neurturin），
# 因此不再需要任何符号映射，109 个基因全部可直接注释。


# --------------------------------------------------------------- 1. 本体
def parse_obo(path):
    """返回 (name, parents)：仅 biological_process 术语。"""
    name, parents = {}, defaultdict(set)
    cur, ns = None, None
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '[Term]':
                cur, ns = None, None
            elif line.startswith('id: '):
                cur = line[4:].strip()
            elif line.startswith('name: '):
                if cur:
                    name[cur] = line[6:].strip()
            elif line.startswith('namespace: '):
                ns = line[11:].strip()
            elif line.startswith('is_a: ') and cur and ns == 'biological_process':
                parents[cur].add(line[6:].split(' ')[0].strip())
            elif (line.startswith('relationship: part_of ')
                  and cur and ns == 'biological_process'):
                parents[cur].add(line.split(' ')[2].strip())
    bp = {t for t in name if t.startswith('GO:')}
    return name, parents


def ancestors_of(parents_bp):
    """按拓扑序计算每个术语的全部祖先（true-path rule）。

    注意：边建为 t -> parent(t)，拓扑序会把子节点排在父节点之前，因此必须
    **逆拓扑序** 遍历，才能保证用到的 anc[parent] 已经算好。
    """
    G = nx.DiGraph()
    for t, ps in parents_bp.items():
        for p in ps:
            G.add_edge(t, p)          # t is_a p
    anc = {}
    for t in reversed(list(nx.topological_sort(G))):
        s = set()
        for p in G.successors(t):
            s.add(p)
            s |= anc.get(p, set())
        anc[t] = s
    return anc


# --------------------------------------------------------------- 2. 注释
def parse_gaf(path, bp_terms):
    """返回 gene_symbol -> set(直接注释的 BP 术语)。"""
    g2t = defaultdict(set)
    n = 0
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('!'):
                continue
            c = line.rstrip('\n').split('\t')
            if len(c) < 9:
                continue
            qual, goid, aspect, sym = c[3], c[4], c[8], c[2]
            if aspect != 'P' or goid not in bp_terms:
                continue
            if 'NOT' in qual.split('|'):
                continue
            g2t[sym].add(goid)
            n += 1
    return g2t, n


def main():
    print('=' * 74)
    print('  GO BP enrichment of LROT LR genes — v2 (real background, deduped)')
    print('=' * 74)

    print('\n[1/5] 解析 GO 本体 …')
    name, parents = parse_obo(OBO)
    bp_terms = set(parents) | {t for t in name if t.startswith('GO:')}
    # 有些 BP 术语没有父（根/顶层）
    bp_terms = {t for t in name if t.startswith('GO:')
                and (t in parents or t in {p for ps in parents.values() for p in ps})}
    print('    BP 术语 %d 个' % len(bp_terms))
    anc = ancestors_of({t: parents.get(t, set()) for t in bp_terms})
    print('    祖先表就绪（平均每术语 %.1f 个祖先）'
          % (np.mean([len(v) for v in anc.values()])))

    print('\n[2/5] 解析 MGI GAF（小鼠注释）…')
    g2t, n_ann = parse_gaf(GAF, bp_terms)
    print('    注释行 %d；基因 %d 个' % (n_ann, len(g2t)))

    print('\n[3/5] 沿本体传播（true-path rule）…')
    def propagate(terms):
        s = set(terms)
        for t in terms:
            s |= anc.get(t, set())
        return s

    bg_sets = {g: propagate(ts) for g, ts in g2t.items()}
    background = set(bg_sets)
    print('    背景基因集 %d 个（有 BP 注释的小鼠基因）' % len(background))

    bg_count = defaultdict(int)
    for g, s in bg_sets.items():
        for t in s:
            bg_count[t] += 1

    lr = sorted({g for g in (set(LIGANDS) | set(RECEPTORS))})
    lr_canon = lr
    print('    LR 基因 %d 个（符号已为标准小鼠符号 Artn / Nrtn 等）' % len(lr))
    fg_sets = {g: bg_sets.get(g, set()) for g in lr_canon}
    annotated = [g for g in lr_canon if fg_sets[g]]
    missing = [g for g in lr_canon if not fg_sets[g]]
    print('    注释成功 %d/%d %s' % (len(annotated), len(lr_canon),
                                    ('缺失 %s' % missing) if missing else ''))
    fg_count = defaultdict(int)
    fg_genes = defaultdict(list)
    for g in annotated:
        for t in sorted(fg_sets[g]):        # 排序：避免 set 迭代顺序影响并列名次
            fg_count[t] += 1
            fg_genes[t].append(g)

    print('\n[4/5] Fisher 精确检验 + BH 校正 …')
    n_fg = len(annotated)
    n_bg = len(background)
    tests = []
    for t, a in fg_count.items():
        c = bg_count.get(t, 0)
        if c < 1:
            continue
        b = n_fg - a
        d = n_bg - c
        if d < 0:
            continue
        orr, p = fisher_exact([[a, b], [c, d]], alternative='greater')
        tests.append(dict(go=t, term=name.get(t, ''), a=a, c=c, p=p,
                          fold=(a / n_fg) / (c / n_bg)))
    # 以 GO ID 作为并列名次的确定型次键（否则 set 迭代顺序会让同 p 值的术语
    # 在不同进程间互换行序，结果文件无法逐字节复现）
    tests.sort(key=lambda r: (r['p'], r['go']))
    m = len(tests)
    for i, r in enumerate(tests):
        r['padj'] = min(1.0, r['p'] * m / (i + 1))
    # BH 单调化（从大到小取前缀最小值）
    for i in range(m - 2, -1, -1):
        tests[i]['padj'] = min(tests[i]['padj'], tests[i + 1]['padj'])
    print('    检验术语 %d 个（分子=%d 基因，分母=背景 %d 基因）'
          % (m, n_fg, n_bg))

    sig = [r for r in tests if r['padj'] < 0.05 and r['a'] >= 3]
    print('    padj<0.05 且命中≥3 基因：%d 个术语' % len(sig))

    print('\n[5/5] 写出结果 …')
    lines = []
    lines.append('=' * 78)
    lines.append('  GO BP enrichment — LROT LR genes (v2: real background, deduped)')
    lines.append('=' * 78)
    lines.append('')
    lines.append('Annotation source : GO go-basic.obo + mgi.gaf.gz (MGI, mouse)')
    lines.append('Aspect            : biological_process only; NOT qualifiers excluded')
    lines.append('Foreground        : %d LR genes (%d/%d annotated)'
                 % (n_fg, len(annotated), len(lr_canon)))
    lines.append('Background        : %d mouse genes with >=1 BP annotation' % n_bg)
    lines.append('True-path rule    : annotations propagated to ancestors (is_a/part_of)')
    lines.append('Test              : Fisher exact (one-sided greater); '
                 'BH over all %d tested terms' % m)
    lines.append('Reported          : padj < 0.05 AND >=3 foreground genes')
    lines.append('')
    lines.append('%-12s %-54s %4s %7s %8s %8s %10s'
                 % ('GO ID', 'Term', 'fg', 'bg', 'fold', 'p', 'p_adj'))
    lines.append('-' * 112)
    for r in sig:
        lines.append('%-12s %-54s %4d %7d %8.1f %8.1e %10.1e'
                     % (r['go'], r['term'][:54], r['a'], r['c'], r['fold'],
                        r['p'], r['padj']))
    lines.append('')
    lines.append('--- Top 25 by p (all tested terms) ---')
    for r in tests[:25]:
        lines.append('%-12s %-54s fg=%3d bg=%6d fold=%6.1f p=%.2e padj=%.2e'
                     % (r['go'], r['term'][:54], r['a'], r['c'], r['fold'],
                        r['p'], r['padj']))
    lines.append('')
    lines.append('--- 前景命中基因（padj<0.05, >=3 基因）---')
    for r in sig[:30]:
        lines.append('%s  %s' % (r['go'], r['term']))
        lines.append('    %s' % ', '.join(sorted(fg_genes[r['go']])))
    out_txt = os.path.join(OUT, 'lrot_go_enrichment_results.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('    -> %s' % out_txt)

    with open(os.path.join(OUT, 'lrot_go_enrichment_data.pkl'), 'wb') as f:
        pickle.dump(dict(sig=sig, tests=tests, n_fg=n_fg, n_bg=n_bg,
                         fg_genes=dict(fg_genes), annotated=annotated,
                         missing=missing), f)

    print('\n★ 关键术语（供正文引用）★')
    keys = ['axon guidance', 'angiogenesis', 'canonical Wnt signaling pathway',
            'Notch signaling pathway', 'positive regulation of cell population '
            'proliferation', 'cell migration', 'positive regulation of cell '
            'migration', 'ephrin receptor signaling pathway',
            'semaphorin-plexin signaling pathway', 'viral process']
    for k in keys:
        hit = [r for r in tests if r['term'].lower() == k.lower()]
        if hit:
            r = hit[0]
            print('   %-52s fg=%2d bg=%5d fold=%5.1f padj=%.1e%s'
                  % (r['term'][:52], r['a'], r['c'], r['fold'], r['padj'],
                     '' if r['padj'] < 0.05 else '   (不显著)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
