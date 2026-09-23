# -*- coding: utf-8 -*-
"""lrot_families.py — LR 数据库家族划分的**唯一权威来源**

背景：`lrot_core.LR_DB` 只给出「配体–受体对 → 置信度权重」，家族归属若只写在 `lrot_core.py`
的代码注释里（`# --- Wnt Pathway ---` 之类），注释不参与任何校验，改动数据库时极易漏改。
需要保证正文「11 个信号家族」与图 S3(B) 的 7 类归并口径一致。

本模块把这件事显式化，并在 **导入时** 对 `LR_DB` 做三项完备性校验：

1. 每一对恰好属于一个家族；
2. 11 个家族的配体–受体对并集 == `LR_DB` 的全部 61 对；
3. 11 个家族的基因并集 == `lrot_core.ALL_LR_GENES`（109 个）。

任何一条不成立都会直接 `AssertionError`，从而在论文数据被改动时立刻暴露。

用法::

    from lrot_families import (
        LR_FAMILIES, LR_FAMILY_CN, LR_FAMILY_GROUPS, LR_FAMILY_GROUP_CN,
        LR_SYMBOL_FIX, family_gene_counts, group_gene_counts,
    )

    family_gene_counts()   # {'Neurotrophin / Growth Factor': 34, 'Wnt': 7, ...}  共 11 项
    group_gene_counts()    # {'Neurotrophin / Growth Factor': 34, 'Wnt / Notch': 14, ...} 共 7 项

自检：python -X utf8 lrot_families.py
"""
from collections import OrderedDict

from lrot_core import LR_DB, ALL_LR_GENES

# ---------------------------------------------------------------------------
# 11 个信号家族（顺序即论文 §3.1/图 S3 中的叙述顺序）
# ---------------------------------------------------------------------------
LR_FAMILIES = OrderedDict([
    ('Neurotrophin / Growth Factor', [
        ('Ntng1', 'Ntrk2'), ('Bdnf', 'Ntrk2'), ('Ntf3', 'Ntrk3'),
        ('Ngf', 'Ntrk1'), ('Gdnf', 'Gfra1'), ('Artn', 'Gfra3'),
        ('Nrtn', 'Gfra2'), ('Fgf8', 'Fgfr1'), ('Fgf15', 'Fgfr2'),
        ('Fgf10', 'Fgfr2'), ('Fgf2', 'Fgfr1'), ('Egf', 'Egfr'),
        ('Hgf', 'Met'), ('Vegfa', 'Kdr'), ('Vegfb', 'Flt1'),
        ('Pdgfa', 'Pdgfra'), ('Pdgfb', 'Pdgfrb'), ('Igf1', 'Igf1r'),
        ('Igf2', 'Igf1r'),
    ]),
    ('Wnt', [
        ('Wnt3a', 'Fzd1'), ('Wnt5a', 'Fzd5'), ('Wnt7a', 'Fzd10'),
        ('Wnt1', 'Fzd1'),
    ]),
    ('Notch', [
        ('Dll1', 'Notch1'), ('Dll4', 'Notch4'), ('Jag1', 'Notch1'),
        ('Jag2', 'Notch2'),
    ]),
    ('Ephrin', [
        ('Efnb1', 'Ephb2'), ('Efna1', 'Epha4'), ('Efnb2', 'Ephb4'),
        ('Efna5', 'Epha3'),
    ]),
    ('Semaphorin', [
        ('Sema3a', 'Nrp1'), ('Sema3f', 'Nrp2'), ('Sema4d', 'Plxnb1'),
        ('Sema6a', 'Plxna2'),
    ]),
    ('Slit/Robo', [
        ('Slit1', 'Robo1'), ('Slit2', 'Robo2'), ('Slit3', 'Robo2'),
    ]),
    ('Chemokine / Cytokine', [
        ('Cxcl12', 'Cxcr4'), ('Cxcl13', 'Cxcr5'), ('Ccl2', 'Ccr2'),
        ('Ccl5', 'Ccr5'), ('Il1b', 'Il1r1'), ('Tnf', 'Tnfrsf1a'),
        ('Tgfb1', 'Tgfbr1'), ('Tgfb2', 'Tgfbr2'),
    ]),
    ('BMP', [
        ('Bmp4', 'Bmpr1a'), ('Bmp7', 'Bmpr2'), ('Bmp2', 'Acvr1'),
    ]),
    ('Cell Adhesion', [
        ('Cdh1', 'Cdh1'), ('Cdh2', 'Cdh2'), ('Ncam1', 'Ncam1'),
        ('L1cam', 'L1cam'), ('Lama1', 'Itga1'), ('Col1a1', 'Itga2'),
        ('Vtn', 'Itgav'),
    ]),
    ('Hedgehog', [
        ('Shh', 'Ptch1'), ('Ihh', 'Ptch2'), ('Dhh', 'Ptch1'),
    ]),
    ('Reelin', [
        ('Reln', 'Lrp8'), ('Reln', 'Vldlr'),
    ]),
])

# 中文名（供论文正文/图注生成使用）
LR_FAMILY_CN = OrderedDict([
    ('Neurotrophin / Growth Factor', '神经营养因子/生长因子'),
    ('Wnt', 'Wnt'),
    ('Notch', 'Notch'),
    ('Ephrin', 'Ephrin'),
    ('Semaphorin', 'Semaphorin'),
    ('Slit/Robo', 'Slit/Robo'),
    ('Chemokine / Cytokine', '趋化因子/细胞因子'),
    ('BMP', 'BMP'),
    ('Cell Adhesion', '细胞黏附'),
    ('Hedgehog', 'Hedgehog'),
    ('Reelin', 'Reelin'),
])

# ---------------------------------------------------------------------------
# 绘图用的 7 类归并（图 S3B）：11 个家族按通路相近性合并
# ---------------------------------------------------------------------------
LR_FAMILY_GROUPS = OrderedDict([
    ('Neurotrophin / Growth Factor', ['Neurotrophin / Growth Factor']),
    ('Wnt / Notch', ['Wnt', 'Notch']),
    ('Ephrin / Semaphorin', ['Ephrin', 'Semaphorin']),
    ('Chemokine / Cytokine', ['Chemokine / Cytokine']),
    ('Cell Adhesion / ECM', ['Cell Adhesion']),
    ('BMP / Hedgehog / Reelin', ['BMP', 'Hedgehog', 'Reelin']),
    ('Slit/Robo', ['Slit/Robo']),
])

LR_FAMILY_GROUP_CN = OrderedDict([
    ('Neurotrophin / Growth Factor', '神经营养/生长因子'),
    ('Wnt / Notch', 'Wnt/Notch'),
    ('Ephrin / Semaphorin', 'Ephrin/Semaphorin'),
    ('Chemokine / Cytokine', '趋化因子/细胞因子'),
    ('Cell Adhesion / ECM', '细胞黏附/ECM'),
    ('BMP / Hedgehog / Reelin', 'BMP/Hedgehog/Reelin'),
    ('Slit/Robo', 'Slit/Robo'),
])

# ---------------------------------------------------------------------------
# 基因符号约定
# ---------------------------------------------------------------------------
# Artn / Nrtn 若写成人源符号 Artemin / Neurturin，小鼠 Visium 数据里这两对恒不激活
# （10x 小鼠数据的基因名是 Artn / Nrtn）。库内已统一为标准小鼠符号，
# 因此不需要任何符号映射；下方断言用于防回归。
NONSTANDARD_SYMBOLS = {'Artemin': 'Artn', 'Neurturin': 'Nrtn'}


# ---------------------------------------------------------------------------
# 完备性校验（导入即执行）
# ---------------------------------------------------------------------------
_pairs_in_families = [p for pairs in LR_FAMILIES.values() for p in pairs]

assert len(_pairs_in_families) == len(set(_pairs_in_families)), \
    '有配体-受体对重复出现在多个家族中'
assert set(_pairs_in_families) == set(LR_DB), \
    '家族划分与 LR_DB 不一致：缺少 %s，多出 %s' % (
        sorted(set(LR_DB) - set(_pairs_in_families)),
        sorted(set(_pairs_in_families) - set(LR_DB)))
assert len(_pairs_in_families) == 61, len(_pairs_in_families)

_genes = {g for p in _pairs_in_families for g in p}
assert _genes == set(ALL_LR_GENES), \
    '家族基因并集(%d) 与 ALL_LR_GENES(%d) 不一致' % (len(_genes),
                                                    len(ALL_LR_GENES))
assert len(_genes) == 109, len(_genes)
assert set(LR_FAMILY_CN) == set(LR_FAMILIES), '中文名与家族列表不匹配'
assert {f for g in LR_FAMILY_GROUPS.values() for f in g} == set(LR_FAMILIES), \
    '7 类归并未覆盖全部 11 个家族（或有多余项）'
assert not (_genes & set(NONSTANDARD_SYMBOLS)), \
    'LR 库中又出现了非标准符号：%s' % sorted(_genes & set(NONSTANDARD_SYMBOLS))


# ---------------------------------------------------------------------------
# 统计辅助
# ---------------------------------------------------------------------------
def family_gene_counts():
    """11 个家族各自的唯一基因数（OrderedDict）。"""
    return OrderedDict(
        (f, len({g for p in pairs for g in p}))
        for f, pairs in LR_FAMILIES.items())


def group_gene_counts():
    """7 类归并各自的唯一基因数（OrderedDict），用于图 S3B 饼图。"""
    out = OrderedDict()
    for g, fams in LR_FAMILY_GROUPS.items():
        genes = {x for f in fams for p in LR_FAMILIES[f] for x in p}
        out[g] = len(genes)
    return out


def group_of_family(family):
    """给定家族名，返回其所属的绘图归并类名。"""
    for g, fams in LR_FAMILY_GROUPS.items():
        if family in fams:
            return g
    raise KeyError(family)


def _selfcheck():
    fc = family_gene_counts()
    gc = group_gene_counts()
    print('LR_DB：%d 对 / %d 个基因 / %d 个信号家族'
          % (len(LR_DB), len(ALL_LR_GENES), len(LR_FAMILIES)))
    print('\n%-34s %6s %8s' % ('家族', '对数', '基因数'))
    for f in LR_FAMILIES:
        print('%-34s %6d %8d' % (f, len(LR_FAMILIES[f]), fc[f]))
    print('%-34s %6d %8d' % ('合计', len(LR_DB), len(ALL_LR_GENES)))
    print('\n归并为 %d 类（图 S3B）：' % len(gc))
    for g, n in gc.items():
        print('  %-34s %3d（%s）'
              % (g, n, ' + '.join('%s=%d' % (f, fc[f])
                                  for f in LR_FAMILY_GROUPS[g])))
    print('  合计 %d' % sum(gc.values()))
    assert sum(gc.values()) == len(ALL_LR_GENES)


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _selfcheck()
