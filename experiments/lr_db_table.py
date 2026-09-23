# -*- coding: utf-8 -*-
"""lr_db_table.py — 生成补充表 S9「LR 数据库完整清单」的数据（61 对 × 权重 × 家族 × 来源）

可复现性用途：把论文 §2.2 的 61 对 LR 数据库导出为逐对清单（权重、家族、来源），
便于逐条核验。本脚本从唯一权威来源导出：

    lrot_core.LR_DB        → 61 对与置信度权重
    lrot_families          → 11 个信号家族 + 7 类归并（导入时已做完备性校验）
    data/CellChatDB.mouse.rda → 官方 CellChatDB（2,019 条交互记录），用于标注来源

⚠️ 归属判定不能用简单键匹配：CellChatDB 中大量受体写作**复合物**（GFRA1_RET、
   NRP1_PLXNA1、PTCH1_SMO…）且多为**大写**符号，而本库用标准小鼠符号（Gfra1、
   Nrp1、Ptch1）。因此这里统一：
       配体不区分大小写相等，且受体为该记录受体复合物的组分之一
   例：Gdnf-Gfra1 命中 ('Gdnf','GFRA1_RET')；Sema3a-Nrp1 命中 ('Sema3a','NRP1_PLXNA1')。

输出：
    lrot_output/lr_db_table.csv   机器可读清单
    lrot_output/lr_db_table.txt   便于核对的对齐文本表（同时打印到stdout）

用法：python -X utf8 experiments/lr_db_table.py
"""
import csv
import os
import sys

import rdata
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


sys.path.insert(0, _R)
from lrot_core import LR_DB                                   # noqa: E402
from lrot_families import (LR_FAMILIES, LR_FAMILY_CN,          # noqa: E402
                           LR_FAMILY_GROUPS)


ROOT = _R
OUT = os.path.join(ROOT, 'lrot_output')
CC_RDA = os.path.join(ROOT, 'data', 'CellChatDB.mouse.rda')


def load_cellchat(path):
    """返回 {配体小写: {受体复合物中各组分的大写符号}}。"""
    conv = rdata.conversion.convert(rdata.parser.parse_file(path))
    inter = conv['CellChatDB.mouse']['interaction']
    lig = inter['ligand'].astype(str).values
    rec = inter['receptor'].astype(str).values
    out = {}
    for l, r in zip(lig, rec):
        units = {u.strip().upper() for u in r.split('_')
                 if u.strip() and u.strip().lower() not in ('co', 'i')}
        out.setdefault(l.strip().lower(), set()).update(units)
    return out, len(lig)


def main():
    cc, n_cc_rows = load_cellchat(CC_RDA)
    print('CellChatDB.mouse：%d 条交互记录，%d 个不同配体'
          % (n_cc_rows, len(cc)))

    rows = []
    n = 0
    for fam, pairs in LR_FAMILIES.items():
        for lig, rec in pairs:
            n += 1
            w = LR_DB[(lig, rec)]
            in_cc = (lig.lower() in cc) and (rec.upper() in cc[lig.lower()])
            rows.append(dict(
                idx=n, ligand=lig, receptor=rec, weight=w,
                family=fam, family_cn=LR_FAMILY_CN[fam],
                group=next(g for g, fs in LR_FAMILY_GROUPS.items() if fam in fs),
                in_cellchat=('是' if in_cc else '否'),
                homotypic=('是' if lig == rec else ''),
            ))
    assert n == len(LR_DB) == 61, (n, len(LR_DB))
    n_cc = sum(1 for r in rows if r['in_cellchat'] == '是')
    ws = [r['weight'] for r in rows]
    print('清单 %d 对；其中属 CellChatDB.mouse 的 %d 对（%.0f%%）'
          % (n, n_cc, 100.0 * n_cc / n))
    print('权重区间 %.2f–%.2f，中位 %.3f，均值 %.3f'
          % (min(ws), max(ws), sorted(ws)[len(ws) // 2], sum(ws) / len(ws)))
    print('同型对 %d 个：%s'
          % (sum(1 for r in rows if r['homotypic']),
             [r['ligand'] for r in rows if r['homotypic']]))

    csv_path = os.path.join(OUT, 'lr_db_table.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['idx', 'ligand', 'receptor',
                                           'weight', 'family', 'family_cn',
                                           'group', 'in_cellchat',
                                           'homotypic'])
        wr.writeheader()
        wr.writerows(rows)
    print('-> %s' % csv_path)

    lines = []
    lines.append('%-4s %-10s %-10s %6s  %-26s %-10s %s'
                 % ('#', 'Ligand', 'Receptor', 'w', 'Family', 'CellChatDB',
                    '同型'))
    lines.append('-' * 74)
    for r in rows:
        lines.append('%-4d %-10s %-10s %6.2f  %-26s %-10s %s'
                     % (r['idx'], r['ligand'], r['receptor'], r['weight'],
                        r['family'], r['in_cellchat'], r['homotypic']))
    lines.append('-' * 74)
    lines.append('合计 %d 对 / %d 个基因 / 11 个家族；'
                 '与 CellChatDB.mouse 重叠 %d 对'
                 % (n, len({g for r in rows for g in (r["ligand"], r["receptor"])}),
                    n_cc))
    txt_path = os.path.join(OUT, 'lr_db_table.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('-> %s' % txt_path)
    print()
    print('\n'.join(lines))
    return rows


if __name__ == '__main__':
    main()
