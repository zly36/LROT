"""
build_cellchat_db.py — 从 CellChatDB.mouse.rda 构建大规模 LR 数据库
并评估其在 LROT 数据中的有效覆盖, 用于"更大数据库"实验
"""
import os
import rdata
import numpy as np
import sys
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

sys.path.insert(0, _R)
from real_st_loader import RealisticSTGenerator


RDA = os.path.join(_R, 'data', 'CellChatDB.mouse.rda')

# ========== 1. 解析 CellChatDB ==========
parsed = rdata.parser.parse_file(RDA)
converted = rdata.conversion.convert(parsed)
inter = converted['CellChatDB.mouse']['interaction']
lig = inter['ligand'].astype(str).values
rec = inter['receptor'].astype(str).values

# ========== 2. 构建 (ligand, receptor) 对 ==========
# CellChatDB 的异源复合物用 "_" 连接，但个别第二亚基写成了缩写，直接按 "_" 切会得到
# 残缺的假基因名（如 ('Tgfb1','R2')、('Tgfb1','TGFbR') 这类非法符号）：
#     TGFbR1_R2    -> 'R2'    应为 TGFbR2（前缀承接）
#     ACVR1_TGFbR  -> 'TGFbR' 应为 TGFbR2（同家族另有 ACVR1B_TGFbR2 / ACVR1C_TGFbR2）
# 另有非基因条目混在受体字段：LXA4（脂氧素 A4，脂质介体，不是基因）。
# 已逐一看过全部 140 个复合物受体串，缩写就这两种。
SUBUNIT_FIX = {'R2': 'TGFbR2', 'TGFbR': 'TGFbR2'}
# 注意：该集合必须写成 mouse_symbol() 归一后的形式（首字母大写、其余小写），
# 因为比对发生在符号归一之后。
NON_GENE = {'Lxa4'}



def mouse_symbol(sym):
    """CellChatDB 里符号大小写不统一（GFRA1 / NRP1 / PTCH1 全大写，Tgfb1 / Lrp5
    首字母大写），而 10x 小鼠数据的基因名是标准小鼠写法（Gfra1 / Nrp1 / Ptch1）。
    不统一就会被当成不同基因、在数据里永远匹配不上（否则会低估
    “大库在数据内有效对”的数量）。标准小鼠符号即为首字母大写、其余小写。"""
    s = str(sym).strip()
    return s[:1].upper() + s[1:].lower()


db = {}
n_fixed = n_dropped = 0
for l, r in zip(lig, rec):
    l = mouse_symbol(l)
    if l in NON_GENE:
        n_dropped += 1
        continue
    if '_' in r:
        # 复合体受体拆成多个单基因
        for gene in r.split('_'):
            if gene in SUBUNIT_FIX:
                gene = SUBUNIT_FIX[gene]
                n_fixed += 1
            gene = mouse_symbol(gene)
            if not gene or gene in NON_GENE:
                n_dropped += 1
                continue
            key = (l, gene)
            db[key] = max(db.get(key, 0), 0.6)
    else:
        gene = mouse_symbol(r)
        if not gene or gene in NON_GENE:
            n_dropped += 1
            continue
        key = (l, gene)
        db[key] = max(db.get(key, 0), 0.9)

print(f"CellChatDB 派生 LR 数据库: {len(db)} 对")
print(f"  唯一配体: {len(set(k[0] for k in db))}, 唯一受体: {len(set(k[1] for k in db))}")
print(f"  亚基缩写修正 {n_fixed} 处，剔除非基因条目 {n_dropped} 处")

# 校验：拆分后不应再出现残缺的亚基名，且符号必须已是标准小鼠写法
stub = sorted({t for k in db for t in k if t in SUBUNIT_FIX})
assert not stub, f'仍有未修正的亚基缩写: {stub}'
non_gene_left = sorted({t for k in db for t in k} & NON_GENE)
assert not non_gene_left, f'仍有非基因条目: {non_gene_left}'
bad_case = sorted({t for k in db for t in k if t != mouse_symbol(t)})[:10]
assert not bad_case, f'符号大小写未统一: {bad_case}'

# ========== 3. 与现有 LR_DB 对比 ==========
from lrot_core import LR_DB
print(f"\n现有 LR_DB: {len(LR_DB)} 对")
overlap = len(set(db.keys()) & set(LR_DB.keys()))
print(f"  与现有 61 对重叠: {overlap} 对")

# ========== 4. 在当前数据中的有效覆盖 ==========
gen = RealisticSTGenerator(n_spots_A=1000, n_spots_B=1000, n_genes=150, seed=42)
paired = gen.generate_paired_slices(rotation=5.0, batch_effect=0.15)
gene_names = list(paired.slice_A.gene_names)
gene_set = set(gene_names)
print(f"\n数据基因集: {len(gene_names)} 个 (含 {len([g for g in gene_names if g != 'Gene'])} LR/other)")
# 数据中有效的 CellChatDB 对
valid = {k: v for k, v in db.items() if k[0] in gene_set and k[1] in gene_set}
print(f"  CellChatDB 大库在数据内有效对: {len(valid)} / {len(db)}")
valid_61 = {k: v for k, v in LR_DB.items() if k[0] in gene_set and k[1] in gene_set}
print(f"  现有 61 对在数据内有效对: {len(valid_61)} / {len(LR_DB)}")
# 覆盖的数据内基因数
covered_genes = set()
for (l, r) in valid:
    covered_genes.add(l); covered_genes.add(r)
print(f"  CellChatDB 大库覆盖数据内基因: {len(covered_genes)} 个")
covered61 = set()
for (l, r) in valid_61:
    covered61.add(l); covered61.add(r)
print(f"  现有 61 对覆盖数据内基因: {len(covered61)} 个")

# ========== 5. 保存构建好的数据库 ==========
import pickle
save_path = os.path.join(_R, 'data', 'CellChatDB_mouse_lr.pkl')
with open(save_path, 'wb') as f:
    pickle.dump({'db_full': db, 'db_valid_data': valid}, f)
print(f"\n已保存: {save_path}")
