# -*- coding: utf-8 -*-
"""
重建"上下合并"型补充图(把大写编号的分面板重新竖排拼接)。

合并方式: 按最大宽度等比归一 + 36 px 白缝(vstack)。
组件均取自 lrot_output/figures_png/, 输出写回同目录同名合并图。
各分量脚本的编号为连续大写(例如 S11 = 上(A)(B)(C) + 下(D)(E))。

⚠️ 使用前请先跑校验: python -X utf8 figures/rebuild_supp_composites.py --verify
   `--verify` 不写文件, 只把"由当前组件重算出的尺寸"与"文档内实际图片尺寸"逐张对比;
   文档内尺寸可直接读 docx 的 word/media 得到。

注意: `figures_png/` 里的组件与 docx 内的合成图可能不同步——docx 里的 S2/S4/S11/S15
由某一批组件烘焙; 若组件其后被重渲染过(图内注释、suptitle 等有差异), 直接重跑本脚本
会覆盖 figures_png/ 下的同名合成图。若 --verify 报尺寸不符, 先重跑对应的分量脚本:
    python -X utf8 experiments/pathway_activity_gamma_sweep.py
    python -X utf8 experiments/pathway_activity_dlpfc.py
    python -X utf8 experiments/batch_effect_scan.py
    python -X utf8 experiments/spatial_distortion_scan.py
    python -X utf8 experiments/robustness_cost_diagnosis.py
    python -X utf8 experiments/robustness_failure_scan.py
这些分量脚本把 png 写到 lrot_output/ 根目录, 需手动归位到 figures_png/;
重跑前请先备份 lrot_output/*.txt——分量脚本实测的运行时间会覆盖随包的存档数值。
"""
import io
import os
import sys
import zipfile
import fnmatch

from PIL import Image
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


D = os.path.join(_R, 'lrot_output', 'figures_png')
# 文档内图片尺寸的参照文件（仅 --verify 使用）：优先取最终提交的中文补充材料，
# 依次回退到更早的中文版本；也可用环境变量 LROT_SUPP_DOCX 直接指定。
# 论文 docx 不随代码包分发, 这里只按名字模式在目录中查找最近修改的一份;
# 也可用环境变量 LROT_SUPP_DOCX 直接指定。
_SUPP_GLOB = 'LROT_*中文论文*补充材料*.docx'


def _supp_dirs():
    # 论文 docx 不随代码包分发：既可能在包内 lrot_output/，也可能在与包同级的 lrot_output/。
    dirs = [os.path.join(_R, 'lrot_output'),
            os.path.join(os.path.dirname(_R), 'lrot_output')]
    out = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def _pick_supp():
    env = os.environ.get('LROT_SUPP_DOCX')
    if env:
        return env
    for d in _supp_dirs():
        if not os.path.isdir(d):
            continue
        cands = [os.path.join(d, n) for n in os.listdir(d)
                 if fnmatch.fnmatch(n, _SUPP_GLOB)]
        if cands:
            cands.sort(key=os.path.getmtime, reverse=True)   # 取最近修改的一份
            return cands[0]
    return os.path.join(_supp_dirs()[0], 'LROT_supplementary.docx')


SUPP = _pick_supp()
GAP = 36

MERGES = [
    # 输出名                        组件(顺序即上下顺序)                     docx 内 media
    ('lrot_supp_S2_db_merged.png', ['lrot_lr_sensitivity.png', 'lrot_cellchat_comparison_ABC.png'], 'image15.png'),
    ('lrot_supp_S4_pathway_merged.png', ['lrot_pathway_gamma_sweep.png', 'lrot_pathway_dlpfc.png'], 'image17.png'),
    ('lrot_supp_S11_robustness_merged.png', ['lrot_batch_effect.png', 'lrot_spatial_distortion.png'], 'image25.png'),
    ('lrot_supp_S15_boundary_merged.png', ['lrot_cost_diagnosis.png', 'lrot_failure_domain.png'], 'image45.png'),
    # S14 的 docx 图 (image44.png) 实际是 lrot_supp_S14_breast_ABC.png(4980x3027)，与本
    # 脚本能拼出的 lrot_supp_S14_breast_truecoord.png(2791x3071) 并非同一张；它由另一条
    # 流程产出，故组件列置 None 表示“非本脚本产物”，不参与合成与校验。
    ('lrot_supp_S14_breast_ABC.png', None, 'image44.png'),
]

def compose(paths):
    ims = [Image.open(os.path.join(D, p)).convert('RGB') for p in paths]
    w = max(im.width for im in ims)
    ims = [im if im.width == w else
           im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
           for im in ims]
    H = sum(im.height for im in ims) + GAP * (len(ims) - 1)
    c = Image.new('RGB', (w, H), 'white')
    y = 0
    for im in ims:
        c.paste(im, (0, y))
        y += im.height + GAP
    return c

def verify():
    print('参照文档: %s\n' % os.path.basename(SUPP))
    z = zipfile.ZipFile(SUPP)
    bad = skip = 0
    print('%-40s %-16s %-16s %s' % ('输出图', '文档内', '由组件重算', '结论'))
    for out, comps, media in MERGES:
        old = Image.open(io.BytesIO(z.read('word/media/' + media))).size
        if comps is None:
            skip += 1
            print('%-40s %-16s %-16s %s'
                  % (out, '%dx%d' % old, '—', '跳过（非本脚本产物）'))
            continue
        lack = [c for c in comps if not os.path.exists(os.path.join(D, c))]
        if lack:
            print('%-40s 缺组件: %s' % (out, lack))
            bad += 1
            continue
        new = compose(comps).size
        same = new == old
        if not same:
            bad += 1
        print('%-40s %-16s %-16s %s'
              % (out, '%dx%d' % old, '%dx%d' % new,
                 '一致 ✓' if same else '*** 不一致, 勿覆盖 ***'))
    n = len(MERGES) - skip
    print('\n%d/%d 张可复现（另 %d 张非本脚本产物，已跳过）' % (n - bad, n, skip))
    return bad == 0

if '--verify' in sys.argv:
    ok = verify()
    raise SystemExit(0 if ok else 1)

missing = []
for out, comps, _media in MERGES:
    if comps is None:
        print('SKIP %s (非本脚本产物，跳过)' % out)
        continue
    lack = [c for c in comps if not os.path.exists(os.path.join(D, c))]
    if lack:
        missing.append((out, lack))
        print('SKIP %s (缺少组件: %s)' % (out, ', '.join(lack)))
        continue
    c = compose(comps)
    c.save(os.path.join(D, out))
    print('%-40s %s  ar=%.3f' % (out, c.size, c.width / c.height))

print('\n完成; 缺组件而未生成的:', [m[0] for m in missing] or '无')
print('提示: 如需确认覆盖是否安全，先运行 python -X utf8 figures/rebuild_supp_composites.py --verify')
