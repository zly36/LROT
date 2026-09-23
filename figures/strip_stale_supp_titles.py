# -*- coding: utf-8 -*-
"""strip_stale_supp_titles.py — 修掉"脚本已去标题、但成品图还留着旧标题"的那批组件图。

背景："去标题"只改在绘图脚本里，figures_png/ 中的图未再重新渲染，
于是 4 张合并型补充图里仍烘焙着 fig.suptitle。

做法：对下面 7 个组件按相同规则裁掉顶部标题带（内容上方留 50 px），
再按 36 px 白缝竖排重拼 S2/S4/S11/S15。默认只写到 staging 目录，不动 canonical。
    python -X utf8 figures/strip_stale_supp_titles.py            # 预演(写到 lrot_output/_中间产物/_retitle_out)
    python -X utf8 figures/strip_stale_supp_titles.py --apply    # 覆盖 figures_png/ 里的组件与合成图
"""
import io, os, sys
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)


ROOT = _R
sys.path.insert(0, os.path.join(ROOT, "figures"))
from fig_crop_title import find_crop

D = os.path.join(ROOT, r"lrot_output\figures_png")
STAGE = os.path.join(ROOT, os.path.join("lrot_output", "_中间产物", "_retitle_out"))
GAP = 36
APPLY = "--apply" in sys.argv

COMPONENTS = ["lrot_pathway_gamma_sweep.png", "lrot_pathway_dlpfc.png",
              "lrot_batch_effect.png", "lrot_spatial_distortion.png",
              "lrot_cost_diagnosis.png", "lrot_failure_domain.png",
              "lrot_cellchat_comparison_ABC.png"]
MERGES = [("lrot_supp_S2_db_merged.png", ["lrot_lr_sensitivity.png", "lrot_cellchat_comparison_ABC.png"]),
          ("lrot_supp_S4_pathway_merged.png", ["lrot_pathway_gamma_sweep.png", "lrot_pathway_dlpfc.png"]),
          ("lrot_supp_S11_robustness_merged.png", ["lrot_batch_effect.png", "lrot_spatial_distortion.png"]),
          ("lrot_supp_S15_boundary_merged.png", ["lrot_cost_diagnosis.png", "lrot_failure_domain.png"])]

def cropped(im, path):
    crop, why = find_crop(im, margin=50)
    if crop <= 0:
        print("    %-38s 无需裁切（%s）" % (os.path.basename(path), why))
        return im, 0, why
    box = (0, crop, im.size[0], im.size[1])
    out = im.crop(box)
    assert np.array_equal(np.array(im.convert('RGB'))[crop:], np.array(out.convert('RGB')))
    print("    %-38s %dx%d → %dx%d（裁 %d；%s）"
          % (os.path.basename(path), im.size[0], im.size[1], out.size[0], out.size[1],
             crop, why.split('→')[0].strip()))
    return out, crop, why

def compose(ims):
    w = max(i.width for i in ims)
    ims = [i if i.width == w else i.resize((w, round(i.height * w / i.width)), Image.LANCZOS)
           for i in ims]
    H = sum(i.height for i in ims) + GAP * (len(ims) - 1)
    c = Image.new("RGB", (w, H), "white")
    y = 0
    for i in ims:
        c.paste(i, (0, y)); y += i.height + GAP
    return c

os.makedirs(STAGE, exist_ok=True)
print("=== 1) 裁掉组件的旧标题带 (模式下: %s) ===" % ("APPLY 覆盖 canonical" if APPLY else "预演, 写 staging"))
fixed = {}
for fn in COMPONENTS:
    p = os.path.join(D, fn)
    im = Image.open(p).convert("RGB")
    out, crop, _ = cropped(im, p)
    fixed[fn] = out
    if crop > 0:
        tgt = p if APPLY else os.path.join(STAGE, fn)
        out.save(tgt)

print()
print("=== 2) 重拼 4 张合并补充图 ===")
for out_name, comps in MERGES:
    ims = [fixed.get(c) or Image.open(os.path.join(D, c)).convert("RGB") for c in comps]
    c = compose(ims)
    old = Image.open(os.path.join(D, out_name)).size
    tgt = os.path.join(D, out_name) if APPLY else os.path.join(STAGE, out_name)
    c.save(tgt)
    print("  %-38s 原 %dx%d → 新 %dx%d" % (out_name, old[0], old[1], c.size[0], c.size[1]))
print()
print("输出目录: %s" % (D if APPLY else STAGE))
