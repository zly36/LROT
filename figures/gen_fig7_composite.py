# -*- coding: utf-8 -*-
"""图7 横排合成图:把三张分面板横向拼成一张,**面板像素内容原样保留**,只把面板编号
由小写 a/b/c 改为 NAR 规范的大写 A/B/C。

要点:
  · 三个面板取自 figures_png/_docx_panels/ 中的原始 PNG（即横排合成图所用的同一批
    面板）,拼接参数与该合成图完全一致(H1=1250 统一高, GAP=52, PAD=26),
    因此除字母外逐像素不变。
  · 脚本末尾自动与旧合成图做差分,报告"变化区域"应仅落在 3 个字母框内。
输出: lrot_output/figures_png/lrot_fig7_3d_panel_combined_ABC.png
      (旧文件 lrot_fig7_3d_panel_combined.png 保留不动, 供比对)
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)



D = os.path.join(_R, 'lrot_output', 'figures_png')
PREV = os.path.join(D, '_docx_panels')
SRC = [(os.path.join(PREV, 'image7.png'), 'A'),      # 3D 重建
       (os.path.join(PREV, 'image8.png'), 'B'),      # 跨切片一致性
       (os.path.join(PREV, 'image9.png'), 'C')]      # 多 seed 相邻精度
OUT = os.path.join(D, 'lrot_fig7_3d_panel_combined_ABC.png')
OLD = os.path.join(D, 'lrot_fig7_3d_panel_combined.png')

GAP = 52        # 子图间隙(保持原图几何)
PAD = 26        # 画布外白边(保持原图几何)
H1 = 1250       # 三面板统一高(保持原图几何)
FS = 112        # 面板字母字号(保持原图几何)


def scaled(im, h):
    return im.resize((round(im.width * h / im.height), h), Image.LANCZOS)


imgs = [(scaled(Image.open(p).convert('RGB'), H1), L) for p, L in SRC]
W = sum(im.width for im, _ in imgs) + GAP * (len(imgs) - 1) + 2 * PAD
canvas = Image.new('RGB', (W, H1 + 2 * PAD), 'white')

xs, x = [], PAD
for im, _ in imgs:
    xs.append(x)
    canvas.paste(im, (x, PAD))
    x += im.width + GAP

draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype('arialbd.ttf', FS)
except Exception:
    font = ImageFont.load_default()
for (im, label), px in zip(imgs, xs):
    box = draw.textbbox((0, 0), label, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    lx, ly = px + 14, PAD + 14
    draw.rectangle([lx - 6, ly - 6, lx + tw + 10, ly + th + 8], fill='white')
    draw.text((lx, ly), label, font=font, fill=(20, 30, 40))

canvas.save(OUT, dpi=(300, 300))
print('Saved:', OUT, canvas.size)

# ---- 与旧合成图差分:确认只有字母区域变了 ----
if os.path.exists(OLD):
    old = Image.open(OLD).convert('RGB')
    print('old size', old.size)
    if old.size == canvas.size:
        diff = np.asarray(ImageChops.difference(old, canvas)).sum(axis=2) > 12
        ys, xsr = np.where(diff)
        print('changed pixels: %d (%.4f%%)' % (diff.sum(), 100.0 * diff.sum() / diff.size))
        if diff.sum():
            lab = np.zeros(diff.shape, int)
            print('changed bbox: x[%d,%d] y[%d,%d]' % (xsr.min(), xsr.max(), ys.min(), ys.max()))
            # 检查:所有变化像素是否都位于面板左上角字母框内
            for (im, L), px in zip(imgs, xs):
                box = (slice(PAD - 10, PAD + 260), slice(px - 10, px + 300))
                n_in = diff[box].sum()
                print('  panel %s letter-box pixels: %d' % (L, n_in))
            print('  outside letter boxes:', int(diff.sum() - sum(
                diff[PAD - 10:PAD + 260, px - 10:px + 300].sum() for _, px in zip(imgs, xs))))
    else:
        print('size differs -> 版面参数与原图不一致')
