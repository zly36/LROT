# -*- coding: utf-8 -*-
"""fig_crop_title.py — 把「去标题裁切」烘焙进绘图脚本，使**一次运行**就能产出发布版那张图。

为什么需要它
------------
目标：图内不留最上方那条总标题（`fig.suptitle`）。做法是两步：
    ① 脚本照常渲染（带 suptitle）；
    ② 事后把"标题带"及其下方预留空白裁掉（顶部留 50 px）。
第 ② 步不在绘图脚本里，只重跑绘图脚本不会得到同一张图
（实测 lrot_dlpfc_layer_ratio.png：发布版 3919×1442，直接重跑得到 3919×1496）。

本模块把第 ② 步做成函数：绘图脚本在 savefig 之后调用 `crop_baked_title(path)` 即可，
裁切规则固定（同一套逐行扫描 + 同样的 PIL 保存参数），重跑即可逐像素复现发布版图件。

用法（在绘图脚本里）
--------------------
    sys.path.insert(0, os.path.join(ROOT, 'figures'))
    from fig_crop_title import crop_baked_title
    ...
    fig.savefig(out_png, dpi=360, bbox_inches='tight')
    crop_baked_title(out_png)          # ← 必须在 savefig 之后、plt.close 之前/之后都行
"""
import os
import sys

import numpy as np
from PIL import Image


def find_crop(im, min_gap=12, margin=50, max_scan=800, dpi=300.0):
    """返回 (裁切行, 说明)。规则：图顶部第一段墨迹 = 总标题带。

    `min_gap` 默认为 12 行。取值偏大时，在"标题下缘离轴框很近"的图上会
    静默判为"图里没有标题"，于是**带标题的图被原样保存**（S8 就踩过这个坑：标题下缘到
    轴框只隔 23 行）。对已发表的那几张图结果不变，例如 S13：标题带 36..99、面板内容起始
    244 → 裁切线仍是 194。
    """
    rgb = np.array(im.convert('RGB')).astype(np.int16)
    ink = rgb.min(axis=2) < 240
    rows = ink.sum(axis=1)
    nz = np.nonzero(rows > 0)[0]
    if len(nz) == 0:
        return 0, '无内容'
    y0 = int(nz[0])
    if y0 > 0.15 * len(rows):
        return 0, '首段墨迹不在图顶部(y0=%d)，不是图内总标题' % y0
    y = y0
    blank = 0
    title_end = None
    while y < len(rows) and y < max_scan:
        if rows[y] == 0:
            blank += 1
            if blank >= min_gap:
                title_end = y - blank + 1
                break
        else:
            blank = 0
        y += 1
    if title_end is None:
        return 0, '未找到标题段（图里可能本来就没有 suptitle）'
    # 标题带的形状自检：必须是横向居中、不顶到左右边缘的一条窄带（防止误把整幅内容当标题裁掉）
    band = ink[y0:title_end]
    cols = np.nonzero(band.sum(axis=0) > 0)[0]
    x0, x1 = int(cols[0]), int(cols[-1])
    w = im.size[0]
    if (title_end - y0) > 0.12 * len(rows) or x0 < 0.02 * w or x1 > 0.98 * w:
        return 0, '顶部这段墨迹不像总标题（高 %d px，x %d..%d / %d）' % (title_end - y0, x0, x1, w)
    yy = title_end + min_gap
    while yy < len(rows) and rows[yy] == 0:
        yy += 1
    crop = max(title_end, yy - margin)
    return crop, ('标题段 %d..%d，面板内容起始 %d → 裁切线 %d（留上边距 %d px ≈ %.2f in）'
                  % (y0, title_end - 1, yy, crop, yy - crop, (yy - crop) / float(dpi)))


def crop_baked_title(path, min_gap=12, margin=50, dpi=None, verbose=True, expect_title=True):
    """把 path 顶部烘焙的标题带裁掉并原位写回。

    返回 (裁切行数, 说明)。裁切行数为 0 表示没找到标题带。

    `dpi=None`（默认）表示**沿用原图内嵌的 dpi**——切勿无条件写 300：
    部分图是按 360 dpi 渲染的（NAR 要求线稿≥600 dpi、彩色≥300 dpi）；
    写死 300 会把 360 静默降成 300：像素不变，但 dpi 元数据随之改变，
    重跑产物与论文图就不再逐字节一致。

    `expect_title=True`（默认）时，找不到标题带会**抛 RuntimeError**，而不是静默跳过——
    这是 NAR 投稿的硬要求（图内不得出现总标题），静默跳过等于把带标题的图放进论文。
    真正不需要裁切的图，请显式传 `expect_title=False`。
    """
    im = Image.open(path)
    if dpi is None:
        dpi = im.info.get('dpi', (300.0, 300.0))
    if isinstance(dpi, (int, float)):
        dpi = (dpi, dpi)
    crop, why = find_crop(im, min_gap=min_gap, margin=margin, dpi=dpi[0])
    if crop <= 0:
        if expect_title:
            raise RuntimeError(
                'crop_baked_title 没找到可裁的标题带：{0}；若确认这张图本来就不带标题，'
                '请显式传 expect_title=False。文件：{1}'.format(why, path))
        if verbose:
            print('  [crop_title] 跳过：%s' % why)
        return 0, why
    w, h = im.size
    a = np.array(im.convert('RGBA'))
    out = im.crop((0, crop, w, h))
    b = np.array(out.convert('RGBA'))
    assert a[crop:, :, :].shape == b.shape and np.array_equal(a[crop:, :, :], b), '裁切像素校验失败'
    ink = a[:, :, :3].min(axis=2) < 240
    assert ink[:crop].any(), '裁切区内没有墨迹，说明裁切位置不对'
    assert ink[crop:].any(), '裁切后图内已无内容'
    out.save(path, dpi=dpi)
    # 落盘后自检：裁切线上方留的空白不得超过 1.25×margin（否则说明没裁干净）
    top_ink = int(np.nonzero(np.array(Image.open(path).convert('RGB')).astype(np.int16)
                             .min(axis=2).min(axis=1) < 240)[0][0])
    if top_ink > 1.25 * margin:
        raise RuntimeError('裁切后顶部仍留 %d px 空白（>1.25×%d），标题可能没裁干净：%s'
                           % (top_ink, margin, path))
    if verbose:
        print('  [crop_title] %s  %dx%d → %dx%d（高 -%d，顶部留白 %d px，dpi %g）| %s'
              % (os.path.basename(path), w, h, out.size[0], out.size[1],
                 h - out.size[1], top_ink, dpi[0], why))
    return crop, why


def patch_scripts(scripts, dry_run=True):
    """（工具函数）批量给绘图脚本补上 suptitle 行 + crop 调用。"""
    raise NotImplementedError('该批量入口未随代码包提供；如需重裁请直接调用本模块的 crop_baked_title')
