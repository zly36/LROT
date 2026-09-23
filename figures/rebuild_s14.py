# -*- coding: utf-8 -*-
"""
rebuild_s14.py — 重建补充图 S14「乳腺癌 ERBB2/HER2 消融(A) + 12 家族消融谱(B)」。

背景
----
图 S14 是两张子图竖排的合成图（docx 内 word/media/image44.png，4980×3027）。
其下半 (B) 并非 `figures_png/lrot_cancer_ablation_scan_truecoord.png` 那张独立图，
而是另行渲染的 4980×1439 面板：家族名水平两行排列；数值标签只标两根非中性柱
（+0.48 / +0.28），其余 10 个家族以「neutral (|Δ| ≤ 0.015)」灰带 + 图例表示。

做法（最小改动、可核对）
------------------------
  * 上半 (A) 区域 行 0..1482 原样保留（像素不动）；
  * 中间 105 px 白缝保持；
  * 下半 (B) 行 1588..3026 重新渲染为 4980×1439，并且：
      - 坐标框与 (A) 完全对齐（左 373 / 右 4931 / 上 66 / 下 1303，由该合成图量得）；
      - 家族名水平两行排列（不旋转）；
      - 数值标签与中性灰带如上；
  * (B) 面板字母置于该面板左上角。

数值一律取自 lrot_output/lrot_cancer_ablation_scan_truecoord_results.txt（12 家族 Δ），
以及 lrot_cancer_erbb2_ablation_truecoord_results.txt 的 0.494% 完整库熵降。

用法: python -X utf8 figures/rebuild_s14.py
输出: lrot_output/figures_png/lrot_supp_S14_breast_ABC.png  (4980x3027)
"""
import io
import os
import shutil
import sys
import zipfile
import fnmatch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                                        # noqa: E402
import numpy as np                                                          # noqa: E402
from matplotlib.patches import Patch                                         # noqa: E402
from PIL import Image, ImageDraw, ImageFont                                  # noqa: E402


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
# 自定位仓库根：代码包解压到任意路径均可运行
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

ROOT = _R
FIGD = os.path.join(ROOT, 'lrot_output', 'figures_png')
# 面板 A 的回退来源（仅当 _中间产物/_img_orig/S14_A.png 不存在时才读取）。
# 优先取最终提交的中文补充材料，可用环境变量 LROT_SUPP_DOCX 直接指定。
# 论文 docx 不随代码包分发, 这里只按名字模式在目录中查找最近修改的一份;
# 也可用环境变量 LROT_SUPP_DOCX 直接指定。
_SUPP_GLOB = 'LROT_*中文论文*补充材料*.docx'


def _pick_supp():
    env = os.environ.get('LROT_SUPP_DOCX')
    if env:
        return env
    dirs = [os.path.join(ROOT, 'lrot_output'),
            os.path.join(os.path.dirname(ROOT), 'lrot_output')]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        cands = [os.path.join(d, n) for n in os.listdir(d)
                 if fnmatch.fnmatch(n, _SUPP_GLOB)]
        if cands:
            cands.sort(key=os.path.getmtime, reverse=True)
            return cands[0]
    return os.path.join(dirs[0], 'LROT_supplementary.docx')


SUPP = _pick_supp()
MEDIA = 'word/media/image44.png'
OUT_FINAL = os.path.join(FIGD, 'lrot_supp_S14_breast_ABC.png')

# ---- 原图几何（由逐行白带分析量得）----
W, H = 4980, 3027
CUT_A = 1483          # 行 0..1482 = 面板 A
SEAM = 105            # 行 1483..1587 = 白缝
H_B = 1439            # 行 1588..3026 = 面板 B
FRAME = dict(left=373, right=4931, top=66, bottom=1303)   # 相对面板B区域的行/列

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.facecolor'] = 'white'

RED = '#D6604D'       # 干扰（移除该家族后熵降反而增大）
GREEN = '#4DAF4A'     # 正贡献（本图无该情形，仅用于完整呈现配色编码）
GRAY = '#888888'      # 中性

NAMES = ['ERBB2/HER', 'Cell Adhesion\n', 'Semaphorin', 'Neurotrophin/\nGrowth',
         'Ephrin', 'Notch', 'Slit/Robo', 'Hedgehog', 'Wnt', 'BMP',
         'Reelin', 'Chemokine/\nCytokine']
DS = [0.480, 0.277, 0.015, 0.013, 0.007, 0.003, 0.001, 0.000, 0.000, 0.000,
      -0.000, -0.004]
FULL = 0.494          # 完整库熵降（%），结果文件 0.494


def render_panelB(path):
    fig = plt.figure(figsize=(W / 300.0, H_B / 300.0), dpi=300)
    ax = fig.add_axes([FRAME['left'] / float(W),
                       1.0 - FRAME['bottom'] / float(H_B),
                       (FRAME['right'] - FRAME['left']) / float(W),
                       (FRAME['bottom'] - FRAME['top']) / float(H_B)])
    x = np.arange(len(NAMES))
    cols = [GREEN if d < -0.1 else (RED if d > 0.1 else GRAY) for d in DS]
    ax.bar(x, DS, color=cols, width=0.62, edgecolor='#333333', lw=0.6, zorder=3)
    ax.axhline(0, color='black', lw=1.0, zorder=4)
    ax.axhline(FULL, ls='--', color='#444444', lw=1.2, zorder=4)
    ax.axhspan(-0.015, 0.015, color='#cccccc', alpha=0.55, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(NAMES, fontsize=10.5)
    ax.set_xlim(-0.75, len(NAMES) - 0.25)
    ax.set_ylim(-0.06, 0.70)
    ax.set_yticks(np.arange(0.0, 0.61, 0.1))
    ax.set_yticklabels(['%.1f' % v for v in np.arange(0.0, 0.61, 0.1)], fontsize=11)
    ax.set_ylabel(u'\u0394 entropy reduction (vs full library, %)', fontsize=12)

    # 只标两根非中性柱的数值（其余 10 个家族由灰带 + 图例统一说明，避免 "+0.00" 堆叠）
    for i, d in enumerate(DS):
        if d > 0.1:
            ax.text(i, d + 0.012, '%+.2f' % d, ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='#222222', zorder=5)
    ax.text(len(NAMES) - 1.15, FULL + 0.014, u'Full-library reduction %.3f%%' % FULL,
            ha='right', va='bottom', fontsize=10.5, color='#333333', zorder=5)

    ax.legend(handles=[Patch(facecolor=RED, edgecolor='#333333',
                             label=u'interference (\u0394 > 0.1)'),
                       Patch(facecolor=GRAY, edgecolor='#333333',
                             label=u'neutral (|\u0394| \u2264 0.015)'),
                       Patch(facecolor=GREEN, edgecolor='#333333',
                             label=u'reduction weakens (\u0394 < \u22120.1)')],
              fontsize=10.5, loc='upper center', ncol=3, frameon=False)
    ax.set_title('Breast-cancer pathway ablation spectrum '
                 '(leave-one-family-out, 12 families)', fontsize=11.5, pad=8)
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    sz = Image.open(path).size
    assert sz == (W, H_B), '面板B 尺寸不符: %s' % (sz,)
    print('[panelB] %s  %dx%d' % (os.path.basename(path), sz[0], sz[1]))


def main():
    tmp_b = os.path.join(ROOT, 'lrot_output', '_中间产物', '_S14_panelB_new.png')
    render_panelB(tmp_b)

    # 面板 A 的原图：优先用已抽出的独立 PNG（代码包随附），否则回落到补充材料 docx。
    # 本脚本只重画面板 B，面板 A 原样沿用；因此面板 A 来自哪都行，只要几何一致。
    a_png = os.path.join(ROOT, 'lrot_output', '_中间产物', '_img_orig', 'S14_A.png')
    if os.path.exists(a_png):
        orig = Image.new('RGB', (W, H), 'white')
        orig.paste(Image.open(a_png).convert('RGB'), (0, 0))
    elif os.path.exists(SUPP):
        z = zipfile.ZipFile(SUPP)
        orig = Image.open(io.BytesIO(z.read(MEDIA))).convert('RGB')
        z.close()
    else:
        raise SystemExit(
            '  [SKIP] 本脚本只用于重画补充图S14 的面板B，需要面板A 的原图。请二选一：\n'
            '         · 保持 lrot_output/_中间产物/_img_orig/S14_A.png 存在（代码包已随附）；\n'
            '         · 或用 LROT_SUPP_DOCX 指向补充材料 docx。\n'
            '         论文最终版图已在 lrot_output/figures_png/lrot_supp_S14_breast_ABC.png。')
    assert orig.size == (W, H), 'docx 内 image44 尺寸变了: %s' % (orig.size,)

    canvas = Image.new('RGB', (W, H), 'white')
    canvas.paste(orig.crop((0, 0, W, CUT_A)), (0, 0))                    # 面板 A 原样
    canvas.paste(Image.open(tmp_b).convert('RGB'), (0, CUT_A + SEAM))    # 新面板 B

    # ---- 抹掉原图误压的 (B) 字母，重新压到 (B) 面板左上角 ----
    try:
        font = ImageFont.truetype('arialbd.ttf', int(H * 0.032))
    except Exception:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(canvas)
    lx, ly = int(W * 0.012), int(H * 0.455)              # 原脚本里被误放的 (B) 位置
    bb = d.textbbox((0, 0), '(B)', font=font)
    d.rectangle([lx - 6, ly - 6, lx + bb[2] + 10, ly + bb[3] + 8], fill='white')
    ny = CUT_A + SEAM + int(H * 0.008)                  # 与 (A) 相同的相对偏移
    d.rectangle([lx - 6, ny - 6, lx + bb[2] + 10, ny + bb[3] + 8], fill='white')
    d.text((lx, ny), '(B)', font=font, fill=(20, 30, 40))
    print('[letters] 抹掉旧 (B) @ y=%d, 新 (B) @ y=%d' % (ly, ny))

    if os.path.exists(OUT_FINAL):
        shutil.copy2(OUT_FINAL, OUT_FINAL.replace('.png', '_bak.png'))
    canvas.save(OUT_FINAL, dpi=(300, 300))
    assert Image.open(OUT_FINAL).size == (W, H)
    print('[saved] %s  %dx%d' % (OUT_FINAL, W, H))


main()
