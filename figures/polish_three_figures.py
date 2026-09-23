# -*- coding: utf-8 -*-
"""
polish_three_figures.py — 重绘三张图的版式（不改语义/数值口径）。

调整说明
--------
① 图1 `lrot_cancer_ablation_scan_truecoord.png`（乳腺癌通路消融谱）
   改动前: 竖直柱 + 12 个长家族名旋转 45°；11 个 Δ≈0 的数值标签画在同一高度。
   改动后: 横向条形图，家族名水平排列不旋转；每个数值标签独占一行；
           颜色为 绿=正贡献 / 红=干扰 / 灰=中性；加图例与「灰=中性 |Δ|≤0.015」说明。

② 图2 `lrot_dlpfc_full_vs_sampled.png`（DLPFC 全量 vs 采样）
   改动前: (A) 的 −17.6%/−14.4% 与柱顶数值标签挤在一起；(C) 对数轴上两个柱标签都显示成 "5"。
   改动后: 抬高留白（ylim ×1.35），降幅标注放到更高处；(C) 标签改用一位小数区分。
           (C) 的耗值与内存一律取自权威结果文件，使图—表—正文同源：
           全量 FGW 307.5 s / LROT 225.4 s、峰值内存 4.4 GB
           （`lrot_output/dlpfc_full_scale_results.txt`）；
           采样 FGW 10.2 s / LROT 18.5 s（补充表 S9，与 (A)(B) 面板为同一次运行）。

③ 图3 `lrot_runtime_comparison.png`（运行时间对比）
   改动前: 两个加速比标注框压在柱体/轴标签上。
   改动后: 标注移到柱子上方的空白区，不画穿柱箭头；抬高 ylim 留白；x 轴标签不再挤。

保持长宽比（关键）
------------------
改动后 figsize 按**原 PNG 像素尺寸/300dpi** 设定，且**不加 bbox_inches='tight'**
⇒ 输出像素与长宽比与改动前一致，docx 里不会被拉伸变形。
"""
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'lrot_output', 'figures_png')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.facecolor'] = 'white'

FGW_C = '#E68A8A'      # 红 - FGW
LROT_C = '#2E8B57'     # 绿 - LROT
GRAY_C = '#888888'


def finish(fig, path, px_w, dpi=300):
    """按原图像素宽定 figsize，保存时不用 bbox_inches（保住长宽比）。"""
    fig.savefig(path, dpi=dpi)
    from PIL import Image
    w, h = Image.open(path).size
    print('  saved %-46s %4dx%-5d (原 %d px 宽)' % (os.path.basename(path), w, h, px_w))
    plt.close(fig)


# ============================================================ ① 乳腺癌通路消融谱
def fig_ablation():
    names = ['ERBB2/HER', 'Cell Adhesion', 'Semaphorin', 'Neurotrophin/Growth',
             'Ephrin', 'Notch', 'Slit/Robo', 'Hedgehog', 'Wnt', 'BMP',
             'Reelin', 'Chemokine/Cytokine']
    ds = [0.480, 0.277, 0.015, 0.013, 0.007, 0.003, 0.001, 0.000, 0.000, 0.000,
          -0.000, -0.004]
    cols = ['#4DAF4A' if d < -0.1 else ('#D6604D' if d > 0.1 else GRAY_C) for d in ds]

    # 原 PNG 2791x1342 → figsize = 2791/300, 1342/300
    fig, ax = plt.subplots(figsize=(2791 / 300.0, 1342 / 300.0))
    y = np.arange(len(names))[::-1]
    ax.barh(y, ds, color=cols, alpha=0.9, height=0.62,
            edgecolor='#333333', lw=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.axvline(0, color='black', lw=0.9, zorder=4)
    for yi, d in zip(y, ds):
        off = 0.008 if d >= 0 else -0.008
        ax.text(d + off, yi, '%+.3f' % d, va='center',
                ha='left' if d >= 0 else 'right', fontsize=8, zorder=5)
    ax.set_xlim(-0.075, 0.575)
    ax.set_ylim(-0.75, len(names) - 0.25)
    ax.set_xlabel('\u0394 entropy reduction vs full library (%)', fontsize=9.5)
    ax.set_title('(B) Breast-cancer pathway ablation spectrum '
                 '(leave-one-family-out)', fontsize=10.5, pad=8)
    ax.grid(True, axis='x', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=[Patch(facecolor='#4DAF4A', label='reduction weakens (positive)'),
                       Patch(facecolor='#D6604D', label='interference'),
                       Patch(facecolor=GRAY_C, label='neutral (|\u0394| \u2264 0.015)')],
              fontsize=7.5, loc='lower right', frameon=False)
    ax.text(0.995, 0.055, 'n = 12 families; full-library reduction 0.494%',
            transform=ax.transAxes, ha='right', fontsize=7.5, color='#555555')
    fig.tight_layout()
    finish(fig, os.path.join(OUT, 'lrot_cancer_ablation_scan_truecoord.png'), 2791)


# ==================================================== ② DLPFC 全量 vs 采样
def fig_dlpfc():
    ent = {'s': (6.0119, 4.9524), 'f': (7.4938, 6.4135)}
    same = {'s': (0.3766, 0.3742), 'f': (0.3882, 0.3886)}
    # 全量耗时取自 lrot_output/dlpfc_full_scale_results.txt（307.5 / 225.4 s）
    times = {'s': (10.2, 18.5), 'f': (307.5, 225.4)}

    fig, axes = plt.subplots(1, 3, figsize=(3865 / 300.0, 1257 / 300.0))
    groups = ['Sampled\n(1,000 spots)', 'Full-scale\n(4,221/4,381)']

    def panel(ax, data, ylab, fmt, log=False, annot=None, title='',
              head=1.35, w=0.34):
        x = np.arange(2)
        fgw = [data[k][0] for k in ('s', 'f')]
        lrot = [data[k][1] for k in ('s', 'f')]
        ax.bar(x - w / 2, fgw, w, label='FGW (\u03b3=0)', color=FGW_C,
               alpha=0.9, edgecolor='#333333', lw=0.6, zorder=3)
        ax.bar(x + w / 2, lrot, w, label='LROT (\u03b3=0.1)', color=LROT_C,
               alpha=0.9, edgecolor='#333333', lw=0.6, zorder=3)
        for xi, (fv, lv) in enumerate(zip(fgw, lrot)):
            ax.text(xi - w / 2, fv * 1.03, fmt.format(fv), ha='center',
                    va='bottom', fontsize=8.5, zorder=4)
            ax.text(xi + w / 2, lv * 1.03, fmt.format(lv), ha='center',
                    va='bottom', fontsize=8.5, zorder=4)
        if annot:
            for xi, txt in annot.items():
                top = max(fgw[xi], lrot[xi])
                ax.text(xi, top * 1.22, txt, ha='center', fontsize=9.5,
                        color='#B22222', fontweight='bold', zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=10, pad=6)
        if log:
            ax.set_yscale('log')
            ax.set_ylim(1, 1500)          # 抬高留白，避免 178/194 顶到边框
        else:
            ax.set_ylim(0, max(fgw + lrot) * head)
        ax.grid(True, axis='y', alpha=0.25, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    panel(axes[0], ent, 'Transport entropy', '{:.2f}', annot={0: '\u221217.6%', 1: '\u221214.4%'},
          title='(A) Transport entropy')
    axes[0].legend(fontsize=8, loc='upper left', frameon=False)
    panel(axes[1], same, 'Same-layer mass', '{:.3f}', title='(B) Layer concordance')
    panel(axes[2], times, 'Runtime (s, log scale)', '{:.1f}', log=True,
          title='(C) Runtime (peak mem. 4.4 GB)')

    fig.tight_layout(pad=1.1)
    finish(fig, os.path.join(OUT, 'lrot_dlpfc_full_vs_sampled.png'), 3865)


# ============================================================ ③ 运行时间对比
def fig_runtime():
    times = [3.20, 3.96, 8.58, 2.91, 171.3]
    colors = [LROT_C, FGW_C, '#D67236', '#B08BC9', '#8B1A1A']
    for lang, sub in (('cn', ['(γ=0.1)\n软FGW+LR', '(γ=0)\n软FGW', '\nCG硬匹配',
                              '\n部分FGW', '\n(训练)\n深度训练']),
                      ('en', ['(γ=0.1)\nsoft FGW+LR', '(γ=0)\nsoft FGW', '\nCG hard matching',
                              '\npartial FGW', '\n(training)\ndeep learning'])):
        methods = ['LROT' + ('\n' if lang == 'cn' else '\n') + sub[0],
                   'FGW' + ('\n' if lang == 'cn' else '\n') + sub[1],
                   'Official\nPASTE' + sub[2], 'Official\nPASTE2' + sub[3],
                   'STAligner' + sub[4]]
        fig, ax = plt.subplots(figsize=(3840 / 300.0, 2237 / 300.0))
        bars = ax.bar(methods, times, color=colors, edgecolor='gray',
                      linewidth=0.8, width=0.62, zorder=3)
        ax.set_yscale('log')
        ax.set_ylim(1, 3000)                     # 抬高留白，给标注腾地方
        ax.set_ylabel('Runtime (s, log scale) \u2193', fontsize=11)
        ax.grid(axis='y', which='major', linestyle='--', alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.10,
                    '%.1fs' % v, ha='center', fontsize=10.5, fontweight='bold', zorder=4)
        ax.tick_params(axis='x', labelsize=9)
        # 加速比标注：放在柱子上方的空白区，不画穿柱箭头；比例由数据推导
        # ⇒ 官方 PASTE 8.58/3.20 与 STAligner 171.3/3.20 分别=2.7 与 53.5，
        #   与图S6 题注「慢2.7倍 / 慢约54倍」一致。
        ax.text(2, 260, '%.1f\u00d7 faster' % (times[2] / times[0]), ha='center',
                fontsize=10.5, color=LROT_C, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor=LROT_C, alpha=0.95), zorder=5)
        ax.text(4, 900, '%.0f\u00d7 faster' % (times[4] / times[0]), ha='center',
                fontsize=10.5, color='#8B1A1A', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='#8B1A1A', alpha=0.95), zorder=5)
        for xi in (2, 4):
            top = times[xi] * 1.12
            ax.annotate('', xy=(xi, top if xi == 4 else 8.58 * 1.06),
                        xytext=(xi, 250 if xi == 2 else 880),
                        arrowprops=dict(arrowstyle='-', color='#999999',
                                        lw=0.9, linestyle=':'), zorder=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        name = 'lrot_runtime_comparison%s.png' % ('' if lang == 'cn' else '_en')
        finish(fig, os.path.join(OUT, name), 3840)


if __name__ == '__main__':
    print('① 乳腺癌通路消融谱（横条图）')
    fig_ablation()
    print('② DLPFC 全量 vs 采样')
    fig_dlpfc()
    print('③ 运行时间对比（中/英两版）')
    fig_runtime()
