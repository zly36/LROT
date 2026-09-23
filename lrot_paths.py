# -*- coding: utf-8 -*-
"""LROT 统一路径解析（自定位，不写死绝对路径）。

用法
----
    ROOT     仓库根目录（= 本文件所在目录），可用环境变量 LROT_ROOT 覆盖
    OUT      产出目录（默认 ROOT/lrot_output），可用 LROT_OUT 覆盖
    FIG      OUT/figures_png
    NPZ      OUT/数据_npz
    find_input(name, *extra_dirs)
             按候选目录顺序查找输入文件；找不到时抛错并列出所有已查找位置。

使用时只需：
    1) 把代码包解压到任意路径（例如 D:\\LROT_code）
    2) 把下载的数据放到 ROOT/data_raw/（见 README.md 与 data/README.md）
    3) 其余路径全部自定位，无需改任何脚本
如数据放在别处，设置环境变量即可，例如
    set LROT_ROOT=D:\\LROT_code
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('LROT_ROOT', Path(__file__).resolve().parent)).resolve()
OUT = Path(os.environ.get('LROT_OUT', ROOT / 'lrot_output')).resolve()
FIG = OUT / 'figures_png'
NPZ = OUT / '数据_npz'

# 输入可能落在的次级目录（部分产物按主题分子目录存放）
_SEARCH = [
    OUT,
    NPZ,
    FIG,
    OUT / '_中间产物',
    OUT / '_中间产物' / '结果txt',
    OUT / '_中间产物' / 'S17_support',
    OUT / '中间产物',
    ROOT,
    ROOT / 'data',
    ROOT / 'cancer_data',
    ROOT / 'dlpfc_data',
    ROOT / 'data_raw',
]

ROOT_S, OUT_S, FIG_S, NPZ_S = str(ROOT), str(OUT), str(FIG), str(NPZ)


def _all_dirs():
    """候选项：固定候选 + 递归扫描 lrot_output/ 下两层子目录。"""
    dirs = list(_SEARCH)

    def sweep(base):
        try:
            subs = sorted(d for d in base.iterdir() if d.is_dir())
        except OSError:
            return
        for d in subs:
            dirs.append(d)
            sweep(d)

    sweep(OUT)
    seen, uniq = set(), []
    for d in dirs:
        if str(d) not in seen:
            seen.add(str(d))
            uniq.append(d)
    return uniq


def find_input(name, *extra_dirs):
    """按候选目录顺序查找输入文件，返回 Path。找不到时抛出带检索清单的 FileNotFoundError。"""
    tried = []
    for d in [Path(x) for x in extra_dirs] + _all_dirs():
        p = d / name
        tried.append(str(p))
        if p.exists():
            if p != Path(name):
                print('  [INPUT] %s -> %s' % (name, p))
            return p
    raise FileNotFoundError(
        '找不到输入文件 %s，已查找以下位置：\n  %s\n'
        '请确认数据已下载（见 data/README.md），或用环境变量 LROT_ROOT / LROT_OUT 指定位置。'
        % (name, '\n  '.join(tried)))


def ensure_out(*sub):
    """确保输出目录存在并返回（Path）。"""
    p = OUT.joinpath(*sub) if sub else OUT
    p.mkdir(parents=True, exist_ok=True)
    return p
