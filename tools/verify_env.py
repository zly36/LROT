# -*- coding: utf-8 -*-
"""verify_env.py — 一键环境自检 + 最小复现验证。

用途：解压后**先跑这一条**，即可确认环境是否齐备、能否复现图件。

    python -X utf8 tools/verify_env.py

检查项
------
1. 解释器与关键包版本（与补充表 S14 / requirements.txt 对比）
2. 仓库自定位是否指向本包（而非包外的其他副本）
3. 绘图脚本所需的 Tier 1 输入是否就位
4. 实跑 2 个秒级脚本，并逐字节比对产物是否与随包文件相同

退出码 0 = 全部通过；非 0 = 有缺项（会列出缺什么、怎么补）。
"""
import hashlib
import importlib.metadata
import importlib.util
import io
import os
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import lrot_paths as P                                                        # noqa: E402

FAIL = []

# ---------- 1. 解释器与依赖 ----------
print('=' * 78)
print('1. 解释器与依赖')
print('=' * 78)
print('  解释器      %s' % sys.executable)
print('  Python      %s' % sys.version.split()[0])
if sys.version_info[:2] != (3, 10):
    print('  ⚠ 本包在本机实测于 Python 3.10.11；其他版本可能因 matplotlib/scanpy '
          'API 变化导致个别脚本失败（尤其不要用 3.12）。')

NEED = {
    'numpy': '2.2.6', 'scipy': '1.15.3', 'matplotlib': '3.10.7', 'pandas': '2.3.3',
    'sklearn': '1.7.2', 'PIL': '12.0.0', 'h5py': '3.16.0', 'ot': '0.9.6.post1',
    'anndata': '0.11.4', 'scanpy': '1.11.5', 'networkx': '3.4.2', 'statsmodels': '0.14.6',
}
DIST = {'sklearn': 'scikit-learn', 'PIL': 'pillow', 'ot': 'POT'}
miss, drift = [], []
for mod, want in NEED.items():
    if importlib.util.find_spec(mod) is None:
        miss.append(mod)
        print('  ✘ %-14s 未安装（期望 %s）' % (mod, want))
        continue
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            __import__(mod)
            got = importlib.metadata.version(DIST.get(mod, mod))
    except Exception as e:                                     # noqa: BLE001
        print('  ✘ %-14s 导入失败: %s' % (mod, e))
        miss.append(mod)
        continue
    flag = '✔' if got == want else '⚠'
    if got != want:
        drift.append((mod, want, got))
    print('  %s %-14s %-12s（期望 %s）' % (flag, mod, got, want))
if miss:
    FAIL.append('缺包: %s —— 请先 `pip install -r requirements.txt`' % ', '.join(miss))
if drift:
    print('  ⚠ 版本与实测环境不同：%s。多数情况可运行，若出图不一致请按本行装回原版本。'
          % ', '.join('%s %s→%s' % (m, w, g) for m, w, g in drift))

# ---------- 2. 自定位 ----------
print('')
print('=' * 78)
print('2. 仓库自定位')
print('=' * 78)
print('  ROOT        %s' % P.ROOT)
print('  OUT         %s' % P.OUT)
if not (P.ROOT / 'lrot_core.py').exists():
    FAIL.append('ROOT 下没有 lrot_core.py，包可能不完整')
if not (P.ROOT / 'lrot_output').is_dir():
    FAIL.append('ROOT 下没有 lrot_output/，结果目录缺失')
print('  ✔ ROOT 内已含 lrot_core.py 与 lrot_output/，解压路径无关')

# ---------- 3. Tier 1 输入 ----------
print('')
print('=' * 78)
print('3. Tier 1 输入（绘图脚本所需）')
print('=' * 78)
TIER1 = ['lrot_data_dlpfc.npz', 'lrot_data_real_visium.npz', 'lrot_data_synthetic.npz',
         'lrot_data_realistic.npz', 'lrot_dlpfc_results.txt', 'lrot_synthetic_results.txt',
         'baseline_ceiling_sweep.json', 'eps_sensitivity.json', 'lr_program_control.json']
for name in TIER1:
    try:
        p = P.find_input(name)
        print('  ✔ %-32s %s' % (name, p))
    except FileNotFoundError:
        FAIL.append('找不到 %s（Tier 1 输入）' % name)
        print('  ✘ %-32s 未找到' % name)

# ---------- 4. 最小复现 ----------
print('')
print('=' * 78)
print('4. 最小复现（实跑 + 逐字节比对）')
print('=' * 78)
CASES = [('figures/gen_method_flow.py', 'lrot_output/figures_png/lrot_method_flow.png'),
         ('figures/gen_scaling_from_archive.py', 'lrot_output/figures_png/lrot_scaling.png')]
for script, rel in CASES:
    ap = P.ROOT / script
    target = P.ROOT / rel.replace('/', os.sep)
    before = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    r = subprocess.run([sys.executable, '-X', 'utf8', script], cwd=str(P.ROOT),
                       capture_output=True)
    if r.returncode != 0:
        FAIL.append('%s 运行失败' % script)
        print('  ✘ %-38s 退出码 %d' % (script, r.returncode))
        print('      %s' % r.stderr.decode('utf-8', 'replace').strip().split('\n')[-1][:100])
        continue
    after = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    same = (before == after)
    print('  %s %-38s 产物 %s' % ('✔' if same else '⚠', script,
                                  '字节完全相同' if same else '与随包文件不同（见 RUN_REPRO.md §5）'))

print('')
print('=' * 78)
if FAIL:
    print('结论：%d 项需要处理' % len(FAIL))
    for x in FAIL:
        print('  - %s' % x)
    print('=' * 78)
    sys.exit(1)
print('结论：环境与包自检全部通过，可以开始复现（见 RUN_REPRO.md）。')
print('=' * 78)
