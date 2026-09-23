# -*- coding: utf-8 -*-
"""download_dlpfc_data.py — 下载 DLPFC 真实数据 (Maynard et al. 2021)
来源:
  - 表达矩阵 + 坐标: spatialLIBD 官方 AWS (spatial-dlpfc.s3.us-east-2.amazonaws.com)
  - 层标注: LieberInstitute/HumanPilot (barcode_level_layer_map.tsv)
下载到 dlpfc_data/ 供 run_dlpfc_*.py 使用
"""
import os
import requests
import sys

# 自定位仓库根：向上找到含 lrot_core.py 的目录（代码包解压到任意路径均可运行）
_R = os.path.dirname(os.path.abspath(__file__))
while _R != os.path.dirname(_R) and not os.path.exists(os.path.join(_R, 'lrot_core.py')):
    _R = os.path.dirname(_R)

DATA_DIR = os.path.join(_R, 'dlpfc_data')
os.makedirs(DATA_DIR, exist_ok=True)

SAMPLES = ["151507", "151508", "151509"]
BASE = "https://spatial-dlpfc.s3.us-east-2.amazonaws.com"
GH = "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def download(url, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  [skip] {os.path.basename(out_path)} 已存在")
        return True
    print(f"  [下载] {url}")
    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
        print(f"    -> {os.path.getsize(out_path) / 1e6:.1f} MB")
        return True
    except Exception as e:
        print(f"    !! 失败: {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 60)
    print("  DLPFC 数据下载 (Maynard et al. 2021, Br8325)")
    print("=" * 60, flush=True)

    ok = True
    for s in SAMPLES:
        # h5 表达矩阵 (AWS, 需 /h5/ 前缀)
        url = f"{BASE}/h5/{s}_filtered_feature_bc_matrix.h5"
        out = os.path.join(DATA_DIR, f"{s}_filtered_feature_bc_matrix.h5")
        ok &= download(url, out)
        # tissue_positions 坐标 (HumanPilot GitHub, 10X/{sample}/ 目录)
        url = f"{GH}/10X/{s}/tissue_positions_list.txt"
        out = os.path.join(DATA_DIR, f"{s}_tissue_positions_list.txt")
        ok &= download(url, out)

    # 层标注 (HumanPilot GitHub, 10X/barcode_level_layer_map.tsv)
    lm_out = os.path.join(DATA_DIR, "barcode_level_layer_map.tsv")
    if not (os.path.exists(lm_out) and os.path.getsize(lm_out) > 0):
        got = False
        for url in [f"{GH}/10X/barcode_level_layer_map.tsv"]:
            print(f"  [下载] {url}")
            try:
                with requests.get(url, headers=HEADERS, timeout=300) as r:
                    if r.status_code == 200:
                        with open(lm_out, "wb") as f:
                            f.write(r.content)
                        print(f"    -> {os.path.getsize(lm_out) / 1e3:.0f} KB")
                        got = True
                        break
                    else:
                        print(f"    !! HTTP {r.status_code}")
            except Exception as e:
                print(f"    !! {type(e).__name__}: {e}")
        if not got:
            print("  !! 层标注下载失败: 请手动获取 barcode_level_layer_map.tsv")
            ok = False

    print("=" * 60)
    if ok:
        print("  下载完成。dlpfc_data/ 内容:")
        for fn in sorted(os.listdir(DATA_DIR)):
            print(f"    - {fn} ({os.path.getsize(os.path.join(DATA_DIR, fn))/1e6:.1f} MB)")
    else:
        print("  部分文件下载失败，请检查网络后重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
