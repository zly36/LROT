# LROT — ligand–receptor-guided optimal transport

Code and reproduction material for the spatial transcriptomics slice-alignment method LROT,
described in the paper **LROT: ligand–receptor-guided optimal transport for spatial
transcriptomics slice alignment**.

> **Start with [`RUN_REPRO.md`](RUN_REPRO.md)** — it covers environment setup, data acquisition,
> per-figure reproduction commands, expected values and known deviations.
>
> 中文版：[`README_zh.md`](README_zh.md) · [`RUN_REPRO_zh.md`](RUN_REPRO_zh.md)

## Quick start

```bash
pip install -r requirements.txt
python experiments/dlpfc_layer_ratio.py     # e.g. reproduce Figure S13
```

* Scripts locate the repository root by searching upwards for `lrot_core.py`, so the package **runs
  from any path** with no path editing;
* The Tier 1 intermediate results are shipped with the package, so **most figures do not require
  re-running the experiments** (minutes at most);
* Steps that need raw data or oversized intermediate arrays are marked **Tier 2** in
  `RUN_REPRO.md` §4, together with measured runtimes;
* Requires **Python 3.10** (measured with 3.10.11 on Windows 11). Raw datasets are downloaded from the
  public sources listed in `RUN_REPRO.md` §2 into `cancer_data/` and `dlpfc_data/`.

## Package contents

| Path | Contents |
|---|---|
| `lrot_core.py` / `real_st_loader.py` | Core library and data loading (LR strength matrix, FGW/LROT solvers, metrics) |
| `lrot_families.py` | Authoritative LR-family grouping (11 signalling families); LR-database completeness is checked on import |
| `lrot_paths.py` | Unified path resolution: `ROOT/OUT/FIG/NPZ` plus `find_input()` multi-candidate lookup |
| `experiments/` | Experiment scripts (results are written to `lrot_output/`) |
| `figures/` | Plotting scripts, including `fig_crop_title.py` (removal of baked-in top titles) |
| `tools/` | `verify_env.py` (environment and input check) and `build_cellchat_db.py` (LR-database construction) |
| `lrot_output/figures_png/` | 52 PNGs: paper and supplementary figures and their parts (27 of them embedded directly in the docx) |
| `lrot_output/*.txt / *.csv / *.json` | Numerical results, directly diffable line by line against a re-run |
| `data/` | Ligand–receptor database sources |

## Naming

* Python files are lowercase `snake_case` throughout — no version suffixes such as `_v2`;
* Scripts use the lowercase figure id (`experiments/supp/_supp_s17_plot.py` → Figure S17), while the PNG keeps the manuscript label (`lrot_supp_S17_*.png`), so each file maps one-to-one onto a supplementary figure;
* Numerical results are `<figure-or-experiment>_results.txt`.

## Confidence in the results (measured)

* `experiments/lr_db_table.py` / `figures/gen_scaling_from_archive.py`: **101/101 outputs byte-identical**
* `experiments/run_cancer_visium_truecoord.py`: all values reproduced; all 15 `.npz` arrays **bit-identical**
* `figures/gen_fig7_composite.py`: output **md5-identical** to the figure embedded in the paper
* `experiments/dlpfc_layer_ratio.py`: one run yields `3919×1442 @360 dpi`, **pixel-identical** to the paper figure
* Figure overview: 11 pixel-identical, plus 2 identical after the cropping step; the remaining 16 keep the same numerical convention and differ only in layout (a different revision)

## Citation and archive

* Code repository: <https://github.com/zly36/LROT>
* Zenodo archive (v1.0.4, permanent DOI): <https://doi.org/10.5281/zenodo.22915948>
* Concept DOI covering all versions: <https://doi.org/10.5281/zenodo.22898129>

## License

See [`LICENSE`](LICENSE).
