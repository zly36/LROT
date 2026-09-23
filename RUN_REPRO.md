# LROT code package — running and reproduction guide

This package accompanies the paper **LROT: ligand–receptor-guided optimal transport for spatial
transcriptomics slice alignment**. It is meant to let a reviewer reproduce the figures and key
numbers of the paper **without modifying a single path**.

**Archive DOI**: <https://doi.org/10.5281/zenodo.22908788> (the v1.0.3 snapshot); the concept DOI
<https://doi.org/10.5281/zenodo.22898129> always resolves to the latest version.

> Chinese version: [`RUN_REPRO_zh.md`](RUN_REPRO_zh.md).

---

## 0. In one line

```
pip install -r requirements.txt
python tools/verify_env.py                     # (1) self-check: interpreter / dependencies / self-location / inputs / minimal reproduction
python experiments/dlpfc_layer_ratio.py        # (2) example: reproduce Figure S13
```

`tools/verify_env.py` checks, in order: the interpreter and the versions of 12 key packages; whether
repository self-location points at this package; whether the Tier 1 inputs required for plotting are
in place. It then **actually runs two second-scale scripts and compares them byte by byte**. It exits
with code 0 when everything passes; otherwise it states plainly what is missing and how to supply it.

* Every script locates the repository root by searching upwards for `lrot_core.py`, so the package **runs from any path**;
* The **Tier 1** intermediate results are included, so **the great majority of figures need no re-run of the experiments** (minutes at most);
* Steps requiring raw data or oversized intermediate arrays are marked **Tier 2** in §3 (measured runtimes included).

## 1. Directory structure

```
LROT_code/
├── README.md                 English overview
├── README_zh.md              Chinese overview
├── RUN_REPRO.md              this file: per-figure reproduction checklist (commands + expected results + known deviations)
├── RUN_REPRO_zh.md           Chinese reproduction checklist
├── requirements.txt          Python dependencies (versions measured locally)
├── CITATION.cff              citation metadata (CFF 1.2.0)
├── LICENSE                   MIT
├── MANIFEST.sha256           checksums of every file
├── MANIFEST_size.txt         file list with sizes
├── lrot_core.py              core library: LR strength matrix, FGW/LROT solvers, metrics
├── real_st_loader.py         real / realistic data loading and generation
├── lrot_families.py          LR-family grouping (authoritative source; DB completeness checked on import)
├── lrot_paths.py             unified path resolution (ROOT/OUT/FIG/NPZ + find_input multi-candidate lookup)
├── experiments/              experiment scripts (results written to lrot_output/)
│   └── supp/               Figure S17/S18 reproduction scripts, DLPFC data download script
├── figures/                  plotting scripts (including fig_crop_title.py)
├── tools/                    LR database construction and similar
├── data/                     LR database sources (CellChatDB.mouse.rda)
├── cancer_data/  dlpfc_data/ ← empty; place the downloaded raw data here (see §2)
└── lrot_output/
    ├── figures_png/          paper and supplementary figures plus their parts (52 PNGs, 27 embedded directly in the docx)
    ├── *.txt / *.csv / *.json numerical results
    ├── 数据_npz/             Tier 1 intermediate arrays (inputs to the plotting scripts)
    ├── dlpfc_donor2/        donor-2 replication results (Figure S18)
    ├── 验证_CellChatDB重建/   LR-database rebuild and verification record
    └── _中间产物/
        ├── 结果txt/            results.txt needed by Figure 9
        ├── S17_support/        the 3 JSON files needed by Figure S17
        └── _img_orig/          the raw panels needed to compose Figure S14
```

## 2. Environment and data

### 2.1 Python environment

```
python -m pip install -r requirements.txt
```

Measured locally: Python 3.10.11 (Windows 11 / AMD64).

> ⚠ **Do not use a Python 3.12 environment** (such as the Anaconda default base). Measured with
> `matplotlib 3.8.4`, `figures/gen_paste2_multiseed_figure.py` fails immediately with
> `Axes.boxplot() got an unexpected keyword argument 'tick_labels'`, and the figure it produces
> differs from the paper version.

The three external baselines are **separate repositories** and must be installed following their own
instructions (their source code is not included here):

| Baseline | Package / repository | Used for |
|---|---|---|
| Official PASTE | `POT` + the PASTE source (imported under a renamed module) | Figure 4 / Figure S6 etc. |
| Official PASTE2 | `pip install paste2==1.0.1` | Figure S8 / Figure S17 |
| STAligner | `pip install STAligner==1.0.0` (or its source) | Figure S6 |

### 2.2 Raw data (needed only for Tier 2)

```bash
python experiments/supp/download_dlpfc_data.py     # DLPFC (spatialLIBD / HumanPilot)
```

| Dataset | How to obtain | Target location |
|---|---|---|
| Mouse brain Visium CytAssist | 10x Genomics website, `Visium_Mouse_Olfactory_Bulb` and similar | `cancer_data/mouse_brain.h5` |
| Human breast cancer Visium Block A | 10x Genomics website, `V1_Breast_Cancer_Block_A_Section_1/2` | `cancer_data/V1_Breast_Cancer_Block_A_Section_*.h5` |
| DLPFC (spatialLIBD) | `experiments/supp/download_dlpfc_data.py` | `dlpfc_data/` |
| GO ontology | `go-basic.obo`, `mgi.gaf.gz` (Gene Ontology website) | `data/go/` |

## 3. Reproducing the figures (Tier 1: bundled results, minutes)

The finished figures in `lrot_output/figures_png/` cover every figure of the paper and its
supplementary material and can be used directly for comparison: 27 of them are pixel-identical to
the images embedded in the docx, and the rest are either parts of composite figures (to be assembled
by `figures/rebuild_*.py`) or downsampled masters of main-text figures.
The table below records the **measured** status (pixel-level comparison):

| Figure | In paper as | Generating script | Measured status | Notes |
|---|---|---|---|---|
| Fig. 1 | image1 | `figures/gen_method_flow.py` | pixel-identical | |
| Fig. 2 | image2 | `experiments/lrot_prototype.py` | not identical | the script needs lrot_synthetic_results.txt (shipped in lrot_output/); the paper version, 5220x2973, is a downsample of lrot_full_panel.png (9482x5359), whereas a re-run gives 9486x5572 |
| Fig. 3 | image3 | `experiments/lrot_real_data.py` | not identical | the paper version is the re-run figure with the top 280 px overall-title band removed; a layout difference remains after cropping |
| Fig. 4 | image4 | `figures/gen_sota_figure.py` | not identical | re-run is 84 px taller; a 6.2% pixel difference remains after cropping (annotations and layout belong to a different revision) |
| Fig. 5 | image5 | `experiments/lrot_real_data.py` | not identical | dpi=380 preserved; re-run is 52 px taller, a difference remains after cropping |
| Fig. 6 | image6 | `experiments/_render_mouse_visium_fig6.py` | not identical | re-run gives 5200x1631 vs the paper 5154x1546 |
| Fig. 7 | image7 | `figures/gen_fig7_composite.py` | pixel-identical | |
| Fig. 8 | image42 | `experiments/_render_breast_realcoord_official.py` | not identical | a re-run yields the 11337x6537 master (i.e. figures_png/lrot_breast_cancer_realcoord.png); the paper version, 5220x2939, is a downsample of it |
| Fig. 9 | image11 | `experiments/_render_dlpfc_fig9.py` | not identical | the script needs lrot_dlpfc_results.txt (shipped in _中间产物/结果txt/); the paper version, 4950x2840, is a downsample of figures_png/lrot_dlpfc.png (17007x9735) |
| Fig. 10 | image12 | `experiments/run_dlpfc_3d.py` | not identical | dpi=240 preserved; re-run is 127 px taller, a difference remains after cropping |
| Fig. 11 | image13 | `experiments/run_lrot_entropy_calibration.py` | not identical | dpi=270 preserved; re-run is 176 px taller, a difference remains after cropping |
| Fig. S1 | image14 | `experiments/lrot_prototype.py` | not identical | as for Fig. 2: the script writes into figures_png, same size but different content |
| Fig. S2 | image15 | `figures/rebuild_supp_composites.py` | pixel-identical | |
| Fig. S3 | image37 | `experiments/go_enrichment_fig.py` | pixel-identical | |
| Fig. S4 | image17 | `figures/rebuild_supp_composites.py` | pixel-identical | |
| Fig. S5 | image38 | `experiments/lr_parameter_sensitivity.py` | not identical | re-run 3600x1440 vs the paper 3600x1220 (220 px taller) |
| Fig. S6 | image19 | `figures/polish_three_figures.py` | pixel-identical | the Chinese and English versions come from the same script; the Chinese version also matches the canonical file on disk |
| Fig. S7 | image21 | `figures/gen_scaling_from_archive.py` | pixel-identical | |
| Fig. S8 | image22 | `figures/gen_paste2_multiseed_figure.py` | title-free on re-run (99.77%) | dpi=540 preserved; on 2026-09-21 the figure-level ax.set_title was removed and a rect=[0,0,1,0.957] placeholder added, so the re-run output is 99.77% pixel-identical to the paper version (the residual is sub-pixel text anti-aliasing); the original paper figure is unchanged |
| Fig. S9 | image23 | `figures/polish_three_figures.py` | pixel-identical | |
| Fig. S10 | image43 | `experiments/run_cancer_calibration_truecoord.py` | not identical | dpi=360 preserved; re-run is 105 px taller, a difference remains after cropping |
| Fig. S11 | image25 | `figures/rebuild_supp_composites.py` | pixel-identical | |
| Fig. S12 | image35 | `experiments/official_paste_convergence.py` | not identical | dpi=360 preserved; re-run is 104 px taller, a difference remains after cropping |
| Fig. S13 | image29 | `experiments/dlpfc_layer_ratio.py` | title-free on re-run and pixel-identical | dpi=360 preserved; the suptitle serves only as a layout placeholder and is cropped after rendering by crop_baked_title (expect_title=True, which raises an error if no title is found); after cropping 194 px the result is pixel-identical to the original paper figure |
| Fig. S14 | image44 | `figures/rebuild_s14.py` | pixel-identical | |
| Fig. S15 | image45 | `figures/rebuild_supp_composites.py` | pixel-identical | |
| Fig. S16 | image36 | `experiments/lr_gene_set_fairness.py` | not identical | dpi=360 preserved; re-run is 105 px taller, a 7.7% pixel difference remains after cropping |
| Fig. S17 | image40 | `experiments/supp/_supp_s17_plot.py` | not identical | the script needs 3 JSON files (shipped in lrot_output/_中间产物/S17_support/, produced by experiments/supp/_supp_s17_checks.py); re-run 4604x2730 vs the paper 3936x2367 |
| Fig. S18 | image41 | `experiments/supp/_supp_s18_plot.py` | not identical | re-run 4770x2789 vs the paper 4131x2410 |
| Fig. S19 | image46 | `experiments/_render_dlpfc_fullscale_tissue.py` | not run this time | full-scale DLPFC (4221x4381) solve, very time-consuming; the input lrot_data_dlpfc_fullscale.npz is present |

**Totals: 11 pixel-identical plus 2 that match after the extra top-crop step = 13/30 reproducible in one command.**
The other 16 keep the **same numerical convention** and differ only in layout and annotations (a
different revision); one full-scale figure was not run because of its runtime.

### 3.1 Top-title cropping (now baked into the scripts)
The journal requires that a figure contain no overall title. The historical approach was to crop
after rendering, which is why a single script run produced a figure different from the paper
version. That step is now `figures/fig_crop_title.py:crop_baked_title()`, called by the plotting
scripts after `savefig`:

* the rule is unchanged: the first band of ink within the top 15% is taken to be the title band, the first run of at least 12 blank rows below it is located, and the image is cropped to `max(bottom edge of the title, start of panel content − 50 px)`;
* **the original dpi is preserved** (never hard-coded to 300 — paper figures are rendered at 360 dpi, and the journal requires at least 600 dpi for line art and 300 dpi for colour);
* if no title band is found the function **raises an error** rather than skipping silently, so a titled figure cannot slip into the paper.

Example: one run of `experiments/dlpfc_layer_ratio.py` gives `3919×1442 @360 dpi`, pixel-identical to the paper `image29.png`.

### 3.2 Clean-room run (2026-09-21)
The package was copied to **a different drive under an all-ASCII path**, `D:\repro_test\LROT_code`, and 17 scripts were run there with the local interpreter:

* **15/17 ran successfully.** Both failures were caused by missing **raw data or an upstream result** (Tier 2, not a code problem):

| Script | Figure | Missing |
|---|---|---|
| `experiments/dlpfc_layer_ratio.py` | Fig. S13 | DLPFC raw data such as `dlpfc_data/151507_filtered_feature_bc_matrix.h5` (download per §2.2) |
| `experiments/go_enrichment_fig.py` | Fig. S3 | requires a prior run of `experiments/go_enrichment.py` and a populated `data/go/` |

* The whole package was then compared with the shipped original, file by file, by sha256:

| Result | Count | Comment |
|---|---|---|
| byte-identical | **239 / 240** | paper figures and result files reproduced byte for byte **from a different path** |
| byte-different | 1 | `figures_png/lrot_paste2_multiseed.png` (Fig. S8), a 0.1% difference from sub-pixel text anti-aliasing |
| new files | 16 | re-run outputs written into the `lrot_output/` root (side by side with the paper versions in `figures_png/`, for easy comparison), plus `__pycache__` |
| missing | 0 | — |

> Conclusion: **the path problem is gone** — unpacking anywhere needs no code edit, and most figures reproduce at byte level.

## 4. Reproducing the numerical results (Tier 2: raw data required; measured runtimes)
Main result scripts in the order of the paper. Values after `→` are **measured reproduction results**
(local machine: Intel i5-11300H / 8 threads / 16 GB).

| Step | Command | Measured time | Output and checkpoints |
|---|---|---|---|
| Synthetic main experiment | `python experiments/lrot_prototype.py` | a few minutes | `lrot_output/lrot_synthetic_results.txt`, Fig. 2 / Fig. S1 |
| Realistic data | `python experiments/lrot_real_data.py` | a few minutes | inputs to Fig. 3 / Fig. 5 |
| Ablation and statistics | `python experiments/lrot_control_experiment.py`, `experiments/pathway_significance_test.py` | minutes | Table 1, Fig. S5 |
| Mouse brain Visium | `python experiments/run_real_visium.py` | minutes | Fig. 6 |
| Human breast cancer | `python experiments/run_cancer_visium_truecoord.py` | 171 s | FGW entropy 5.8465 / LROT 5.8176 (−0.49%); cost 0.1144→0.1143; 10/68 effective LR pairs; 9 iterations; `lrot_data_breast_cancer_truecoord.npz` (186 MB, not shipped) |
| DLPFC main experiment | `python experiments/run_dlpfc_visium.py` | minutes | Fig. 9, Fig. S13 |
| DLPFC full scale | `python experiments/run_dlpfc_full_scale.py` | 178–194 s per pair, peak about 4.3 GB | 14.4% entropy reduction; `lrot_data_dlpfc_fullscale.npz` (141 MB, not shipped) |
| DLPFC, two donors | `python experiments/run_dlpfc_visium.py` (donor 2) / `experiments/supp/_supp_s18_dlpfc_donor2.py` | minutes | Fig. S18 |
| 3D stacked reconstruction | `python experiments/lrot_3d_reconstruction.py`, `experiments/lrot_3d_multi_seed.py` | minutes | Fig. 7 |
| Ablation: weights / preprocessing | `python experiments/lr_weight_sensitivity.py`, `experiments/lr_parameter_sensitivity.py`, `experiments/preproc_sensitivity.py` | minutes | Fig. S5, Fig. S11, Supplementary Table S15 |
| Baselines | `python experiments/run_dlpfc_baselines.py`, `experiments/official_paste_convergence.py`, `experiments/lr_cellchat_comparison.py` | minutes (external baselines from §2.1 required) | Fig. 4, Fig. S2, Fig. S6, Fig. S12 |
| Enrichment and pathways | `python experiments/go_enrichment.py`, `experiments/pathway_activity_dlpfc.py` | minutes (`data/go/` required) | Fig. S3, Fig. S4 |
| LR database tables | `python experiments/lr_db_table.py` | seconds | Supplementary Tables S11–S13; **measured 101/101 outputs byte-identical** |
| Runtime and scaling | `python experiments/run_runtime_benchmark.py`, `figures/gen_scaling_from_archive.py` | minutes | Fig. S6, Fig. S7 |

> Note: the `*_results.txt` files in `lrot_output/` are shipped and can be diffed **line by line**
> against a re-run, without repeating the experiments first.

## 5. Known deviations (an honest list)

1. **Wall-clock time**: the runtimes recorded in `lrot_output/lrot_cancer_truecoord_results.txt` (43.7 s → 68.8 s and so on) depend on machine load and are **not a reproducible quantity**; every other value (entropy, cost, effective pair count, iteration count) is identical digit for digit.
2. **16 of 30 figures differ in layout**: a re-run yields **another revision** under the same numerical convention; the difference from the paper version is font, margins and annotation placement, not values (see the "Measured status" column of the §3 table).
3. **The paper version may be a downsampled master**: the paper versions of Fig. 2 / Fig. 8 / Fig. 9 were obtained by downsampling an extra-large master, so a direct re-run produces a larger original size.
4. **The dpi discrepancy is fixed**: the top-crop step in early scripts silently reduced 360 dpi to 300 dpi; it now keeps the original dpi.

## 6. Self-checking

```bash
# verify package integrity
python -c "import hashlib,pathlib;[print(hashlib.sha256(p.read_bytes()).hexdigest(), p.as_posix()) for p in pathlib.Path('.').rglob('*') if p.is_file()]" > /tmp/now.txt
# compare against MANIFEST.sha256 (order does not matter; a line-by-line diff is enough)

# figures vs. paper: lrot_output/figures_png/*.png should be pixel-identical to the media embedded in the docx
```

**Line endings**: the package is stored with `* text=auto eol=lf` and with binary types explicitly
marked in `.gitattributes`, so regardless of platform and of the `core.autocrlf` setting the bytes
checked out by `git clone` match `MANIFEST.sha256`.

**Numerical checking**: after re-running any `experiments/*.py`, the resulting
`lrot_output/*_results.txt` should agree numerically with the shipped file of the same name
(timing fields excepted).

## 7. Frequently asked questions

**Q1: The editor shows red squiggles under the scripts, but they run fine?**

That is **Pylance static analysis**, not a runtime error. The usual causes:

* scripts under `experiments/` and `figures/` add the repository root at runtime with `sys.path.insert`, which static analysis cannot see, so it reports `Import "lrot_3d_multi_seed" could not be resolved`;
* the interpreter selected in the editor is not the one where the dependencies were installed (often because an empty `.venv` is left in the directory).

Fix: (1) run `Python: Select Interpreter` from the command palette and choose the right interpreter; (2) if warnings persist, run `Developer: Reload Window` (or Pylance's `Restart Language Server`) to refresh the cache. **Do not judge the environment by squiggles** — use `python tools/verify_env.py`, which reaches its verdict by actually running code.

**Q2: The script printed tables of numbers but produced no figure?**

`experiments/*.py` are experiment scripts: they compute values and print tables, and only draw and
save the figure at the **last step**. `figures/*.py` are the scripts that produce a figure directly,
within seconds. The output path is printed by the script (`图已保存: ...`, "figure saved: ...").

**Q3: Why can't I see images in the terminal?**

A terminal outputs text only. PNGs are written to disk; open them with an image viewer.
`lrot_output/figures_png/` already contains the paper versions, so you can inspect them without
re-running anything.

## 8. License

See `LICENSE`.
