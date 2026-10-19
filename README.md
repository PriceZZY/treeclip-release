# TreeCLIP

*Do frozen 2D vision foundation-model features cluster individual-tree LiDAR point clouds by species more coherently than hand-crafted geometric descriptors — with no species labels, no training, and with acquisition-source confounding controlled?*

This repository contains the complete code, configuration, result files, and the pre-registration of the frozen 3D baseline for the paper:

> *Frozen 2D Foundation-Model Features Cluster Tree Species in LiDAR Point Clouds More Coherently Than the Evaluated Geometric Descriptors: A Source-Controlled Evaluation.* Accepted at *PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science* (Springer), in press. Preprint of the first version: [10.21203/rs.3.rs-10238136/v1](https://doi.org/10.21203/rs.3.rs-10238136/v1).

It mirrors the citable archive of the version used for the paper, [Zenodo 10.5281/zenodo.22216659](https://doi.org/10.5281/zenodo.22216659) (v2.0.0); the few differences are listed under [Changes from the archived release](#changes-from-the-archived-release).

## The pipeline (fully frozen; no training at any stage)

1. Each individual-tree point cloud is voxel-downsampled on a fixed 0.1 m grid (one point per occupied voxel), then — only if more than 4,096 points remain — uniformly subsampled to exactly 4,096 points with a fixed seed. Smaller clouds are kept as they are (a cap, not an exact size; no upsampling).
2. The cloud is rendered into **six PCA-aligned orthographic 224×224 depth views**. The renderer is a transparent projection: each pixel takes the **maximum normalized height** of the points that project into it, empty pixels are 0. There is no view-ray z-buffer; pixel intensity encodes height, not viewing distance.
3. Each view is encoded by a **frozen `open_clip` ViT-B/16 (OpenAI weights)** image tower → 512-d per view; the six views are L2-normalized and mean-fused into one 512-d embedding per tree. (DINOv2, supervised-ImageNet, and a random-initialized ViT run through the identical pipeline as controls.)
4. Embeddings are clustered with **KMeans at the true species count** (oracle-K; 15 seeds, mean ± s.d.), and scored with **matching-free NMI** (primary) and ARI, plus macro-F1/OA/mIoU under a post-hoc Hungarian mapping that maximises OA (it uses the evaluation labels, so these scores are optimistic, not deployable accuracy). A Leiden community-detection variant removes the oracle-K assumption as a robustness check.
5. **Source control is first-class.** Because acquisition source (sensor/campaign) is partly confounded with species, a global gap can include source structure, and controlling for source can shrink a gap or widen it. The primary estimate is the **within-source residual** (the advantage that survives when clustering is refit inside each acquisition block), backed by a per-block breakdown, a leave-one-source-out sweep, and a data-level bootstrap.

The CLIP text tower is never used: species structure is recovered by clustering, not text prompts.

## Baselines evaluated under the identical harness

- **geo-11d** — the whole-tree single-scale scale-free covariance/height descriptor (11-d).
- **Weinmann-local (PCA-11)** — the stronger local multi-scale variant: per-point eigenfeatures at neighbourhood sizes 10/20/40, averaged per tree, PCA-matched to 11-d.
- **OpenShape PointBERT** (ShapeNet-pretrained, frozen) — a frozen **3D** point-cloud foundation model given the native geometry directly; model choice, preprocessing, and reporting rules were **pre-registered before any 3D result was seen** (`PREREG_3d_baseline.md`; courtesy English translation in `PREREG_3d_baseline_EN.md`).

## Findings (short version)

- **Global picture** (matched 11-d, oracle-K): frozen CLIP features lead the single-scale descriptor by +0.173 / +0.249 NMI (FOR-species20K / BioDiv-3DTrees-ULS) but lead the stronger local multi-scale Weinmann baseline by only **+0.048 / +0.158**. Part of a global gap can reflect acquisition source rather than species, which is why the source-controlled residual below is the primary estimate. A within-species probe recovers the acquisition source from CLIP features (FOR-species20K, 11-d) with a balanced accuracy of 0.73, averaged over species and weighted by species size; for reference, the per-species chance level (1/K) averages 0.26 with equal species weights, so the two numbers are not weighted the same way.
- **Source-controlled residual (the primary estimate)**: **+0.044** NMI on the multi-sensor FOR-species20K and **+0.204** on the single-sensor BioDiv-3DTrees-ULS against the single-scale descriptor (bootstrap-significant; leave-one-source-out range never crosses zero), and **+0.092 / +0.104** against the local multi-scale baseline (these two within-source values come from a single clustering run, seed 0 with 10 initialisations, not a 15-seed mean).
- **The frozen 3D foundation model trails the 2D features on both datasets** (global pooled-token NMI 0.204 / 0.272 vs CLIP's 0.275 / 0.350), an availability comparison among off-the-shelf frozen models, not an architectural verdict.
- **Absolute quality stays modest**: in the headline table the evaluated features reach a mean NMI of about 0.10–0.36 (the random-initialisation control is lower). This is a relative cluster-coherence result, not a deployable classifier.

### Headline table (15-seed mean ± s.d.; matched 11-d unless noted; ±0.000 = s.d. < 0.0005)

| Dataset | Feature | NMI | ARI | macro-F1† | OA† | mIoU† |
|---|---|---|---|---|---|---|
| **FOR-species20K** (K=15, n=13,497) | CLIP-512 | **0.278** ± 0.001 | **0.147** ± 0.003 | 0.209 ± 0.002 | **0.275** ± 0.003 | 0.126 ± 0.001 |
| | CLIP-11d (PCA) | 0.275 ± 0.001 | 0.145 ± 0.003 | **0.211** ± 0.002 | 0.273 ± 0.004 | **0.127** ± 0.001 |
| | DINOv2-11d | 0.273 ± 0.001 | 0.129 ± 0.000 | 0.196 ± 0.001 | 0.241 ± 0.002 | 0.116 ± 0.001 |
| | ImageNet-11d | 0.207 ± 0.001 | 0.084 ± 0.001 | 0.167 ± 0.002 | 0.215 ± 0.001 | 0.096 ± 0.001 |
| | random-11d | 0.108 ± 0.003 | 0.046 ± 0.001 | 0.123 ± 0.003 | 0.175 ± 0.003 | 0.070 ± 0.002 |
| | geo-11d | 0.102 ± 0.002 | 0.059 ± 0.003 | 0.132 ± 0.003 | 0.198 ± 0.005 | 0.074 ± 0.002 |
| | Weinmann-local (PCA-11) | 0.227 ± 0.000 | 0.102 ± 0.001 | 0.173 ± 0.001 | 0.247 ± 0.003 | 0.101 ± 0.001 |
| **BioDiv-3DTrees-ULS** (K=6, n=2,250) | CLIP-512 | **0.355** ± 0.001 | **0.295** ± 0.001 | **0.287** ± 0.001 | **0.420** ± 0.002 | **0.191** ± 0.001 |
| | CLIP-11d (PCA) | 0.350 ± 0.001 | 0.291 ± 0.002 | **0.287** ± 0.001 | 0.419 ± 0.004 | **0.191** ± 0.001 |
| | DINOv2-11d | 0.298 ± 0.002 | 0.265 ± 0.002 | 0.265 ± 0.006 | 0.414 ± 0.002 | 0.178 ± 0.004 |
| | ImageNet-11d | 0.209 ± 0.005 | 0.162 ± 0.005 | 0.263 ± 0.003 | 0.359 ± 0.001 | 0.169 ± 0.003 |
| | random-11d | 0.059 ± 0.000 | 0.036 ± 0.000 | 0.194 ± 0.000 | 0.277 ± 0.000 | 0.115 ± 0.000 |
| | geo-11d | 0.101 ± 0.000 | 0.073 ± 0.000 | 0.204 ± 0.000 | 0.326 ± 0.000 | 0.124 ± 0.000 |
| | Weinmann-local (PCA-11) | 0.192 ± 0.000 | 0.144 ± 0.000 | 0.244 ± 0.000 | 0.408 ± 0.000 | 0.157 ± 0.000 |

† Hungarian-matched scores use the evaluation labels post hoc to pick the cluster→species assignment that maximises OA, so they are optimistic; macro-F1 and mIoU are computed under that same mapping and are not separately maximised. This is why matching-free NMI/ARI lead the report. A random clustering scores NMI ≈ 0.003. Older text in this repository calls all three scores upper bounds: the docstrings of `scripts/protocol_v2/gen_unsup_robust.py` and `scripts/protocol_v2/gen_unsup.py`, the note string that `gen_unsup.py` writes, the note fields of `runs/protocol_v2/gen/unsup_for.json` and `unsup_biodiv.json`, and §6 (Evaluation) of the locked pre-registration (`PREREG_3d_baseline.md` / `PREREG_3d_baseline_EN.md`). This footnote supersedes that wording; the cited wording is retained from the archive.

### What this is *not*

- **Not a deployable tree-species classifier.** Oracle-K, global fit, modest absolute quality.
- **Not a claim about all geometric descriptors.** The claim is scoped to the two evaluated ones — hence "the evaluated geometric descriptors" in the title.
- **Not a clean morphology result at face value.** Part of a global advantage can be acquisition-source structure; the within-source residual is the species-attributable estimate and is the number the paper leads with.
- **Not an architectural 2D-vs-3D verdict.** The frozen 3D baseline is ShapeNet-pretrained and out-of-domain for forest LiDAR (pre-registered caveat).

## Reproducing

These are the scripts that produced the paper's numbers. They run in stages, each reading caches that an earlier stage wrote under `data/raw/`, and they were not re-run end to end for this repository (that needs the raw datasets and a CUDA GPU). Each script's docstring states its inputs and outputs.

1. **Environment.** Some scripts use packages that `pyproject.toml` does not declare, so add them after syncing:

   ```bash
   uv sync                      # Python ≥3.11; uv.lock pins the CUDA 12.8 PyTorch build used for the paper (Linux / Windows)
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   uv pip install zstandard torch-redstone einops matplotlib igraph leidenalg
   ```

2. **Data.** Download the datasets (table below) and place them as the scripts expect:
   - FOR-species20K: `dev.zip` and `tree_metadata_dev.csv` into `data/raw/for_species20k/`;
   - BioDiv-3DTrees: the dataset's `ULS.tar.zst` saved as `data/raw/biodiv_3dtrees/biodiv_ULS.tar.zst`, and its `labels.csv` saved as `data/raw/biodiv_3dtrees/biodiv_labels.csv` (the scripts use its `treeID`, `species` and `ep` columns).

3. **Features.** `gen_for20k_features.py --matched` writes `geo_features_matched.npz`; `gen_multiview_backbones.py` writes the frozen-encoder features (`clip_features`, `for_dino`, `for_extra` for FOR-species20K; `biodiv_features`, `biodiv_dino`, `biodiv_extra` for BioDiv-3DTrees):

   ```bash
   python scripts/protocol_v2/gen_for20k_features.py --matched
   python scripts/protocol_v2/gen_multiview_backbones.py for --nviews 6
   python scripts/protocol_v2/gen_multiview_backbones.py biodiv --nviews 6
   ```

4. **Scores.** Several scripts take the dataset as an argument:

   ```bash
   python scripts/protocol_v2/gen_unsup_robust.py for            # headline NMI rows (repeat with biodiv)
   python scripts/protocol_v2/gen_ari.py                         # ARI column
   python scripts/protocol_v2/gen_source_control_multi.py for    # within-source residuals (repeat with biodiv)
   python scripts/protocol_v2/gen_3dfm_baseline.py               # frozen 3D baseline
   ```

On macOS or a CPU-only machine the locked CUDA wheels do not install. Install the declared dependencies from PyPI instead and run the unit tests (no datasets needed):

```bash
uv venv
uv pip install --no-sources -r pyproject.toml --extra dev
.venv/bin/python -m pytest -q
```

This path resolves current releases rather than the exact versions in `uv.lock`; the locked environment is the closer match to the one behind the reported numbers (the six extra packages above are not version-pinned either).

Inference cost (single RTX 4080): rendering six views + one batched CLIP forward ≈ 30 ms per tree, voxel downsampling ≈ 30 ms median, geometric descriptor < 1 ms; the full analyzable pool processes in under one hour.

### Datasets (not redistributed here)

Download each from its DOI under its own license and place the raw data locally (the pipeline reads from `data/`).

| Dataset | Role | DOI | License | Reference |
|---|---|---|---|---|
| **FOR-species20K** | Unsupervised comparison (TLS/MLS pooled) | [10.5281/zenodo.13255198](https://doi.org/10.5281/zenodo.13255198) | GPL-3.0-or-later | Puliti et al. 2025, *MEE* — [10.1111/2041-210X.14503](https://doi.org/10.1111/2041-210X.14503) |
| **BioDiv-3DTrees** (ULS subset) | Unsupervised comparison (ULS) | [10.25625/8PB1IF](https://doi.org/10.25625/8PB1IF) | CC-BY-4.0 | Griese et al. 2025, *Sci. Data* — [10.1038/s41597-025-06421-7](https://doi.org/10.1038/s41597-025-06421-7) |
| **Weiser / SYSSIFOSS** (ALS subset) | Provenance only (enters via FOR-species20K blocks) | [10.1594/PANGAEA.942856](https://doi.org/10.1594/PANGAEA.942856) | CC-BY-SA-4.0 | Weiser et al. 2022, *ESSD* — [10.5194/essd-14-2989-2022](https://doi.org/10.5194/essd-14-2989-2022) |
| **NEON-MLBS** | Depth-vs-RGB boundary probe only | via the NEON/IDTReeS benchmark | CC BY 4.0 ([NEON data use policy](https://data.neonscience.org/data-api/data-use-policy/)) | Graves et al. 2023 |

### Script → result map (what produces which paper number)

| Paper artifact | Script(s) | Result file(s) in `runs/protocol_v2/gen/` |
|---|---|---|
| Table 2 (headline, NMI/F1/OA/mIoU) | `gen_unsup_robust.py`, `gen_unsup_robust_multi.py` | `unsup_robust_{for,biodiv}.json`, `unsup_robust_multi_{for,biodiv}.json` |
| Table 2 ARI column | `gen_ari.py` | `ari_multi.json` |
| Table 2 Weinmann-local rows | `gen_local_weinmann.py`, `gen_weinmann_row.py` | `local_weinmann.json`, `weinmann_row.json` |
| Table 4 (view-count ablation) | `gen_view_ablation.py` | `view_ablation*.json` |
| Table 3 (clustering algorithms) | `gen_experiment_battery.py` (+`_dino`) | `clustering_methods_*.json` |
| Table 5 (frozen 3D baseline) | `gen_3dfm_baseline.py` (1,024 points); `gen_3dfm_np.py` (point-count robustness, default 10,000); `gen_3dfm_ws_seeds.py` | `unsup_3dfm.json`; `unsup_3dfm_n{N}.json`; `tdfm_ws_seeds.json` |
| Table 6 (within-source control) | `gen_source_control_multi.py`, `gen_source_control_robust.py` | `source_control_multi_*.json`, `source_control_robust*.json` |
| Leiden (no oracle-K) | `gen_leiden.py` | `leiden*.json` |
| Residual bootstrap + LOO | `gen_bootstrap_residual.py`, `gen_source_control_perblock.py` | `bootstrap_residual_*.json`, `source_control_perblock*.json` |
| Source-recovery probe (FOR-species20K, CLIP at 11-d: size-weighted balanced accuracy 0.73; equal-weight mean chance level 0.26) | `gen_source_control_robust.py` | `source_control_robust.json` |
| Fig. 9 (gap decomposition) | `gen_fig_decomp.py` | written to `runs/protocol_v2/figs/` (not shipped) |
| PCA dimension sweep (Fig. 4) | `gen_experiment_battery.py` | `dim_sweep_*.json` |
| Permutation nulls | `gen_permutation_multi.py` | `perm_test*.json` |
| Inference timing | `time_inference.py` | `inference_timing.json` |
| Renderer validation | `render_validation.py`, `check_renderer.py` | — |

**Seeds.** Repetition counts differ between analyses (for example, the clustering-method comparison and the PCA and K sweeps use 5 seeds, and some algorithms run once), so the generating script of each result file is the reference. Two cases need care. The within-source NMI in `unsup_3dfm.json`, `unsup_3dfm_n10000.json` and `local_weinmann.json`, and both the global and the within-source NMI in `source_control.json`, come from a single clustering run (seed 0, `n_init=10`); the "15 seeds" in `unsup_3dfm.json` refers to its global scores only. `tdfm_ws_seeds.json` holds 15-seed within-source values for the frozen 3D baseline (BioDiv-3DTrees pooled: 0.1873 mean, against 0.1891 for seed 0).

Every number in the manuscript traces to a JSON under `runs/protocol_v2/gen/` (the manuscript names the file next to each principal table/figure). These result files ship in this repository; they are evidence, not a substitute for downloading the source data.

## Pre-registration

The frozen 3D baseline was pre-registered before any 3D result was seen: model choice, preprocessing, evaluation harness, reporting rules (the comparison enters the main text win or lose), and a deviation log. `PREREG_3d_baseline.md` is the locked original (Chinese, normative); `PREREG_3d_baseline_EN.md` is a courtesy translation.

## Layout

```
treeclip/                  # the package: preprocessing, renderers, encoders, fusion
scripts/protocol_v2/       # every gen_* analysis script behind the paper
runs/protocol_v2/gen/      # result JSONs (provenance for every paper number)
runs/protocol_v2/figs/     # generated figures
tests/                     # unit tests
PREREG_3d_baseline.md      # pre-registration (locked 2026-07-01) + _EN courtesy translation
```

## Changes from the archived release

This repository mirrors the Zenodo archive v2.0.0. It differs only in the following ways. No result JSON or reported number was changed. Existing code changed only in the path constants and docstrings described below, in a license note at the top of `scripts/protocol_v2/oshape_model.py`, in the legend and docstring of `scripts/protocol_v2/gen_fig_decomp.py` (Fig. 9), in seed descriptions (docstrings and one output note string) of `gen_3dfm_baseline.py`, `gen_local_weinmann.py` and `gen_3dfm_np.py`, which called within-source values 15-seed results, and in the docstring of `gen_view_ablation.py`, which said that two views reproduce the headline values (the headline values correspond to six views).

- `treeclip/data/` (two files, unchanged since 2026-05-28) was left out of the archive by mistake and is restored here. `treeclip/encoders/precompute_embeddings.py` and three tests import it.
- Figures that render third-party data are not included: 15 files under `runs/protocol_v2/figs/` (`fig1_asset_cloud`, `fig1_asset_views`, `fig1_pipeline`, `figgallery_species`, `figrender_demo`, `hsi_NEON-MLBS`, `hsi_NEON-OSBS`, `rgb_NEON-MLBS`, `rgb_NEON-OSBS`, `structure_NEON-MLBS`, `structure_NEON-OSBS` and `structure_Weiser`, as PNG and, for three of them, also PDF). The scripts that draw them are included.
- `LICENSES/OpenShape-Apache-2.0.txt` was added, and `scripts/protocol_v2/oshape_model.py` now states at the top that it contains Apache-2.0 code from OpenShape.
- The author line in `LICENSE` and in the two pre-registration files shows the repository handle. The locked pre-registration originals are in the archive.
- `CITATION.cff` is not included; cite via the DOIs below.
- Five scripts under `scripts/protocol_v2/` had hard-coded local paths in a constant or a docstring; these now point to repo-relative locations. The two poster-figure scripts (`gen_fig1_pipeline.py`, `gen_fig_gallery.py`) read assets from `poster/assets/`, which is not shipped.
- `scripts/protocol_v2/gen_fig_decomp.py` labelled the global gap "(upper bound)" in the Fig. 9 legend, and its docstring called the source component negative for the local multi-scale baseline in general; the legend now reads "global gap", and the docstring says the component is negative on FOR-species20K and positive on BioDiv-3DTrees-ULS. Its docstring also called every value a 15-seed mean; it now says that the local multi-scale within-source values come from one clustering run.
- This README was reworded for the repository and corrected in ten places: the reproduction steps; the Python version (≥3.11, as declared in `pyproject.toml`) and the macOS / CPU install path; the provenance of the 0.73 / 0.26 source-recovery numbers and of Table 5; the description of global gaps (the archived README called every global gap an upper bound, but controlling for source widens one of them); the NEON license entry in the dataset table, which was blank; the note that the local multi-scale within-source values are a single clustering run; the **Seeds** note under the script → result map; the description of the Hungarian-matched scores, which the archived README called upper bounds although the mapping maximises OA only (the footnote under the headline table lists the archived files that keep the old wording); the source-recovery probe, whose two numbers are weighted differently; and the NMI range, now limited to the evaluated features in the headline table.

## License and citation

MIT (see `LICENSE`), except `scripts/protocol_v2/oshape_model.py`, which contains code from [OpenShape](https://github.com/Colin97/OpenShape_code) under the Apache License 2.0 (see `LICENSES/OpenShape-Apache-2.0.txt`; the file's docstring names the upstream commit and lists the changes). The datasets are not redistributed here, and neither are figures that render them. To cite, use the DOIs: the software archive [10.5281/zenodo.22216659](https://doi.org/10.5281/zenodo.22216659) (v2.0.0, the version used for the paper) and the manuscript preprint [10.21203/rs.3.rs-10238136/v1](https://doi.org/10.21203/rs.3.rs-10238136/v1).
