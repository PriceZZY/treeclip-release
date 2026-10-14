# Pre-registration: Frozen 3D-SSL baseline (TreeCLIP supplementary experiment)

> **Courtesy English translation.** The normative original is `PREREG_3d_baseline.md` (Chinese), locked on 2026-07-01 before any 3D result was seen. In case of any discrepancy the Chinese original governs.

*Lock date: 2026-07-01 (before any results) · Author: PriceZZY · Status: pre-registered, outcome unknown*

> This document locks the protocol and reporting rules **before any 3D-SSL result is seen**, as an honesty anchor. The critic gate requires preventing (a) crippling the baseline in order to win, (b) reporting only if the outcome is favorable, and (c) picking the best of several analysis variants. Any deviation from this protocol must be explicitly recorded and justified in the final manuscript / response to reviewers.

## 1. Research question (why this experiment)
The paper's core claim is that "frozen vision foundation-model features are a *stronger label-free representation* than hand-crafted geometry." The most predictable and most legitimate reviewer objection: *the only competitor is a single hand-crafted Weinmann geometric descriptor*, while the field's real label-free competitor — a **frozen self-supervised 3D point-cloud encoder** (Point-MAE/PointNeXt) — is cited but never evaluated.

This experiment adds that apples-to-apples competitor and answers: **is the advantage of frozen 2D foundation models specific to 2D foundation models, or can any pre-trained 3D encoder reach it or better?**

## 2. Pre-committed reporting rules (non-negotiable)
- **The head-to-head comparison enters the main text regardless of outcome** (2D-FM vs 3D-SSL vs geometry). Whether 3D wins, ties, or loses.
- *Only* additional per-source decompositions / ablations may be held as rebuttal reserve; the main comparison may not be held back.
- All three outcomes are honestly publishable:
  - 2D-FM still wins → the claim is markedly strengthened ("depth-projected FM beats a frozen 3D-SSL encoder without any 3D pre-training").
  - 2D-FM ties 3D-SSL → honestly reframe as a label-free representation-landscape study.
  - 3D-SSL wins → honestly reframe (the "Foundation Model-based" title must be rewritten into a different but still honest paper — see the §7 contingency).

## 3. Models (frozen, no fine-tuning)
- **Primary**: Point-MAE, official ShapeNet-pretrained checkpoint, **frozen** (no fine-tuning, no probing training).
- **Secondary (time permitting)**: PointNeXt or PointMLP, ScanObjectNN-pretrained (real scans, domain closer to LiDAR).
- Feature = globally pooled token of the frozen forward pass (consistent in spirit with the paper's pooling of 2D features).
- **No crippling**: use the officially recommended input specification, a reasonable checkpoint, and fair preprocessing. Deliberately weakening the 3D baseline is as forbidden as burying an unfavorable result.

## 4. Data and alignment
- Raw point clouds from local LAZ: FOR = `data/raw/for_species20k/dev.zip` (`{treeID}.laz`); BioDiv = `data/raw/biodiv_3dtrees/biodiv_ULS.tar.zst` (`ULS/{treeID}_ULS.laz`).
- Aligned by **treeID to the same analyzable subset as the existing experiments**: the same `analyzable_mask` (≥30 trees per species, ≥2 blocks) + BioDiv's Fagus/dominant-species cap of 0.39 (bit-identical to `gen_score.py` / the existing pipeline), guaranteeing comparison on the same trees as CLIP/geo.
- Species labels, source (FOR `dataset` / BioDiv `region`), and sensor type are taken from the existing metadata CSVs.

## 5. Preprocessing (pre-specified, single variant)
- Each tree cloud: random downsample if points > N, upsample with replacement if < N, to **N = 1024** points (Point-MAE's standard input).
- Normalization: **translate to centroid + scale to the unit sphere** (zero-mean, unit max-norm).
- Random seeds fixed; downsampling RNG seed = 0.
- *No* trying several N or several normalizations and picking the best — N=1024 + unit sphere is the single pre-specified scheme.

## 6. Evaluation (reuses the existing harness exactly, nothing new)
- Clustering: `StandardScaler → PCA(11-d) → KMeans(K = true species count, n_init=10)`, identical to `gen_permutation_test.cluster_11d` / `gen_unsup_robust`.
- Primary metric: **matching-free NMI** (15 seeds, mean ± SD = init jitter).
- Secondary metrics (Hungarian-matched, explicitly labeled upper bounds): macro-F1 / OA / mIoU.
- **Source control**: same as the existing `gen_source_control_*` — report the within-source residual NMI (what remains after removing source), side by side with CLIP/geo.
- Reporting: one 3D-SSL row added into the existing Table 2 / Table 5 structure, all datasets, all metrics.

## 7. Domain-gap caveat (pre-committed interpretation)
Standard Point-MAE/PointNeXt checkpoints are pre-trained on synthetic CAD (ShapeNet) or indoor scans (ScanObjectNN), and are **out-of-domain for forest LiDAR**. Therefore:
- If 3D-SSL loses: the conclusion is limited to "**this out-of-domain 3D pre-training** is weak on this task" and **must not** be generalized to "3D pre-training is generally weaker than 2D-FM." This ambiguity must be stated explicitly in the main text.
- If available, prefer/supplement a point-cloud checkpoint pre-trained on vegetation or real outdoor scans to reduce the ambiguity.

## 8. Contingency for "3D wins" (decided before results, against burying pressure)
If the frozen 3D-SSL encoder robustly beats the 2D-FM: the paper's center of gravity shifts from "2D foundation models" to "frozen pre-trained representations (2D or 3D) both beat hand-crafted geometry," and the title drops the exclusive sense of "Foundation Model-based." That is a different but equally honest, equally publishable paper; committing to this path in advance removes any incentive to bury an unfavorable result.

## 9. Deviation log
- **2026-07-01 — Model: Point-MAE → OpenShape PointBERT (frozen 3D foundation model, ShapeNet-pretrained `openshape-pointbert-shapenet`).** Reasons: (1) no Point-MAE checkpoint on HF, and this avoids compiling custom CUDA ops on Windows (pointnet2_ops/knn_cuda); (2) OpenShape is a CLIP-aligned 3D foundation model — a **stronger and more relevant** competitor than Point-MAE SSL, directly matched to the paper's "2D foundation model" claim; using the stronger baseline is the **conservative / anti-crippling** direction (not a weakening). Grouping (FPS/kNN) implemented in pure torch to avoid CUDA ops; input is xyz (+ a constant color channel, OpenShape's standard no-color convention). The §7 domain-gap caveat is unchanged (ShapeNet CAD pre-training → out-of-domain for forest LiDAR). The §2 reporting rules, §6 evaluation harness, and §7 caveat are all unchanged. Fallback ladder (if loading fails): Uni3D → pure-torch port of Point-MAE. All outcomes still enter the main text per §2, win or lose.
