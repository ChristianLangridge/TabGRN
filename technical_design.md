# Technical Design Document
## TRACE: Trajectory Representation And Composition Estimator

**Version:** 1.9.0  
**Status:** Run 17 submitted — full Myriad config finalised  
**Project:** Joint rotation — Queen Mary University London / University College London  
**Supervisors:** Dr. Julien Gautrot · Dr. Yanlan Mao · Dr. Isabel Palacios  
**Author:** Christian Langridge  
**Last Updated:** May 2026

**Changelog v1.9.0**
- Project renamed from TabGRN → TRACE (Trajectory Representation And Composition Estimator) throughout
- §1.1: Framing updated — TRACE is a developmental coordinate prediction tool; GRN inference is a future direction, not the rotation deliverable
- §3.1: `n_bins` removed entirely from `ContextConfig` (dead config — sampler always bins by the 5 active collection days at runtime); validation now uses hardcoded `5 active days × cells_per_bin`; `ExplainabilityConfig`, `AblationTarget`, `PerturbationConfig`, `BenchmarkConfig` removed from `ExperimentConfig` (stripped from codebase)
- §3.1 presets: `rotation_finetune()`, `rotation_baselines()`, `scratch_preset()`, `no_icl_preset()` removed; `full_finetune()` updated to 2000 genes; `full_finetune_dirichlet` updated to `supervised_batch_size=4`
- §3.3: `ContextSampler` updated — three independent anchor-selection components (day-stratified, pseudotime-stratified, composition-stratified); all three live in the same `sample()` call
- §5.1: `DirichletCompositionHead.forward` updated to include `.clamp(max=80.0)` on softplus input (overflow fix)
- §7.1: `HARDWARE_TIERS` updated — all tiers `max_context_cells=50` (5 active days × 10 cells); `full` tier `max_genes=2000`
- §8.2a: Test counts updated — 405 passing, 0 failing
- §9.1: Myriad run configuration updated for Run 18 (2000 genes, 50 context cells, warmup 2500 steps)
- §9.3: Milestones updated through Run 17
- §10: New decisions C1–C6 (context sampling ablation, n_bins removal, warmup fix, gene count) added
- §11 new: Future directions — TRACE 2.0, multimodal GRN pipeline, developmental coordinate tool use case

**Changelog v1.8.0**
- §3.1: `lr_col` corrected to `1e-6`; `composition_loss_type` field added to `ModelConfig`; `rotation_finetune_dirichlet()` preset added
- §3.5: `lr_col` in parameter groups table updated; note added on `composition_loss_type` switching
- §3.7: `DirichletCompositionHead` documented as parallel sub-module
- §3.8: `DirichletDualHeadLoss` documented; `lambda_comp=0.1` fixed-weight design decision recorded
- §3.12: `SupervisedTrainer` documented
- §10: New decisions D1–D4 added

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Component Specifications](#3-component-specifications)
4. [Workflow & Data Flow](#4-workflow--data-flow)
5. [Implementation Details](#5-implementation-details)
6. [Integration Guide](#6-integration-guide)
7. [Configuration Reference](#7-configuration-reference)
8. [Test Architecture](#8-test-architecture)
9. [Deployment & Compute](#9-deployment--compute)
10. [Decision Log](#10-decision-log)
11. [Future Directions](#11-future-directions)

---

## 1. System Overview

### 1.1 Architectural Goals

TRACE (Trajectory Representation And Composition Estimator) is a tabular in-context learning model for developmental coordinate prediction from single-cell RNA sequencing data. It is trained on the Jain et al. 2025 (Nature) brain organoid time-course dataset and targets two simultaneous prediction objectives:

| Objective | Output | Head | Status |
|---|---|---|---|
| Pseudotime regression | Scalar ∈ (0, 1) — developmental progression | `PseudotimeHead` | **Active — rotation deliverable** |
| Cell state composition | K-vector of Dirichlet parameters — lineage identity | `DirichletCompositionHead` | **Active — rotation deliverable** |

The model uses the **TabICLv2** pre-trained backbone, adapted for continuous regression targets via a dual-head output architecture. The two heads capture orthogonal information: **pseudotime measures progression** (distance from root along the main developmental axis), while **composition measures identity** (which lineage(s) a cell belongs to). On a branching trajectory, two cells equidistant from root on different branches get similar pseudotime but distinct composition vectors.

**The rotation-scope deliverable is developmental coordinate prediction.** A biologist provides unannotated scRNA-seq data from the same neurectoderm organoid system. TRACE assigns each cell a pseudotime coordinate and a lineage composition vector in a single forward pass, without requiring manual annotation or re-clustering. Column-wise attention (stage 1) encodes gene co-expression structure as a byproduct of both prediction tasks — this is the foundation for future GRN inference work, but GRN extraction is explicitly out of scope for the July 2026 rotation report.

### 1.2 Core Design Principles

- **Explicit over implicit.** Every hyperparameter lives in `ExperimentConfig`. No magic numbers in implementation code.
- **Schema contracts at construction.** `ProcessedDataset` validates its own schema at build time. Failures surface immediately, not mid-training.
- **Dual-head from start.** Both heads are active during training from step one. No phase gate.
- **Tests as first-class artifacts.** RED phase tests written before implementation. All 405 tests pass as of v1.9.0.
- **Hardware-tier portability.** Three named hardware tiers (`debug`, `standard`, `full`) ensure reproducibility across laptop and A100 without code changes.

### 1.3 Scientific Context

The model operates on the Matrigel-only condition of the Jain et al. 2025 time-course, which tracks brain organoid development across six collection days: 5, 7, 11, 16, 21, 30 (~41,000 cells). The trajectory branches after day 11 into telencephalic vs. non-telencephalic lineages.

**Pseudotime** is recomputed via diffusion map on the full trajectory (days 5–30), replacing the published DC1 which only covers days 5–11 neuroectodermal cells. Expression matrices from different batches were integrated with the Cluster Similarity Spectrum (CSS) technique outlined by the Treutlein Lab. Day 11 cell labels (pseudotime and hard classification labels) are withheld from model training and context — they represent the neuroectoderm-to-neuroepithelial transition, the hardest interpolation point.

```
D5   n= 5481  median pseudotime=0.1167
D7   n= 8183  median pseudotime=0.2534
D11  n= 4912  median pseudotime=0.3112  ← withheld test set
D16  n= 6571  median pseudotime=0.7128
D21  n= 7962  median pseudotime=0.7837
D30  n= 7950  median pseudotime=0.6526
```

**Known limitation — Prosencephalic progenitors:** D11 Prosencephalic cells (n=207, ~4.2% of D11) sit at pt~0.47 but the bulk of training Prosencephalic cells live at D16/D21 (pt 0.71–0.78). The transcriptional profile at pt~0.47 overlaps with Neurectoderm/Late Neurectoderm territory, causing the model to systematically under-predict Prosencephalic affinity (W-dist ~4.8 vs baseline 4.54) regardless of anchor sampling strategy. This is a dataset limitation, not an architecture limitation, and is documented as a known caveat in the rotation report.

---

## 2. Directory Structure

```
TRACE/
├── pyproject.toml                      # Package registration — pip install -e .
├── myriad_trace.sh                     # SGE job script — full Myriad GPU run
├── myriad_inference.sh                 # SGE job script — inference only
├── LICENSE
├── README.md
├── technical_design.md
│
├── experiments/                        # Auto-created at runtime
│   └── {run_id}/
│       ├── config.json                 # Serialised ExperimentConfig (every run)
│       └── checkpoints/
│           ├── gene_coexpression.pt    # Saved at warmup boundary — col/row attention only
│           ├── final.pt                # Saved after all N_STEPS — inference checkpoint
│           └── loss_curve.png          # 3-panel loss curve: total / pseudotime / composition
│
├── data/
│   ├── TabICLv2_checkpoint/
│   │   └── tabicl-regressor-v2-20260212.ckpt   # Pretrained backbone
│   ├── training_data/
│   │   └── AnnData/
│   │       └── neurectoderm_with_pseudotime.h5ad  # Jain et al. 2025 (Matrigel condition)
│   ├── HPC run logs/                   # Archived Myriad run stdout/stderr
│   └── WLS_ko_validation/             # External GRN validation — future direction
│
├── path/
│   └── spatialmt/                      # Installable package root (pip install -e .)
│       ├── config/                     # ✓ Implemented
│       │   ├── paths.py                # Filesystem path resolution
│       │   └── experiment.py           # ExperimentConfig + sub-configs + HARDWARE_TIERS
│       │
│       ├── data_preparation/           # ✓ Implemented
│       │   ├── prep.py                 # HVG selection, expression extraction
│       │   ├── diffusion_trajectory.py # CSS integration, diffusion pseudotime
│       │   └── dataset.py              # ProcessedDataset — from_anndata, soft labels, validation
│       │
│       ├── context/                    # ✓ Implemented
│       │   ├── sampler.py              # ContextSampler — day/pseudotime/composition anchor sampling
│       │   ├── builder.py              # CellTableBuilder + CellTable + TrainingTargets
│       │   └── collate.py              # ICLBatch + icl_collate — DataLoader collate fn
│       │
│       ├── model/                      # ✓ Implemented
│       │   ├── trace.py                # TabICLRegressor + AnchorLabelEmbedder + SharedTrunk
│       │   │                           # + PseudotimeHead + CompositionHead + DirichletCompositionHead
│       │   └── loss.py                 # DualHeadLoss + DirichletDualHeadLoss
│       │
│       ├── training/                   # ✓ Implemented
│       │   ├── trainer.py              # Trainer + SupervisedTrainer + MuonAdamW
│       │   ├── callbacks.py            # CheckpointCallback
│       │   └── muon.py                 # Muon optimizer (Newton-Schulz spectral orthogonalisation)
│       │
│       └── eval/
│           └── metrics.py              # spearman_r, mae, per_day_spearman, wasserstein_1,
│                                       # wasserstein_baseline, brier_score, jsd
│
├── src/
│   └── train/
│       ├── trace_debug_run.py          # Local debug run — 1000 steps, MPS/CPU, sampler ablation
│       └── trace_myriad_run.py         # Full Myriad run — 25,000 steps, checkpointing, loss curve
│
└── tests/
    ├── conftest.py                     # Fixtures: debug_config, synthetic_dataset,
    │                                   #           synthetic_dataset_with_labels
    ├── unit/
    │   ├── test_experiment_config.py   # ExperimentConfig — serialisation, hash, presets ✓
    │   ├── test_dataset.py             # ProcessedDataset schema contract ✓
    │   ├── test_trace_model.py         # TabICLRegressor + sub-modules ✓
    │   ├── test_dual_head_loss.py      # DualHeadLoss — Kendall weighting, KL divergence ✓
    │   ├── test_dirichlet_loss.py      # DirichletDualHeadLoss ✓
    │   ├── test_dirichlet_composition_head.py  # DirichletCompositionHead ✓
    │   ├── test_context_sampler.py     # ContextSampler — day/pt/comp sampling ✓
    │   ├── test_cell_table_builder.py  # CellTableBuilder ✓
    │   ├── test_icl_collate.py         # ICLBatch + icl_collate ✓
    │   ├── test_trainer.py             # Trainer + MuonAdamW ✓
    │   ├── test_supervised_trainer.py  # SupervisedTrainer ✓
    │   ├── test_callbacks.py           # CheckpointCallback ✓
    │   └── test_wasserstein.py         # Wasserstein distance metric ✓
    └── integration/
        ├── test_dataset.py             # ProcessedDataset.from_anndata ✓
        └── test_prep.py                # prep.py HVG, pseudotime pipeline ✓
```

---

## 3. Component Specifications

### 3.1 `ExperimentConfig`
**File:** `path/spatialmt/config/experiment.py`

**Purpose:** Single source of truth for all hyperparameters. Serialised to `experiments/{run_id}/config.json` at training startup. Every run is fully reproducible from its config file alone.

**Sub-configs:**

| Sub-config | Key fields | Notes |
|---|---|---|
| `DataConfig` | `max_genes`, `test_timepoint=11`, `hardware_tier`, `n_cell_states=8`, `label_softening_temperature=1.0` | `log1p_transform` validated `True` at construction |
| `ContextConfig` | `cells_per_bin=5`, `max_context_cells=50`, `allow_replacement=True`, `n_composition_anchors=0`, `n_pseudotime_anchors=0`, `n_pseudotime_bins=5` | Validates `5 × cells_per_bin + n_pseudotime_anchors + n_composition_anchors ≤ max_context_cells`; the 5 is hardcoded — the dataset always has 5 active collection days after D11 exclusion; `n_bins` config field has been removed entirely (it was dead — the sampler always bins by actual dataset days) |
| `ModelConfig` | `lr_col=1e-6`, `lr_row=1e-4`, `lr_icl=5e-5`, `lr_emb=1e-3`, `lr_head=1e-3`, `warmup_col_steps=500`, `warmup_icl_steps=100`, `output_head_init_bias=0.5`, `output_head_init_std=0.01`, `composition_loss_type="kl"`, `supervised_batch_size=64` | `composition_loss_type` switches `TabICLRegressor` between `CompositionHead` (`"kl"`) and `DirichletCompositionHead` (`"dirichlet"`); `bio_plausibility_passed` populated post-training |

**Named presets:**

```python
ExperimentConfig.debug_preset()                  # 256 genes, 5 days × 5 cells, debug tier
ExperimentConfig.rotation_finetune_dirichlet()   # 512 genes, standard tier, Dirichlet NLL (rotation_002)
ExperimentConfig.full_finetune()                 # 2000 genes, 5 days × 10 cells, full tier, KL
ExperimentConfig.full_finetune_dirichlet()       # 2000 genes, 5 days × 10 cells, full tier, Dirichlet NLL
```

**`config_hash`:** SHA-256 of serialised hyperparameters excluding `run_id`. Two runs with identical hyperparameters but different `run_id` values produce the same hash.

**Dependencies:** `spatialmt.config.paths.Paths`, `dataclasses`, `json`, `hashlib`

---

### 3.2 `ProcessedDataset`
**File:** `path/spatialmt/data_preparation/dataset.py`

**Purpose:** Immutable, schema-validated container for one experiment's training data. Every downstream component receives this object; raw files are never accessed after construction.

**Fields:**

| Field | Shape | Type | Notes |
|---|---|---|---|
| `expression` | `(n_cells, n_genes)` | `np.float32` | log-normalised (CP10k+log1p), validated max < 20.0 |
| `gene_names` | `(n_genes,)` | `list[str]` | HVG names in column order (flavor=`seurat`) |
| `pseudotime` | `(n_cells,)` | `np.float32` | Diffusion pseudotime ∈ [0, 1]; loaded from `adata.obs["rank-transformed-pseudotime"]` |
| `collection_day` | `(n_cells,)` | `np.int32` | ∈ {5, 7, 11, 16, 21, 30} |
| `cell_ids` | `(n_cells,)` | `list[str]` | Unique identifiers |
| `cell_type_labels` | `(n_cells,)` | `pd.Series` | `class3` annotations — 8 states |
| `soft_labels` | `(n_cells, K)` | `np.float32` | Distance-to-centroid softmax, K=8 |
| `centroid_distances` | `(K, K)` | `np.float64` | PCA-space inter-centroid distances — ground metric for Wasserstein evaluation |
| `manifest_hash` | scalar | `str` | SHA-256 of `sorted(gene_names)` |

**Validation assertions (all checked at construction):**
- `expression.max() < 20.0`
- `pseudotime ∈ [0, 1]`
- No NaN or Inf in expression or pseudotime
- `soft_labels.sum(axis=1) ≈ 1.0` ± 1e-5
- Shape consistency across all arrays

---

### 3.3 `ContextSampler`
**File:** `path/spatialmt/context/sampler.py` ✓ Implemented

**Purpose:** Samples anchor cells for the ICL context window. Supports three independent anchor-selection strategies that can be combined in any proportion. Day-11 cells are excluded from all anchor pools.

**Three components:**

| Component | Config field | Mechanism |
|---|---|---|
| Day-stratified | `cells_per_bin` | `cells_per_bin` anchors drawn from each active collection day (D5, D7, D16, D21, D30). 5 active days at runtime after D11 exclusion. |
| Pseudotime-stratified | `n_pseudotime_anchors` | Anchors distributed across `n_pseudotime_bins` equal-width bins over [0, 1]. Covers developmental progression independent of collection day. |
| Composition-stratified | `n_composition_anchors` | Anchors distributed evenly across dominant-class pools (argmax of `soft_labels`). Ensures each cell-type archetype is represented in every context window. |

**Active components:** Any combination is valid. A component is inactive when its count is 0. The `debug_preset` uses day-stratified only by default; individual sampler modes are selected via the `SAMPLER` env var in `trace_debug_run.py`.

**Pre-built index structures (constructor):**
- `_day_to_indices`: `dict[int, list[int]]` — per collection day (D11 excluded)
- `_pt_bin_to_indices`: `dict[int, list[int]]` — equal-width pseudotime bins; `np.searchsorted` assignment; pt=1.0 clamped to last bin
- `_class_to_indices`: `dict[int, list[int]]` — dominant class = `soft_labels.argmax(axis=1)`

**Sparse bin guard:** When a bin contains fewer eligible cells than requested, sampling proceeds with replacement and a `UserWarning` is raised. `allow_replacement=False` raises `ValueError` instead.

**Primary method:**
```python
sampler.sample(query_cell_id: str, rng: Generator | int | None = None)
    -> (anchor_cell_ids: list[str], anchor_pseudotimes: np.ndarray[float32])
```

---

### 3.4 `CellTableBuilder`, `CellTable`, `TrainingTargets`
**File:** `path/spatialmt/context/builder.py` ✓ Implemented

**Purpose:** Assembles model inputs and training targets for a single ICL training step. Query labels are never placed in `CellTable`; they are held separately in `TrainingTargets`.

```python
@dataclass
class CellTable:
    context_expression:  np.ndarray  # (n_anchors, n_genes) float32
    context_pseudotime:  np.ndarray  # (n_anchors,)         float32
    context_soft_labels: np.ndarray  # (n_anchors, K)       float32
    query_expression:    np.ndarray  # (n_genes,)           float32

@dataclass
class TrainingTargets:
    query_pseudotime:  np.floating   # scalar float32 — loss only
    query_soft_labels: np.ndarray    # (K,) float32   — loss only
```

---

### 3.4b `ICLBatch` and `icl_collate`
**File:** `path/spatialmt/context/collate.py` ✓ Implemented

```python
@dataclass
class ICLBatch:
    context_expression:  torch.Tensor  # (B, n_anchors, n_genes)
    context_pseudotime:  torch.Tensor  # (B, n_anchors)
    context_soft_labels: torch.Tensor  # (B, n_anchors, K)
    query_expression:    torch.Tensor  # (B, n_genes)
    query_pseudotime:    torch.Tensor  # (B,)
    query_soft_labels:   torch.Tensor  # (B, K)
```

**Ragged guard:** Raises `ValueError` if any item has a different `n_anchors` than item 0.

---

### 3.5 `TabICLRegressor`
**File:** `path/spatialmt/model/trace.py` ✓ Implemented

**Forward pass (five stages):**

| Stage | Module | LR | Warmup | Role |
|---|---|---|---|---|
| 1 — Column | `col_embedder` | 1e-6 | 500 steps | Gene co-expression encoding |
| 2 — Row | `row_interactor` | 1e-4 | None | Cell vector aggregation |
| 3 — Label inject | `anchor_label_embedder` | 1e-3 | None | Injects anchor targets (ICL protocol) |
| 4 — ICL | `tf_icl` (pretrained) | 5e-5 | 100 steps | Query attends to anchor context |
| 5 — Heads | `shared_trunk` → dual heads | 1e-3 | None | Dual-head output |

**Critical implementation details:**
- `emb.clone()` passed to `row_interactor` — prevents in-place CLS write severing autograd to `col_embedder`
- `col_embedder.*` always excluded from `load_backbone()` — pretrained weights are position-indexed, not gene-name-indexed
- `DirichletCompositionHead.forward` clamps softplus input to max=80.0 — prevents float32 overflow producing `inf` alpha values (fix: commit `041f20f`)

---

### 3.6 `PseudotimeHead`
**File:** `path/spatialmt/model/trace.py` ✓ Implemented

**Architecture:** `Linear(d_model, 1)` → `sigmoid` → `squeeze(-1)`  
**Output:** `(batch,)` predicted pseudotime ∈ `(0, 1)`  
**Init:** weight std=0.01, bias=0.5 (sigmoid(0.5)≈0.62 — trajectory midpoint prior)

---

### 3.7 `CompositionHead` and `DirichletCompositionHead`
**File:** `path/spatialmt/model/trace.py` ✓ Implemented

#### `CompositionHead` (`composition_loss_type="kl"`)
**Architecture:** `Linear(d_model, K)` → `softmax(dim=-1)`  
**Output:** `(batch, K)` probability vector, rows sum to 1.0

#### `DirichletCompositionHead` (`composition_loss_type="dirichlet"`)
**Architecture:** `Linear(d_model, K)` → `softplus(clamp(x, max=80.0))`  
**Output:** `(batch, K)` concentration parameters α_k > 0

**Inference-time readouts:**
```
mean prediction  : α_k / Σα_k
total precision  : α₀ = Σα_k
per-class variance: α_k(α₀ − α_k) / (α₀²(α₀ + 1))
```

**K=8 cell state index mapping:**

| Index | State | Dominant timepoint |
|---|---|---|
| 0 | Neurectoderm | Day 5 |
| 1 | Late Neurectoderm | Day 7–11 |
| 2 | Unknown proliferating cells | Mixed |
| 3 | Prosencephalic progenitors | Day 11 |
| 4 | Late Prosencephalic progenitors | Day 11–16 |
| 5 | Telencephalic progenitors | Day 16–21 |
| 6 | Diencephalic progenitors | Day 16–21 |
| 7 | Tel/Die neurons | Day 21–30 |

---

### 3.8 `DualHeadLoss` and `DirichletDualHeadLoss`
**File:** `path/spatialmt/model/loss.py` ✓ Implemented

#### `DualHeadLoss` (`rotation_001`)
Kendall uncertainty weighting on both MSE and KL. Both `log_sigma_sq_pt` and `log_sigma_sq_comp` are learnable.

```python
total = exp(–s_pt) · MSE(pt_pred, pt_target) + ½·s_pt
      + exp(–s_comp) · KL(comp_target ∥ comp_pred) + ½·s_comp
```

#### `DirichletDualHeadLoss` (`rotation_002`)
Kendall weighting on pseudotime MSE only; fixed `lambda_comp=0.1` on Dirichlet NLL.

```python
total = exp(–s_pt) · MSE(pt_pred, pt_target) + ½·s_pt
      + 0.1 · (−mean log Dir(comp_target ; α))
```

`lambda_comp=0.1` is fixed because Dirichlet NLL is unbounded below — Kendall σ_comp would diverge as concentrations sharpen during training.

---

### 3.9 `Trainer`
**File:** `path/spatialmt/training/trainer.py` ✓ Implemented

Step-budget fine-tuning loop. Each step draws a fresh stochastic `(query_cell, context_cells)` ICL pair. Day-11 cells excluded from query pool.

**Warmup schedule:**

| Component | Frozen while | Rationale |
|---|---|---|
| `col_embedder` | `step < warmup_col_steps` (500) | Stabilise gene embeddings before perturbing pretrained backbone |
| `tf_icl` | `step < warmup_icl_steps` (100) | Let heads establish gradient signal before ICL unfreezes |

**`fit()` return:**
```python
{"train_loss": float, "pt_loss": float, "comp_loss": float,
 "loss_history": list[{"step", "train_loss", "pt_loss", "comp_loss"}]}
```

---

### 3.10 `MuonAdamW`
Routes `ndim ≥ 2` parameters (weight matrices) to Muon and `ndim < 2` (biases, LayerNorm, Kendall scalars) to AdamW.

---

### 3.11 `CheckpointCallback`
Saves `{global_step, model_state, optimizer_state, loss_fn_state}` every `every` steps.

---

### 3.12 `SupervisedTrainer`
Batched supervised alternative — uses population-mean anchor, `tf_icl` permanently frozen. Used for head-only fine-tuning or debugging head initialisation independently of the ICL pathway.

---

## 4. Workflow & Data Flow

### 4.1 Training Data Flow

```
h5ad file
    └──► ProcessedDataset.from_anndata(h5ad_path, cfg.data)
              HVG selection (2000 genes, seurat flavor)
              log1p normalisation
              soft labels (distance-to-centroid softmax)
              schema validation (6 invariants)
         │
         ▼
    ContextSampler(dataset, cfg.context)
         Pre-builds: _day_to_indices, _pt_bin_to_indices, _class_to_indices
         │
         ▼ per training step
    sampler.sample(query_cell_id)
         → anchor_ids (day + pseudotime + composition components)
         │
         ▼
    CellTableBuilder.build(query_id, anchor_ids)
         → CellTable (model input), TrainingTargets (loss only)
         │
         ▼
    icl_collate([(table, targets)])
         → ICLBatch (B=1 tensors on CPU)
         │
         ▼
    TabICLRegressor.forward(batch)
         Stage 1: col_embedder  — gene × gene attention
         Stage 2: row_interactor — feature → cell repr
         Stage 3: anchor_label_embedder — inject anchor targets
         Stage 4: tf_icl — query attends to anchor context
         Stage 5: shared_trunk → PseudotimeHead + DirichletCompositionHead
         │
         ▼
    DirichletDualHeadLoss(pt_pred, alpha, targets)
         → (total_loss, pt_loss, comp_loss)
         │
         ▼
    MuonAdamW.step()
```

### 4.2 Inference Flow (Day-11 Evaluation)

```
For each day-11 cell:
    sampler.sample(cid)           → anchor context (training cells only)
    builder.build(cid, anchors)   → CellTable
    icl_collate(...)              → ICLBatch
    model(batch)                  → (pt_pred, alpha)
    alpha / alpha.sum()           → normalised composition prediction

Metrics:
    Pseudotime : Spearman ρ, MAE
    Composition: Wasserstein-1 (PCA ground metric), JSD, Brier score
    Stratified : W-dist per dominant true class
```

---

## 5. Implementation Details

### 5.1 Composition Head Activation

**KL path (`rotation_001`, `composition_loss_type="kl"`):**
```python
def forward(self, x):
    return torch.softmax(self.linear(x), dim=-1)   # (B, K), rows sum to 1.0

# DualHeadLoss.composition_loss
kl = (comp_target * (log_target - log_pred)).sum(dim=-1)
return kl.mean()
```

**Dirichlet NLL path (`rotation_002`, `composition_loss_type="dirichlet"`):**
```python
def forward(self, x):
    # clamp prevents float32 softplus overflow (exp > 88 → inf; inf/inf = NaN in α/Σα)
    return F.softplus(self.linear(x).clamp(max=80.0))

# DirichletDualHeadLoss.composition_loss
return -torch.distributions.Dirichlet(concentrations).log_prob(comp_target).mean()
```

### 5.2 Bias Initialisation

```python
# PseudotimeHead: sigmoid(0.5) ≈ 0.62 — trajectory midpoint prior
nn.init.constant_(self.linear.bias, 0.5)

# CompositionHead: uniform prior at init
nn.init.constant_(self.linear.bias, 0.0)

# Both heads: near-zero weights prevent large initial gradients
nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
```

### 5.3 Soft Label Generation

K=8 states from `class3` annotations. Cluster centroids in PCA space; soft labels via temperature-scaled softmax of negative distances (`label_softening_temperature=1.0`). `centroid_distances` (inter-centroid PCA distances) is stored in `ProcessedDataset` and used as the ground metric for Wasserstein evaluation.

---

## 6. Integration Guide

### 6.1 Training Loop

```python
from spatialmt.config.experiment import ExperimentConfig
from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.context.sampler import ContextSampler
from spatialmt.context.builder import CellTableBuilder
from spatialmt.model.trace import TabICLRegressor
from spatialmt.model.loss import DirichletDualHeadLoss
from spatialmt.training.trainer import Trainer

cfg = ExperimentConfig.full_finetune_dirichlet(run_id="rotation_002")
dataset = ProcessedDataset.from_anndata(h5ad_path, cfg.data)
sampler = ContextSampler(dataset, cfg.context)
builder = CellTableBuilder(dataset)
model = TabICLRegressor(n_genes=dataset.n_genes, k=cfg.data.n_cell_states,
                        composition_loss_type="dirichlet", ...).to(device)
model.load_backbone(cfg.model.backbone_path)
loss_fn = DirichletDualHeadLoss().to(device)

trainer = Trainer(model=model, dataset=dataset, sampler=sampler,
                  builder=builder, loss_fn=loss_fn, config=cfg,
                  n_steps=25000, eval_every=500)
metrics = trainer.fit()
```

### 6.2 Checkpoint Loading (Inference)

```python
ckpt = torch.load("final.pt", weights_only=True)
model.load_state_dict(ckpt["model_state"])
# loss_fn_state included for training resumption; not required for inference
```

---

## 7. Configuration Reference

### 7.1 Hardware Tier Defaults

```python
HARDWARE_TIERS = {
    # max_context_cells = 5 active days × cells_per_bin (D11 excluded at runtime)
    "debug":    {"max_genes": 256,  "max_context_cells": 50},
    "standard": {"max_genes": 512,  "max_context_cells": 50},
    "full":     {"max_genes": 2000, "max_context_cells": 50},
}
```

All tiers share `max_context_cells=50`. This reflects the dataset reality: there are exactly 5 active collection days after D11 exclusion (D5, D7, D16, D21, D30). At `cells_per_bin=10` (full/standard tier) the maximum context window is 5 × 10 = 50. `cells_per_bin` is the only lever; `n_bins` has been removed as dead config.

### 7.2 Myriad Run Configuration (Run 18)

```
N_STEPS              = 25,000
N_ICL_WARMUP_STEPS   = 2,500   (10% warmup ratio — matches working debug runs)
cells_per_bin        = 10      (5 active days × 10 = 50 context cells)
max_genes            = 2000    (full HVG set)
supervised_batch_size = 4      (2000 × 4 = 8k col ops — safe on A100 80GB)
composition_loss_type = dirichlet
```

---

## 8. Test Architecture

### 8.1 Test Layers

| Layer | Location | Runs in CI |
|---|---|---|
| Unit | `tests/unit/` | Yes |
| Integration | `tests/integration/` | Partially |

### 8.2 Current Test Status

**405 tests passing, 0 failing** (as of v1.9.0).

| File | Count | Status |
|---|---|---|
| `test_experiment_config.py` | ~34 | ✓ |
| `test_dataset.py` | ~26 | ✓ |
| `test_trace_model.py` | ~53 | ✓ |
| `test_dual_head_loss.py` | ~26 | ✓ |
| `test_dirichlet_loss.py` | ~6 | ✓ |
| `test_dirichlet_composition_head.py` | ~7 | ✓ |
| `test_context_sampler.py` | ~29 | ✓ |
| `test_cell_table_builder.py` | ~27 | ✓ |
| `test_icl_collate.py` | ~28 | ✓ |
| `test_trainer.py` | ~57 | ✓ |
| `test_supervised_trainer.py` | ~8 | ✓ |
| `test_callbacks.py` | ~16 | ✓ |
| `test_wasserstein.py` | ~19 | ✓ |
| `test_prep.py` (integration) | ~47 | ✓ |
| `test_dataset.py` (integration) | ~18 | ✓ |

### 8.3 Critical Tests

```python
# ProcessedDataset schema
assert dataset.expression.max() < 20.0
assert not np.any(np.isnan(dataset.expression))
assert np.allclose(dataset.soft_labels.sum(axis=1), 1.0, atol=1e-5)

# ContextConfig validation — 5 active days × cells_per_bin + anchors ≤ max_context_cells
with pytest.raises(ValueError):
    ContextConfig(cells_per_bin=11, max_context_cells=50)   # 55 > 50

# Checkpoint round-trip
ckpt = torch.load("final.pt", weights_only=True)
missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=True)
assert missing == [] and unexpected == []

# Config hash excludes run_id
cfg1 = ExperimentConfig.debug_preset(run_id="run_1")
cfg2 = ExperimentConfig.debug_preset(run_id="run_2")
assert cfg1.config_hash == cfg2.config_hash
```

---

## 9. Deployment & Compute

### 9.1 UCL Myriad Job Submission

```bash
# Sync code to Myriad
rsync -avz --exclude='data/' --exclude='experiments/' --exclude='.git/' \
    ./ zcbtcl9@myriad.rc.ucl.ac.uk:~/Scratch/TRACE/

# Submit job
ssh zcbtcl9@myriad.rc.ucl.ac.uk "cd ~/Scratch/TRACE && qsub myriad_trace.sh"

# Retrieve logs
rsync -avz "zcbtcl9@myriad.rc.ucl.ac.uk:~/Scratch/TRACE/logs/trace_JOBID.*" ./
```

**Resources (A100 node):** `h_rt=2:00:00`, `mem=4G`, `gpu=1`, `pe smp 4`, `tmpfs=20G`  
**h5ad staging:** Copied to `$TMPDIR` at job start for faster random access than Lustre home.

### 9.2 Environment Setup

```bash
# On Myriad (handled by myriad_trace.sh)
module load python/miniconda3/24.3.0-0
module load cuda/11.8.0/gnu-10.2.0
conda activate trace
pip install -e "$PROJECT_ROOT" --quiet   # registers spatialmt

# Local
pip install -e .
pytest tests/unit/ -v
```

### 9.3 Run History and Milestones

| Run | Status | Notes |
|---|---|---|
| Runs 1–9 | OOM / config errors | A100 OOM; fixed by `supervised_batch_size=8` |
| Runs 10–15 | Head collapse / nan ρ | `warmup_icl_steps=10000` (40% of training) caused over-adaptation |
| Run 16 | Head collapse | `N_ICL_WARMUP_STEPS=10000` — confirmed cause; warmup reduced to 2500 |
| Run 17 | Crash — `ModuleNotFoundError: No module named 'spatialmt'` | `pip install -e .` missing from job script; warmup still at 1000 |
| **Run 18** | **Submitted** | pip install fixed; warmup=2500; 2000 genes; 50 context cells |

| Date | Milestone | Status |
|---|---|---|
| Week 2 (Apr 1) | AnnData inspected, class3 labels confirmed | ✓ |
| Week 4 (Apr 15) | prep.py + ProcessedDataset implemented | ✓ |
| Week 6 (Apr 29) | Diffusion pseudotime integrated | ✓ |
| Week 7 (May 6) | Full model + training loop implemented; 300+ tests GREEN | ✓ |
| Week 8 (May 13) | Dirichlet head + DirichletDualHeadLoss implemented | ✓ |
| Week 8–9 (May 20) | Context sampling ablation (6 modes) complete; 50-cell config confirmed; 405 tests GREEN | ✓ |
| Week 10 (May 27) | Run 18 results — full config validation on A100 | In progress |
| Week 15 (Jul 3) | Rotation report + talk submitted | Target |

---

## 10. Decision Log

### Original Architecture Decisions (v1.0.0)

| Decision | Rationale |
|---|---|
| TabICLv2 as backbone | Native regression pre-training; clean two-stage attention; ICL with target-aware embeddings |
| Day 11 withheld as test set | Hardest interpolation point — neuroectoderm-to-neuroepithelial transition |
| Dirichlet head over softmax | Models uncertainty over composition; α₀ = Σα_k is a meaningful confidence readout |
| Distance-to-centroid softening | Preserves paper's biologically validated cluster identities; deterministic |
| Matrigel-only training | Linear early trajectory; clear directional ground truth for validation |
| 500-step column attention warmup | Column embeddings re-initialised for gene count; pretrained patterns must stabilise before fine-tuning |

### Training Layer Decisions (v1.7.0)

| # | Decision | Rationale |
|---|---|---|
| T1 | `col_embedder.*` always excluded from `load_backbone()` | Pretrained column embeddings are position-indexed, not gene-name-indexed |
| T2 | Effective batch size = 1 | `Trainer.fit()` always calls `icl_collate([(table, targets)])` — one pair per step |
| T3 | `fit()` returns `loss_history` | Interval averages smooth noisy batch=1 gradients; enables convergence diagnosis without TensorBoard |
| T4 | 3-panel loss curve PNG saved to checkpoint dir | Post-training convergence check without interactive tooling on Myriad |
| T5 | `CheckpointCallback` saves `loss_fn.state_dict()` | Kendall `log_sigma_sq_pt` is a learnable parameter; omitting it resets task weighting on resume |

### Dirichlet NLL Decisions (v1.8.0)

| # | Decision | Rationale |
|---|---|---|
| D1 | Fixed `lambda_comp=0.1` for Dirichlet NLL | Dirichlet NLL is unbounded below; Kendall σ_comp would diverge as concentrations sharpen |
| D2 | Both KL and Dirichlet paths retained | Enables direct comparison; `composition_loss_type` is the single switch |
| D3 | `lr_col` reduced 1e-5 → 1e-6 | Column attention instability on unfreeze; 1e-6 preserves pretrained GRN signal |
| D4 | `SupervisedTrainer` alongside `Trainer` | Batched head-only fine-tuning path; useful for debugging head initialisation |

### Context Sampling & Config Decisions (v1.9.0)

| # | Decision | Rationale |
|---|---|---|
| C1 | `n_bins` removed entirely from `ContextConfig` | Field was dead — `ContextSampler` always bins by actual collection days in the dataset (5 active days). The validator used `n_bins × cells_per_bin` but `n_bins` never matched runtime behaviour. Replaced with hardcoded `5 active days` in the validator. |
| C2 | `max_context_cells=50` for all tiers | There are exactly 5 active collection days (D5, D7, D16, D21, D30). Maximum context window is 5 × cells_per_bin. Old `full` tier value of 100 assumed a nonexistent 10-day dataset. |
| C3 | `cells_per_bin=10` for full/standard presets | 6-mode sampling ablation (1000 steps, 256 genes, seed=42) showed day-50-cell run achieves ρ=0.648 vs ρ=0.584 at 25 cells. More anchors per day gives the ICL transformer a richer developmental timeline. |
| C4 | Pure day-stratified sampling for Myriad run | Ablation confirmed: composition-aware sampling does not fix Prosencephalic failure (structural data limitation); day-stratified preserves best pseudotime accuracy (ρ) which is the primary deliverable metric. |
| C5 | `N_ICL_WARMUP_STEPS=2500` (10% of 25,000) | Run 16 used 10,000 warmup steps (40% ratio) — heads over-adapted to frozen tf_icl; on unfreeze the residual stream shift caused head collapse (ρ=NaN). Debug runs at 100/1000 (10%) consistently achieve ρ~0.65. Fixing the ratio to 10% resolves the collapse. |
| C6 | `max_genes=2000` for full tier | Pipeline natively selects 2000 HVGs via `select_highly_variable_genes`. 1024 was a VRAM engineering cap, not a biological optimum. At `supervised_batch_size=4` the col operation is 2000 × 4 = 8k — safe on A100 80GB. Additional genes may help the ColEmbedding discriminate Prosencephalic lineage-specifying markers (OTX2, FOXG1) from the general neurectoderm background. |
| C7 | Softplus overflow fix: `.clamp(max=80.0)` on `DirichletCompositionHead` input | Float32 `exp(x)` overflows for x > 88, producing `inf` alpha values and `inf/inf = NaN` in normalised composition predictions. Pre-fix baseline metrics (W-dist 1.96) were partially numerical artifacts; post-fix metrics (W-dist 2.09 at 50 cells) are the honest benchmark. |

---

## 11. Future Directions

### 11.1 TRACE as a Biologist Tool

The rotation deliverable establishes TRACE as a developmental coordinate predictor for unannotated scRNA-seq data from the same neurectoderm organoid system. The practical workflow:

1. Biologist generates a new scRNA-seq dataset from an NE organoid experiment.
2. Standard Seurat preprocessing (normalise, log1p, HVG selection against the training gene list).
3. TRACE forward pass: each cell gets a pseudotime coordinate ∈ (0, 1) and a K-vector of lineage affinities.
4. Uncertainty readout via α₀ = Σα_k — cells with low precision are transitional or out-of-distribution.
5. **In-silico perturbation:** zero a gene's expression in the query row, run a second forward pass, read the shift in pseudotime and composition. Composition shift is the primary readout — lineage-specific effects are visible even when pseudotime shift is small.

This makes TRACE useful for drug effect prediction: a compound that perturbs WNT signalling would produce a detectable shift in telencephalic vs. non-telencephalic composition, quantifiable without wet-lab KO experiments.

### 11.2 Known Limitations for Report

- **Prosencephalic progenitors:** W-dist ~4.8 vs baseline 4.54 across all sampler modes. Structural data limitation (207 D11 cells, pseudotime ambiguity at pt~0.47). Recommend reporting as a known caveat; not addressable within the rotation scope.
- **Model generalisation:** TRACE is trained on one dataset (Jain et al. 2025, Matrigel condition). Generalisation to other organoid protocols or other labs' data is unvalidated.
- **col_num_inds=128 bottleneck:** At 2000 genes the ColEmbedding Set Transformer compresses 2000 → 128 inducing points. Whether pretrained weights generalise to this regime is an empirical question answered by Run 18.

### 11.3 TRACE 2.0 — Composition-Aware Sampling

The 6-mode ablation established that composition-stratified anchors do not fix the Prosencephalic failure at debug scale. However, the design of composition-aware sampling guarantees that every cell-type archetype is represented in every context window regardless of class frequency. At full scale (2000 genes, 25,000 steps), this could improve the model's calibration for rare lineages. The three-component sampler infrastructure is already implemented in `ContextSampler`; a future run can test composition-aware sampling at full scale by setting `n_composition_anchors > 0` in the full preset.

### 11.4 Multimodal GRN Pipeline

TRACE is one component of a longer-term multimodal strategy:

1. **TRACE** — pseudotime + composition prediction from scRNA-seq (this rotation)
2. **ATAC model** — chromatin accessibility encoding; identifies open regulatory regions
3. **Dedicated GRN model** — integrates TRACE developmental coordinates with ATAC accessibility to produce a state-dependent, lineage-specific gene regulatory network

In this architecture, TRACE's role is to provide high-quality developmental coordinates that condition the GRN model's regulatory graph on cell state. The column attention weights from TRACE are a supplementary certificate of biological grounding, not the primary GRN output.

### 11.5 State-Dependent Perturbation Sensitivity

The ICL mechanism conditions every forward pass on the context window, making TRACE's predictions state-dependent: the same query cell paired with early-stage anchors vs. late-stage anchors will produce different outputs. This makes in-silico perturbation interpretable in a developmental context — the magnitude of a gene knockdown's effect on composition is a function of the cell's developmental state, not just its expression level. Mapping this state-dependence systematically (perturbation × developmental state → composition shift) is the foundation for the GRN inference strategy described in §11.4.

---

*This document reflects all architectural decisions through v1.9.0 (May 2026). The rotation deliverable is developmental coordinate prediction on NE organoid scRNA-seq; target submission July 3rd.*
