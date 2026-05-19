"""
trace_debug_run.py — Local debug training run for TRACE-ICL.

Uses ExperimentConfig.debug_preset() (256 genes, max_context_cells=30).
Runs 1000 gradient steps to verify the pipeline end-to-end before the full
Myriad run. Prints per-step loss and a summary at the end.

Usage
-----
    python src/train/trace_debug_run.py

Optional env vars:
    H5AD_PATH   — path to the h5ad file
                  (default: data/training_data/AnnData/neurectoderm_with_pseudotime.h5ad)
    BACKBONE    — path to a TabICLv2 .ckpt to load pretrained weights (optional)
    DEVICE      — "cpu" | "mps" | "cuda"  (auto-detected if not set)
    COMP_LOSS   — "kl" | "dirichlet"  (default: "dirichlet")
                  Selects composition head and paired loss function.
                  "kl"        → CompositionHead (softmax) + DualHeadLoss (Kendall on both heads)
                  "dirichlet" → DirichletCompositionHead (softplus) + DirichletDualHeadLoss (fixed λ)
"""
import os

# macOS OpenMP workaround — must be set before ANY library import.
# PyTorch, sklearn, and numpy each bundle their own libomp.dylib. On macOS the
# dynamic linker loads all three, causing a duplicate-runtime segfault when any
# of them spawns threads. Capping every threading layer to 1 thread prevents the
# crash. MPS (Metal) is unaffected — it has its own threading model.
# None of these vars are needed on Linux/Myriad.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time

import numpy as np
import torch

from spatialmt.config.experiment import ExperimentConfig
from spatialmt.config.paths import Dirs
from spatialmt.context.builder import CellTableBuilder
from spatialmt.context.sampler import ContextSampler
from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.model.loss import DirichletDualHeadLoss, DualHeadLoss
from spatialmt.model.trace import TabICLRegressor
from spatialmt.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------
N_STEPS    = 1000   # enough to see a loss trend; ~3–5 min on M4
EVAL_EVERY = 25    # print a summary line every N steps
SEED       = 42    # set SEED env var to "" to disable

H5AD_PATH = os.environ.get(
    "H5AD_PATH",
    str(Dirs.model_data_anndata / "neurectoderm_with_pseudotime.h5ad"),
)
# BACKBONE env var overrides cfg.model.backbone_path.
# Leave unset to use the path stored in the preset config.
_BACKBONE_OVERRIDE = os.environ.get("BACKBONE", None)

_seed_str = os.environ.get("SEED", str(SEED))
SEED = int(_seed_str) if _seed_str else None

COMP_LOSS = os.environ.get("COMP_LOSS", "dirichlet").lower()
if COMP_LOSS not in ("kl", "dirichlet"):
    raise ValueError(f"COMP_LOSS must be 'kl' or 'dirichlet', got {COMP_LOSS!r}")

# Sampler ablation — controls anchor selection strategy.
# Each mode configures (cells_per_bin, n_pseudotime_anchors, n_composition_anchors):
#   day                  — 5 days × 5 = 25 day-stratified only (original baseline)
#   pseudotime           — 25 pseudotime-binned, no day or composition
#   composition          — 25 composition-stratified, no day or pseudotime
#   day_pseudotime       — 20 day + 20 pseudotime = 40
#   pseudotime_composition — 20 pseudotime + 20 composition = 40
_SAMPLER_CONFIGS: dict[str, dict] = {
    "day":                    {"cells_per_bin": 5, "n_pseudotime_anchors":  0, "n_composition_anchors":  0, "max_context_cells": 30},
    "pseudotime":             {"cells_per_bin": 0, "n_pseudotime_anchors": 25, "n_composition_anchors":  0, "max_context_cells": 30},
    "composition":            {"cells_per_bin": 0, "n_pseudotime_anchors":  0, "n_composition_anchors": 25, "max_context_cells": 30},
    "day_pseudotime":         {"cells_per_bin": 4, "n_pseudotime_anchors": 20, "n_composition_anchors":  0, "max_context_cells": 40},
    "day_composition":        {"cells_per_bin": 4, "n_pseudotime_anchors":  0, "n_composition_anchors": 20, "max_context_cells": 40},
    "pseudotime_composition": {"cells_per_bin": 0, "n_pseudotime_anchors": 20, "n_composition_anchors": 20, "max_context_cells": 40},
}

SAMPLER = os.environ.get("SAMPLER", "day_pseudotime").lower()
if SAMPLER not in _SAMPLER_CONFIGS:
    raise ValueError(
        f"SAMPLER must be one of {list(_SAMPLER_CONFIGS)}, got {SAMPLER!r}"
    )

# Ablation flags
# FREEZE_ICL=1  — keep tf_icl frozen for all N_STEPS (simulates post-Phase-1.5 checkpoint)
# NULL_CONTEXT=1 — run a second inference pass with zeroed context tensors
FREEZE_ICL   = os.environ.get("FREEZE_ICL",   "").lower() in ("1", "true")
NULL_CONTEXT = os.environ.get("NULL_CONTEXT",  "").lower() in ("1", "true")


def _detect_device() -> torch.device:
    env = os.environ.get("DEVICE", "").lower()
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Callback — step-progress heartbeat
# The Trainer calls on_epoch_end(model, dataset, step) — per-step loss values
# are not passed to callbacks; the final averaged metrics come from fit().
# ---------------------------------------------------------------------------

class LogCallback:
    def on_epoch_end(self, _model, _dataset, step: int) -> None:
        print(f"  checkpoint step {step:>4d}/{N_STEPS}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = _detect_device()
    cfg    = ExperimentConfig.debug_preset(run_id=f"debug_local_{COMP_LOSS}_{SAMPLER}")

    # Apply sampler mode — overrides context config from preset
    sc = _SAMPLER_CONFIGS[SAMPLER]
    cfg.context.cells_per_bin            = sc["cells_per_bin"]
    cfg.context.n_pseudotime_anchors     = sc["n_pseudotime_anchors"]
    cfg.context.n_composition_anchors    = sc["n_composition_anchors"]
    cfg.context.max_context_cells        = sc["max_context_cells"]

    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    print("=" * 60)
    print("TRACE debug run")
    print(f"  device      : {device}")
    print(f"  seed        : {SEED if SEED is not None else '(none)'}")
    print(f"  n_steps     : {N_STEPS}")
    print(f"  max_genes   : {cfg.data.max_genes}")
    print(f"  n_bins      : {cfg.context.n_bins}  "
          f"cells_per_bin: {cfg.context.cells_per_bin}")
    # Resolve backbone path: CLI env var > config > none
    backbone_path = _BACKBONE_OVERRIDE or cfg.model.backbone_path
    print(f"  h5ad        : {H5AD_PATH}")
    print(f"  backbone    : {backbone_path or '(random init)'}")
    print(f"  embed_dim   : {cfg.model.embed_dim}  "
          f"d_model: {cfg.model.num_cls * cfg.model.embed_dim}  "
          f"n_heads: {cfg.model.n_heads}")
    print(f"  comp_loss   : {COMP_LOSS}")
    print(f"  sampler     : {SAMPLER}  "
          f"(day={cfg.context.cells_per_bin * len([d for d in [5,7,16,21,30]])}"
          f"  pt={cfg.context.n_pseudotime_anchors}"
          f"  comp={cfg.context.n_composition_anchors}"
          f"  total={cfg.context.cells_per_bin * 5 + cfg.context.n_pseudotime_anchors + cfg.context.n_composition_anchors})")
    if FREEZE_ICL:
        print("  freeze_icl  : ON  (tf_icl frozen throughout — ablation mode)")
    if NULL_CONTEXT:
        print("  null_context: ON  (second inference pass with zeroed context)")
    print("=" * 60)

    # 1. Data
    print("\n[1/4] Loading data ...")
    t0 = time.time()
    dataset = ProcessedDataset.from_anndata(H5AD_PATH, cfg.data)
    print(
        f"  {dataset.n_cells} cells  |  {dataset.n_genes} genes  "
        f"({time.time() - t0:.1f}s)"
    )

    # 2. Context layer
    print("\n[2/4] Building context layer ...")
    sampler = ContextSampler(dataset, cfg.context)
    builder = CellTableBuilder(dataset)

    # 3. Model — arch dims come entirely from cfg.model (locked to TabICLv2 checkpoint)
    print("\n[3/4] Initialising model ...")
    m = cfg.model
    model = TabICLRegressor(
        n_genes               = dataset.n_genes,
        k                     = cfg.data.n_cell_states,
        embed_dim             = m.embed_dim,
        n_heads               = m.n_heads,
        num_cls               = m.num_cls,
        col_num_inds          = m.col_num_inds,
        n_layers_col          = m.n_layers_col,
        n_layers_row          = m.n_layers_row,
        n_layers_icl          = m.n_layers_icl,
        composition_loss_type = COMP_LOSS,
    ).to(device)

    if backbone_path:
        print(f"  Loading backbone from {backbone_path}")
        model.load_backbone(backbone_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    loss_fn = (DirichletDualHeadLoss() if COMP_LOSS == "dirichlet" else DualHeadLoss()).to(device)

    if FREEZE_ICL:
        cfg.model.warmup_icl_steps = N_STEPS + 1  # warmup never completes → tf_icl stays frozen

    # 4. Train
    print(f"\n[4/4] Training for {N_STEPS} steps (eval every {EVAL_EVERY}) ...\n")
    cb = LogCallback()
    trainer = Trainer(
        model      = model,
        dataset    = dataset,
        sampler    = sampler,
        builder    = builder,
        loss_fn    = loss_fn,
        config     = cfg,
        n_steps    = N_STEPS,
        eval_every = EVAL_EVERY,
        callbacks  = [cb],
        seed       = SEED,
    )

    t0 = time.time()
    metrics = trainer.fit()
    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 60)
    print("Training complete")
    print(f"  elapsed     : {elapsed:.1f}s  ({elapsed / N_STEPS:.2f}s/step)")
    print(f"  train_loss  : {metrics['train_loss']:.4f}")
    print(f"  pt_loss     : {metrics['pt_loss']:.4f}")
    print(f"  comp_loss   : {metrics['comp_loss']:.4f}")
    print("=" * 60)

    _sanity_checks(metrics, loss_fn, model, cfg)
    _inference_check(model, dataset, sampler, builder, device)

    cfg.save()
    print(f"\nConfig saved to experiments/{cfg.run_id}/config.json")


def _sanity_checks(
    metrics: dict,
    loss_fn: DirichletDualHeadLoss,
    model: TabICLRegressor,
    cfg: ExperimentConfig,
) -> None:
    import math

    print("\nSanity checks:")

    # Scalar loss values finite
    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        tag = "OK  " if math.isfinite(v) else "FAIL"
        print(f"  {tag}  {k} = {v:.4f}")

    # Kendall uncertainty weights / fixed composition weight
    val = loss_fn.log_sigma_sq_pt.item()
    tag = "OK  " if math.isfinite(val) else "FAIL"
    print(f"  {tag}  log_sigma_sq_pt = {val:.4f}")
    if hasattr(loss_fn, "log_sigma_sq_comp"):
        val = loss_fn.log_sigma_sq_comp.item()
        tag = "OK  " if math.isfinite(val) else "FAIL"
        print(f"  {tag}  log_sigma_sq_comp = {val:.4f}")
    else:
        print(f"  OK    lambda_comp (fixed) = {loss_fn.lambda_comp:.4f}")

    # Freeze schedule: 200 steps < warmup_col_steps (500) → col_embedder frozen
    col_frozen = all(not p.requires_grad for p in model.col_embedder.parameters())
    warmup_remaining = cfg.model.warmup_col_steps - 200
    if col_frozen:
        print(f"  OK    col_embedder frozen ({warmup_remaining} warmup steps remaining)")
    else:
        print("  OK    col_embedder unfrozen (warmup complete)")

    icl_frozen = all(not p.requires_grad for p in model.tf_icl.parameters())
    if FREEZE_ICL:
        tag = "OK  " if icl_frozen else "FAIL"
        print(f"  {tag}  tf_icl frozen throughout (FREEZE_ICL ablation)")
    else:
        status = "frozen" if icl_frozen else "unfrozen (warmup complete)"
        print(f"  OK    tf_icl {status}")

    # Heads always trainable
    head_ok = all(p.requires_grad for p in model.pseudotime_head.parameters())
    tag = "OK  " if head_ok else "FAIL"
    print(f"  {tag}  pseudotime_head trainable")

    # Loss progression
    history = metrics.get("loss_history", [])
    if history:
        print("\n  Loss progression:")
        print(f"  {'step':>6}  {'total':>8}  {'pt':>8}  {'comp':>8}")
        for entry in history:
            print(
                f"  {entry['step']:>6d}  "
                f"{entry['train_loss']:>8.4f}  "
                f"{entry['pt_loss']:>8.4f}  "
                f"{entry['comp_loss']:>8.4f}"
            )


def _inference_check(
    model: TabICLRegressor,
    dataset: ProcessedDataset,
    sampler: ContextSampler,
    builder: CellTableBuilder,
    device: torch.device,
) -> None:
    """Evaluate over all day-11 held-out cells and report dual-head metrics.

    Day 11 is the held-out test set — these cells were never used as queries
    during training, so this is the first time the model sees them.

    Pseudotime head : Spearman ρ (primary), per-day ρ (diagnostic), MAE (error).
    Composition head: Wasserstein EMD (primary), Brier score (diagnostic), JSD (error).
    Wasserstein baseline printed alongside EMD for null-model reference.
    """
    from spatialmt.context.collate import icl_collate
    from spatialmt.eval.metrics import (
        brier_score,
        jsd,
        mae,
        per_day_spearman,
        spearman_r,
        wasserstein_1,
        wasserstein_baseline,
    )
    from spatialmt.training.trainer import _batch_to_device

    print("\n" + "=" * 60)
    print("Inference check — day 11 held-out cells")

    day11_mask = dataset.collection_day == 11
    day11_ids  = [cid for cid, m in zip(dataset.cell_ids, day11_mask) if m]

    if not day11_ids:
        print("  No day-11 cells found in dataset — skipping.")
        return

    cats = dataset.cell_type_categories

    # Per-day pseudotime medians — biological plausibility check
    print("\n  Pseudotime medians by collection day:")
    for day in [5, 7, 11, 16, 21, 30]:
        mask = dataset.collection_day == day
        if mask.any():
            med = float(np.median(dataset.pseudotime[mask]))
            print(f"    D{day:>2d} : {med:.4f}  (n={mask.sum()})")

    def _progress(done: int, total: int, prefix: str, width: int = 30) -> None:
        filled = int(width * done / total)
        bar = "█" * filled + "░" * (width - filled)
        print(f"\r  {prefix} [{bar}] {done}/{total}", end="", flush=True)
        if done == total:
            print()

    # Forward pass over all day-11 cells
    n_day11 = len(day11_ids)
    print(f"\n  Running forward pass over {n_day11} day-11 cells ...")
    model.eval()
    pt_preds:   list[float] = []
    comp_preds: list[list[float]] = []

    with torch.no_grad():
        for i, cid in enumerate(day11_ids):
            anchor_ids, _ = sampler.sample(cid)
            table, label  = builder.build(cid, anchor_ids)
            batch         = icl_collate([(table, label)])
            batch         = _batch_to_device(batch, device)
            pt_out, comp_out = model(batch)
            pt_preds.append(pt_out[0].item())
            alpha = comp_out[0]
            comp_preds.append((alpha / alpha.sum()).cpu().tolist())
            _progress(i + 1, n_day11, prefix="  forward")

    model.train()

    pt_pred_arr   = np.array(pt_preds,  dtype=np.float32)
    comp_pred_arr = np.array(comp_preds, dtype=np.float32)

    day11_indices  = [dataset.cell_ids.index(cid) for cid in day11_ids]
    pt_true_arr    = dataset.pseudotime[day11_indices]
    comp_true_arr  = dataset.soft_labels[day11_indices]
    day11_days_arr = dataset.collection_day[day11_indices]

    train_mask = dataset.collection_day != 11

    # --- Pseudotime metrics ---
    rho       = spearman_r(pt_pred_arr, pt_true_arr)
    pt_mae    = mae(pt_pred_arr, pt_true_arr)
    day_rho   = per_day_spearman(pt_pred_arr, pt_true_arr, day11_days_arr)

    # --- Composition metrics ---
    cost_m = dataset.centroid_distances.astype(np.float64)
    emds: list[float] = []
    for i in range(len(day11_ids)):
        emds.append(wasserstein_1(
            comp_pred_arr[i].astype(np.float64),
            comp_true_arr[i].astype(np.float64),
            cost_m,
        ))
        _progress(i + 1, len(day11_ids), prefix="  emd    ")
    mean_emd  = float(np.mean(emds))
    baseline  = wasserstein_baseline(dataset.soft_labels, train_mask, cost_m)
    bs_mean, bs_per_class = brier_score(comp_pred_arr, comp_true_arr)
    js        = jsd(comp_pred_arr, comp_true_arr)

    # --- Print summary ---
    print("\n  -- Pseudotime head --")
    print(f"  Spearman ρ   : {rho:+.4f}   (primary accuracy)")
    print(f"  MAE          : {pt_mae:.4f}   (error magnitude)")
    if day_rho:
        print("  per-day ρ    :", "  ".join(f"D{d}: {v:+.3f}" for d, v in sorted(day_rho.items())))

    print("\n  -- Composition head --")
    print(f"  Wasserstein  : {mean_emd:.4f}   (primary accuracy, PCA ground metric)")
    print(f"  W. baseline  : {baseline:.4f}   (null model — training mean composition)")
    print(f"  Brier score  : {bs_mean:.4f}   (diagnostic)")
    bs_str = "  ".join(f"{cats[i]}: {v:.4f}" for i, v in enumerate(bs_per_class))
    print(f"  Brier/class  : {bs_str}")
    print(f"  JSD          : {js:.4f}   (error, bounded [0,1])")

    # --- Stratified composition metrics by dominant true class ---
    dominant_class = comp_true_arr.argmax(axis=1)  # (n_day11,) — index into cats
    print("\n  -- Composition head: stratified by dominant true class --")
    print(f"  {'class':<40} {'n':>5}  {'W-dist':>7}  {'JSD':>7}  {'Brier':>7}  {'pt_med':>7}")
    worse_than_baseline: list[tuple[str, np.ndarray]] = []  # (name, stratum_mask)
    for cls_idx, cls_name in enumerate(cats):
        mask = dominant_class == cls_idx
        if not mask.any():
            continue
        cls_emds  = [emds[i] for i, m in enumerate(mask) if m]
        cls_bs, _ = brier_score(comp_pred_arr[mask], comp_true_arr[mask])
        cls_js    = jsd(comp_pred_arr[mask], comp_true_arr[mask])
        cls_w     = float(np.mean(cls_emds))
        cls_pt    = float(np.median(pt_true_arr[mask]))
        flag      = "  *** below baseline" if cls_w >= baseline else ""
        print(
            f"  {cls_name:<40} {mask.sum():>5d}  "
            f"{cls_w:>7.4f}  {cls_js:>7.4f}  {cls_bs:>7.4f}  {cls_pt:>7.4f}"
            f"{flag}"
        )
        if cls_w >= baseline:
            worse_than_baseline.append((cls_name, mask))

    # For any stratum worse than the null baseline, print mean true vs pred composition
    # so we can see what the model is collapsing to.
    for cls_name, mask in worse_than_baseline:
        mean_true = comp_true_arr[mask].mean(axis=0)
        mean_pred = comp_pred_arr[mask].mean(axis=0)
        mean_pt_true = float(pt_true_arr[mask].mean())
        mean_pt_pred = float(pt_pred_arr[mask].mean())
        print(f"\n  -- Failure analysis: '{cls_name}' (W-dist ≥ baseline) --")
        print(f"  pseudotime  true mean {mean_pt_true:.4f}  pred mean {mean_pt_pred:.4f}")
        col_w = 36
        print(f"  {'class':<{col_w}}  {'true_mean':>9}  {'pred_mean':>9}  {'delta':>9}")
        for i, cat in enumerate(cats):
            delta = mean_pred[i] - mean_true[i]
            flag = "  <-- overfit" if delta > 0.05 else ("  <-- underfit" if delta < -0.05 else "")
            print(f"  {cat:<{col_w}}  {mean_true[i]:>9.4f}  {mean_pred[i]:>9.4f}  {delta:>+9.4f}{flag}")

    # Qualitative example — first day-11 cell
    print("\n  -- Example: first day-11 cell --")
    true_comp_str = "  |  ".join(f"{cats[i]}: {v:.3f}" for i, v in enumerate(comp_true_arr[0]))
    pred_comp_str = "  |  ".join(f"{cats[i]}: {v:.3f}" for i, v in enumerate(comp_pred_arr[0]))
    print(f"  cell id         : {day11_ids[0]}")
    print(f"  true pseudotime : {pt_true_arr[0]:.4f}")
    print(f"  pred pseudotime : {pt_pred_arr[0]:.4f}  (|error| {abs(pt_pred_arr[0] - pt_true_arr[0]):.4f})")
    print(f"  true composition:\n    {true_comp_str}")
    print(f"  pred composition:\n    {pred_comp_str}")
    anchor_ids_ex, _ = sampler.sample(day11_ids[0])
    print(f"  n context cells : {len(anchor_ids_ex)}")

    # -- Null-context ablation --
    if NULL_CONTEXT:
        print("\n  -- Null-context ablation (context zeroed) --")
        null_pt_preds:   list[float] = []
        null_comp_preds: list[list[float]] = []

        with torch.no_grad():
            for i, cid in enumerate(day11_ids):
                anchor_ids, _ = sampler.sample(cid)
                table, label  = builder.build(cid, anchor_ids)
                batch         = icl_collate([(table, label)])
                batch         = _batch_to_device(batch, device)
                # Zero all context tensors — expression, pseudotime, soft labels
                batch.context_expression  = torch.zeros_like(batch.context_expression)
                batch.context_pseudotime  = torch.zeros_like(batch.context_pseudotime)
                batch.context_soft_labels = torch.zeros_like(batch.context_soft_labels)
                pt_out, comp_out = model(batch)
                null_pt_preds.append(pt_out[0].item())
                alpha = comp_out[0]
                null_comp_preds.append((alpha / alpha.sum()).cpu().tolist())
                _progress(i + 1, n_day11, prefix="  null  ")

        null_pt_arr   = np.array(null_pt_preds,   dtype=np.float32)
        null_comp_arr = np.array(null_comp_preds,  dtype=np.float32)

        null_rho     = spearman_r(null_pt_arr, pt_true_arr)
        null_mae     = mae(null_pt_arr, pt_true_arr)
        null_emds    = [
            wasserstein_1(
                null_comp_arr[i].astype(np.float64),
                comp_true_arr[i].astype(np.float64),
                cost_m,
            )
            for i in range(len(day11_ids))
        ]
        null_emd  = float(np.mean(null_emds))
        null_bs, _  = brier_score(null_comp_arr, comp_true_arr)
        null_js     = jsd(null_comp_arr, comp_true_arr)

        print(f"  Spearman ρ   : {null_rho:+.4f}   Δ {null_rho - rho:+.4f} vs real context")
        print(f"  MAE          : {null_mae:.4f}   Δ {null_mae - pt_mae:+.4f}")
        print(f"  Wasserstein  : {null_emd:.4f}   Δ {null_emd - mean_emd:+.4f}")
        print(f"  Brier score  : {null_bs:.4f}   Δ {null_bs - bs_mean:+.4f}")
        print(f"  JSD          : {null_js:.4f}   Δ {null_js - js:+.4f}")
        print("  (positive Δ = null context is worse, negative = real context was hurting)")

    print("=" * 60)


if __name__ == "__main__":
    main()
