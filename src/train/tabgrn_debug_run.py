"""
tabgrn_debug_run.py — Local debug training run for TabGRN-ICL.

Uses ExperimentConfig.debug_preset() (256 genes, max_context_cells=30).
Runs 200 gradient steps to verify the pipeline end-to-end before the full
Myriad run. Prints per-step loss and a summary at the end.

Usage
-----
    python src/train/tabgrn_debug_run.py

Optional env vars:
    H5AD_PATH   — path to the h5ad file
                  (default: data/training_data/AnnData/neurectoderm_with_pseudotime.h5ad)
    BACKBONE    — path to a TabICLv2 .ckpt to load pretrained weights (optional)
    DEVICE      — "cpu" | "mps" | "cuda"  (auto-detected if not set)
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
from spatialmt.model.loss import DualHeadLoss
from spatialmt.model.tabgrn import TabICLRegressor
from spatialmt.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

N_STEPS    = 500   # enough to see a loss trend; ~3–5 min on M4
EVAL_EVERY = 25    # print a summary line every N steps
SEED       = 42    # set SEED env var to "" to disable

H5AD_PATH = os.environ.get(
    "H5AD_PATH",
    str(Dirs.model_data_anndata / "neurectoderm_with_pseudotime.h5ad"),
)
# BACKBONE env var overrides cfg.model.backbone_path.
# Leave unset to use the path stored in the preset config.
_BACKBONE_OVERRIDE = os.environ.get("tabicl-regressor-v2-20260212.ckpt", None)

_seed_str = os.environ.get("SEED", str(SEED))
SEED = int(_seed_str) if _seed_str else None


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
    cfg    = ExperimentConfig.debug_preset(run_id="debug_local")

    if SEED is not None:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

    print("=" * 60)
    print("TabGRN debug run")
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
        n_genes      = dataset.n_genes,
        k            = cfg.data.n_cell_states,
        embed_dim    = m.embed_dim,
        n_heads      = m.n_heads,
        num_cls      = m.num_cls,
        col_num_inds = m.col_num_inds,
        n_layers_col = m.n_layers_col,
        n_layers_row = m.n_layers_row,
        n_layers_icl = m.n_layers_icl,
    ).to(device)

    if backbone_path:
        print(f"  Loading backbone from {backbone_path}")
        model.load_backbone(backbone_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    loss_fn = DualHeadLoss().to(device)

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
    loss_fn: DualHeadLoss,
    model: TabICLRegressor,
    cfg: ExperimentConfig,
) -> None:
    import math

    print("\nSanity checks:")

    # Loss values finite
    for k, v in metrics.items():
        tag = "OK  " if math.isfinite(v) else "FAIL"
        print(f"  {tag}  {k} = {v:.4f}")

    # Kendall uncertainty weights learned
    for name in ("log_sigma_sq_pt", "log_sigma_sq_comp"):
        val = getattr(loss_fn, name).item()
        tag = "OK  " if math.isfinite(val) else "FAIL"
        print(f"  {tag}  {name} = {val:.4f}")

    # Freeze schedule: 200 steps < warmup_col_steps (500) → col_embedder frozen
    col_frozen = all(not p.requires_grad for p in model.col_embedder.parameters())
    warmup_remaining = cfg.model.warmup_col_steps - 200
    if col_frozen:
        print(f"  OK    col_embedder frozen ({warmup_remaining} warmup steps remaining)")
    else:
        print("  OK    col_embedder unfrozen (warmup complete)")

    # Heads always trainable
    head_ok = all(p.requires_grad for p in model.pseudotime_head.parameters())
    tag = "OK  " if head_ok else "FAIL"
    print(f"  {tag}  pseudotime_head trainable")


def _inference_check(
    model: TabICLRegressor,
    dataset: ProcessedDataset,
    sampler: ContextSampler,
    builder: CellTableBuilder,
    device: torch.device,
) -> None:
    """Run one forward pass with a day-11 query cell and print predictions.

    Day 11 is the held-out test set — these cells were never used as queries
    during training, so this is the first time the model sees them.
    """
    from spatialmt.context.collate import icl_collate

    print("\n" + "=" * 60)
    print("Inference check — day 11 query cell")

    day11_ids = [
        cid for cid, day in zip(dataset.cell_ids, dataset.collection_day)
        if int(day) == 11
    ]
    if not day11_ids:
        print("  No day-11 cells found in dataset — skipping.")
        return

    query_id = day11_ids[0]
    q_idx    = dataset.cell_ids.index(query_id)

    cats = dataset.cell_type_categories

    # Per-day pseudotime summary (median) to check biological plausibility
    print("\n  Pseudotime medians by collection day:")
    for day in [5, 7, 11, 16, 21, 30]:
        mask = dataset.collection_day == day
        if mask.any():
            med = float(np.median(dataset.pseudotime[mask]))
            print(f"    D{day:>2d} : {med:.4f}  (n={mask.sum()})")

    print(f"\n  query cell      : {query_id}")
    print(f"  true pseudotime : {dataset.pseudotime[q_idx]:.4f}")
    true_comp = dataset.soft_labels[q_idx]
    comp_str = "  |  ".join(f"{cats[i]}: {v:.3f}" for i, v in enumerate(true_comp))
    print(f"  true composition:\n    {comp_str}")

    anchor_ids, _ = sampler.sample(query_id)
    table, _      = builder.build(query_id, anchor_ids)
    batch         = icl_collate([(table, _)])

    # Move to device
    from spatialmt.training.trainer import _batch_to_device
    batch = _batch_to_device(batch, device)

    model.eval()
    with torch.no_grad():
        pt_pred, comp_pred = model(batch)
    model.train()

    pt_val   = pt_pred[0].item()
    comp_val = comp_pred[0].cpu().tolist()

    pred_comp_str = "  |  ".join(f"{cats[i]}: {v:.3f}" for i, v in enumerate(comp_val))
    print(f"\n  pred pseudotime : {pt_val:.4f}")
    print(f"  pred composition:\n    {pred_comp_str}")
    print(f"  |pt error|      : {abs(pt_val - dataset.pseudotime[q_idx]):.4f}")
    print(f"  n context cells : {len(anchor_ids)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
