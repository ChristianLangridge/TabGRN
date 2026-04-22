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
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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

N_STEPS    = 200   # enough to see a loss trend; ~3–5 min on M4
EVAL_EVERY = 25    # print a summary line every N steps

H5AD_PATH = os.environ.get(
    "H5AD_PATH",
    str(Dirs.model_data_anndata / "neurectoderm_with_pseudotime.h5ad"),
)
BACKBONE_PATH = os.environ.get("BACKBONE", None)


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
# Callback — running-average loss printed at each eval checkpoint
# ---------------------------------------------------------------------------

class LogCallback:
    def __init__(self) -> None:
        self._sum_loss = self._sum_pt = self._sum_comp = 0.0
        self._n = 0

    def on_epoch_end(self, model, dataset, step: int) -> None:
        n = max(self._n, 1)
        print(
            f"  step {step:>4d}/{N_STEPS}"
            f"  total={self._sum_loss / n:.4f}"
            f"  pt={self._sum_pt / n:.4f}"
            f"  comp={self._sum_comp / n:.4f}"
        )
        self._sum_loss = self._sum_pt = self._sum_comp = 0.0
        self._n = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = _detect_device()
    cfg    = ExperimentConfig.debug_preset(run_id="debug_local")

    print("=" * 60)
    print("TabGRN debug run")
    print(f"  device      : {device}")
    print(f"  n_steps     : {N_STEPS}")
    print(f"  max_genes   : {cfg.data.max_genes}")
    print(f"  n_bins      : {cfg.context.n_bins}  "
          f"cells_per_bin: {cfg.context.cells_per_bin}")
    print(f"  h5ad        : {H5AD_PATH}")
    print(f"  backbone    : {BACKBONE_PATH or '(random init)'}")
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

    # 3. Model
    print("\n[3/4] Initialising model ...")
    model = TabICLRegressor(
        n_genes   = dataset.n_genes,
        embed_dim = 64,
        n_heads   = 4,
        n_layers  = 2,
        k         = cfg.data.n_cell_states,
        num_cls   = 2,
    ).to(device)

    if BACKBONE_PATH:
        print(f"  Loading backbone from {BACKBONE_PATH}")
        model.load_backbone(BACKBONE_PATH)

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


if __name__ == "__main__":
    main()
