"""
tabgrn_myriad_run.py — Full training run for TabGRN-ICL on Myriad HPC.

Uses ExperimentConfig.rotation_finetune() (512 genes, batch=16, standard tier).
Runs 10 000 gradient steps with checkpointing every 1 000 steps.

Usage
-----
    python src/train/tabgrn_myriad_run.py

Required env vars (set in SLURM script):
    PROJECT_ROOT   — absolute path to the cloned repo on Myriad
    H5AD_PATH      — absolute path to neurectoderm_with_pseudotime.h5ad
    BACKBONE       — absolute path to tabicl-regressor-v2-20260212.ckpt

Optional env vars:
    SEED           — integer RNG seed (default: 42)
    DEVICE         — "cpu" | "cuda"  (auto-detected if not set)
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from spatialmt.config.experiment import ExperimentConfig
from spatialmt.config.paths import PROJECT_ROOT
from spatialmt.context.builder import CellTableBuilder
from spatialmt.context.sampler import ContextSampler
from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.model.loss import DualHeadLoss
from spatialmt.model.tabgrn import TabICLRegressor
from spatialmt.training.callbacks import CheckpointCallback
from spatialmt.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

N_STEPS        = 10_000
EVAL_EVERY     = 1_000   # checkpoint + progress log every N steps
SEED           = int(os.environ.get("SEED", "42"))

H5AD_PATH = os.environ.get("H5AD_PATH")
if not H5AD_PATH:
    raise RuntimeError(
        "H5AD_PATH env var is not set.\n"
        "Add: export H5AD_PATH=/path/to/neurectoderm_with_pseudotime.h5ad"
    )

BACKBONE_PATH = os.environ.get("BACKBONE")
if not BACKBONE_PATH:
    raise RuntimeError(
        "BACKBONE env var is not set.\n"
        "Add: export BACKBONE=/path/to/tabicl-regressor-v2-20260212.ckpt"
    )


def _detect_device() -> torch.device:
    env = os.environ.get("DEVICE", "").lower()
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Callback — step-progress heartbeat
# ---------------------------------------------------------------------------

class ProgressCallback:
    def on_epoch_end(self, _model, _dataset, step: int) -> None:
        pct = 100 * step / N_STEPS
        print(f"  [step {step:>6d}/{N_STEPS}]  {pct:.0f}% complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = _detect_device()
    cfg    = ExperimentConfig.rotation_finetune(run_id="rotation_001")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("TabGRN full training run — Myriad")
    print(f"  device      : {device}")
    print(f"  seed        : {SEED}")
    print(f"  n_steps     : {N_STEPS}")
    print(f"  eval_every  : {EVAL_EVERY}  (checkpoint interval)")
    print(f"  max_genes   : {cfg.data.max_genes}")
    print(f"  n_bins      : {cfg.context.n_bins}  "
          f"cells_per_bin: {cfg.context.cells_per_bin}")
    print(f"  h5ad        : {H5AD_PATH}")
    print(f"  backbone    : {BACKBONE_PATH}")
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

    # 3. Model
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

    print(f"  Loading backbone from {BACKBONE_PATH}")
    model.load_backbone(BACKBONE_PATH)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    loss_fn = DualHeadLoss().to(device)

    # 4. Train
    print(f"\n[4/4] Training for {N_STEPS} steps "
          f"(checkpoint every {EVAL_EVERY}) ...\n")

    ckpt_dir = PROJECT_ROOT / "experiments" / "rotation_001" / "checkpoints"
    progress_cb   = ProgressCallback()
    checkpoint_cb = CheckpointCallback(
        trainer  = None,   # set after Trainer is constructed below
        loss_fn  = loss_fn,
        out_dir  = ckpt_dir,
        every    = EVAL_EVERY,
    )
    trainer = Trainer(
        model      = model,
        dataset    = dataset,
        sampler    = sampler,
        builder    = builder,
        loss_fn    = loss_fn,
        config     = cfg,
        n_steps    = N_STEPS,
        eval_every = EVAL_EVERY,
        callbacks  = [progress_cb, checkpoint_cb],
        seed       = SEED,
    )
    checkpoint_cb.trainer = trainer   # wire back after construction

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

    cfg.save()
    print(f"\nConfig saved to experiments/rotation_001/config.json")

    # Loss curve
    history = metrics["loss_history"]
    if history:
        steps      = [r["step"]       for r in history]
        train_loss = [r["train_loss"] for r in history]
        pt_loss_h  = [r["pt_loss"]    for r in history]
        comp_loss_h = [r["comp_loss"] for r in history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, values, title in zip(
            axes,
            [train_loss, pt_loss_h, comp_loss_h],
            ["Total loss (Kendall-weighted)", "Pseudotime loss (MSE)", "Composition loss (KL)"],
        ):
            ax.plot(steps, values, linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        fig.suptitle("TabGRN training — rotation_001", y=1.02)
        fig.tight_layout()
        curve_path = ckpt_dir / "loss_curve.png"
        fig.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Loss curve saved to {curve_path}")


if __name__ == "__main__":
    main()
