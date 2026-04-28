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
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_dir() -> "Path":
    from spatialmt.config.paths import PROJECT_ROOT
    d = PROJECT_ROOT / "experiments" / "rotation_001" / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_checkpoint(model: TabICLRegressor, loss_fn: DualHeadLoss, step: int) -> None:
    path = _checkpoint_dir() / f"step_{step:06d}.pt"
    torch.save(
        {
            "step":            step,
            "model_state":     model.state_dict(),
            "loss_fn_state":   loss_fn.state_dict(),
        },
        path,
    )
    print(f"  [ckpt] saved → {path}")


# ---------------------------------------------------------------------------
# Callback — step-progress heartbeat + checkpoint
# ---------------------------------------------------------------------------

class MyriadCallback:
    def __init__(self, model: TabICLRegressor, loss_fn: DualHeadLoss) -> None:
        self._model   = model
        self._loss_fn = loss_fn

    def on_epoch_end(self, _model, _dataset, step: int) -> None:
        pct = 100 * step / N_STEPS
        print(f"  [step {step:>6d}/{N_STEPS}]  {pct:.0f}% complete")
        _save_checkpoint(self._model, self._loss_fn, step)


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

    cb = MyriadCallback(model, loss_fn)
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

    # Save final checkpoint and config
    _save_checkpoint(model, loss_fn, N_STEPS)
    cfg.save()
    print(f"\nConfig saved to experiments/rotation_001/config.json")


if __name__ == "__main__":
    main()
