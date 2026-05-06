"""
tabgrn_myriad_run.py — Full training run for TabGRN on Myriad HPC (Option B).

Training strategy
-----------------
Phase 1 — Supervised fine-tuning (this script):
  col_embedder, row_interactor, and the dual output heads are fine-tuned via
  standard mini-batch gradient descent on ALL non-day-11 cells.  tf_icl is
  permanently frozen and excluded from the optimizer.

Phase 2 — ICL inference (separate eval script):
  The fine-tuned model is loaded and run in full ICL mode against day-11 cells:
  context = other day-11 cells (pseudotime + soft_labels provided as anchors),
  query   = one day-11 cell (labels withheld).

Two experimental presets selected by the RUN_PRESET env var:

    RUN_PRESET=kl        (default)  rotation_001 — KL divergence composition loss
    RUN_PRESET=dirichlet            rotation_002 — Dirichlet NLL composition loss

Usage
-----
    python src/train/tabgrn_myriad_run.py

Required env vars (set in SLURM script):
    PROJECT_ROOT   — absolute path to the cloned repo on Myriad
    H5AD_PATH      — absolute path to neurectoderm_with_pseudotime.h5ad
    BACKBONE       — absolute path to tabicl-regressor-v2-20260212.ckpt

Optional env vars:
    RUN_PRESET     — "kl" (default) | "dirichlet"
    N_EPOCHS       — passes over the non-day-11 cell pool (default: 3)
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
from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.model.loss import DirichletDualHeadLoss, DualHeadLoss
from spatialmt.model.tabgrn import TabICLRegressor
from spatialmt.training.callbacks import CheckpointCallback
from spatialmt.training.trainer import SupervisedTrainer

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

N_EPOCHS = int(os.environ.get("N_EPOCHS", "3"))
SEED     = int(os.environ.get("SEED", "42"))

_RUN_PRESET = os.environ.get("RUN_PRESET", "kl").lower()
if _RUN_PRESET not in {"kl", "dirichlet"}:
    raise RuntimeError(
        f"RUN_PRESET must be 'kl' or 'dirichlet', got {_RUN_PRESET!r}"
    )

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


def _build_config(preset: str) -> ExperimentConfig:
    if preset == "dirichlet":
        return ExperimentConfig.full_finetune_dirichlet()
    return ExperimentConfig.full_finetune()


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
    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps

    def on_epoch_end(self, _model, _dataset, step: int) -> None:
        pct = 100 * step / self._n_steps
        print(f"  [step {step:>6d}/{self._n_steps}]  {pct:.0f}% complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = _detect_device()
    cfg    = _build_config(_RUN_PRESET)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Data
    print("\n[1/3] Loading data ...")
    t0 = time.time()
    dataset = ProcessedDataset.from_anndata(H5AD_PATH, cfg.data)
    print(
        f"  {dataset.n_cells} cells  |  {dataset.n_genes} genes  "
        f"({time.time() - t0:.1f}s)"
    )

    n_train_cells  = sum(1 for d in dataset.collection_day if int(d) != 11)
    batch_size     = cfg.model.supervised_batch_size
    steps_per_epoch = int(np.ceil(n_train_cells / batch_size))
    n_steps         = N_EPOCHS * steps_per_epoch
    eval_every      = steps_per_epoch   # checkpoint once per epoch

    print("=" * 60)
    print("TabGRN supervised fine-tuning — Myriad (Option B)")
    print(f"  preset          : {_RUN_PRESET}  (run_id={cfg.run_id})")
    print(f"  device          : {device}")
    print(f"  seed            : {SEED}")
    print(f"  n_train_cells   : {n_train_cells}")
    print(f"  supervised_batch: {batch_size}")
    print(f"  n_epochs        : {N_EPOCHS}")
    print(f"  steps_per_epoch : {steps_per_epoch}")
    print(f"  n_steps         : {n_steps}")
    print(f"  eval_every      : {eval_every}  (one checkpoint per epoch)")
    print(f"  max_genes       : {cfg.data.max_genes}")
    print(f"  h5ad            : {H5AD_PATH}")
    print(f"  backbone        : {BACKBONE_PATH}")
    print(f"  tf_icl          : FROZEN (pretrained weights preserved for ICL inference)")
    print("=" * 60)

    # 2. Model
    print("\n[2/3] Initialising model ...")
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
        composition_loss_type = m.composition_loss_type,
    ).to(device)

    print(f"  Loading backbone from {BACKBONE_PATH}")
    model.load_backbone(BACKBONE_PATH)

    n_params         = sum(p.numel() for p in model.parameters())
    n_frozen_params  = sum(p.numel() for p in model.tf_icl.parameters())
    print(f"  Parameters total  : {n_params:,}")
    print(f"  tf_icl (frozen)   : {n_frozen_params:,}")
    print(f"  Trainable         : {n_params - n_frozen_params:,}")

    loss_fn = (
        DirichletDualHeadLoss() if m.composition_loss_type == "dirichlet"
        else DualHeadLoss()
    ).to(device)

    # 3. Train
    print(f"\n[3/3] Training for {n_steps} steps "
          f"({N_EPOCHS} epochs × {steps_per_epoch} steps/epoch) ...\n")

    ckpt_dir = PROJECT_ROOT / "experiments" / cfg.run_id / "checkpoints"
    checkpoint_cb = CheckpointCallback(
        trainer  = None,
        loss_fn  = loss_fn,
        out_dir  = ckpt_dir,
        every    = eval_every,
    )
    trainer = SupervisedTrainer(
        model      = model,
        dataset    = dataset,
        loss_fn    = loss_fn,
        config     = cfg,
        n_epochs   = N_EPOCHS,
        eval_every = eval_every,
        callbacks  = [ProgressCallback(n_steps), checkpoint_cb],
        seed       = SEED,
    )
    checkpoint_cb.trainer = trainer

    t0 = time.time()
    metrics = trainer.fit()
    elapsed = time.time() - t0

    # Save final model weights as a clean inference artifact
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir / "final_model.pt"
    torch.save(
        {
            "model_state":           model.state_dict(),
            "run_id":                cfg.run_id,
            "composition_loss_type": m.composition_loss_type,
            "global_step":           trainer.global_step,
        },
        final_path,
    )
    print(f"\nFinal model saved to {final_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Training complete")
    print(f"  elapsed     : {elapsed:.1f}s  ({elapsed / n_steps:.3f}s/step)")
    print(f"  train_loss  : {metrics['train_loss']:.4f}")
    print(f"  pt_loss     : {metrics['pt_loss']:.4f}")
    print(f"  comp_loss   : {metrics['comp_loss']:.4f}")
    print("=" * 60)

    cfg.save()
    print(f"\nConfig saved to experiments/{cfg.run_id}/config.json")

    # Loss curve
    history = metrics["loss_history"]
    if history:
        steps       = [r["step"]       for r in history]
        train_loss  = [r["train_loss"] for r in history]
        pt_loss_h   = [r["pt_loss"]    for r in history]
        comp_loss_h = [r["comp_loss"]  for r in history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        comp_label = "Dirichlet NLL" if _RUN_PRESET == "dirichlet" else "KL divergence"
        for ax, values, title in zip(
            axes,
            [train_loss, pt_loss_h, comp_loss_h],
            ["Total loss (Kendall-weighted)", "Pseudotime loss (MSE)",
             f"Composition loss ({comp_label})"],
        ):
            ax.plot(steps, values, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"TabGRN training — {cfg.run_id}  [{comp_label}]", y=1.02)
        fig.tight_layout()
        curve_path = ckpt_dir / "loss_curve.png"
        fig.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Loss curve saved to {curve_path}")


if __name__ == "__main__":
    main()
