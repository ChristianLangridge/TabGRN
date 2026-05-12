"""
tabgrn_myriad_run.py — Full training run for TabGRN on Myriad HPC (Option B).

Training strategy
-----------------
Phase 1   — Supervised fine-tuning on all non-day-11 cells; tf_icl frozen.
Phase 1.5 — ICL warm-up; trains anchor_label_embedder with tf_icl still frozen.
Phase 2   — ICL inference on day-11 queries (separate eval script).

Two experimental presets selected by the RUN_PRESET env var:

    RUN_PRESET=dirichlet (default)  rotation_002 — Dirichlet NLL composition loss
    RUN_PRESET=kl                   rotation_001 — KL divergence composition loss

Usage
-----
    python src/train/tabgrn_myriad_run.py

Required env vars (set in job script):
    PROJECT_ROOT   — absolute path to the cloned repo on Myriad
    H5AD_PATH      — absolute path to neurectoderm_with_pseudotime.h5ad
    BACKBONE       — absolute path to tabicl-regressor-v2-20260212.ckpt

Optional env vars:
    RUN_PRESET          — "dirichlet" (default) | "kl"
    N_EPOCHS            — passes over the non-day-11 cell pool (default: 3)
    N_ICL_WARMUP_STEPS  — ICL warm-up steps (default: 1000)
    SEED                — integer RNG seed (default: 42)
    DEVICE              — "cpu" | "cuda"  (auto-detected if not set)
"""
import math
import os
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from spatialmt.config.experiment import ExperimentConfig
from spatialmt.config.paths import PROJECT_ROOT
from spatialmt.context.builder import CellTableBuilder
from spatialmt.context.sampler import ContextSampler
from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.model.loss import DirichletDualHeadLoss, DualHeadLoss
from spatialmt.model.tabgrn import TabICLRegressor
from spatialmt.training.callbacks import CheckpointCallback
from spatialmt.training.trainer import SupervisedTrainer, Trainer

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

N_EPOCHS           = int(os.environ.get("N_EPOCHS", "3"))
N_ICL_WARMUP_STEPS = int(os.environ.get("N_ICL_WARMUP_STEPS", "1000"))
SEED               = int(os.environ.get("SEED", "42"))
JOB_ID             = os.environ.get("JOB_ID", "local")

_RUN_PRESET = os.environ.get("RUN_PRESET", "dirichlet").lower()
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
# Callback — step-progress heartbeat (matches debug run style)
# ---------------------------------------------------------------------------

class ProgressCallback:
    def __init__(self, n_steps: int, phase: str = "") -> None:
        self._n_steps = n_steps
        self._phase   = f"[{phase}] " if phase else ""

    def on_epoch_end(self, _model, _dataset, step: int) -> None:
        print(f"  {self._phase}checkpoint step {step:>6d}/{self._n_steps}")


# ---------------------------------------------------------------------------
# Sanity checks — mirrors debug run; called after each training phase
# ---------------------------------------------------------------------------

def _sanity_checks(
    metrics: dict,
    loss_fn,
    model: TabICLRegressor,
    global_step: int,
    warmup_col_steps: int,
    phase: str,
) -> None:
    print(f"\nSanity checks ({phase}):")

    for k, v in metrics.items():
        if not isinstance(v, (int, float)):
            continue
        tag = "OK  " if math.isfinite(v) else "FAIL"
        print(f"  {tag}  {k} = {v:.4f}")

    val = loss_fn.log_sigma_sq_pt.item()
    tag = "OK  " if math.isfinite(val) else "FAIL"
    print(f"  {tag}  log_sigma_sq_pt = {val:.4f}")
    if hasattr(loss_fn, "log_sigma_sq_comp"):
        val = loss_fn.log_sigma_sq_comp.item()
        tag = "OK  " if math.isfinite(val) else "FAIL"
        print(f"  {tag}  log_sigma_sq_comp = {val:.4f}")
    else:
        print(f"  OK    lambda_comp (fixed) = {loss_fn.lambda_comp:.4f}")

    col_frozen = all(not p.requires_grad for p in model.col_embedder.parameters())
    if col_frozen:
        remaining = warmup_col_steps - global_step
        print(f"  OK    col_embedder frozen ({remaining} warmup steps remaining)")
    else:
        print("  OK    col_embedder unfrozen (warmup complete)")

    head_ok = all(p.requires_grad for p in model.pseudotime_head.parameters())
    tag = "OK  " if head_ok else "FAIL"
    print(f"  {tag}  pseudotime_head trainable")

    history = metrics.get("loss_history", [])
    if history:
        print(f"\n  Loss progression ({phase}):")
        print(f"  {'step':>6}  {'total':>10}  {'pt':>8}  {'comp':>10}")
        for entry in history:
            print(
                f"  {entry['step']:>6d}  "
                f"{entry['train_loss']:>10.4f}  "
                f"{entry['pt_loss']:>8.4f}  "
                f"{entry['comp_loss']:>10.4f}"
            )


# ---------------------------------------------------------------------------
# Inference check — mirrors debug run exactly
# ---------------------------------------------------------------------------

def _inference_check(
    model: TabICLRegressor,
    dataset: ProcessedDataset,
    sampler: ContextSampler,
    builder: CellTableBuilder,
    device: torch.device,
    loss_type: str,
) -> None:
    """Evaluate over all day-11 held-out cells and report dual-head metrics."""
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

    print("\n  Pseudotime medians by collection day:")
    for day in [5, 7, 11, 16, 21, 30]:
        mask = dataset.collection_day == day
        if mask.any():
            med = float(np.median(dataset.pseudotime[mask]))
            print(f"    D{day:>2d} : {med:.4f}  (n={mask.sum()})")

    n_day11 = len(day11_ids)
    print(f"\n  Running forward pass over {n_day11} day-11 cells ...")
    model.eval()
    pt_preds:   list[float] = []
    comp_preds: list[list[float]] = []
    log_every = max(1, n_day11 // 10)

    with torch.no_grad():
        for i, cid in enumerate(day11_ids):
            anchor_ids, _ = sampler.sample(cid)
            table, label  = builder.build(cid, anchor_ids)
            batch         = icl_collate([(table, label)])
            batch         = _batch_to_device(batch, device)
            pt_out, comp_out = model(batch)
            pt_preds.append(pt_out[0].item())
            if loss_type == "dirichlet":
                alpha = comp_out[0]
                comp_preds.append((alpha / alpha.sum()).cpu().tolist())
            else:
                comp_preds.append(comp_out[0].cpu().tolist())
            if (i + 1) % log_every == 0 or (i + 1) == n_day11:
                print(f"  forward {i + 1}/{n_day11}")

    model.train()

    pt_pred_arr   = np.array(pt_preds,  dtype=np.float32)
    comp_pred_arr = np.array(comp_preds, dtype=np.float32)

    _id_to_idx     = {cid: i for i, cid in enumerate(dataset.cell_ids)}
    day11_indices  = [_id_to_idx[cid] for cid in day11_ids]
    pt_true_arr    = dataset.pseudotime[day11_indices]
    comp_true_arr  = dataset.soft_labels[day11_indices]
    day11_days_arr = dataset.collection_day[day11_indices]
    train_mask     = dataset.collection_day != 11

    rho     = spearman_r(pt_pred_arr, pt_true_arr)
    pt_mae  = mae(pt_pred_arr, pt_true_arr)
    day_rho = per_day_spearman(pt_pred_arr, pt_true_arr, day11_days_arr)

    cost_m = dataset.centroid_distances.astype(np.float64)
    emds: list[float] = []
    for i in range(len(day11_ids)):
        emds.append(wasserstein_1(
            comp_pred_arr[i].astype(np.float64),
            comp_true_arr[i].astype(np.float64),
            cost_m,
        ))
        if (i + 1) % log_every == 0 or (i + 1) == len(day11_ids):
            print(f"  emd     {i + 1}/{len(day11_ids)}")
    mean_emd             = float(np.mean(emds))
    baseline             = wasserstein_baseline(dataset.soft_labels, train_mask, cost_m)
    bs_mean, bs_per_class = brier_score(comp_pred_arr, comp_true_arr)
    js                   = jsd(comp_pred_arr, comp_true_arr)

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
    print("=" * 60)


# ---------------------------------------------------------------------------
# Loss curve helper
# ---------------------------------------------------------------------------

def _save_loss_curve(
    history: list[dict],
    ckpt_dir,
    run_id: str,
    comp_label: str,
    phase: str,
) -> None:
    if not history:
        return
    steps       = [r["step"]       for r in history]
    train_loss  = [r["train_loss"] for r in history]
    pt_loss_h   = [r["pt_loss"]    for r in history]
    comp_loss_h = [r["comp_loss"]  for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, values, title in zip(
        axes,
        [train_loss, pt_loss_h, comp_loss_h],
        ["Total loss", "Pseudotime loss (MSE)", f"Composition loss ({comp_label})"],
    ):
        ax.plot(steps, values, linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"TabGRN {phase} — {run_id}  [{comp_label}]", y=1.02)
    fig.tight_layout()
    curve_path = ckpt_dir / f"loss_curve_{phase.replace(' ', '_')}.png"
    fig.savefig(str(curve_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved to {curve_path}")


# ---------------------------------------------------------------------------
# Checkpoint sync
# ---------------------------------------------------------------------------

def _git_sync(path, label: str) -> None:
    """Stage, commit, and push a checkpoint file/directory to the remote."""
    rel = str(path)
    cmds = [
        ["git", "-C", str(PROJECT_ROOT), "add", rel],
        ["git", "-C", str(PROJECT_ROOT), "commit", "-m", f"[hpc] {label} checkpoint  job={JOB_ID}"],
        ["git", "-C", str(PROJECT_ROOT), "push"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [sync] WARNING: '{' '.join(cmd[2:])}' failed:\n{result.stderr.strip()}")
            return
    print(f"  [sync] pushed {label} checkpoint to remote")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = _detect_device()
    cfg    = _build_config(_RUN_PRESET)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    comp_label = "Dirichlet NLL" if _RUN_PRESET == "dirichlet" else "KL divergence"

    # 1. Data
    print("\n[1/3] Loading data ...")
    t0 = time.time()
    dataset = ProcessedDataset.from_anndata(H5AD_PATH, cfg.data)  # type: ignore[arg-type]
    print(
        f"  {dataset.n_cells} cells  |  {dataset.n_genes} genes  "
        f"({time.time() - t0:.1f}s)"
    )

    n_train_cells   = sum(1 for d in dataset.collection_day if int(d) != 11)
    batch_size      = cfg.model.supervised_batch_size
    steps_per_epoch = int(np.ceil(n_train_cells / batch_size))
    n_steps         = N_EPOCHS * steps_per_epoch
    eval_every      = max(50, steps_per_epoch // 4)

    print("=" * 60)
    print("TabGRN Myriad run")
    print(f"  preset          : {_RUN_PRESET}  ({comp_label})  run_id={cfg.run_id}")
    print(f"  device          : {device}")
    print(f"  seed            : {SEED}")
    print(f"  max_genes       : {cfg.data.max_genes}")
    print(f"  n_bins          : {cfg.context.n_bins}  cells_per_bin: {cfg.context.cells_per_bin}")
    print(f"  embed_dim       : {cfg.model.embed_dim}  "
          f"d_model: {cfg.model.num_cls * cfg.model.embed_dim}  "
          f"n_heads: {cfg.model.n_heads}")
    print(f"  h5ad            : {H5AD_PATH}")
    print(f"  backbone        : {BACKBONE_PATH}")
    print("  --- Phase 1: supervised fine-tuning ---")
    print(f"  n_train_cells   : {n_train_cells}")
    print(f"  supervised_batch: {batch_size}")
    print(f"  n_epochs        : {N_EPOCHS}")
    print(f"  steps_per_epoch : {steps_per_epoch}")
    print(f"  n_steps         : {n_steps}")
    print(f"  tf_icl          : FROZEN")
    print("  --- Phase 1.5: ICL warm-up ---")
    print(f"  n_icl_warmup    : {N_ICL_WARMUP_STEPS}")
    print(f"  col_embedder    : immediately trainable (warmup_col_steps=0)")
    print(f"  tf_icl          : FROZEN")
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

    n_params        = sum(p.numel() for p in model.parameters())
    n_frozen_params = sum(p.numel() for p in model.tf_icl.parameters())
    print(f"  Parameters total  : {n_params:,}")
    print(f"  tf_icl (frozen)   : {n_frozen_params:,}")
    print(f"  Trainable         : {n_params - n_frozen_params:,}")

    loss_fn = (
        DirichletDualHeadLoss() if m.composition_loss_type == "dirichlet"
        else DualHeadLoss()
    ).to(device)

    # 3. Phase 1 — supervised fine-tuning
    print(f"\n[3/3] Phase 1: supervised fine-tuning "
          f"({N_EPOCHS} epochs × {steps_per_epoch} steps/epoch) ...\n")

    ckpt_dir = PROJECT_ROOT / "experiments" / cfg.run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        trainer = None,
        loss_fn = loss_fn,
        out_dir = ckpt_dir,
        every   = eval_every,
    )
    trainer = SupervisedTrainer(
        model      = model,
        dataset    = dataset,
        loss_fn    = loss_fn,
        config     = cfg,
        n_epochs   = N_EPOCHS,
        eval_every = eval_every,
        callbacks  = [ProgressCallback(n_steps, "Phase 1"), checkpoint_cb],
        seed       = SEED,
    )
    checkpoint_cb.trainer = trainer

    t0 = time.time()
    metrics = trainer.fit()
    elapsed_p1 = time.time() - t0

    print("\n" + "=" * 60)
    print("Phase 1 complete")
    print(f"  elapsed     : {elapsed_p1:.1f}s  ({elapsed_p1 / n_steps:.3f}s/step)")
    print(f"  train_loss  : {metrics['train_loss']:.4f}")
    print(f"  pt_loss     : {metrics['pt_loss']:.4f}")
    print(f"  comp_loss   : {metrics['comp_loss']:.4f}")
    print("=" * 60)

    _sanity_checks(metrics, loss_fn, model, trainer.global_step,
                   cfg.model.warmup_col_steps, "Phase 1")
    _save_loss_curve(metrics["loss_history"], ckpt_dir, cfg.run_id, comp_label, "phase1")

    final_path = ckpt_dir / "phase1_final.pt"
    torch.save(
        {
            "model_state":           model.state_dict(),
            "loss_fn_state":         loss_fn.state_dict(),
            "run_id":                cfg.run_id,
            "composition_loss_type": m.composition_loss_type,
            "global_step":           trainer.global_step,
        },
        final_path,
    )
    print(f"\nPhase 1 model saved to {final_path}")
    _git_sync(final_path, "phase1")

    # Phase 1.5 — ICL warm-up
    print(f"\n[Phase 1.5] ICL warm-up for {N_ICL_WARMUP_STEPS} steps ...\n")

    warmup_cfg = ExperimentConfig.icl_warmup_preset(
        run_id=cfg.run_id + "_warmup",
        composition_loss_type=m.composition_loss_type,
        n_warmup_steps=N_ICL_WARMUP_STEPS,
    )
    warmup_ckpt_dir = PROJECT_ROOT / "experiments" / cfg.run_id / "warmup_checkpoints"
    warmup_ckpt_dir.mkdir(parents=True, exist_ok=True)

    warmup_checkpoint_cb = CheckpointCallback(
        trainer = None,
        loss_fn = loss_fn,
        out_dir = warmup_ckpt_dir,
        every   = N_ICL_WARMUP_STEPS // 2,
    )
    sampler = ContextSampler(dataset, warmup_cfg.context)
    builder = CellTableBuilder(dataset)
    warmup_trainer = Trainer(
        model      = model,
        dataset    = dataset,
        sampler    = sampler,
        builder    = builder,
        loss_fn    = loss_fn,
        config     = warmup_cfg,
        n_steps    = N_ICL_WARMUP_STEPS,
        eval_every = N_ICL_WARMUP_STEPS // 10,
        callbacks  = [ProgressCallback(N_ICL_WARMUP_STEPS, "Phase 1.5"), warmup_checkpoint_cb],
        seed       = SEED,
    )
    warmup_checkpoint_cb.trainer = warmup_trainer

    t_warmup = time.time()
    warmup_metrics = warmup_trainer.fit()
    elapsed_warmup = time.time() - t_warmup

    print("\n" + "=" * 60)
    print("Phase 1.5 complete")
    print(f"  elapsed     : {elapsed_warmup:.1f}s  ({elapsed_warmup / N_ICL_WARMUP_STEPS:.3f}s/step)")
    print(f"  train_loss  : {warmup_metrics['train_loss']:.4f}")
    print(f"  pt_loss     : {warmup_metrics['pt_loss']:.4f}")
    print(f"  comp_loss   : {warmup_metrics['comp_loss']:.4f}")
    print("=" * 60)

    _sanity_checks(warmup_metrics, loss_fn, model, warmup_trainer.global_step,
                   warmup_cfg.model.warmup_col_steps, "Phase 1.5")
    _save_loss_curve(warmup_metrics["loss_history"], warmup_ckpt_dir, cfg.run_id,
                     comp_label, "phase1.5")

    warmup_final_path = warmup_ckpt_dir / "warmup_final.pt"
    torch.save(
        {
            "model_state":           model.state_dict(),
            "loss_fn_state":         loss_fn.state_dict(),
            "run_id":                cfg.run_id,
            "composition_loss_type": m.composition_loss_type,
            "global_step":           warmup_trainer.global_step,
        },
        warmup_final_path,
    )
    print(f"\nPre-ICL checkpoint saved to {warmup_final_path}")
    _git_sync(warmup_final_path, "phase1.5")

    # Inference check on day-11 held-out cells
    _inference_check(model, dataset, sampler, builder, device, _RUN_PRESET)

    cfg.save()
    warmup_cfg.save()
    print(f"\nConfig saved to experiments/{cfg.run_id}/config.json")
    print(f"Warmup config saved to experiments/{warmup_cfg.run_id}/config.json")
    _git_sync(PROJECT_ROOT / "experiments" / cfg.run_id, "final (configs + loss curves)")


if __name__ == "__main__":
    main()
