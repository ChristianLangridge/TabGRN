"""
trace_myriad_run.py — Full training run for TRACE on Myriad HPC.

Single ICL training run using Trainer throughout. tf_icl is frozen for the
first N_WARMUP_STEPS (col_embedder, tf_col, tf_row, and prediction heads
condition on real ICL representations before the ICL mechanism is exposed).
After N_WARMUP_STEPS, tf_icl unfreezes and all components train together.

gene_coexpression.pt is saved at the N_WARMUP_STEPS boundary — this is the
canonical GRN extraction checkpoint (tf_col / tf_row uncontaminated by the
ICL mechanism).

Two experimental presets selected by the RUN_PRESET env var:

    RUN_PRESET=dirichlet (default)  rotation_002 — Dirichlet NLL composition loss
    RUN_PRESET=kl                   rotation_001 — KL divergence composition loss

Usage
-----
    python src/train/trace_myriad_run.py

Required env vars (set in job script):
    PROJECT_ROOT   — absolute path to the cloned repo on Myriad
    H5AD_PATH      — absolute path to neurectoderm_with_pseudotime.h5ad
    BACKBONE       — absolute path to tabicl-regressor-v2-20260212.ckpt

Optional env vars:
    RUN_PRESET      — "dirichlet" (default) | "kl"
    N_STEPS         — total training steps (default: 15000)
    N_WARMUP_STEPS  — steps before tf_icl unfreezes (default: 10000)
    SEED            — integer RNG seed (default: 42)
    DEVICE          — "cpu" | "cuda"  (auto-detected if not set)
"""
import math
import os
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from spatialmt.config.experiment import ExperimentConfig
from spatialmt.config.paths import PROJECT_ROOT
from spatialmt.context.builder import CellTableBuilder
from spatialmt.context.sampler import ContextSampler
from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.model.loss import DirichletDualHeadLoss, DualHeadLoss
from spatialmt.model.trace import TabICLRegressor
from spatialmt.training.callbacks import CheckpointCallback, WarmupBoundaryCallback
from spatialmt.training.trainer import Trainer

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

N_STEPS        = int(os.environ.get("N_STEPS",        "25000"))
N_WARMUP_STEPS = int(os.environ.get("N_WARMUP_STEPS", "2500"))
SEED           = int(os.environ.get("SEED", "42"))
JOB_ID         = os.environ.get("JOB_ID", "local")

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
# Callbacks
# ---------------------------------------------------------------------------

class ProgressCallback:
    def __init__(self, n_steps: int) -> None:
        self._n_steps = n_steps

    def on_epoch_end(self, _model, _dataset, step: int) -> None:
        print(f"  checkpoint step {step:>6d}/{self._n_steps}")


class _SyncingWarmupBoundaryCallback(WarmupBoundaryCallback):
    """WarmupBoundaryCallback that also git-syncs after saving."""

    def on_epoch_end(self, model, dataset, step: int) -> None:
        already_saved = self._saved
        super().on_epoch_end(model, dataset, step)
        if self._saved and not already_saved:
            _git_sync(self.out_path, "gene_coexpression (GRN extraction point)")



# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _sanity_checks(
    metrics: dict,
    loss_fn,
    model: TabICLRegressor,
    global_step: int,
) -> None:
    print("\nSanity checks:")

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
    status = "frozen" if col_frozen else "unfrozen (warmup complete)"
    print(f"  OK    col_embedder {status}")

    icl_frozen = all(not p.requires_grad for p in model.tf_icl.parameters())
    if global_step < N_WARMUP_STEPS:
        tag = "OK  " if icl_frozen else "FAIL"
        print(f"  {tag}  tf_icl frozen ({N_WARMUP_STEPS - global_step} warmup steps remaining)")
    else:
        tag = "OK  " if not icl_frozen else "FAIL"
        print(f"  {tag}  tf_icl unfrozen (warmup complete at step {N_WARMUP_STEPS})")

    head_ok = all(p.requires_grad for p in model.pseudotime_head.parameters())
    tag = "OK  " if head_ok else "FAIL"
    print(f"  {tag}  pseudotime_head trainable")

    history = metrics.get("loss_history", [])
    if history:
        print("\n  Loss progression:")
        print(f"  {'step':>6}  {'total':>10}  {'pt':>8}  {'comp':>10}")
        for entry in history:
            print(
                f"  {entry['step']:>6d}  "
                f"{entry['train_loss']:>10.4f}  "
                f"{entry['pt_loss']:>8.4f}  "
                f"{entry['comp_loss']:>10.4f}"
            )


# ---------------------------------------------------------------------------
# Inference check
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

    n_day11   = len(day11_ids)
    log_every = max(1, n_day11 // 10)
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
            if loss_type == "dirichlet":
                alpha = comp_out[0].clamp(min=1e-6)
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

    cost_m   = dataset.centroid_distances.astype(np.float64)
    nan_mask = np.isnan(comp_pred_arr).any(axis=1)
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.sum()} / {len(day11_ids)} cells have NaN composition predictions — skipping for EMD")
    emds: list[float] = []
    for i in range(len(day11_ids)):
        if nan_mask[i]:
            continue
        emds.append(wasserstein_1(
            comp_pred_arr[i].astype(np.float64),
            comp_true_arr[i].astype(np.float64),
            cost_m,
        ))
        if (i + 1) % log_every == 0 or (i + 1) == len(day11_ids):
            print(f"  emd     {i + 1}/{len(day11_ids)}")
    mean_emd              = float(np.mean(emds)) if emds else float("nan")
    baseline              = wasserstein_baseline(dataset.soft_labels, train_mask, cost_m)
    bs_mean, bs_per_class = brier_score(comp_pred_arr, comp_true_arr)
    js                    = jsd(comp_pred_arr, comp_true_arr)

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

    fig.suptitle(f"TRACE — {run_id}  [{comp_label}]", y=1.02)
    fig.tight_layout()
    curve_path = ckpt_dir / "loss_curve.png"
    fig.savefig(str(curve_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve saved to {curve_path}")


# ---------------------------------------------------------------------------
# Checkpoint sync
# ---------------------------------------------------------------------------

def _git_sync(path, label: str) -> None:
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
    m = cfg.model

    print("=" * 60)
    print("TRACE Myriad job")
    print(f"  Job ID      : {JOB_ID}")
    print(f"  GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"  RUN_PRESET  : {_RUN_PRESET}")
    print(f"  N_STEPS     : {N_STEPS}")
    print(f"  N_WARMUP    : {N_WARMUP_STEPS}  (tf_icl frozen until this step)")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  H5AD_PATH   : {H5AD_PATH}")
    print(f"  BACKBONE    : {BACKBONE_PATH}")
    print("=" * 60)

    # 1. Data
    print("\n[1/3] Loading data ...")
    t0 = time.time()
    dataset = ProcessedDataset.from_anndata(H5AD_PATH, cfg.data)  # type: ignore[arg-type]
    print(
        f"  {dataset.n_cells} cells  |  {dataset.n_genes} genes  "
        f"({time.time() - t0:.1f}s)"
    )

    print("=" * 60)
    print("TRACE Myriad run")
    print(f"  preset          : {_RUN_PRESET}  ({comp_label})  run_id={cfg.run_id}")
    print(f"  device          : {device}")
    print(f"  seed            : {SEED}")
    print(f"  max_genes       : {cfg.data.max_genes}")
    print(f"  n_bins          : {cfg.context.n_bins}  cells_per_bin: {cfg.context.cells_per_bin}")
    print(f"  embed_dim       : {m.embed_dim}  "
          f"d_model: {m.num_cls * m.embed_dim}  "
          f"n_heads: {m.n_heads}")
    print(f"  h5ad            : {H5AD_PATH}")
    print(f"  backbone        : {BACKBONE_PATH}")
    print(f"  n_steps         : {N_STEPS}  (warmup: {N_WARMUP_STEPS}, icl: {N_STEPS - N_WARMUP_STEPS})")
    print(f"  gene_coexpression.pt : saved at step {N_WARMUP_STEPS}  (GRN extraction point)")
    print("=" * 60)

    # 2. Model
    print("\n[2/3] Initialising model ...")
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
    n_icl_params    = sum(p.numel() for p in model.tf_icl.parameters())
    print(f"  Parameters total  : {n_params:,}")
    print(f"  tf_icl            : {n_icl_params:,}  (frozen until step {N_WARMUP_STEPS})")
    print(f"  Initially trainable: {n_params - n_icl_params:,}")

    loss_fn = (
        DirichletDualHeadLoss() if m.composition_loss_type == "dirichlet"
        else DualHeadLoss()
    ).to(device)

    # 3. Train
    cfg.model.warmup_icl_steps = N_WARMUP_STEPS
    cfg.model.warmup_col_steps = 0  # col_embedder trainable from step 0

    ckpt_dir = PROJECT_ROOT / "experiments" / cfg.run_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eval_every = max(100, N_STEPS // 20)

    ckpt_cb = CheckpointCallback(
        trainer = None,
        loss_fn = loss_fn,
        out_dir = ckpt_dir,
        every   = max(500, N_STEPS // 10),
    )
    warmup_boundary_cb = _SyncingWarmupBoundaryCallback(
        save_step             = N_WARMUP_STEPS,
        out_path              = ckpt_dir / "gene_coexpression.pt",
        loss_fn               = loss_fn,
        run_id                = cfg.run_id,
        composition_loss_type = m.composition_loss_type,
    )
    sampler = ContextSampler(dataset, cfg.context)
    builder = CellTableBuilder(dataset)
    trainer = Trainer(
        model      = model,
        dataset    = dataset,
        sampler    = sampler,
        builder    = builder,
        loss_fn    = loss_fn,
        config     = cfg,
        n_steps    = N_STEPS,
        eval_every = eval_every,
        callbacks  = [ProgressCallback(N_STEPS), ckpt_cb, warmup_boundary_cb],
        seed       = SEED,
    )
    ckpt_cb.trainer = trainer

    print(f"\n[3/3] Training for {N_STEPS} steps ...  [{time.strftime('%H:%M:%S')}]\n")
    t0 = time.time()
    metrics = trainer.fit()
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("Training complete")
    print(f"  elapsed     : {elapsed:.1f}s  ({elapsed / N_STEPS:.3f}s/step)")
    print(f"  train_loss  : {metrics['train_loss']:.4f}")
    print(f"  pt_loss     : {metrics['pt_loss']:.4f}")
    print(f"  comp_loss   : {metrics['comp_loss']:.4f}")
    print("=" * 60)

    _sanity_checks(metrics, loss_fn, model, trainer.global_step)
    _save_loss_curve(metrics["loss_history"], ckpt_dir, cfg.run_id, comp_label)

    final_path = ckpt_dir / "final.pt"
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
    print(f"\nFinal checkpoint saved to {final_path}")
    _git_sync(final_path, "final")

    # Inference on day-11 held-out cells
    print(f"\n[Inference] Day-11 held-out cells ...  [{time.strftime('%H:%M:%S')}]")
    _inference_check(model, dataset, sampler, builder, device, _RUN_PRESET)

    cfg.save()
    print(f"\nConfig saved to experiments/{cfg.run_id}/config.json")
    _git_sync(PROJECT_ROOT / "experiments" / cfg.run_id, "final (configs + loss curves)")


if __name__ == "__main__":
    main()
