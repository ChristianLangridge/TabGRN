"""
checkpoint_inference.py — Evaluate a saved rotation_002 checkpoint on day-11 held-out cells.

Runs all requested conditions in a single forward-pass loop (batch construction
happens once per cell). EMD computations are parallelised across CPU cores.

Usage
-----
    python src/inference/checkpoint_inference.py

Required env vars:
    CKPT_PATH     — path to the .pt checkpoint to evaluate
    H5AD_PATH     — path to neurectoderm_with_pseudotime.h5ad

Optional env vars:
    DEVICE        — "cpu" | "mps" | "cuda"  (auto-detected if not set)
    N_WORKERS     — CPU cores for parallel EMD (default: all available)
    NULL_CONTEXT  — "1" to include null-context condition (zeros context tensors)
    SHUFFLE_GENES — "1" to include permutation control (shuffles query gene expression)

Conditions always run: "normal"
Conditions added by flags: "null_context", "shuffle_genes", "both" (if both flags set)

The checkpoint must contain a "model_state" key (all rotation_002 checkpoints do).
"composition_loss_type" is read from the checkpoint; defaults to "dirichlet".
"""
import dataclasses
import multiprocessing
import os

# macOS OpenMP workaround — must be set before ANY library import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from pathlib import Path

import numpy as np
import torch

from spatialmt.config.experiment import ExperimentConfig
from spatialmt.config.paths import Dirs
from spatialmt.context.builder import CellTableBuilder
from spatialmt.context.collate import ICLBatch, icl_collate
from spatialmt.context.sampler import ContextSampler
from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.eval.metrics import (
    brier_score,
    jsd,
    mae,
    per_day_spearman,
    spearman_r,
    wasserstein_1,
    wasserstein_baseline,
)
from spatialmt.model.trace import TabICLRegressor
from spatialmt.training.trainer import _batch_to_device

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

CKPT_PATH = os.environ.get("CKPT_PATH", "")
if not CKPT_PATH:
    raise RuntimeError(
        "CKPT_PATH env var is not set.\n"
        "Example: export CKPT_PATH=experiments/rotation_002/checkpoints/gene_coexpression.pt"
    )

H5AD_PATH = os.environ.get(
    "H5AD_PATH",
    str(Dirs.model_data_anndata / "neurectoderm_with_pseudotime.h5ad"),
)

NULL_CONTEXT  = os.environ.get("NULL_CONTEXT",  "").lower() in ("1", "true")
SHUFFLE_GENES = os.environ.get("SHUFFLE_GENES", "").lower() in ("1", "true")
N_WORKERS     = int(os.environ.get("N_WORKERS", str(multiprocessing.cpu_count())))


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
# Batch transforms
# ---------------------------------------------------------------------------

def _zero_context(batch: ICLBatch) -> ICLBatch:
    return dataclasses.replace(
        batch,
        context_expression  = torch.zeros_like(batch.context_expression),
        context_pseudotime  = torch.zeros_like(batch.context_pseudotime),
        context_soft_labels = torch.zeros_like(batch.context_soft_labels),
    )


def _shuffle_query_genes(batch: ICLBatch, gene_perm: np.ndarray) -> ICLBatch:
    return dataclasses.replace(
        batch,
        query_expression = batch.query_expression[:, gene_perm],
    )


# ---------------------------------------------------------------------------
# Parallel EMD worker
# ---------------------------------------------------------------------------

def _emd_worker(args: tuple) -> float:
    pred, true, cost_m = args
    return wasserstein_1(pred.astype(np.float64), true.astype(np.float64), cost_m)


def _parallel_emd(
    comp_pred:  np.ndarray,
    comp_true:  np.ndarray,
    cost_m:     np.ndarray,
    n_workers:  int,
    label:      str,
) -> list[float]:
    n = len(comp_pred)
    args = [(comp_pred[i], comp_true[i], cost_m) for i in range(n)]
    print(f"  EMD — {label}  ({n_workers} workers) ...", flush=True)
    t0 = time.time()
    with multiprocessing.Pool(processes=n_workers) as pool:
        emds = pool.map(_emd_worker, args)
    print(f"  done  ({time.time() - t0:.1f}s)", flush=True)
    return emds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device    = _detect_device()
    ckpt_path = Path(CKPT_PATH)

    conditions: list[str] = ["normal"]
    if NULL_CONTEXT:
        conditions.append("null_context")
    if SHUFFLE_GENES:
        conditions.append("shuffle_genes")
    if NULL_CONTEXT and SHUFFLE_GENES:
        conditions.append("both")

    print("=" * 60)
    print("TRACE checkpoint inference")
    print(f"  checkpoint  : {ckpt_path.name}  ({ckpt_path})")
    print(f"  h5ad        : {H5AD_PATH}")
    print(f"  device      : {device}")
    print(f"  n_workers   : {N_WORKERS}  (EMD parallelism)")
    print(f"  conditions  : {', '.join(conditions)}")
    print("=" * 60)

    # 1. Checkpoint
    print("\n[1/4] Loading checkpoint ...")
    ckpt           = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    comp_loss_type = ckpt.get("composition_loss_type", "dirichlet")
    global_step    = ckpt.get("global_step", "?")
    run_id         = ckpt.get("run_id", "?")
    print(f"  run_id           : {run_id}")
    print(f"  global_step      : {global_step}")
    print(f"  composition_loss : {comp_loss_type}")

    # 2. Data
    print("\n[2/4] Loading data ...")
    cfg = ExperimentConfig.full_finetune_dirichlet()
    t0  = time.time()
    dataset = ProcessedDataset.from_anndata(H5AD_PATH, cfg.data)
    print(f"  {dataset.n_cells} cells  |  {dataset.n_genes} genes  ({time.time() - t0:.1f}s)")

    sampler = ContextSampler(dataset, cfg.context)
    builder = CellTableBuilder(dataset)

    # 3. Model
    print("\n[3/4] Building model ...")
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
        composition_loss_type = comp_loss_type,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}  (loaded from checkpoint)")

    # 4. Inference
    print("\n[4/4] Running inference ...")
    _run_inference(model, dataset, sampler, builder, device, comp_loss_type, conditions)


# ---------------------------------------------------------------------------
# Single-pass inference — all conditions in one cell loop
# ---------------------------------------------------------------------------

def _run_inference(
    model:          TabICLRegressor,
    dataset:        ProcessedDataset,
    sampler:        ContextSampler,
    builder:        CellTableBuilder,
    device:         torch.device,
    comp_loss_type: str,
    conditions:     list[str],
) -> None:
    day11_mask = dataset.collection_day == 11
    day11_ids  = [cid for cid, m in zip(dataset.cell_ids, day11_mask) if m]
    if not day11_ids:
        print("  No day-11 cells found — skipping.")
        return

    n_day11       = len(day11_ids)
    day11_indices = [dataset.cell_ids.index(cid) for cid in day11_ids]
    pt_true_arr   = dataset.pseudotime[day11_indices]
    comp_true_arr = dataset.soft_labels[day11_indices]
    days_arr      = dataset.collection_day[day11_indices]
    train_mask    = dataset.collection_day != 11
    cost_m        = dataset.centroid_distances.astype(np.float64)

    gene_perm = np.random.default_rng(0).permutation(dataset.n_genes)

    pt_preds:   dict[str, list[float]]       = {c: [] for c in conditions}
    comp_preds: dict[str, list[list[float]]] = {c: [] for c in conditions}

    def _extract(pt_out, comp_out):
        pt = pt_out[0].item()
        if comp_loss_type == "dirichlet":
            alpha = comp_out[0].clamp(min=1e-6)
            comp  = (alpha / alpha.sum()).cpu().tolist()
        else:
            comp  = comp_out[0].cpu().tolist()
        return pt, comp

    def _progress(done: int, total: int, width: int = 30) -> None:
        filled = int(width * done / total)
        bar    = "█" * filled + "░" * (width - filled)
        print(f"\r  [{bar}] {done}/{total}", end="", flush=True)
        if done == total:
            print()

    print(f"\n  Forward pass — {n_day11} cells × {len(conditions)} condition(s) ...")
    t0 = time.time()
    with torch.no_grad():
        for i, cid in enumerate(day11_ids):
            anchor_ids, _ = sampler.sample(cid)
            table, label  = builder.build(cid, anchor_ids)
            base_batch    = _batch_to_device(icl_collate([(table, label)]), device)

            for cond in conditions:
                if cond == "normal":
                    batch = base_batch
                elif cond == "null_context":
                    batch = _zero_context(base_batch)
                elif cond == "shuffle_genes":
                    batch = _shuffle_query_genes(base_batch, gene_perm)
                elif cond == "both":
                    batch = _zero_context(_shuffle_query_genes(base_batch, gene_perm))

                pt, comp = _extract(*model(batch))
                pt_preds[cond].append(pt)
                comp_preds[cond].append(comp)

            _progress(i + 1, n_day11)

    print(f"  forward pass complete  ({time.time() - t0:.1f}s)", flush=True)

    pt_arr   = {c: np.array(pt_preds[c],   dtype=np.float32) for c in conditions}
    comp_arr = {c: np.array(comp_preds[c], dtype=np.float32) for c in conditions}

    # Parallel EMD — one pool.map per condition
    baseline = wasserstein_baseline(dataset.soft_labels, train_mask, cost_m)
    emds: dict[str, list[float]] = {}
    for cond in conditions:
        emds[cond] = _parallel_emd(comp_arr[cond], comp_true_arr, cost_m, N_WORKERS, cond)

    # Print results
    cats = dataset.cell_type_categories
    for cond in conditions:
        _print_condition(
            cond         = cond,
            pt_pred      = pt_arr[cond],
            comp_pred    = comp_arr[cond],
            pt_true      = pt_true_arr,
            comp_true    = comp_true_arr,
            days         = days_arr,
            cond_emds    = emds[cond],
            baseline     = baseline,
            cats         = cats,
            day11_ids    = day11_ids,
            sampler      = sampler,
            normal_pt    = pt_arr["normal"]                                        if cond != "normal" else None,
            normal_emds  = emds["normal"]                                          if cond != "normal" else None,
            normal_bs    = brier_score(comp_arr["normal"], comp_true_arr)[0]       if cond != "normal" else None,
            normal_js    = jsd(comp_arr["normal"], comp_true_arr)                  if cond != "normal" else None,
        )


# ---------------------------------------------------------------------------
# Per-condition results printer
# ---------------------------------------------------------------------------

def _print_condition(
    cond:        str,
    pt_pred:     np.ndarray,
    comp_pred:   np.ndarray,
    pt_true:     np.ndarray,
    comp_true:   np.ndarray,
    days:        np.ndarray,
    cond_emds:   list[float],
    baseline:    float,
    cats:        list[str],
    day11_ids:   list[str],
    sampler:     ContextSampler,
    normal_pt:   np.ndarray | None,
    normal_emds: list[float] | None,
    normal_bs:   float | None,
    normal_js:   float | None,
) -> None:
    LABELS = {
        "normal":       "Normal inference — day 11 held-out cells",
        "null_context": "Ablation — null context (context tensors zeroed)",
        "shuffle_genes": "Permutation control — query gene expression shuffled",
        "both":         "Ablation — null context + shuffled genes",
    }
    print("\n" + "=" * 60)
    print(LABELS.get(cond, cond))

    rho     = spearman_r(pt_pred, pt_true)
    pt_mae  = mae(pt_pred, pt_true)
    day_rho = per_day_spearman(pt_pred, pt_true, days)

    mean_emd       = float(np.mean(cond_emds))
    bs_mean, bs_pc = brier_score(comp_pred, comp_true)
    js             = jsd(comp_pred, comp_true)

    print("\n  -- Pseudotime head --")
    print(f"  Spearman ρ   : {rho:+.4f}   (primary accuracy)")
    print(f"  MAE          : {pt_mae:.4f}   (error magnitude)")
    if day_rho:
        print("  per-day ρ    :", "  ".join(f"D{d}: {v:+.3f}" for d, v in sorted(day_rho.items())))

    print("\n  -- Composition head --")
    print(f"  Wasserstein  : {mean_emd:.4f}   (primary accuracy, PCA ground metric)")
    print(f"  W. baseline  : {baseline:.4f}   (null model — training mean composition)")
    print(f"  Brier score  : {bs_mean:.4f}   (diagnostic)")
    bs_str = "  ".join(f"{cats[i]}: {v:.4f}" for i, v in enumerate(bs_pc))
    print(f"  Brier/class  : {bs_str}")
    print(f"  JSD          : {js:.4f}   (error, bounded [0,1])")

    if normal_pt is not None:
        normal_rho = spearman_r(normal_pt, pt_true)
        normal_mae = mae(normal_pt, pt_true)
        normal_emd = float(np.mean(normal_emds))
        print("\n  -- Delta vs normal --")
        print(f"  Δ Spearman ρ : {rho - normal_rho:+.4f}   (positive = this condition is better)")
        print(f"  Δ MAE        : {pt_mae - normal_mae:+.4f}   (negative = this condition is better)")
        print(f"  Δ Wasserstein: {mean_emd - normal_emd:+.4f}   (negative = this condition is better)")
        print(f"  Δ Brier      : {bs_mean - normal_bs:+.4f}")
        print(f"  Δ JSD        : {js - normal_js:+.4f}")

    dominant_class = comp_true.argmax(axis=1)
    print("\n  -- Composition: stratified by dominant true class --")
    print(f"  {'class':<40} {'n':>5}  {'W-dist':>7}  {'JSD':>7}  {'Brier':>7}  {'pt_med':>7}")
    worse_than_baseline: list[tuple[str, np.ndarray]] = []
    for cls_idx, cls_name in enumerate(cats):
        mask = dominant_class == cls_idx
        if not mask.any():
            continue
        cls_emds  = [cond_emds[i] for i, m in enumerate(mask) if m]
        cls_bs, _ = brier_score(comp_pred[mask], comp_true[mask])
        cls_js    = jsd(comp_pred[mask], comp_true[mask])
        cls_w     = float(np.mean(cls_emds))
        cls_pt    = float(np.median(pt_true[mask]))
        flag      = "  *** below baseline" if cls_w >= baseline else ""
        print(
            f"  {cls_name:<40} {mask.sum():>5d}  "
            f"{cls_w:>7.4f}  {cls_js:>7.4f}  {cls_bs:>7.4f}  {cls_pt:>7.4f}"
            f"{flag}"
        )
        if cls_w >= baseline:
            worse_than_baseline.append((cls_name, mask))

    for cls_name, mask in worse_than_baseline:
        mean_true    = comp_true[mask].mean(axis=0)
        mean_pred    = comp_pred[mask].mean(axis=0)
        mean_pt_true = float(pt_true[mask].mean())
        mean_pt_pred = float(pt_pred[mask].mean())
        print(f"\n  -- Failure analysis: '{cls_name}' (W-dist ≥ baseline) --")
        print(f"  pseudotime  true mean {mean_pt_true:.4f}  pred mean {mean_pt_pred:.4f}")
        col_w = 36
        print(f"  {'class':<{col_w}}  {'true_mean':>9}  {'pred_mean':>9}  {'delta':>9}")
        for i, cat in enumerate(cats):
            delta = mean_pred[i] - mean_true[i]
            flag  = "  <-- overfit" if delta > 0.05 else ("  <-- underfit" if delta < -0.05 else "")
            print(f"  {cat:<{col_w}}  {mean_true[i]:>9.4f}  {mean_pred[i]:>9.4f}  {delta:>+9.4f}{flag}")

    print("\n  -- Example: first day-11 cell --")
    true_comp_str = "  |  ".join(f"{cats[i]}: {v:.3f}" for i, v in enumerate(comp_true[0]))
    pred_comp_str = "  |  ".join(f"{cats[i]}: {v:.3f}" for i, v in enumerate(comp_pred[0]))
    print(f"  cell id         : {day11_ids[0]}")
    print(f"  true pseudotime : {pt_true[0]:.4f}")
    print(f"  pred pseudotime : {pt_pred[0]:.4f}  (|error| {abs(pt_pred[0] - pt_true[0]):.4f})")
    print(f"  true composition:\n    {true_comp_str}")
    print(f"  pred composition:\n    {pred_comp_str}")
    anchor_ids_ex, _ = sampler.sample(day11_ids[0])
    print(f"  n context cells : {len(anchor_ids_ex)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
