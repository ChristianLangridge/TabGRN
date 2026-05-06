"""spatialmt.eval.metrics — Evaluation metrics for the dual-head TabGRN model.

Pseudotime head
---------------
    spearman_r         Primary accuracy: rank correlation between predicted and true pseudotime.
    per_day_spearman   Diagnostic: Spearman ρ computed within each collection day.
    mae                Error magnitude in pseudotime units [0, 1].

Composition head
----------------
    wasserstein_1      Primary accuracy: earth mover's distance with PCA-centroid ground metric.
    brier_score        Diagnostic: geometry-blind MSE on the simplex, per class and mean.
    jsd                Error: Jensen-Shannon divergence, information-theoretic and bounded.
    wasserstein_baseline  Null-model reference: EMD when always predicting training mean composition.
"""
from __future__ import annotations

import functools

import numpy as np


# ---------------------------------------------------------------------------
# Pseudotime head
# ---------------------------------------------------------------------------

def spearman_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation between predicted and true pseudotime."""
    from scipy.stats import spearmanr
    result = spearmanr(pred, true)
    return float(result.statistic)


def per_day_spearman(
    pred: np.ndarray,
    true: np.ndarray,
    collection_day: np.ndarray,
) -> dict[int, float]:
    """Spearman ρ computed separately within each collection day.

    Days with fewer than 3 cells are skipped (rank correlation undefined).
    """
    days = sorted(np.unique(collection_day).tolist())
    result: dict[int, float] = {}
    for day in days:
        mask = collection_day == day
        if mask.sum() < 3:
            continue
        result[int(day)] = spearman_r(pred[mask], true[mask])
    return result


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean absolute error in pseudotime units."""
    return float(np.mean(np.abs(pred - true)))


# ---------------------------------------------------------------------------
# Composition head
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _transport_constraints(K: int) -> np.ndarray:
    """Equality constraint matrix for the K×K transport LP.

    Returns a (2K, K²) array cached per K — identical for every distribution
    pair of the same size, so it is built once and reused across all cells.

    Row i        (i < K) : selects row-block i of flattened T → source marginal.
    Row K+j (j < K) : selects column-stride j of flattened T → target marginal.
    """
    A = np.zeros((2 * K, K * K), dtype=np.float64)
    for i in range(K):
        A[i, i * K:(i + 1) * K] = 1.0
    for j in range(K):
        A[K + j, j::K] = 1.0
    return A


def wasserstein_1(p: np.ndarray, q: np.ndarray, cost_matrix: np.ndarray) -> float:
    """Earth mover's distance between discrete distributions p and q.

    Uses POT (Python Optimal Transport) network simplex — exact solution,
    substantially faster than a general LP solver for small K.

    Parameters
    ----------
    p, q        : (K,) float — probability distributions.
    cost_matrix : (K, K) float — ground metric (e.g. inter-centroid L2 distances).
    """
    import ot

    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()

    return float(ot.emd2(p, q, cost_matrix.astype(np.float64)))


def brier_score(
    pred: np.ndarray,
    true: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Brier score (MSE on the simplex).

    Parameters
    ----------
    pred : (n_cells, K)
    true : (n_cells, K)

    Returns
    -------
    mean_score  : float — mean over cells and classes.
    per_class   : (K,) float — mean squared error per class column.
    """
    sq = (pred - true) ** 2
    return float(sq.mean()), sq.mean(axis=0)


def jsd(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean Jensen-Shannon divergence between predicted and true compositions.

    Symmetric, bounded in [0, 1] (log base 2). Averaged over cells.

    Parameters
    ----------
    pred : (n_cells, K)
    true : (n_cells, K)
    """
    eps = 1e-12
    m = 0.5 * (pred + true)
    kl_pm = np.sum(np.where(pred > eps, pred * np.log2(pred / (m + eps) + eps), 0.0), axis=1)
    kl_qm = np.sum(np.where(true > eps, true * np.log2(true / (m + eps) + eps), 0.0), axis=1)
    js_per_cell = np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0)
    return float(js_per_cell.mean())


# ---------------------------------------------------------------------------
# Wasserstein baseline
# ---------------------------------------------------------------------------

def wasserstein_baseline(
    true_labels: np.ndarray,
    train_mask: np.ndarray,
    cost_matrix: np.ndarray,
) -> float:
    """Mean Wasserstein distance when predicting the training mean composition for every cell.

    This is the null-model EMD: a predictor that knows the training prior but
    nothing about individual cells. Compare the model's mean Wasserstein against
    this value to confirm it is doing better than the prior.

    Parameters
    ----------
    true_labels  : (n_cells, K) — soft labels for all cells.
    train_mask   : (n_cells,) bool — True for training cells (non-withheld).
    cost_matrix  : (K, K) — inter-centroid ground metric.

    Returns
    -------
    baseline_emd : float — mean EMD over non-training (test) cells.
    """
    train_mean = true_labels[train_mask].mean(axis=0).astype(np.float64)
    train_mean /= train_mean.sum()

    test_labels = true_labels[~train_mask]
    emds = [
        wasserstein_1(train_mean, test_labels[i].astype(np.float64), cost_matrix)
        for i in range(len(test_labels))
    ]
    return float(np.mean(emds))
