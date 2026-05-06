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

def wasserstein_1(p: np.ndarray, q: np.ndarray, cost_matrix: np.ndarray) -> float:
    """Earth mover's distance between discrete distributions p and q.

    Solves the exact transport LP: min_{T≥0} <T, M>
    subject to T·1 = p and T^T·1 = q.

    Parameters
    ----------
    p, q        : (K,) float — probability distributions (must sum to 1).
    cost_matrix : (K, K) float — ground metric (e.g. inter-centroid L2 distances).
    """
    from scipy.optimize import linprog

    K = len(p)
    c = cost_matrix.flatten().astype(np.float64)

    # Build equality constraint matrix: row sums = p, col sums = q
    A_eq = np.zeros((2 * K, K * K), dtype=np.float64)
    for i in range(K):
        A_eq[i, i * K:(i + 1) * K] = 1.0   # row i: sum_j T[i,j] = p[i]
    for j in range(K):
        A_eq[K + j, j::K] = 1.0             # col j: sum_i T[i,j] = q[j]

    b_eq = np.concatenate([p.astype(np.float64), q.astype(np.float64)])
    bounds = [(0.0, None)] * (K * K)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return float(res.fun)


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
