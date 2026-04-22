"""
Unit tests for spatialmt.context.sampler.ContextSampler.

Tests are organised into:
  - Construction (valid inputs build without error)
  - Output contract (types, shapes, value ranges)
  - Day 11 exclusion (withheld bin never appears in output)
  - Bin coverage (all five active days represented every call)
  - Pseudotime alignment (returned pseudotimes correspond to returned cell ids)
  - Query cell exclusion (query cell never selected as its own anchor)
  - Sparse bin — warning path (allow_replacement=True, fewer cells than cells_per_bin)
  - Sparse bin — raise path (allow_replacement=False)
  - Determinism (same seed → same selection)
  - Exact count (5 active bins × cells_per_bin anchors returned)
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from spatialmt.config.experiment import ContextConfig, DataConfig
from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.context.sampler import ContextSampler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K = 8
DAYS = [5, 7, 11, 16, 21, 30]
N_CELLS_PER_DAY = 20          # generous — no sparse bins by default
N_CELLS = N_CELLS_PER_DAY * len(DAYS)   # 120
N_GENES = 10
CELLS_PER_BIN = 4             # fits inside debug max_context_cells with 5 active bins (4×5=20)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_dataset(
    n_cells_per_day: int = N_CELLS_PER_DAY,
    n_genes: int = N_GENES,
    k: int = K,
    seed: int = 0,
) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    days = np.array(DAYS * n_cells_per_day)   # exactly n_cells_per_day cells per day
    n_cells = len(days)

    expression = (rng.random((n_cells, n_genes)) * 4.0).astype(np.float32)

    # Pseudotime increases monotonically with day (scaffold: linear scaling)
    day_to_pt = {5: 0.05, 7: 0.15, 11: 0.30, 16: 0.50, 21: 0.70, 30: 0.90}
    pseudotime = np.array([
        day_to_pt[d] + rng.random() * 0.05 for d in days
    ], dtype=np.float32)
    pseudotime = np.clip(pseudotime, 0.0, 1.0)

    collection_day = days.astype(np.int32)
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"GENE_{i:02d}" for i in range(n_genes)]
    cell_type_labels = pd.Series([f"state_{i % k}" for i in range(n_cells)])
    orig_ident = pd.Series([f"HB4_D{d}" for d in collection_day])

    raw = rng.random((n_cells, k)).astype(np.float32)
    soft_labels = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    manifest_hash = ProcessedDataset._compute_manifest_hash(gene_names)
    categories = sorted(cell_type_labels.unique())

    return ProcessedDataset(
        expression=expression,
        gene_names=gene_names,
        pseudotime=pseudotime,
        collection_day=collection_day,
        cell_ids=cell_ids,
        cell_type_labels=cell_type_labels,
        orig_ident=orig_ident,
        soft_labels=soft_labels,
        cell_type_categories=categories,
        manifest_hash=manifest_hash,
    )


def _make_config(cells_per_bin: int = CELLS_PER_BIN, allow_replacement: bool = True) -> ContextConfig:
    # 6 bins × cells_per_bin must fit within max_context_cells
    return ContextConfig(
        n_bins=6,
        cells_per_bin=cells_per_bin,
        max_context_cells=6 * cells_per_bin,
        allow_replacement=allow_replacement,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_context_sampler_constructs():
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    assert sampler is not None


def test_context_sampler_accepts_custom_bin_edges():
    ds = _make_dataset()
    cfg = _make_config()
    bin_edges = np.array([0.0, 0.12, 0.25, 0.45, 0.65, 0.85, 1.01])
    sampler = ContextSampler(ds, cfg, bin_edges=bin_edges)
    assert sampler is not None


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_sample_returns_tuple():
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[0]
    result = sampler.sample(query_id, rng=np.random.default_rng(0))
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_sample_cell_ids_is_list_of_strings():
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[0]
    cell_ids, _ = sampler.sample(query_id, rng=np.random.default_rng(0))
    assert isinstance(cell_ids, list)
    assert all(isinstance(x, str) for x in cell_ids)


def test_sample_pseudotimes_is_float32_array():
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[0]
    _, pseudotimes = sampler.sample(query_id, rng=np.random.default_rng(0))
    assert isinstance(pseudotimes, np.ndarray)
    assert pseudotimes.dtype == np.float32


def test_sample_all_cell_ids_are_in_dataset():
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[0]
    cell_ids, _ = sampler.sample(query_id, rng=np.random.default_rng(0))
    assert all(cid in ds.cell_ids for cid in cell_ids)


def test_sample_pseudotimes_in_unit_interval():
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[0]
    _, pseudotimes = sampler.sample(query_id, rng=np.random.default_rng(0))
    assert pseudotimes.min() >= 0.0
    assert pseudotimes.max() <= 1.0


# ---------------------------------------------------------------------------
# Exact count — 5 active bins × cells_per_bin
# ---------------------------------------------------------------------------

def test_sample_returns_correct_total_count():
    """5 active bins (day 11 withheld) × cells_per_bin anchors."""
    cells_per_bin = CELLS_PER_BIN
    ds = _make_dataset()
    cfg = _make_config(cells_per_bin=cells_per_bin)
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[0]
    cell_ids, pseudotimes = sampler.sample(query_id, rng=np.random.default_rng(0))
    n_active_bins = 5  # 6 bins minus day 11
    assert len(cell_ids) == n_active_bins * cells_per_bin
    assert len(pseudotimes) == len(cell_ids)


# ---------------------------------------------------------------------------
# Day 11 exclusion
# ---------------------------------------------------------------------------

def test_sample_contains_no_day11_cells():
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    # Build a lookup: cell_id → collection_day
    day_lookup = {cid: d for cid, d in zip(ds.cell_ids, ds.collection_day)}
    for query_id in ds.cell_ids[:10]:
        cell_ids, _ = sampler.sample(query_id, rng=np.random.default_rng(0))
        days_in_context = [day_lookup[cid] for cid in cell_ids]
        assert 11 not in days_in_context, (
            f"Day 11 cell found in context for query {query_id}"
        )


def test_day11_query_cell_still_valid():
    """Day 11 cells can be query cells — they just cannot be context anchors."""
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    day_lookup = {cid: d for cid, d in zip(ds.cell_ids, ds.collection_day)}
    day11_cell = next(cid for cid in ds.cell_ids if day_lookup[cid] == 11)
    cell_ids, pseudotimes = sampler.sample(day11_cell, rng=np.random.default_rng(0))
    assert 11 not in [day_lookup[cid] for cid in cell_ids]
    assert len(cell_ids) > 0


# ---------------------------------------------------------------------------
# Bin coverage
# ---------------------------------------------------------------------------

def test_all_active_days_represented():
    """Each active day (5, 7, 16, 21, 30) must contribute at least one anchor."""
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    day_lookup = {cid: d for cid, d in zip(ds.cell_ids, ds.collection_day)}
    query_id = ds.cell_ids[0]
    cell_ids, _ = sampler.sample(query_id, rng=np.random.default_rng(0))
    days_present = set(day_lookup[cid] for cid in cell_ids)
    expected_days = {5, 7, 16, 21, 30}
    assert expected_days == days_present


def test_each_bin_contributes_cells_per_bin_cells():
    """Verify that each active bin contributes exactly cells_per_bin anchors."""
    cells_per_bin = CELLS_PER_BIN
    ds = _make_dataset()
    cfg = _make_config(cells_per_bin=cells_per_bin)
    sampler = ContextSampler(ds, cfg)
    day_lookup = {cid: d for cid, d in zip(ds.cell_ids, ds.collection_day)}
    query_id = ds.cell_ids[0]
    cell_ids, _ = sampler.sample(query_id, rng=np.random.default_rng(0))
    from collections import Counter
    day_counts = Counter(day_lookup[cid] for cid in cell_ids)
    for day in [5, 7, 16, 21, 30]:
        assert day_counts[day] == cells_per_bin, (
            f"Expected {cells_per_bin} cells from day {day}, got {day_counts[day]}"
        )


# ---------------------------------------------------------------------------
# Pseudotime alignment
# ---------------------------------------------------------------------------

def test_pseudotimes_correspond_to_cell_ids():
    """Returned pseudotimes must match dataset pseudotime for the returned cell ids."""
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    pt_lookup = {cid: pt for cid, pt in zip(ds.cell_ids, ds.pseudotime)}
    query_id = ds.cell_ids[0]
    cell_ids, pseudotimes = sampler.sample(query_id, rng=np.random.default_rng(0))
    expected = np.array([pt_lookup[cid] for cid in cell_ids], dtype=np.float32)
    np.testing.assert_array_equal(pseudotimes, expected)


# ---------------------------------------------------------------------------
# Query cell exclusion
# ---------------------------------------------------------------------------

def test_query_cell_not_in_anchors():
    """The query cell must never appear in its own context window."""
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    # Test several query cells, including ones in active bins
    day_lookup = {cid: d for cid, d in zip(ds.cell_ids, ds.collection_day)}
    for query_id in ds.cell_ids[:30]:
        if day_lookup[query_id] == 11:
            continue
        cell_ids, _ = sampler.sample(query_id, rng=np.random.default_rng(0))
        assert query_id not in cell_ids, (
            f"Query cell {query_id} appeared in its own context window"
        )


# ---------------------------------------------------------------------------
# Sparse bin — warning path (allow_replacement=True)
# ---------------------------------------------------------------------------

def _make_sparse_dataset(sparse_day: int = 5, n_sparse: int = 2, seed: int = 0) -> ProcessedDataset:
    """Dataset where one day has very few cells."""
    rng = np.random.default_rng(seed)
    # Full-size days except for sparse_day
    day_counts = {d: (n_sparse if d == sparse_day else N_CELLS_PER_DAY) for d in DAYS}
    days = np.array([d for d, count in day_counts.items() for _ in range(count)])
    n_cells = len(days)
    n_genes = N_GENES
    k = K

    expression = (rng.random((n_cells, n_genes)) * 4.0).astype(np.float32)
    pseudotime = rng.random(n_cells).astype(np.float32)
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"GENE_{i:02d}" for i in range(n_genes)]
    cell_type_labels = pd.Series([f"state_{i % k}" for i in range(n_cells)])
    orig_ident = pd.Series([f"HB4_D{d}" for d in days])
    raw = rng.random((n_cells, k)).astype(np.float32)
    soft_labels = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    manifest_hash = ProcessedDataset._compute_manifest_hash(gene_names)
    categories = sorted(cell_type_labels.unique())

    return ProcessedDataset(
        expression=expression,
        gene_names=gene_names,
        pseudotime=pseudotime,
        collection_day=days.astype(np.int32),
        cell_ids=cell_ids,
        cell_type_labels=cell_type_labels,
        orig_ident=orig_ident,
        soft_labels=soft_labels,
        cell_type_categories=categories,
        manifest_hash=manifest_hash,
    )


def test_sparse_bin_issues_warning_with_replacement():
    """When a bin has fewer cells than cells_per_bin and allow_replacement=True, a UserWarning is issued."""
    cells_per_bin = 4
    ds = _make_sparse_dataset(sparse_day=5, n_sparse=2)   # 2 < 4 = cells_per_bin
    cfg = _make_config(cells_per_bin=cells_per_bin, allow_replacement=True)
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[-1]  # a non-day-5 cell
    with pytest.warns(UserWarning):
        sampler.sample(query_id, rng=np.random.default_rng(0))


def test_sparse_bin_with_replacement_still_returns_correct_count():
    """Even with replacement, the total anchor count must equal 5 × cells_per_bin."""
    cells_per_bin = 4
    ds = _make_sparse_dataset(sparse_day=5, n_sparse=2)
    cfg = _make_config(cells_per_bin=cells_per_bin, allow_replacement=True)
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cell_ids, pseudotimes = sampler.sample(query_id, rng=np.random.default_rng(0))
    assert len(cell_ids) == 5 * cells_per_bin
    assert len(pseudotimes) == len(cell_ids)


def test_sparse_bin_warning_contains_duplication_info():
    """The warning message must mention the affected bin or day."""
    cells_per_bin = 4
    ds = _make_sparse_dataset(sparse_day=5, n_sparse=2)
    cfg = _make_config(cells_per_bin=cells_per_bin, allow_replacement=True)
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[-1]
    with pytest.warns(UserWarning, match=r"(replacement|duplicate|sparse|bin|day\s*5)"):
        sampler.sample(query_id, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Sparse bin — raise path (allow_replacement=False)
# ---------------------------------------------------------------------------

def test_sparse_bin_raises_when_replacement_disabled():
    """When allow_replacement=False and a bin is sparse, a ValueError must be raised."""
    cells_per_bin = 4
    ds = _make_sparse_dataset(sparse_day=5, n_sparse=2)
    cfg = _make_config(cells_per_bin=cells_per_bin, allow_replacement=False)
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[-1]
    with pytest.raises(ValueError):
        sampler.sample(query_id, rng=np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_same_selection():
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[0]
    ids_a, pts_a = sampler.sample(query_id, rng=np.random.default_rng(42))
    ids_b, pts_b = sampler.sample(query_id, rng=np.random.default_rng(42))
    assert ids_a == ids_b
    np.testing.assert_array_equal(pts_a, pts_b)


def test_different_seeds_different_selection():
    """Two different seeds should (with overwhelming probability) produce different context windows."""
    ds = _make_dataset(n_cells_per_day=50)   # more cells → low collision probability
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[0]
    ids_a, _ = sampler.sample(query_id, rng=np.random.default_rng(1))
    ids_b, _ = sampler.sample(query_id, rng=np.random.default_rng(2))
    assert ids_a != ids_b


# ---------------------------------------------------------------------------
# Integer seed accepted
# ---------------------------------------------------------------------------

def test_integer_seed_accepted():
    """Passing an integer rng seed (not a Generator) must not raise."""
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[0]
    cell_ids, _ = sampler.sample(query_id, rng=42)
    assert len(cell_ids) > 0


def test_none_rng_accepted():
    """rng=None must use a default random source without raising."""
    ds = _make_dataset()
    cfg = _make_config()
    sampler = ContextSampler(ds, cfg)
    query_id = ds.cell_ids[0]
    cell_ids, _ = sampler.sample(query_id, rng=None)
    assert len(cell_ids) > 0
