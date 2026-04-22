"""
Unit tests for spatialmt.context.builder.CellTableBuilder.

CellTableBuilder assembles the ICL input table for a single training step.
It receives a query cell id and a list of anchor cell ids (from ContextSampler)
and returns (CellTable, TrainingTargets) — a strict separation between what
the model sees during the forward pass and the labels used for the loss.

CellTable    — model inputs only; query labels are ABSENT
TrainingTargets — query labels only; never fed into the model

Contract under test
-------------------
CellTableBuilder(dataset)
    .build(query_cell_id, anchor_ids) -> tuple[CellTable, TrainingTargets]

CellTable fields
----------------
  context_expression  : np.ndarray (n_anchors, n_genes), float32
  context_pseudotime  : np.ndarray (n_anchors,),         float32
  context_soft_labels : np.ndarray (n_anchors, K),       float32
  query_expression    : np.ndarray (n_genes,),           float32

TrainingTargets fields
----------------------
  query_pseudotime    : np.float32   (scalar)
  query_soft_labels   : np.ndarray (K,), float32

Tests are organised into:
  - Construction
  - Return type (tuple of CellTable + TrainingTargets)
  - CellTable field types and shapes
  - TrainingTargets field types and shapes
  - Expression value correctness (rows match dataset.expression)
  - Pseudotime value correctness
  - Soft label value correctness
  - Positional alignment (context rows aligned to anchor_ids order)
  - Empty anchor list handled
  - Duplicate anchors (replacement sampling) handled
  - Unknown cell id raises
"""

import numpy as np
import pandas as pd
import pytest

from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.context.builder import CellTableBuilder, CellTable, TrainingTargets

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ANCHORS = 20
N_GENES   = 10
N_CELLS   = 120
K         = 8
DAYS      = [5, 7, 11, 16, 21, 30]


# ---------------------------------------------------------------------------
# Shared factory
# ---------------------------------------------------------------------------

def _make_dataset(seed: int = 0) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    n_cells_per_day = N_CELLS // len(DAYS)
    days = np.array(DAYS * n_cells_per_day, dtype=np.int32)
    n_cells = len(days)

    expression = (rng.random((n_cells, N_GENES)) * 4.0).astype(np.float32)

    day_to_pt = {5: 0.05, 7: 0.15, 11: 0.30, 16: 0.50, 21: 0.70, 30: 0.90}
    pseudotime = np.array(
        [day_to_pt[d] + rng.random() * 0.05 for d in days], dtype=np.float32
    )
    pseudotime = np.clip(pseudotime, 0.0, 1.0)

    cell_ids         = [f"cell_{i}" for i in range(n_cells)]
    gene_names       = [f"GENE_{i:02d}" for i in range(N_GENES)]
    cell_type_labels = pd.Series([f"state_{i % K}" for i in range(n_cells)])
    orig_ident       = pd.Series([f"HB4_D{d}" for d in days])

    raw         = rng.random((n_cells, K)).astype(np.float32)
    soft_labels = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    manifest_hash = ProcessedDataset._compute_manifest_hash(gene_names)
    categories = sorted(cell_type_labels.unique())

    return ProcessedDataset(
        expression=expression,
        gene_names=gene_names,
        pseudotime=pseudotime,
        collection_day=days,
        cell_ids=cell_ids,
        cell_type_labels=cell_type_labels,
        orig_ident=orig_ident,
        soft_labels=soft_labels,
        cell_type_categories=categories,
        manifest_hash=manifest_hash,
    )


def _anchor_ids(ds: ProcessedDataset, n: int = N_ANCHORS, exclude: str | None = None) -> list[str]:
    return [cid for cid in ds.cell_ids if cid != exclude][:n]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_cell_table_builder_constructs():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    assert builder is not None


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_build_returns_tuple():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    result = builder.build(query_id, anchor_ids)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_build_returns_cell_table_and_training_targets():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    table, targets = builder.build(query_id, anchor_ids)
    assert isinstance(table, CellTable)
    assert isinstance(targets, TrainingTargets)


# ---------------------------------------------------------------------------
# CellTable field types and shapes
# ---------------------------------------------------------------------------

def test_context_expression_shape():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, n=N_ANCHORS, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)
    assert table.context_expression.shape == (N_ANCHORS, N_GENES)


def test_context_expression_dtype():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)
    assert table.context_expression.dtype == np.float32


def test_query_expression_shape():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)
    assert table.query_expression.shape == (N_GENES,)


def test_query_expression_dtype():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)
    assert table.query_expression.dtype == np.float32


def test_context_pseudotime_shape():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, n=N_ANCHORS, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)
    assert table.context_pseudotime.shape == (N_ANCHORS,)


def test_context_pseudotime_dtype():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)
    assert table.context_pseudotime.dtype == np.float32


def test_context_soft_labels_shape():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, n=N_ANCHORS, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)
    assert table.context_soft_labels.shape == (N_ANCHORS, K)


def test_context_soft_labels_dtype():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)
    assert table.context_soft_labels.dtype == np.float32


def test_cell_table_has_no_query_pseudotime_attribute():
    """CellTable must NOT expose query_pseudotime — that belongs to TrainingTargets."""
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)
    assert not hasattr(table, "query_pseudotime")


def test_cell_table_has_no_query_soft_labels_attribute():
    """CellTable must NOT expose query_soft_labels — that belongs to TrainingTargets."""
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)
    assert not hasattr(table, "query_soft_labels")


# ---------------------------------------------------------------------------
# TrainingTargets field types and shapes
# ---------------------------------------------------------------------------

def test_query_pseudotime_is_float32_scalar():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    _, targets = builder.build(query_id, anchor_ids)
    assert isinstance(targets.query_pseudotime, (float, np.floating))
    assert np.float32(targets.query_pseudotime) == targets.query_pseudotime


def test_query_soft_labels_shape():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    _, targets = builder.build(query_id, anchor_ids)
    assert targets.query_soft_labels.shape == (K,)


def test_query_soft_labels_dtype():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    _, targets = builder.build(query_id, anchor_ids)
    assert targets.query_soft_labels.dtype == np.float32


# ---------------------------------------------------------------------------
# Value correctness — expression
# ---------------------------------------------------------------------------

def test_context_expression_rows_match_dataset():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, n=N_ANCHORS, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)

    idx_map = {cid: i for i, cid in enumerate(ds.cell_ids)}
    for row, cid in enumerate(anchor_ids):
        np.testing.assert_array_equal(table.context_expression[row], ds.expression[idx_map[cid]])


def test_query_expression_matches_dataset():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[5]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)

    idx = ds.cell_ids.index(query_id)
    np.testing.assert_array_equal(table.query_expression, ds.expression[idx])


# ---------------------------------------------------------------------------
# Value correctness — pseudotime
# ---------------------------------------------------------------------------

def test_context_pseudotime_values_match_dataset():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, n=N_ANCHORS, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)

    idx_map = {cid: i for i, cid in enumerate(ds.cell_ids)}
    expected = np.array([ds.pseudotime[idx_map[cid]] for cid in anchor_ids], dtype=np.float32)
    np.testing.assert_array_equal(table.context_pseudotime, expected)


def test_query_pseudotime_matches_dataset():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[7]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    _, targets = builder.build(query_id, anchor_ids)

    idx = ds.cell_ids.index(query_id)
    assert targets.query_pseudotime == pytest.approx(float(ds.pseudotime[idx]))


# ---------------------------------------------------------------------------
# Value correctness — soft labels
# ---------------------------------------------------------------------------

def test_context_soft_labels_match_dataset():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = _anchor_ids(ds, n=N_ANCHORS, exclude=query_id)
    table, _ = builder.build(query_id, anchor_ids)

    idx_map = {cid: i for i, cid in enumerate(ds.cell_ids)}
    for row, cid in enumerate(anchor_ids):
        np.testing.assert_array_equal(table.context_soft_labels[row], ds.soft_labels[idx_map[cid]])


def test_query_soft_labels_match_dataset():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[3]
    anchor_ids = _anchor_ids(ds, exclude=query_id)
    _, targets = builder.build(query_id, anchor_ids)

    idx = ds.cell_ids.index(query_id)
    np.testing.assert_array_equal(targets.query_soft_labels, ds.soft_labels[idx])


# ---------------------------------------------------------------------------
# Positional alignment
# ---------------------------------------------------------------------------

def test_context_rows_are_positionally_aligned_to_anchor_ids():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    anchor_ids = list(reversed(_anchor_ids(ds, n=N_ANCHORS, exclude=query_id)))
    table, _ = builder.build(query_id, anchor_ids)

    idx_map = {cid: i for i, cid in enumerate(ds.cell_ids)}
    for row, cid in enumerate(anchor_ids):
        ds_idx = idx_map[cid]
        np.testing.assert_array_equal(table.context_expression[row],    ds.expression[ds_idx])
        assert table.context_pseudotime[row] == pytest.approx(float(ds.pseudotime[ds_idx]))
        np.testing.assert_array_equal(table.context_soft_labels[row],   ds.soft_labels[ds_idx])


# ---------------------------------------------------------------------------
# Duplicate anchors (replacement sampling)
# ---------------------------------------------------------------------------

def test_duplicate_anchors_handled():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    base       = _anchor_ids(ds, n=10, exclude=query_id)
    anchor_ids = base + base   # each repeated twice
    table, _ = builder.build(query_id, anchor_ids)
    assert table.context_expression.shape == (20, N_GENES)
    np.testing.assert_array_equal(
        table.context_expression[:10],
        table.context_expression[10:],
    )


# ---------------------------------------------------------------------------
# Empty anchor list
# ---------------------------------------------------------------------------

def test_empty_anchor_list_returns_zero_row_context():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id = ds.cell_ids[0]
    table, targets = builder.build(query_id, [])
    assert table.context_expression.shape  == (0, N_GENES)
    assert table.context_pseudotime.shape  == (0,)
    assert table.context_soft_labels.shape == (0, ds.soft_labels.shape[1])
    # targets still populated for the query
    assert targets.query_soft_labels.shape == (K,)


# ---------------------------------------------------------------------------
# Unknown cell id raises
# ---------------------------------------------------------------------------

def test_unknown_query_cell_raises():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    with pytest.raises((KeyError, ValueError)):
        builder.build("cell_UNKNOWN", _anchor_ids(ds))


def test_unknown_anchor_cell_raises():
    ds = _make_dataset()
    builder = CellTableBuilder(ds)
    query_id   = ds.cell_ids[0]
    bad_anchors = _anchor_ids(ds, n=5, exclude=query_id) + ["cell_UNKNOWN"]
    with pytest.raises((KeyError, ValueError)):
        builder.build(query_id, bad_anchors)
