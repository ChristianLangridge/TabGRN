"""
Unit tests for spatialmt.context.collate.icl_collate and ICLBatch.

icl_collate is a PyTorch collate_fn — it receives a list of
(CellTable, TrainingTargets) pairs produced by CellTableBuilder.build()
and stacks them into a single ICLBatch of torch.Tensors ready for the
model forward pass.

ICLBatch fields (all torch.Tensor)
------------------------------------
  context_expression  : (B, n_anchors, n_genes)  float32
  context_pseudotime  : (B, n_anchors)            float32
  context_soft_labels : (B, n_anchors, K)         float32
  query_expression    : (B, n_genes)              float32
  query_pseudotime    : (B,)                      float32
  query_soft_labels   : (B, K)                    float32

Tests are organised into:
  - ICLBatch field types (all torch.Tensor)
  - ICLBatch field shapes after collation
  - ICLBatch dtypes (float32 throughout)
  - Value correctness (collated values match source CellTable/TrainingTargets)
  - Batch size 1 (degenerate case)
  - Batch size > 1 (standard case)
  - Mixed n_anchors across batch raises (ragged context not supported)
  - Device placement (cpu by default)
"""

import numpy as np
import pandas as pd
import pytest
import torch

from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.context.builder import CellTableBuilder, CellTable, TrainingTargets
from spatialmt.context.collate import icl_collate, ICLBatch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ANCHORS = 20
N_GENES   = 10
N_CELLS   = 120
K         = 8
DAYS      = [5, 7, 11, 16, 21, 30]
BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_dataset(seed: int = 0) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    n_cells_per_day = N_CELLS // len(DAYS)
    days = np.array(DAYS * n_cells_per_day, dtype=np.int32)
    n_cells = len(days)

    expression = (rng.random((n_cells, N_GENES)) * 4.0).astype(np.float32)
    day_to_pt  = {5: 0.05, 7: 0.15, 11: 0.30, 16: 0.50, 21: 0.70, 30: 0.90}
    pseudotime = np.array(
        [day_to_pt[d] + rng.random() * 0.05 for d in days], dtype=np.float32
    )
    pseudotime = np.clip(pseudotime, 0.0, 1.0)

    cell_ids         = [f"cell_{i}" for i in range(n_cells)]
    gene_names       = [f"GENE_{i:02d}" for i in range(N_GENES)]
    cell_type_labels = pd.Series([f"state_{i % K}" for i in range(n_cells)])
    orig_ident       = pd.Series([f"HB4_D{d}" for d in days])
    raw              = rng.random((n_cells, K)).astype(np.float32)
    soft_labels      = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    manifest_hash    = ProcessedDataset._compute_manifest_hash(gene_names)
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


def _make_items(
    ds: ProcessedDataset,
    batch_size: int = BATCH_SIZE,
    n_anchors: int = N_ANCHORS,
) -> list[tuple[CellTable, TrainingTargets]]:
    """Build a list of (CellTable, TrainingTargets) pairs as a DataLoader would."""
    builder = CellTableBuilder(ds)
    items = []
    for i in range(batch_size):
        query_id   = ds.cell_ids[i]
        anchor_ids = [cid for cid in ds.cell_ids if cid != query_id][:n_anchors]
        items.append(builder.build(query_id, anchor_ids))
    return items


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_icl_collate_returns_icl_batch():
    ds    = _make_dataset()
    items = _make_items(ds)
    batch = icl_collate(items)
    assert isinstance(batch, ICLBatch)


# ---------------------------------------------------------------------------
# All fields are torch.Tensor
# ---------------------------------------------------------------------------

def test_context_expression_is_tensor():
    batch = icl_collate(_make_items(_make_dataset()))
    assert isinstance(batch.context_expression, torch.Tensor)


def test_context_pseudotime_is_tensor():
    batch = icl_collate(_make_items(_make_dataset()))
    assert isinstance(batch.context_pseudotime, torch.Tensor)


def test_context_soft_labels_is_tensor():
    batch = icl_collate(_make_items(_make_dataset()))
    assert isinstance(batch.context_soft_labels, torch.Tensor)


def test_query_expression_is_tensor():
    batch = icl_collate(_make_items(_make_dataset()))
    assert isinstance(batch.query_expression, torch.Tensor)


def test_query_pseudotime_is_tensor():
    batch = icl_collate(_make_items(_make_dataset()))
    assert isinstance(batch.query_pseudotime, torch.Tensor)


def test_query_soft_labels_is_tensor():
    batch = icl_collate(_make_items(_make_dataset()))
    assert isinstance(batch.query_soft_labels, torch.Tensor)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def test_context_expression_shape():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    assert batch.context_expression.shape == (BATCH_SIZE, N_ANCHORS, N_GENES)


def test_context_pseudotime_shape():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    assert batch.context_pseudotime.shape == (BATCH_SIZE, N_ANCHORS)


def test_context_soft_labels_shape():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    assert batch.context_soft_labels.shape == (BATCH_SIZE, N_ANCHORS, K)


def test_query_expression_shape():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    assert batch.query_expression.shape == (BATCH_SIZE, N_GENES)


def test_query_pseudotime_shape():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    assert batch.query_pseudotime.shape == (BATCH_SIZE,)


def test_query_soft_labels_shape():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    assert batch.query_soft_labels.shape == (BATCH_SIZE, K)


# ---------------------------------------------------------------------------
# Dtypes — float32 throughout
# ---------------------------------------------------------------------------

def test_context_expression_dtype():
    batch = icl_collate(_make_items(_make_dataset()))
    assert batch.context_expression.dtype == torch.float32


def test_context_pseudotime_dtype():
    batch = icl_collate(_make_items(_make_dataset()))
    assert batch.context_pseudotime.dtype == torch.float32


def test_context_soft_labels_dtype():
    batch = icl_collate(_make_items(_make_dataset()))
    assert batch.context_soft_labels.dtype == torch.float32


def test_query_expression_dtype():
    batch = icl_collate(_make_items(_make_dataset()))
    assert batch.query_expression.dtype == torch.float32


def test_query_pseudotime_dtype():
    batch = icl_collate(_make_items(_make_dataset()))
    assert batch.query_pseudotime.dtype == torch.float32


def test_query_soft_labels_dtype():
    batch = icl_collate(_make_items(_make_dataset()))
    assert batch.query_soft_labels.dtype == torch.float32


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------

def test_context_expression_values_match_source():
    """Each batch row must equal the original CellTable.context_expression."""
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    for i, (table, _) in enumerate(items):
        np.testing.assert_array_equal(
            batch.context_expression[i].numpy(),
            table.context_expression,
        )


def test_query_expression_values_match_source():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    for i, (table, _) in enumerate(items):
        np.testing.assert_array_equal(
            batch.query_expression[i].numpy(),
            table.query_expression,
        )


def test_query_pseudotime_values_match_source():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    for i, (_, targets) in enumerate(items):
        assert batch.query_pseudotime[i].item() == pytest.approx(float(targets.query_pseudotime))


def test_query_soft_labels_values_match_source():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    for i, (_, targets) in enumerate(items):
        np.testing.assert_array_equal(
            batch.query_soft_labels[i].numpy(),
            targets.query_soft_labels,
        )


def test_context_pseudotime_values_match_source():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    for i, (table, _) in enumerate(items):
        np.testing.assert_array_equal(
            batch.context_pseudotime[i].numpy(),
            table.context_pseudotime,
        )


def test_context_soft_labels_values_match_source():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=BATCH_SIZE, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    for i, (table, _) in enumerate(items):
        np.testing.assert_array_equal(
            batch.context_soft_labels[i].numpy(),
            table.context_soft_labels,
        )


# ---------------------------------------------------------------------------
# Batch size 1 (degenerate case)
# ---------------------------------------------------------------------------

def test_batch_size_one():
    ds    = _make_dataset()
    items = _make_items(ds, batch_size=1, n_anchors=N_ANCHORS)
    batch = icl_collate(items)
    assert batch.context_expression.shape  == (1, N_ANCHORS, N_GENES)
    assert batch.query_expression.shape    == (1, N_GENES)
    assert batch.query_pseudotime.shape    == (1,)


# ---------------------------------------------------------------------------
# Device placement — CPU by default
# ---------------------------------------------------------------------------

def test_batch_is_on_cpu_by_default():
    ds    = _make_dataset()
    items = _make_items(ds)
    batch = icl_collate(items)
    assert batch.context_expression.device.type  == "cpu"
    assert batch.query_expression.device.type    == "cpu"
    assert batch.query_pseudotime.device.type    == "cpu"


# ---------------------------------------------------------------------------
# Ragged context (mismatched n_anchors) raises
# ---------------------------------------------------------------------------

def test_ragged_context_raises():
    """All items in a batch must have the same n_anchors."""
    ds      = _make_dataset()
    builder = CellTableBuilder(ds)

    def _item(query_idx: int, n_anchors: int) -> tuple[CellTable, TrainingTargets]:
        query_id   = ds.cell_ids[query_idx]
        anchor_ids = [cid for cid in ds.cell_ids if cid != query_id][:n_anchors]
        return builder.build(query_id, anchor_ids)

    items = [_item(0, N_ANCHORS), _item(1, N_ANCHORS - 1)]
    with pytest.raises((ValueError, RuntimeError)):
        icl_collate(items)
