"""
Unit tests for spatialmt.data_preparation.dataset.ProcessedDataset.

Tests are organised into:
  - Construction (valid dataset builds without error)
  - Schema validation (each invariant raises the right error when violated)
  - Properties (n_cells, n_genes)
  - Manifest hash (_compute_manifest_hash contract)
  - Soft label computation (_compute_soft_labels contract)
  - Memory feasibility (_check_memory_feasibility raises ConfigurationError)
  - from_anndata stub (raises NotImplementedError until Phase 2)
"""
import hashlib

import numpy as np
import pandas as pd
import pytest

from spatialmt.config.experiment import ConfigurationError, DataConfig
from spatialmt.data_preparation.dataset import ProcessedDataset

K = 8


# ---------------------------------------------------------------------------
# Factory — reused across tests
# ---------------------------------------------------------------------------

def _make_valid(n_cells: int = 60, n_genes: int = 10, k: int = K) -> ProcessedDataset:
    rng = np.random.default_rng(42)
    expression = (rng.random((n_cells, n_genes)) * 4.0).astype(np.float32)
    pseudotime = rng.random(n_cells).astype(np.float32)          # ∈ [0, 1)
    days = [5, 7, 11, 16, 21, 30]
    collection_day = np.array(days * (n_cells // 6 + 1))[:n_cells].astype(np.int32)
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"GENE_{i:02d}" for i in range(n_genes)]
    cell_type_labels = pd.Series([f"state_{i % k}" for i in range(n_cells)])
    orig_ident = pd.Series([f"HB4_D{d}" for d in collection_day])
    raw = rng.random((n_cells, k)).astype(np.float32)
    soft_labels = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    manifest_hash = ProcessedDataset._compute_manifest_hash(gene_names)
    return ProcessedDataset(
        expression=expression,
        gene_names=gene_names,
        pseudotime=pseudotime,
        collection_day=collection_day,
        cell_ids=cell_ids,
        cell_type_labels=cell_type_labels,
        orig_ident=orig_ident,
        soft_labels=soft_labels,
        manifest_hash=manifest_hash,
    )


def _replace(ds: ProcessedDataset, **kwargs) -> ProcessedDataset:
    """Return a new ProcessedDataset with selected fields overridden."""
    fields = {
        "expression": ds.expression,
        "gene_names": ds.gene_names,
        "pseudotime": ds.pseudotime,
        "collection_day": ds.collection_day,
        "cell_ids": ds.cell_ids,
        "cell_type_labels": ds.cell_type_labels,
        "orig_ident": ds.orig_ident,
        "soft_labels": ds.soft_labels,
        "manifest_hash": ds.manifest_hash,
    }
    fields.update(kwargs)
    return ProcessedDataset(**fields)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_valid_dataset_constructs():
    ds = _make_valid()
    assert ds is not None


def test_valid_dataset_n_cells():
    ds = _make_valid(n_cells=60)
    assert ds.n_cells == 60


def test_valid_dataset_n_genes():
    ds = _make_valid(n_genes=10)
    assert ds.n_genes == 10


# ---------------------------------------------------------------------------
# Schema validation — expression
# ---------------------------------------------------------------------------

def test_expression_above_20_raises():
    ds = _make_valid()
    bad_expr = ds.expression.copy()
    bad_expr[0, 0] = 25.0
    with pytest.raises(ValueError, match="expression"):
        _replace(ds, expression=bad_expr)


def test_expression_nan_raises():
    ds = _make_valid()
    bad_expr = ds.expression.copy()
    bad_expr[0, 0] = float("nan")
    with pytest.raises(ValueError):
        _replace(ds, expression=bad_expr)


def test_expression_inf_raises():
    ds = _make_valid()
    bad_expr = ds.expression.copy()
    bad_expr[0, 0] = float("inf")
    with pytest.raises(ValueError):
        _replace(ds, expression=bad_expr)


def test_expression_wrong_dtype_is_cast_or_raises():
    """float64 input should either be silently cast to float32 or raise — not silently keep float64."""
    ds = _make_valid()
    expr_f64 = ds.expression.astype(np.float64)
    # Either raises or produces a dataset with float32 expression
    try:
        ds2 = _replace(ds, expression=expr_f64)
        assert ds2.expression.dtype == np.float32
    except (ValueError, TypeError):
        pass  # raising is also acceptable


# ---------------------------------------------------------------------------
# Schema validation — pseudotime
# ---------------------------------------------------------------------------

def test_pseudotime_above_1_raises():
    ds = _make_valid()
    bad_pt = ds.pseudotime.copy()
    bad_pt[0] = 1.1
    with pytest.raises(ValueError, match="pseudotime"):
        _replace(ds, pseudotime=bad_pt)


def test_pseudotime_below_0_raises():
    ds = _make_valid()
    bad_pt = ds.pseudotime.copy()
    bad_pt[0] = -0.01
    with pytest.raises(ValueError, match="pseudotime"):
        _replace(ds, pseudotime=bad_pt)


def test_pseudotime_nan_raises():
    ds = _make_valid()
    bad_pt = ds.pseudotime.copy()
    bad_pt[0] = float("nan")
    with pytest.raises(ValueError):
        _replace(ds, pseudotime=bad_pt)


def test_pseudotime_exactly_0_is_valid():
    ds = _make_valid()
    pt = ds.pseudotime.copy()
    pt[0] = 0.0
    ds2 = _replace(ds, pseudotime=pt)
    assert ds2 is not None


def test_pseudotime_exactly_1_is_valid():
    ds = _make_valid()
    pt = ds.pseudotime.copy()
    pt[0] = 1.0
    ds2 = _replace(ds, pseudotime=pt)
    assert ds2 is not None


# ---------------------------------------------------------------------------
# Schema validation — shape consistency
# ---------------------------------------------------------------------------

def test_cell_ids_length_mismatch_raises():
    ds = _make_valid(n_cells=60)
    short_ids = ds.cell_ids[:50]
    with pytest.raises(ValueError):
        _replace(ds, cell_ids=short_ids)


def test_gene_names_length_mismatch_raises():
    ds = _make_valid(n_genes=10)
    short_genes = ds.gene_names[:5]
    with pytest.raises(ValueError):
        _replace(ds, gene_names=short_genes)


def test_pseudotime_length_mismatch_raises():
    ds = _make_valid(n_cells=60)
    short_pt = ds.pseudotime[:50]
    with pytest.raises(ValueError):
        _replace(ds, pseudotime=short_pt)


# ---------------------------------------------------------------------------
# Schema validation — soft_labels
# ---------------------------------------------------------------------------

def test_soft_labels_wrong_sum_raises():
    ds = _make_valid()
    bad_sl = ds.soft_labels.copy()
    bad_sl[0] = bad_sl[0] * 2.0   # rows no longer sum to 1
    with pytest.raises(ValueError, match="soft_labels"):
        _replace(ds, soft_labels=bad_sl)


def test_soft_labels_correct_sum_is_valid():
    ds = _make_valid()
    assert np.allclose(ds.soft_labels.sum(axis=1), 1.0, atol=1e-5)


def test_soft_labels_shape():
    ds = _make_valid(n_cells=60, k=K)
    assert ds.soft_labels.shape == (60, K)


# ---------------------------------------------------------------------------
# Manifest hash
# ---------------------------------------------------------------------------

def test_manifest_hash_matches_sha256():
    genes = ["GENE_A", "GENE_B", "GENE_C"]
    expected = hashlib.sha256(",".join(sorted(genes)).encode()).hexdigest()
    assert ProcessedDataset._compute_manifest_hash(genes) == expected


def test_manifest_hash_order_independent():
    genes_a = ["GENE_A", "GENE_B", "GENE_C"]
    genes_b = ["GENE_C", "GENE_A", "GENE_B"]
    assert (
        ProcessedDataset._compute_manifest_hash(genes_a)
        == ProcessedDataset._compute_manifest_hash(genes_b)
    )


def test_manifest_hash_changes_with_different_genes():
    assert (
        ProcessedDataset._compute_manifest_hash(["GENE_A", "GENE_B"])
        != ProcessedDataset._compute_manifest_hash(["GENE_A", "GENE_C"])
    )


def test_manifest_hash_is_64_char_hex():
    h = ProcessedDataset._compute_manifest_hash(["SOX2", "POU5F1"])
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Soft label computation
# ---------------------------------------------------------------------------

def test_compute_soft_labels_rows_sum_to_one():
    rng = np.random.default_rng(0)
    n_cells, n_pcs, k = 80, 20, K
    # Tightly clustered centroids — each cluster a separate blob
    expression_pca = np.zeros((n_cells, n_pcs), dtype=np.float32)
    cluster_ids = np.repeat(np.arange(k), n_cells // k)
    for c in range(k):
        mask = cluster_ids == c
        expression_pca[mask] = rng.standard_normal((mask.sum(), n_pcs)) * 0.1 + c * 5.0
    config = DataConfig(max_genes=512, n_cell_states=k)
    soft = ProcessedDataset._compute_soft_labels(expression_pca, cluster_ids, config)
    assert soft.shape == (n_cells, k)
    assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-5)


def test_compute_soft_labels_dominant_state_for_tight_clusters():
    """Each cell should have highest affinity for its own cluster when clusters are well-separated."""
    rng = np.random.default_rng(1)
    n_per_cluster, n_pcs, k = 10, 20, K
    n_cells = n_per_cluster * k
    expression_pca = np.zeros((n_cells, n_pcs), dtype=np.float32)
    cluster_ids = np.repeat(np.arange(k), n_per_cluster)
    for c in range(k):
        mask = cluster_ids == c
        expression_pca[mask] = rng.standard_normal((mask.sum(), n_pcs)) * 0.01 + c * 100.0
    config = DataConfig(max_genes=512, n_cell_states=k)
    soft = ProcessedDataset._compute_soft_labels(expression_pca, cluster_ids, config)
    dominant = soft.argmax(axis=1)
    assert (dominant == cluster_ids).all()


def test_compute_soft_labels_float32_output():
    rng = np.random.default_rng(2)
    n_cells, n_pcs, k = 40, 10, K
    expression_pca = rng.standard_normal((n_cells, n_pcs)).astype(np.float32)
    cluster_ids = np.arange(n_cells) % k
    config = DataConfig(max_genes=512, n_cell_states=k)
    soft = ProcessedDataset._compute_soft_labels(expression_pca, cluster_ids, config)
    assert soft.dtype == np.float32


# ---------------------------------------------------------------------------
# Memory feasibility
# ---------------------------------------------------------------------------

def test_memory_feasibility_raises_config_error_for_large_config():
    with pytest.raises(ConfigurationError):
        ProcessedDataset._check_memory_feasibility(
            n_genes=20_000,
            d_model=128,
            batch_size=32,
            gpu_memory_bytes=16 * 1024 ** 3,
        )


def test_memory_feasibility_ok_for_debug_tier():
    # debug: n_genes=128, d_model=32, batch_size=2, 16 GB budget
    ProcessedDataset._check_memory_feasibility(
        n_genes=128,
        d_model=32,
        batch_size=2,
        gpu_memory_bytes=16 * 1024 ** 3,
    )


def test_memory_feasibility_standard_tier_512_genes():
    # standard: n_genes=512, d_model=512 (typical transformer), batch_size=16, 16 GB V100
    # attn_bytes = 16 * 512^2 * 512 * 4 = ~1.07 GB — well within 60% of 16 GB (~9.6 GB)
    ProcessedDataset._check_memory_feasibility(
        n_genes=512,
        d_model=512,
        batch_size=16,
        gpu_memory_bytes=16 * 1024 ** 3,
    )


# ---------------------------------------------------------------------------
# from_anndata — integration tests are in test/test_dataset.py (require anndata/scanpy)
# ---------------------------------------------------------------------------
