"""
Integration tests for ProcessedDataset.from_anndata.

Requires anndata + scanpy (smt_pipeline conda env).
All tests use synthetic in-memory data — no real .h5ad file required.
"""
import numpy as np
import pandas as pd
import pytest
import anndata as ad

from spatialmt.config.experiment import DataConfig
from spatialmt.data_preparation.dataset import ProcessedDataset

K = 8
TIMEPOINTS = ["HB4_D5", "HB4_D7", "HB4_D11", "HB4_D16", "HB4_D21", "HB4_D30"]
DAYS = [5, 7, 11, 16, 21, 30]
CELL_TYPES = [
    "Neurectoderm", "NPC", "Neuron", "Astrocyte",
    "OPC", "Microglia", "Endothelial", "Unknown proliferating cells",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(
    n_cells: int = 120,
    n_genes: int = 60,
    normalised: bool = True,
    include_pseudotime: bool = True,
    seed: int = 0,
) -> ad.AnnData:
    """
    Synthetic log-normalised AnnData that mirrors the real neurectoderm h5ad.

    Fields set:
      obs["orig.ident"]                    — HB4_D{day} timepoint strings
      obs["class3"]                        — cell type labels (K categories)
      obs["rank-transformed-pseudotime"]   — ∈ (0, 1] (omitted if include_pseudotime=False)
    """
    rng = np.random.default_rng(seed)
    scale = 4.0 if normalised else 10_000.0
    X = (rng.random((n_cells, n_genes)) * scale).astype(np.float32)

    orig_ident = [TIMEPOINTS[i % len(TIMEPOINTS)] for i in range(n_cells)]
    cell_types = [CELL_TYPES[i % K] for i in range(n_cells)]
    obs = pd.DataFrame(
        {"orig.ident": orig_ident, "class3": cell_types},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    if include_pseudotime:
        obs["rank-transformed-pseudotime"] = rng.random(n_cells).astype(np.float32)

    var = pd.DataFrame(index=[f"GENE_{i:04d}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


def _write_h5ad(tmp_path, adata: ad.AnnData, name: str = "test.h5ad") -> str:
    path = tmp_path / name
    adata.write_h5ad(path)
    return str(path)


def _debug_config() -> DataConfig:
    # n_top_genes capped at 20 to keep tests fast
    return DataConfig(max_genes=20, n_cell_states=K)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_from_anndata_returns_processed_dataset(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert isinstance(ds, ProcessedDataset)


def test_from_anndata_n_cells_matches_input(tmp_path):
    n_cells = 120
    adata = _make_adata(n_cells=n_cells)
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert ds.n_cells == n_cells


def test_from_anndata_n_genes_matches_config(tmp_path):
    config = _debug_config()
    adata = _make_adata(n_genes=60)
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=config)
    assert ds.n_genes <= config.max_genes


def test_from_anndata_expression_is_float32(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert ds.expression.dtype == np.float32


def test_from_anndata_expression_max_below_20(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert ds.expression.max() < 20.0


def test_from_anndata_passes_schema_validation(tmp_path):
    """ProcessedDataset._validate() runs at construction — no exception means all invariants hold."""
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ProcessedDataset.from_anndata(path, config=_debug_config())


# ---------------------------------------------------------------------------
# Pseudotime
# ---------------------------------------------------------------------------

def test_from_anndata_pseudotime_in_unit_interval(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert ds.pseudotime.min() >= 0.0
    assert ds.pseudotime.max() <= 1.0


def test_from_anndata_pseudotime_no_nan(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert not np.any(np.isnan(ds.pseudotime))


def test_from_anndata_missing_pseudotime_raises(tmp_path):
    adata = _make_adata(include_pseudotime=False)
    path = _write_h5ad(tmp_path, adata)
    with pytest.raises(ValueError, match="rank-transformed-pseudotime"):
        ProcessedDataset.from_anndata(path, config=_debug_config())


# ---------------------------------------------------------------------------
# collection_day parsing
# ---------------------------------------------------------------------------

def test_from_anndata_collection_day_values(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert set(ds.collection_day).issubset(set(DAYS))


def test_from_anndata_collection_day_dtype(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert ds.collection_day.dtype == np.int32


# ---------------------------------------------------------------------------
# Soft labels
# ---------------------------------------------------------------------------

def test_from_anndata_soft_labels_shape(tmp_path):
    config = _debug_config()
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=config)
    assert ds.soft_labels.shape == (ds.n_cells, K)


def test_from_anndata_soft_labels_rows_sum_to_one(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert np.allclose(ds.soft_labels.sum(axis=1), 1.0, atol=1e-5)


def test_from_anndata_soft_labels_float32(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert ds.soft_labels.dtype == np.float32


# ---------------------------------------------------------------------------
# Manifest hash
# ---------------------------------------------------------------------------

def test_from_anndata_manifest_hash_is_64_char_hex(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert len(ds.manifest_hash) == 64
    assert all(c in "0123456789abcdef" for c in ds.manifest_hash)


def test_from_anndata_manifest_hash_matches_gene_names(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    expected = ProcessedDataset._compute_manifest_hash(ds.gene_names)
    assert ds.manifest_hash == expected


# ---------------------------------------------------------------------------
# Auto-normalisation
# ---------------------------------------------------------------------------

def test_from_anndata_auto_normalises_raw_counts(tmp_path):
    """If X.max() >= 20, from_anndata must normalise before constructing the dataset."""
    adata = _make_adata(normalised=False)   # raw counts, max ~10_000
    path = _write_h5ad(tmp_path, adata)
    ds = ProcessedDataset.from_anndata(path, config=_debug_config())
    assert ds.expression.max() < 20.0


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_from_anndata_missing_cell_type_key_raises(tmp_path):
    adata = _make_adata()
    path = _write_h5ad(tmp_path, adata)
    with pytest.raises(KeyError):
        ProcessedDataset.from_anndata(path, config=_debug_config(), cell_type_key="nonexistent_key")
