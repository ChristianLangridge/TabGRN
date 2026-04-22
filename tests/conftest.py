"""
Session-scoped fixtures shared across the tests/ suite.

Fixtures added here as components are implemented:
  - debug_config         ExperimentConfig.debug_preset()
  - synthetic_dataset    ProcessedDataset — 100 cells, 10 genes, K=8 soft_labels

Planned (added when model layer is implemented):
  - toy_model                   TabICLRegressor(n_layers=2, d_model=32, n_genes=10)
  - synthetic_attention_weights (n_heads=2, n_genes=10) — SOX2 boosted to highest weight
  - correlated_expression       GENE_02 and GENE_03 perfectly correlated (SHAP stability)
"""
import numpy as np
import pandas as pd
import pytest

from spatialmt.config.experiment import ExperimentConfig
from spatialmt.data_preparation.dataset import ProcessedDataset

K = 8
N_CELLS = 100
N_GENES = 10
TIMEPOINTS = [5, 7, 11, 16, 21, 30]
GENE_NAMES = [f"GENE_{i:02d}" for i in range(N_GENES)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_soft_labels(n_cells: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((n_cells, k)).astype(np.float32)
    return (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)


def _make_valid_dataset(
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    k: int = K,
    seed: int = 0,
) -> ProcessedDataset:
    rng = np.random.default_rng(seed)
    expression = (rng.random((n_cells, n_genes)) * 4.0).astype(np.float32)
    pseudotime = rng.random(n_cells).astype(np.float32)
    days = np.array(TIMEPOINTS * (n_cells // len(TIMEPOINTS) + 1))[:n_cells]
    collection_day = days.astype(np.int32)
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"GENE_{i:02d}" for i in range(n_genes)]
    cell_type_labels = pd.Series([f"state_{i % k}" for i in range(n_cells)])
    orig_ident = pd.Series([f"HB4_D{d}" for d in collection_day])
    soft_labels = _make_soft_labels(n_cells, k, seed=seed)
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def debug_config() -> ExperimentConfig:
    return ExperimentConfig.debug_preset()


@pytest.fixture(scope="session")
def synthetic_dataset() -> ProcessedDataset:
    return _make_valid_dataset()


@pytest.fixture(scope="session")
def synthetic_dataset_with_labels() -> ProcessedDataset:
    """Dataset where day-5 cells (collection_day==5) are all labelled state_0."""
    rng = np.random.default_rng(99)
    n_cells, n_genes = 120, N_GENES
    expression = (rng.random((n_cells, n_genes)) * 4.0).astype(np.float32)
    pseudotime = rng.random(n_cells).astype(np.float32)
    collection_day = np.array(TIMEPOINTS * (n_cells // len(TIMEPOINTS) + 1))[:n_cells].astype(np.int32)
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    gene_names = [f"GENE_{i:02d}" for i in range(n_genes)]
    cell_type_labels = pd.Series(
        ["Neurectoderm" if d == 5 else f"state_{i % (K - 1) + 1}" for i, d in enumerate(collection_day)]
    )
    orig_ident = pd.Series([f"HB4_D{d}" for d in collection_day])
    soft_labels = _make_soft_labels(n_cells, K, seed=99)
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
