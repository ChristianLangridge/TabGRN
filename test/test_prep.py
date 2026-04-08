"""
Tests for spatialmt.data_preparation.prep

All tests use synthetic data — no .h5ad file required.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import anndata as ad

from spatialmt.data_preparation.prep import (
    extract_expression_matrix,
    extract_cell_labels,
    extract_gene_labels,
    extract_cell_type_labels,
    generate_pseudotime_labels,
    select_highly_variable_genes,
    prepare_dataset,
)
from spatialmt.data_preparation.diffusion_trajectory import compute_diffusion_pseudotime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n_cells=10, n_genes=5, sparse=False, scale=None):
    if scale == "normalised":
        X = (np.random.rand(n_cells, n_genes) * 4).astype(np.float32)
    elif scale == "raw":
        X = (np.random.rand(n_cells, n_genes) * 10_000).astype(np.float32)
    else:
        X = np.random.rand(n_cells, n_genes).astype(np.float32)
    if sparse:
        X = sp.csr_matrix(X)
    _timepoints = ["HB4_D5", "HB4_D7", "HB4_D11", "HB4_D16", "HB4_D21", "HB4_D30"]
    obs = pd.DataFrame(
        {
            "cell_type": [f"type_{i % 3}" for i in range(n_cells)],
            "orig.ident": [_timepoints[i % len(_timepoints)] for i in range(n_cells)],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# select_highly_variable_genes
# ---------------------------------------------------------------------------

def test_hvg_returns_correct_gene_count():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    result = select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")
    assert result.n_vars == 10


def test_hvg_output_is_copy():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    result = select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")
    result.obs["sentinel"] = 99
    assert "sentinel" not in adata.obs.columns


def test_hvg_preserves_cell_count():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    result = select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")
    assert result.n_obs == adata.n_obs


def test_hvg_selected_genes_subset_of_input():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    result = select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")
    assert set(result.var_names).issubset(set(adata.var_names))


def test_hvg_seurat_on_normalised_no_warning():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")


def test_hvg_seurat_v3_on_normalised_warns():
    adata = _make_adata(n_cells=20, n_genes=50, scale="normalised")
    with pytest.warns(UserWarning):
        select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat_v3")


def test_hvg_seurat_on_raw_warns():
    adata = _make_adata(n_cells=20, n_genes=50, scale="raw")
    with pytest.warns(UserWarning):
        select_highly_variable_genes(adata, n_top_genes=10, flavor="seurat")


def test_hvg_n_top_exceeds_available():
    adata = _make_adata(n_cells=20, n_genes=10, scale="normalised")
    result = select_highly_variable_genes(adata, n_top_genes=999, flavor="seurat")
    assert result.n_vars == adata.n_vars


# ---------------------------------------------------------------------------
# prepare_dataset
# Monkeypatches load_h5ad so no real .h5ad file is needed.
# Tests written against the current dict return type.
# TODO (Batch 2): update assertions to dataclass interface once Issue 5 lands.
# ---------------------------------------------------------------------------

@pytest.fixture()
def normalised_adata():
    """30-cell / 50-gene log-normalised AnnData for use in prepare_dataset tests."""
    return _make_adata(n_cells=30, n_genes=50, scale="normalised")


def test_prepare_dataset_returns_prepared_data(normalised_adata):
    from spatialmt.data_preparation.prep import PreparedData
    result = prepare_dataset(normalised_adata, n_top_genes=10, hvg_flavor="seurat")
    assert isinstance(result, PreparedData)


def test_prepare_dataset_shapes_consistent(normalised_adata):
    result = prepare_dataset(normalised_adata, n_top_genes=10, hvg_flavor="seurat")
    n = result.X.shape[0]
    assert len(result.cell_labels) == n
    assert len(result.y) == n
    assert len(result.orig_ident) == n


def test_prepare_dataset_gene_count(normalised_adata):
    n_top = 10
    result = prepare_dataset(normalised_adata, n_top_genes=n_top, hvg_flavor="seurat")
    assert result.X.shape[1] == len(result.gene_labels)
    assert result.X.shape[1] <= n_top


TIMEPOINTS = ["HB4_D5", "HB4_D7", "HB4_D11", "HB4_D16", "HB4_D21", "HB4_D30"]
EXPECTED   = [0.0,       0.08,      0.24,       0.44,       0.64,       1.0]


# ---------------------------------------------------------------------------
# extract_expression_matrix
# ---------------------------------------------------------------------------

def test_extract_expression_matrix_dense():
    adata = _make_adata()
    X = extract_expression_matrix(adata)
    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float32
    assert X.shape == (10, 5)


def test_extract_expression_matrix_sparse():
    adata = _make_adata(sparse=True)
    X = extract_expression_matrix(adata)
    assert isinstance(X, np.ndarray)
    assert X.shape == (10, 5)


# ---------------------------------------------------------------------------
# extract_expression_matrix — current behaviour with invalid values
# These are DOCUMENTATION tests: they pin what the function does today
# (pass-through) so that Batch 2 validation guards have a failing baseline.
# ---------------------------------------------------------------------------

def test_expression_matrix_with_nan():
    from spatialmt.data_preparation.prep import DataIntegrityError
    X = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    adata = ad.AnnData(X=X)
    with pytest.raises(DataIntegrityError, match="NaN"):
        extract_expression_matrix(adata)


def test_expression_matrix_with_negative():
    # Current behaviour: negative values pass through unchanged.
    # Batch 2 should replace this with pytest.raises(ValueError).
    X = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    adata = ad.AnnData(X=X)
    result = extract_expression_matrix(adata)
    assert (result < 0).any(), "expected negative values to pass through (current behaviour)"


def test_expression_matrix_with_inf():
    from spatialmt.data_preparation.prep import DataIntegrityError
    X = np.array([[1.0, np.inf], [3.0, 4.0]], dtype=np.float32)
    adata = ad.AnnData(X=X)
    with pytest.raises(DataIntegrityError, match="Inf"):
        extract_expression_matrix(adata)


# ---------------------------------------------------------------------------
# extract_cell_labels / extract_gene_labels
# ---------------------------------------------------------------------------

def test_extract_cell_labels():
    adata = _make_adata()
    labels = extract_cell_labels(adata)
    assert list(labels) == [f"cell_{i}" for i in range(10)]


def test_extract_gene_labels():
    adata = _make_adata()
    labels = extract_gene_labels(adata)
    assert list(labels) == [f"gene_{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# extract_cell_type_labels
# ---------------------------------------------------------------------------

def test_extract_cell_type_labels_valid_key():
    adata = _make_adata()
    y = extract_cell_type_labels(adata, cell_type_key="cell_type")
    assert len(y) == 10
    assert set(y.unique()) == {"type_0", "type_1", "type_2"}


def test_extract_cell_type_labels_missing_key():
    adata = _make_adata()
    with pytest.raises(KeyError, match="missing_key"):
        extract_cell_type_labels(adata, cell_type_key="missing_key")


# ---------------------------------------------------------------------------
# generate_pseudotime_labels
# ---------------------------------------------------------------------------

def test_pseudotime_boundary_values():
    s = pd.Series(["HB4_D5", "HB4_D30"])
    pt = generate_pseudotime_labels(s)
    assert pt.iloc[0] == pytest.approx(0.0)
    assert pt.iloc[1] == pytest.approx(1.0)


def test_pseudotime_linear_scaling():
    s = pd.Series(TIMEPOINTS)
    pt = generate_pseudotime_labels(s)
    for val, expected in zip(pt, EXPECTED):
        assert val == pytest.approx(expected, abs=1e-6)


def test_pseudotime_categorical_input():
    """orig.ident is typically a Categorical — must not raise or produce NaN."""
    s = pd.Categorical(TIMEPOINTS)
    s = pd.Series(s)
    pt = generate_pseudotime_labels(s)
    assert not pt.isna().any()


def test_pseudotime_unknown_label_is_nan():
    s = pd.Series(["HB4_D5", "UNKNOWN", "HB4_D30"])
    pt = generate_pseudotime_labels(s)
    assert pt.iloc[0] == pytest.approx(0.0)
    assert np.isnan(pt.iloc[1])
    assert pt.iloc[1+1] == pytest.approx(1.0)


def test_pseudotime_output_name():
    s = pd.Series(["HB4_D5"])
    pt = generate_pseudotime_labels(s)
    assert pt.name == "pseudotime"


def test_pseudotime_length_preserved():
    s = pd.Series(TIMEPOINTS * 100)
    pt = generate_pseudotime_labels(s)
    assert len(pt) == len(s)


# ---------------------------------------------------------------------------
# Pseudotime interface contract
# Parametrized over all pseudotime implementations. Add "diffusion" when
# the diffusion pseudotime function is available.
# ---------------------------------------------------------------------------

def _diffusion_pseudotime_from_orig_ident(orig_ident: pd.Series) -> pd.Series:
    """
    Wrap compute_diffusion_pseudotime to match the scaffold pseudotime
    contract (accepts orig_ident Series, returns named pseudotime Series).
    Builds a minimal synthetic AnnData from the orig_ident index.
    """
    import scanpy as sc
    n = len(orig_ident)
    n_genes = 50
    rng = np.random.default_rng(42)
    X = rng.random((n, n_genes)).astype(np.float32) * 4.0
    obs = pd.DataFrame(
        {
            "orig.ident": orig_ident.values,
            "class3": ["Neurectoderm"] * n,
            "S.Score": rng.random(n),
            "G2M.Score": rng.random(n),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    return compute_diffusion_pseudotime(adata, cell_type_key="class3", n_top_genes=20, n_pcs=5, n_neighbors=5)


@pytest.fixture(params=["scaffold", "diffusion"])
def pseudotime_fn(request):
    if request.param == "scaffold":
        return generate_pseudotime_labels
    if request.param == "diffusion":
        return _diffusion_pseudotime_from_orig_ident


def test_pseudotime_contract_output_is_series(pseudotime_fn):
    s = pd.Series(TIMEPOINTS)
    result = pseudotime_fn(s)
    assert isinstance(result, pd.Series)


def test_pseudotime_contract_name_is_pseudotime(pseudotime_fn):
    s = pd.Series(TIMEPOINTS)
    result = pseudotime_fn(s)
    assert result.name == "pseudotime"


def test_pseudotime_contract_values_in_unit_interval(pseudotime_fn):
    s = pd.Series(TIMEPOINTS)
    result = pseudotime_fn(s)
    valid = result.dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_pseudotime_contract_length_matches_input(pseudotime_fn):
    s = pd.Series(TIMEPOINTS * 10)
    result = pseudotime_fn(s)
    assert len(result) == len(s)


# ---------------------------------------------------------------------------
# check_memory_feasibility
# ---------------------------------------------------------------------------

from spatialmt.data_preparation.prep import check_memory_feasibility


def test_memory_feasibility_no_warn_small_data():
    """Small dataset should not trigger a warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        check_memory_feasibility(n_cells=100, n_genes=500, n_top_genes=50)


def test_memory_feasibility_warns_large_data(monkeypatch):
    """Simulate insufficient RAM: patch available memory to 1 byte."""
    import psutil
    mock_vm = psutil.virtual_memory()._replace(available=1)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: mock_vm)
    with pytest.warns(UserWarning, match="Estimated peak"):
        check_memory_feasibility(n_cells=10_000, n_genes=20_000, n_top_genes=2_000)


def test_memory_feasibility_estimation():
    """Peak estimate formula: sparse + dense bytes."""
    import psutil
    n_cells, n_genes, n_top = 1_000, 10_000, 2_000
    expected = n_cells * n_genes * 0.1 * 8 + n_cells * n_top * 4
    # Should not warn when available memory is very large (patch to 1 TB)
    import unittest.mock as mock
    mock_vm = psutil.virtual_memory()._replace(available=int(1e12))
    with mock.patch.object(psutil, "virtual_memory", return_value=mock_vm):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            check_memory_feasibility(n_cells, n_genes, n_top)
    assert expected == 1_000 * 10_000 * 0.1 * 8 + 1_000 * 2_000 * 4


# ---------------------------------------------------------------------------
# diffusion_trajectory — unit tests
# All use synthetic data; no .h5ad file required.
# ---------------------------------------------------------------------------

import scanpy as sc
from spatialmt.data_preparation.diffusion_trajectory import (
    exclude_proliferating,
    select_hvgs,
    regress_cell_cycle,
    select_root,
    compute_dpt,
    assign_prolif_pseudotime,
)


def _make_diffusion_adata(n_cells=60, n_genes=80, include_prolif=True):
    """Minimal log-normalised AnnData for diffusion trajectory tests."""
    rng = np.random.default_rng(0)
    X = rng.random((n_cells, n_genes)).astype(np.float32) * 4.0
    timepoints = ["HB4_D5", "HB4_D7", "HB4_D11", "HB4_D16", "HB4_D21", "HB4_D30"]
    cell_types = ["Neurectoderm"] * (n_cells - 5) + ["Unknown proliferating cells"] * 5
    obs = pd.DataFrame(
        {
            "orig.ident": [timepoints[i % len(timepoints)] for i in range(n_cells)],
            "class3": cell_types,
            "S.Score": rng.random(n_cells),
            "G2M.Score": rng.random(n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    # Add POU5F1 as a named gene so root selection is exercised
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes - 1)] + ["POU5F1"])
    return sc.AnnData(X=X, obs=obs, var=var)


# --- exclude_proliferating ---

def test_exclude_proliferating_counts():
    adata = _make_diffusion_adata(n_cells=60)
    traj, prolif = exclude_proliferating(adata, cell_type_key="class3")
    assert traj.n_obs == 55
    assert prolif.n_obs == 5
    assert traj.n_obs + prolif.n_obs == adata.n_obs


def test_exclude_proliferating_no_prolif_in_traj():
    adata = _make_diffusion_adata(n_cells=60)
    traj, _ = exclude_proliferating(adata, cell_type_key="class3")
    assert "Unknown proliferating cells" not in traj.obs["class3"].values


def test_exclude_proliferating_cell_ids_partition():
    adata = _make_diffusion_adata(n_cells=60)
    traj, prolif = exclude_proliferating(adata, cell_type_key="class3")
    all_ids = set(traj.obs_names) | set(prolif.obs_names)
    assert all_ids == set(adata.obs_names)


# --- select_hvgs ---

def test_select_hvgs_gene_count():
    adata = _make_diffusion_adata()
    traj, _ = exclude_proliferating(adata)
    result = select_hvgs(traj, n_top_genes=20)
    assert result.n_vars == 20


def test_select_hvgs_preserves_cell_count():
    adata = _make_diffusion_adata()
    traj, _ = exclude_proliferating(adata)
    n_before = traj.n_obs
    result = select_hvgs(traj, n_top_genes=20)
    assert result.n_obs == n_before


# --- regress_cell_cycle ---

def test_regress_cell_cycle_runs_with_scores():
    adata = _make_diffusion_adata()
    traj, _ = exclude_proliferating(adata)
    traj = select_hvgs(traj, n_top_genes=20)
    sc.pp.scale(traj)
    # Should not raise
    regress_cell_cycle(traj)


def test_regress_cell_cycle_warns_without_scores():
    adata = _make_diffusion_adata()
    traj, _ = exclude_proliferating(adata)
    traj = select_hvgs(traj, n_top_genes=20)
    traj.obs.drop(columns=["S.Score", "G2M.Score"], inplace=True)
    with pytest.warns(UserWarning):
        regress_cell_cycle(traj)


# --- select_root ---

def test_select_root_sets_iroot():
    adata = _make_diffusion_adata()
    traj, _ = exclude_proliferating(adata)
    traj = select_hvgs(traj, n_top_genes=20)
    sc.pp.scale(traj)
    sc.tl.pca(traj, n_comps=5)
    sc.pp.neighbors(traj, n_neighbors=5, n_pcs=5)
    sc.tl.diffmap(traj, n_comps=11)
    traj = select_root(traj)
    assert "iroot" in traj.uns
    assert isinstance(traj.uns["iroot"], (int, np.integer))


def test_select_root_iroot_is_d5_cell():
    adata = _make_diffusion_adata()
    traj, _ = exclude_proliferating(adata)
    traj = select_hvgs(traj, n_top_genes=20)
    sc.pp.scale(traj)
    sc.tl.pca(traj, n_comps=5)
    sc.pp.neighbors(traj, n_neighbors=5, n_pcs=5)
    sc.tl.diffmap(traj, n_comps=11)
    traj = select_root(traj)
    root_day = traj.obs["orig.ident"].iloc[traj.uns["iroot"]]
    assert root_day == "HB4_D5"


def test_select_root_fallback_warns_without_pou5f1():
    adata = _make_diffusion_adata()
    traj, _ = exclude_proliferating(adata)
    # Replace POU5F1 gene name so it's absent
    traj.var.index = [f"gene_{i}" for i in range(traj.n_vars)]
    traj = select_hvgs(traj, n_top_genes=20)
    sc.pp.scale(traj)
    sc.tl.pca(traj, n_comps=5)
    sc.pp.neighbors(traj, n_neighbors=5, n_pcs=5)
    sc.tl.diffmap(traj, n_comps=5)
    with pytest.warns(UserWarning, match="POU5F1 not found"):
        traj = select_root(traj)
    assert "iroot" in traj.uns


# --- compute_dpt (rank transform) ---

def _run_to_dpt(adata):
    traj, _ = exclude_proliferating(adata)
    traj = select_hvgs(traj, n_top_genes=20)
    sc.pp.scale(traj)
    sc.tl.pca(traj, n_comps=5)
    sc.pp.neighbors(traj, n_neighbors=5, n_pcs=5)
    sc.tl.diffmap(traj, n_comps=11)   # ≥ n_dcs default (10) required by compute_dpt
    traj = select_root(traj)
    return compute_dpt(traj)


def test_compute_dpt_pseudotime_in_unit_interval():
    adata = _make_diffusion_adata()
    traj = _run_to_dpt(adata)
    pt = traj.obs["pseudotime"]
    assert (pt > 0.0).all() and (pt <= 1.0).all()


def test_compute_dpt_pseudotime_no_nan():
    adata = _make_diffusion_adata()
    traj = _run_to_dpt(adata)
    assert not traj.obs["pseudotime"].isna().any()


def test_compute_dpt_rank_transform_is_monotone_with_raw():
    """Rank-transformed pseudotime must preserve the ordering of raw DPT."""
    adata = _make_diffusion_adata()
    traj = _run_to_dpt(adata)
    raw  = traj.obs["dpt_pseudotime"].rank()
    ranked = traj.obs["pseudotime"].rank()
    assert raw.corr(ranked) == pytest.approx(1.0, abs=1e-6)


# --- assign_prolif_pseudotime ---

def test_assign_prolif_pseudotime_all_assigned():
    adata = _make_diffusion_adata()
    traj = _run_to_dpt(adata)
    _, prolif = exclude_proliferating(adata)
    prolif = assign_prolif_pseudotime(traj, prolif)
    assert not prolif.obs["pseudotime"].isna().any()


def test_assign_prolif_pseudotime_values_in_unit_interval():
    adata = _make_diffusion_adata()
    traj = _run_to_dpt(adata)
    _, prolif = exclude_proliferating(adata)
    prolif = assign_prolif_pseudotime(traj, prolif)
    pt = prolif.obs["pseudotime"]
    assert (pt > 0.0).all() and (pt <= 1.0).all()


def test_assign_prolif_pseudotime_empty_prolif_returns_unchanged():
    adata = _make_diffusion_adata()
    traj = _run_to_dpt(adata)
    empty_prolif = adata[:0].copy()
    result = assign_prolif_pseudotime(traj, empty_prolif)
    assert result.n_obs == 0


# ---------------------------------------------------------------------------
# CSS embedding integration — R↔Python boundary tests
#
# These tests guard against the failure modes introduced when css_pseudotime.R
# produces css_embedding.csv and Python consumes it via
# compute_dpt_from_css_embedding.  All use synthetic in-memory CSVs — no real
# R output required.
# ---------------------------------------------------------------------------

from spatialmt.data_preparation.diffusion_trajectory import compute_dpt_from_css_embedding
import io


def _make_css_adata(n_cells=60, n_genes=80):
    """Log-normalised AnnData whose cell barcodes match _make_css_df."""
    rng = np.random.default_rng(1)
    X = rng.random((n_cells, n_genes)).astype(np.float32) * 4.0
    timepoints = ["HB4_D5", "HB4_D7", "HB4_D11", "HB4_D16", "HB4_D21", "HB4_D30"]
    cell_types = ["Neurectoderm"] * (n_cells - 5) + ["Unknown proliferating cells"] * 5
    obs = pd.DataFrame(
        {
            "orig.ident": [timepoints[i % len(timepoints)] for i in range(n_cells)],
            "class3": cell_types,
            "S.Score": rng.random(n_cells),
            "G2M.Score": rng.random(n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes - 1)] + ["POU5F1"])
    return sc.AnnData(X=X, obs=obs, var=var)


def _make_css_df(n_cells=60, n_dims=10, cell_prefix="cell_"):
    """Synthetic CSS embedding DataFrame matching _make_css_adata barcodes."""
    rng = np.random.default_rng(2)
    data = rng.standard_normal((n_cells, n_dims)).astype(np.float32)
    index = [f"{cell_prefix}{i}" for i in range(n_cells)]
    cols = [f"CSSPCACC_{d+1}" for d in range(n_dims)]
    return pd.DataFrame(data, index=index, columns=cols)


def _write_css_csv(tmp_path, df):
    """Write CSS DataFrame to a temp CSV the way css_pseudotime.R does it."""
    path = tmp_path / "css_embedding.csv"
    # R write.csv uses row names; we mirror that with index=True
    df.to_csv(path, index=True, index_label="cell_id")
    return path


# --- CSV format guard: cell_id written as first column by R write.csv ---

def test_css_csv_cell_id_as_index(tmp_path):
    """css_embedding.csv must parse correctly when cell_id is the first column."""
    df = _make_css_df()
    path = _write_css_csv(tmp_path, df)
    loaded = pd.read_csv(path, index_col=0)
    assert loaded.index[0] == "cell_0"
    assert loaded.shape == (60, 10)


def test_css_csv_no_extra_whitespace_in_column_names(tmp_path):
    """Column names must not carry leading/trailing whitespace after round-trip."""
    df = _make_css_df()
    path = _write_css_csv(tmp_path, df)
    loaded = pd.read_csv(path, index_col=0)
    for col in loaded.columns:
        assert col == col.strip(), f"Column '{col}' has surrounding whitespace"


# --- Cell barcode alignment ---

def test_css_full_overlap_runs(tmp_path):
    """Happy path: all barcodes match → no warning, returns Series."""
    adata = _make_css_adata()
    df = _make_css_df()
    path = _write_css_csv(tmp_path, df)
    result = compute_dpt_from_css_embedding(adata, path, cell_type_key="class3", n_neighbors=5)
    assert isinstance(result, pd.Series)
    assert result.name == "pseudotime"


def test_css_partial_overlap_warns(tmp_path):
    """Barcodes present in adata but absent from CSS → UserWarning."""
    adata = _make_css_adata(n_cells=60)
    df = _make_css_df(n_cells=50)          # last 10 cells missing from CSS
    path = _write_css_csv(tmp_path, df)
    with pytest.warns(UserWarning, match="not found"):
        compute_dpt_from_css_embedding(adata, path, cell_type_key="class3", n_neighbors=5)


def test_css_partial_overlap_result_length(tmp_path):
    """Result must contain only cells present in both adata and CSS."""
    adata = _make_css_adata(n_cells=60)
    df = _make_css_df(n_cells=50)
    path = _write_css_csv(tmp_path, df)
    with pytest.warns(UserWarning):
        result = compute_dpt_from_css_embedding(adata, path, cell_type_key="class3", n_neighbors=5)
    assert len(result) == 50


def test_css_barcode_suffix_mismatch_raises(tmp_path):
    """
    Seurat barcodes often gain a '-1' suffix (e.g. 'cell_0-1' vs 'cell_0').
    Zero shared cells should raise ValueError rather than silently return
    empty pseudotime.
    """
    adata = _make_css_adata(n_cells=60)
    df = _make_css_df(n_cells=60, cell_prefix="cell_")
    # Simulate the R '-1' suffix that Seurat appends
    df.index = [f"{b}-1" for b in df.index]
    path = _write_css_csv(tmp_path, df)
    with pytest.raises((ValueError, RuntimeError)):
        compute_dpt_from_css_embedding(adata, path, cell_type_key="class3", n_neighbors=5)


# --- Dimension robustness ---

def test_css_arbitrary_dimension_count(tmp_path):
    """Python must not hard-code 10 dims — works with any n_dims from R."""
    adata = _make_css_adata()
    for n_dims in (5, 10, 15):
        df = _make_css_df(n_dims=n_dims)
        path = _write_css_csv(tmp_path, df)
        result = compute_dpt_from_css_embedding(adata, path, cell_type_key="class3", n_neighbors=5)
        assert isinstance(result, pd.Series), f"Failed for n_dims={n_dims}"


# --- Non-finite values in CSS embedding (R preprocessing divergence) ---

def test_css_nan_in_embedding_raises(tmp_path):
    """NaN in the CSS matrix (R numerical issue) must raise, not produce silent NaN pseudotime."""
    adata = _make_css_adata()
    df = _make_css_df()
    df.iloc[0, 0] = float("nan")
    path = _write_css_csv(tmp_path, df)
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        compute_dpt_from_css_embedding(adata, path, cell_type_key="class3", n_neighbors=5)


def test_css_inf_in_embedding_raises(tmp_path):
    """Inf in CSS matrix must raise, not silently propagate."""
    adata = _make_css_adata()
    df = _make_css_df()
    df.iloc[5, 2] = float("inf")
    path = _write_css_csv(tmp_path, df)
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        compute_dpt_from_css_embedding(adata, path, cell_type_key="class3", n_neighbors=5)


# --- assign_prolif_pseudotime with pre-computed embedding (CSS path) ---

def test_assign_prolif_pseudotime_with_embedding_no_pca_needed():
    """
    CSS path passes prolif_embedding directly — must not require varm['PCs'].
    Regression test for the KeyError introduced before the fix.
    """
    adata = _make_diffusion_adata()
    traj = _run_to_dpt(adata)
    _, prolif = exclude_proliferating(adata)
    # Deliberately delete PCs to confirm the embedding path does not fall back
    if "PCs" in traj.varm:
        del traj.varm["PCs"]
    rng = np.random.default_rng(3)
    n_dims = traj.obsm["X_pca"].shape[1]
    fake_css = rng.standard_normal((prolif.n_obs, n_dims)).astype(np.float32)
    result = assign_prolif_pseudotime(traj, prolif, prolif_embedding=fake_css)
    assert not result.obs["pseudotime"].isna().any()


def test_assign_prolif_pseudotime_embedding_values_in_unit_interval():
    """Pseudotime assigned via embedding must remain in (0, 1]."""
    adata = _make_diffusion_adata()
    traj = _run_to_dpt(adata)
    _, prolif = exclude_proliferating(adata)
    rng = np.random.default_rng(4)
    n_dims = traj.obsm["X_pca"].shape[1]
    fake_css = rng.standard_normal((prolif.n_obs, n_dims)).astype(np.float32)
    result = assign_prolif_pseudotime(traj, prolif, prolif_embedding=fake_css)
    pt = result.obs["pseudotime"]
    assert (pt > 0.0).all() and (pt <= 1.0).all()


def test_assign_prolif_pseudotime_embedding_shape_mismatch_raises():
    """Embedding with wrong n_dims must raise, not silently produce garbage."""
    adata = _make_diffusion_adata()
    traj = _run_to_dpt(adata)
    _, prolif = exclude_proliferating(adata)
    wrong_dims = traj.obsm["X_pca"].shape[1] + 99
    bad_css = np.random.default_rng(5).standard_normal((prolif.n_obs, wrong_dims)).astype(np.float32)
    with pytest.raises((ValueError, Exception)):
        assign_prolif_pseudotime(traj, prolif, prolif_embedding=bad_css)
