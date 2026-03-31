import scanpy as sc
import numpy as np
import pandas as pd
from spatialmt.config.paths import Dirs, setup_output_dirs, validate_raw_inputs


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def load_h5ad(path) -> sc.AnnData:
    """Load an .h5ad file and return the AnnData object."""
    adata = sc.read_h5ad(path)
    return adata


def extract_expression_matrix(adata: sc.AnnData) -> np.ndarray:
    """Return the expression matrix as a dense numpy array (cells x genes)."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X.astype(np.float32)


def extract_cell_labels(adata: sc.AnnData) -> pd.Index:
    """Return cell (observation) labels."""
    return adata.obs_names.copy()


def extract_gene_labels(adata: sc.AnnData) -> pd.Index:
    """Return gene (variable) labels."""
    return adata.var_names.copy()


def extract_cell_type_labels(adata: sc.AnnData, cell_type_key: str = "cell_type") -> pd.Series:
    """
    Return the target cell-type label for each cell.

    Parameters
    ----------
    adata : AnnData
    cell_type_key : str
        Key in adata.obs that holds the cell-type annotation.
    """
    if cell_type_key not in adata.obs.columns:
        raise KeyError(
            f"Cell-type key '{cell_type_key}' not found in adata.obs. "
            f"Available keys: {list(adata.obs.columns)}"
        )
    return adata.obs[cell_type_key].copy()


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_highly_variable_genes(
    adata: sc.AnnData,
    n_top_genes: int = 2000,
    flavor: str = "seurat_v3",
) -> sc.AnnData:
    """
    Filter adata to the top highly variable genes.

    Parameters
    ----------
    adata : AnnData
        Raw counts expected for flavor='seurat_v3'; log-normalised for 'seurat'/'cell_ranger'.
    n_top_genes : int
        Number of HVGs to retain.
    flavor : str
        HVG method passed to scanpy. 'seurat_v3' works directly on raw counts.

    Returns
    -------
    AnnData
        Filtered to HVG columns only (copy).
    """
    adata = adata.copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
    adata = adata[:, adata.var["highly_variable"]]
    return adata


# ---------------------------------------------------------------------------
# Pseudotime labelling
# ---------------------------------------------------------------------------


# this function generates a linear mapping of pseudotime labels in the following order:
# "HB4_D5" = 0.0
# "HB4_D7" = 
# "HB4_D11"
# "HB4_D16" = 
# "HB4_D21" = 0.64
# "HB4_D30" = 1.0

def generate_pseudotime_labels(
    orig_ident: pd.Series,
    min_label: str = "HB4_D5",
    max_label: str = "HB4_D30",
) -> pd.Series:
    """
    Assign a linearly scaled pseudotime value to each cell based on its
    ``orig.ident`` timepoint string (e.g. ``"HB4_D7"``).

    Day numbers are parsed from the trailing integer in each label.
    ``min_label`` maps to 0.0 and ``max_label`` maps to 1.0; all other
    recognised ``HB4_D{N}`` values are scaled proportionally.

    Parameters
    ----------
    orig_ident : pd.Series
        Per-cell timepoint strings, e.g. from ``adata.obs["orig.ident"]``.
    min_label : str
        The timepoint string that should map to 0.0.
    max_label : str
        The timepoint string that should map to 1.0.

    Returns
    -------
    pd.Series
        Float pseudotime values in [0, 1], named ``"pseudotime"``.
        Cells whose ``orig_ident`` cannot be parsed receive ``NaN``.
    """
    def _parse_day(label: str) -> float:
        parts = str(label).split("_D")
        if len(parts) == 2 and parts[1].isdigit():
            return float(parts[1])
        return float("nan")

    days = orig_ident.astype(str).map(_parse_day)
    min_day = _parse_day(min_label)
    max_day = _parse_day(max_label)

    pseudotime = (days - min_day) / (max_day - min_day)
    pseudotime.name = "pseudotime"
    return pseudotime


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def prepare_dataset(
    h5ad_path,
    cell_type_key: str = "cell_type",
    n_top_genes: int = 2000,
    hvg_flavor: str = "seurat_v3",
) -> dict:
    """
    Full preprocessing pipeline: load → extract → HVG filter.

    Returns
    -------
    dict with keys:
        X           : np.ndarray, shape (n_cells, n_hvgs)
        cell_labels : pd.Index
        gene_labels : pd.Index   (HVG subset)
        y           : pd.Series  (cell-type labels)
    """
    adata = load_h5ad(h5ad_path)

    adata_hvg = select_highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=hvg_flavor)

    X = extract_expression_matrix(adata_hvg)
    cell_labels = extract_cell_labels(adata_hvg)
    gene_labels = extract_gene_labels(adata_hvg)
    y = extract_cell_type_labels(adata_hvg, cell_type_key=cell_type_key)

    return {
        "X": X,
        "cell_labels": cell_labels,
        "gene_labels": gene_labels,
        "y": y,
        "orig_ident": adata_hvg.obs["orig.ident"].copy(),
    }



