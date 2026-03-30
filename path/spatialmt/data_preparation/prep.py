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
    }



