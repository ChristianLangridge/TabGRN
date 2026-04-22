"""
spatialmt.data_preparation.dataset — ProcessedDataset

Immutable, schema-validated container for one experiment's training data.
Every downstream component receives this object; raw files are never accessed
after construction.

Wraps the output of data_preparation/prep.py extraction functions with schema
validation, soft label computation, and manifest hashing.

Schema invariants (all checked at construction via _validate):
  - expression.max() < 20.0            guards against raw counts
  - pseudotime ∈ [0, 1]
  - No NaN or Inf in expression or pseudotime
  - soft_labels.sum(axis=1) ≈ 1.0 ± 1e-5
  - X.shape[0] == len(cell_ids) == len(pseudotime) == len(cell_type_labels)
  - X.shape[1] == len(gene_names)
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from spatialmt.config.experiment import ConfigurationError, DataConfig


@dataclass
class ProcessedDataset:
    expression: np.ndarray        # (n_cells, n_genes), float32, log-normalised
    gene_names: list[str]         # HVG names in column order
    pseudotime: np.ndarray        # (n_cells,), float32, ∈ [0, 1]
    collection_day: np.ndarray    # (n_cells,), int32, ∈ {5, 7, 11, 16, 21, 30}
    cell_ids: list[str]           # unique cell barcodes
    cell_type_labels: pd.Series   # class3 annotations
    orig_ident: pd.Series         # timepoint strings (HB4_D5, etc.)
    soft_labels: np.ndarray       # (n_cells, K), float32, rows sum to 1.0
    cell_type_categories: list[str]  # sorted class3 label names; index i → soft_labels[:, i]
    manifest_hash: str            # SHA-256 of sorted(gene_names)

    def __post_init__(self) -> None:
        # Cast expression to float32 if float64 supplied
        if self.expression.dtype != np.float32:
            object.__setattr__(self, "expression", self.expression.astype(np.float32))
        self._validate()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return self.expression.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expression.shape[1]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        n = self.expression.shape[0]

        # --- expression ---
        if np.any(np.isnan(self.expression)):
            raise ValueError("expression contains NaN values.")
        if np.any(np.isinf(self.expression)):
            raise ValueError("expression contains Inf values.")
        if self.expression.max() >= 20.0:
            raise ValueError(
                f"expression.max() = {self.expression.max():.2f} ≥ 20.0. "
                "This suggests raw counts. Apply normalize_total() + log1p() first."
            )

        # --- pseudotime ---
        if np.any(np.isnan(self.pseudotime)):
            raise ValueError("pseudotime contains NaN values.")
        if np.any(self.pseudotime < 0.0) or np.any(self.pseudotime > 1.0):
            raise ValueError(
                f"pseudotime must be in [0, 1]. "
                f"Got range [{self.pseudotime.min():.4f}, {self.pseudotime.max():.4f}]."
            )

        # --- shape consistency ---
        if len(self.cell_ids) != n:
            raise ValueError(
                f"cell_ids length ({len(self.cell_ids)}) != n_cells ({n})."
            )
        if len(self.pseudotime) != n:
            raise ValueError(
                f"pseudotime length ({len(self.pseudotime)}) != n_cells ({n})."
            )
        if len(self.gene_names) != self.expression.shape[1]:
            raise ValueError(
                f"gene_names length ({len(self.gene_names)}) != n_genes ({self.expression.shape[1]})."
            )

        # --- soft_labels ---
        row_sums = self.soft_labels.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5):
            raise ValueError(
                f"soft_labels rows must sum to 1.0 ± 1e-5. "
                f"Max deviation: {np.abs(row_sums - 1.0).max():.2e}."
            )

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_manifest_hash(gene_names: list[str]) -> str:
        """SHA-256 of sorted gene names, comma-joined. Order-independent."""
        payload = ",".join(sorted(gene_names))
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _fit_soft_label_centroids(
        train_pca: np.ndarray,
        train_cluster_ids: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """
        Estimate cluster centroids from training cells only.

        Must be called exclusively on non-withheld cells (i.e. not day 11)
        to prevent evaluation label leakage.

        Parameters
        ----------
        train_pca : (n_train, n_pcs) float32
            PCA coordinates of training cells only.
        train_cluster_ids : (n_train,) int
            Hard cluster assignment for each training cell (0 … K-1).
        k : int
            Number of cell states.

        Returns
        -------
        centroids : (K, n_pcs) float32
        """
        return np.stack([
            train_pca[train_cluster_ids == c].mean(axis=0)
            for c in range(k)
        ]).astype(np.float32)  # (K, n_pcs)

    @staticmethod
    def _compute_soft_labels(
        expression_pca: np.ndarray,
        centroids: np.ndarray,
        config: DataConfig,
    ) -> np.ndarray:
        """
        Distance-to-centroid softmax soft labels for all cells.

        Centroids must be pre-fitted on training cells only via
        _fit_soft_label_centroids — never estimated from all cells including
        the withheld evaluation timepoint.

        Parameters
        ----------
        expression_pca : (n_cells, n_pcs) float32
            PCA coordinates of ALL cells (training + withheld).
        centroids : (K, n_pcs) float32
            Cluster centroids fitted on training cells only.
        config : DataConfig
            Provides label_softening_temperature.

        Returns
        -------
        soft_labels : (n_cells, K) float32
            Each row sums to 1.0.
        """
        tau = config.label_softening_temperature

        # Euclidean distance from each cell to each centroid
        diff      = expression_pca[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)  # (n_cells, K)

        # Temperature-scaled softmax of negative distances
        scaled  = -distances / tau
        scaled -= scaled.max(axis=1, keepdims=True)   # numerical stability
        exp_s   = np.exp(scaled)
        soft_labels = exp_s / exp_s.sum(axis=1, keepdims=True)

        return soft_labels.astype(np.float32)

    @staticmethod
    def _check_memory_feasibility(
        n_genes: int,
        d_model: int,
        batch_size: int,
        gpu_memory_bytes: int,
    ) -> None:
        """
        Raise ConfigurationError if the column attention matrix exceeds 60% of
        GPU memory budget.

        Column attention shape: (batch, n_heads≈d_model, n_genes, n_genes) × float32.
        Approximated as batch × n_genes² × d_model × 4 bytes.
        """
        attn_bytes = batch_size * (n_genes ** 2) * d_model * 4
        budget = gpu_memory_bytes * 0.60
        if attn_bytes > budget:
            raise ConfigurationError(
                f"Column attention on {n_genes} genes requires "
                f"~{attn_bytes / 1e9:.1f} GB "
                f"(60% budget: {budget / 1e9:.1f} GB of {gpu_memory_bytes / 1e9:.0f} GB). "
                "Reduce max_genes or use hardware_tier='full' on A100 (Myriad)."
            )

    # ------------------------------------------------------------------
    # Primary constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        h5ad_path: str,
        config: DataConfig,
        cell_type_key: str = "class3",
        n_pca_components: int = 50,
    ) -> "ProcessedDataset":
        """
        Load an h5ad file and construct a validated ProcessedDataset.

        The h5ad must already contain 'rank-transformed-pseudotime' in obs
        (written by compute_dpt_from_css_embedding → merge_and_save).
        Normalisation (normalize_total + log1p) is applied if X.max() >= 20.

        Parameters
        ----------
        h5ad_path : str
            Path to the h5ad file (typically neurectoderm_with_pseudotime.h5ad).
        config : DataConfig
            Provides max_genes (HVG count) and soft-label hyperparameters.
        cell_type_key : str
            obs column for cell-type annotations (default 'class3').
        n_pca_components : int
            PCA dimensions used for soft-label centroid distances.
        """
        import scanpy as sc
        from sklearn.decomposition import PCA
        from spatialmt.data_preparation.prep import (
            load_h5ad,
            extract_expression_matrix,
            extract_cell_labels,
            extract_gene_labels,
            extract_cell_type_labels,
            check_memory_feasibility,
            select_highly_variable_genes,
        )

        # 1. Load and normalise
        adata = load_h5ad(h5ad_path)
        xmax = adata.X.max() if not hasattr(adata.X, "toarray") else adata.X.toarray().max()
        if xmax >= 20.0:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        check_memory_feasibility(adata.n_obs, adata.n_vars, config.max_genes)

        # 2. HVG selection on all cells (day-11 holdout is ContextSampler's concern)
        adata_hvg = select_highly_variable_genes(
            adata, n_top_genes=config.max_genes, flavor="seurat"
        )

        # 3. Extract arrays
        X = extract_expression_matrix(adata_hvg)
        cell_ids = list(extract_cell_labels(adata_hvg))
        gene_names = list(extract_gene_labels(adata_hvg))
        cell_type_labels = extract_cell_type_labels(adata_hvg, cell_type_key=cell_type_key)
        orig_ident = adata_hvg.obs["orig.ident"].copy()

        # 4. Parse collection_day from orig_ident (e.g. "HB4_D11" → 11)
        def _parse_day(s: str) -> int:
            m = re.search(r"D(\d+)$", s)
            if m is None:
                raise ValueError(f"Cannot parse collection day from orig_ident value: {s!r}")
            return int(m.group(1))

        collection_day = orig_ident.map(_parse_day).values.astype(np.int32)

        # 5. Load pseudotime from adata.obs (written by merge_and_save)
        pt_col = "rank-transformed-pseudotime"
        if pt_col not in adata.obs.columns:
            raise ValueError(
                f"'{pt_col}' not found in adata.obs. "
                "Run compute_dpt_from_css_embedding first, which calls merge_and_save "
                "to write pseudotime into the h5ad."
            )
        pseudotime = adata.obs.loc[cell_ids, pt_col].values.astype(np.float32)

        # 6. Soft labels: PCA → distance-to-centroid softmax
        #    PCA and centroids are fitted on training cells only (non-withheld
        #    days) to prevent day-11 evaluation cells from influencing the
        #    centroid positions used to compute their own soft labels.
        n_pcs = min(n_pca_components, X.shape[1], X.shape[0] - 1)

        train_mask = collection_day != config.test_timepoint  # excludes day 11

        pca = PCA(n_components=n_pcs)
        pca.fit(X[train_mask])
        expression_pca = pca.transform(X).astype(np.float32)   # ALL cells

        categories = sorted(cell_type_labels.unique())
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}
        cluster_ids = cell_type_labels.map(cat_to_idx).values.astype(int)

        centroids = cls._fit_soft_label_centroids(
            train_pca=expression_pca[train_mask],
            train_cluster_ids=cluster_ids[train_mask],
            k=config.n_cell_states,
        )

        soft_labels = cls._compute_soft_labels(expression_pca, centroids, config)

        return cls(
            expression=X,
            gene_names=gene_names,
            pseudotime=pseudotime,
            collection_day=collection_day,
            cell_ids=cell_ids,
            cell_type_labels=cell_type_labels,
            orig_ident=orig_ident,
            soft_labels=soft_labels,
            cell_type_categories=categories,
            manifest_hash=cls._compute_manifest_hash(gene_names),
        )
