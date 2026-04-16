"""
spatialmt.data.dataset — ProcessedDataset

Immutable, schema-validated container for one experiment's training data.
Every downstream component receives this object; raw files are never accessed
after construction.

Evolving from PreparedData in data_preparation/prep.py into a full dataclass
wrapper. The primary constructor (from_anndata) is a stub until Phase 2.

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
from dataclasses import dataclass
from typing import Optional

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
    def _compute_soft_labels(
        expression_pca: np.ndarray,
        cluster_ids: np.ndarray,
        config: DataConfig,
    ) -> np.ndarray:
        """
        Distance-to-centroid softmax soft labels.

        Parameters
        ----------
        expression_pca : (n_cells, n_pcs) float32
            Cell coordinates in PCA space.
        cluster_ids : (n_cells,) int
            Hard cluster assignment for each cell (0 … K-1).
        config : DataConfig
            Provides n_cell_states (K) and label_softening_temperature.

        Returns
        -------
        soft_labels : (n_cells, K) float32
            Each row sums to 1.0.
        """
        k = config.n_cell_states
        tau = config.label_softening_temperature
        n_cells = expression_pca.shape[0]

        # Cluster centroids in PCA space
        centroids = np.stack([
            expression_pca[cluster_ids == c].mean(axis=0)
            for c in range(k)
        ])  # (K, n_pcs)

        # Euclidean distance from each cell to each centroid
        diff = expression_pca[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, K, pcs)
        distances = np.linalg.norm(diff, axis=2)  # (n_cells, K)

        # Temperature-scaled softmax of negative distances
        scaled = -distances / tau
        scaled -= scaled.max(axis=1, keepdims=True)  # numerical stability
        exp_scaled = np.exp(scaled)
        soft_labels = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

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
    # Primary constructor (stub — Phase 2)
    # ------------------------------------------------------------------

    @classmethod
    def from_anndata(
        cls,
        h5ad_path: str,
        config: Optional[DataConfig],
    ) -> "ProcessedDataset":
        """
        Load an h5ad file and construct a validated ProcessedDataset.

        This is the primary constructor. Currently a stub — the full
        implementation (HVG selection, soft label computation, holdout-safe
        pseudotime assignment) is tracked in Phase 2.
        """
        raise NotImplementedError(
            "ProcessedDataset.from_anndata is not yet implemented. "
            "Use the dataclass constructor directly with pre-processed arrays, "
            "or wait for Phase 2."
        )
