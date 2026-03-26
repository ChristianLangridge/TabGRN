"""
spatialmt.data.dataset
======================
ProcessedDataset — validated contract between preprocessing and model.

ROTATION SCOPE
--------------
Active fields: expression, gene_names, pseudotime, collection_day,
               cell_ids, train_cells, val_cells, test_cells,
               manifest_hash, preprocessing_config

FULL PROJECT (Phase 5A — July onwards)
---------------------------------------
Add: soft_labels field (shape: n_cells × K, rows sum to 1.0)
     Computed by from_tpm_files() when DataConfig.n_cell_states > 0
     and cluster annotations are present in the AnnData.

Phase gate: soft_labels is None in the rotation scope.
            _validate() skips soft_labels checks if it is None.
            CompositionHead will assert soft_labels is not None at training time.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spatialmt.config.experiment import DataConfig

logger = logging.getLogger(__name__)

VALID_COLLECTION_DAYS: frozenset[int] = frozenset({5, 7, 11, 16, 21})
MEMORY_SAFETY_FACTOR: float = 0.60
DEFAULT_GPU_MEMORY_BYTES: int = 16 * 1024 ** 3  # V100


@dataclass
class ProcessedDataset:
    """
    Immutable, schema-validated container for one experiment's data.

    ROTATION SCOPE fields
    ---------------------
    expression         : (n_cells, n_genes)  log1p TPM
    gene_names         : list[str]            HVG names in column order
    pseudotime         : (n_cells,)           DC1 normalised to [0, 1]
    collection_day     : (n_cells,)           int ∈ {5, 7, 11, 16, 21}
    cell_ids           : list[str]
    train/val/test     : bool masks, mutually exclusive
    manifest_hash      : str
    preprocessing_config : dict

    FULL PROJECT addition (Phase 5A)
    ---------------------------------
    soft_labels        : (n_cells, K) or None
                         Cell state composition targets.
                         None during rotation scope — set by from_tpm_files()
                         once cluster annotations are available.
    """

    expression: np.ndarray
    gene_names: list[str]
    pseudotime: np.ndarray
    collection_day: np.ndarray
    cell_ids: list[str]

    train_cells: np.ndarray
    val_cells: np.ndarray
    test_cells: np.ndarray

    manifest_hash: str
    preprocessing_config: dict

    # Phase 5A — None during rotation scope
    soft_labels: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return self.expression.shape[0]

    @property
    def n_genes(self) -> int:
        return self.expression.shape[1]

    @property
    def n_cell_states(self) -> int | None:
        if self.soft_labels is None:
            return None
        return self.soft_labels.shape[1]

    @property
    def has_soft_labels(self) -> bool:
        return self.soft_labels is not None

    @property
    def train_expression(self) -> np.ndarray:
        return self.expression[self.train_cells]

    @property
    def train_pseudotime(self) -> np.ndarray:
        return self.pseudotime[self.train_cells]

    @property
    def train_soft_labels(self) -> np.ndarray | None:
        if self.soft_labels is None:
            return None
        return self.soft_labels[self.train_cells]

    @property
    def test_expression(self) -> np.ndarray:
        return self.expression[self.test_cells]

    @property
    def test_pseudotime(self) -> np.ndarray:
        return self.pseudotime[self.test_cells]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_tpm_files(
        cls,
        tpm_dir: str | Path,
        config: "DataConfig",
        gpu_memory_bytes: int = DEFAULT_GPU_MEMORY_BYTES,
    ) -> "ProcessedDataset":
        """
        Build a ProcessedDataset from processed TPM CSV files.

        ROTATION SCOPE implementation
        ------------------------------
        1. Load processed_tpm.csv from tpm_dir
        2. Apply log1p transform
        3. HVG selection, cap at config.max_genes
        4. Memory pre-check
        5. Load DC1 pseudotime, normalise to [0, 1]
        6. Build hold-out splits (day 11 test, stratified val)
        7. soft_labels = None  ← rotation scope
        8. Validate schema
        9. Return instance

        PHASE 5A ADDITION (July onwards)
        ---------------------------------
        After step 6, add:
        7. Load cluster annotations from AnnData .obs
        8. soft_labels = _compute_soft_labels(
               expression_pca, cluster_ids, config
           )
        9. Validate soft_labels schema
        """
        raise NotImplementedError(
            "Implement in GREEN phase (Phase 1, Weeks 3–4).\n"
            "See docstring for rotation scope steps.\n"
            "soft_labels = None for rotation scope."
        )

    # ------------------------------------------------------------------
    # Soft label computation (Phase 5A)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_soft_labels(
        expression_pca: np.ndarray,
        cluster_ids: np.ndarray,
        config: "DataConfig",
    ) -> np.ndarray:
        """
        Distance-to-centroid softmax label softening.

        FULL PROJECT — Phase 5A. Do not implement during rotation.

        For each cell:
          1. Compute Euclidean distance to each cluster centroid in PCA space
          2. Apply temperature-scaled softmax to negative distances:
             soft_label = softmax(-distances / temperature)

        Returns
        -------
        soft_labels : (n_cells, K)  rows sum to 1.0, all values ≥ 0
        """
        raise NotImplementedError(
            "Implement in Phase 5A (July onwards).\n\n"
            "Steps:\n"
            "  1. Compute centroids: centroid[k] = mean of PCA coords in cluster k\n"
            "  2. For each cell: distances = [||cell_pca - centroid_k|| for k in K]\n"
            "  3. soft_label = softmax(-distances / config.label_softening_temperature)\n"
            "  4. Assert soft_labels.sum(axis=1) ≈ 1.0\n"
            "  5. Assert soft_labels.min() >= 0.0\n"
            "  6. Biological sanity check:\n"
            "       day5_cells = collection_day == 5\n"
            "       assert soft_labels[day5_cells, 0].mean() > 0.5\n"
            "       (neuroectodermal is state 0, day 5 cells should dominate it)"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def _validate(cls, instance: "ProcessedDataset") -> None:
        """Schema contract validation. Called inside from_tpm_files()."""
        n, g = instance.n_cells, instance.n_genes

        # Shape consistency
        assert instance.expression.shape == (n, g)
        assert len(instance.gene_names) == g
        assert instance.pseudotime.shape == (n,)
        assert instance.collection_day.shape == (n,)
        assert len(instance.cell_ids) == n

        # Split masks — mutually exclusive
        assert np.sum(instance.train_cells & instance.test_cells) == 0, \
            "train/test overlap"
        assert np.sum(instance.train_cells & instance.val_cells) == 0, \
            "train/val overlap"
        assert np.sum(instance.val_cells & instance.test_cells) == 0, \
            "val/test overlap"

        # Test set is day 11 only
        test_days = set(instance.collection_day[instance.test_cells].tolist())
        assert test_days == {11}, \
            f"test_cells contains days {test_days}, expected {{11}}"

        # Expression is log-transformed
        assert float(instance.expression.max()) < 20.0, \
            "Max expression > 20 suggests raw TPM. Apply log1p."

        # Pseudotime in [0, 1]
        assert float(instance.pseudotime.min()) >= 0.0
        assert float(instance.pseudotime.max()) <= 1.0

        # Valid collection days
        invalid = set(instance.collection_day.tolist()) - VALID_COLLECTION_DAYS
        assert not invalid, f"Invalid collection days: {invalid}"

        # No NaN or Inf
        assert not np.any(np.isnan(instance.expression))
        assert not np.any(np.isinf(instance.expression))
        assert not np.any(np.isnan(instance.pseudotime))

        # Soft labels validation (Phase 5A — skipped if None)
        if instance.soft_labels is not None:
            K = instance.soft_labels.shape[1]
            assert instance.soft_labels.shape == (n, K), \
                f"soft_labels shape {instance.soft_labels.shape} != ({n}, {K})"
            row_sums = instance.soft_labels.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-5), \
                f"soft_labels rows do not sum to 1.0. Max deviation: {abs(row_sums - 1.0).max():.6f}"
            assert float(instance.soft_labels.min()) >= 0.0, \
                "soft_labels contains negative values"

    # ------------------------------------------------------------------
    # Memory pre-check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_memory_feasibility(
        n_genes: int,
        d_model: int,
        batch_size: int,
        gpu_memory_bytes: int,
    ) -> None:
        attn_bytes = batch_size * (n_genes ** 2) * d_model * 4
        budget = gpu_memory_bytes * MEMORY_SAFETY_FACTOR
        if attn_bytes > budget:
            raise ConfigurationError(
                f"Column attention on {n_genes} genes requires "
                f"~{attn_bytes / 1e9:.1f} GB "
                f"(budget: {budget / 1e9:.1f} GB at {MEMORY_SAFETY_FACTOR:.0%}).\n"
                f"Reduce max_genes or use hardware_tier='full' on A100 (Myriad)."
            )

    # ------------------------------------------------------------------
    # Manifest hash
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_manifest_hash(
        gene_names: list[str],
        preprocessing_config: dict,
    ) -> str:
        payload = json.dumps(
            {"genes": sorted(gene_names), "config": preprocessing_config},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


class ConfigurationError(ValueError):
    pass


import json  # noqa: E402 — placed after class to avoid circular at top