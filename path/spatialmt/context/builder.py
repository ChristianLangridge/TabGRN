"""
spatialmt.context.builder — CellTableBuilder, CellTable, TrainingTargets

Assembles the ICL input table for a single training step.
Not yet implemented — tests define the contract.

Separation of concerns
-----------------------
CellTable       — model inputs only (query labels are absent)
TrainingTargets — query labels only (never fed into the model)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spatialmt.data_preparation.dataset import ProcessedDataset


@dataclass
class CellTable:
    """Model input for one training step.

    Fields
    ------
    context_expression  : (n_anchors, n_genes) float32
    context_pseudotime  : (n_anchors,)          float32
    context_soft_labels : (n_anchors, K)        float32
    query_expression    : (n_genes,)            float32
    """
    context_expression:  np.ndarray
    context_pseudotime:  np.ndarray
    context_soft_labels: np.ndarray
    query_expression:    np.ndarray


@dataclass
class TrainingTargets:
    """Query labels used for loss computation only — never fed into the model.

    Fields
    ------
    query_pseudotime  : float32 scalar
    query_soft_labels : (K,) float32
    """
    query_pseudotime:  np.floating
    query_soft_labels: np.ndarray


class CellTableBuilder:
    def __init__(self, dataset: ProcessedDataset) -> None:
        raise NotImplementedError("CellTableBuilder is not yet implemented.")

    def build(
        self,
        query_cell_id: str,
        anchor_ids: list[str],
    ) -> tuple[CellTable, TrainingTargets]:
        raise NotImplementedError("CellTableBuilder.build is not yet implemented.")
