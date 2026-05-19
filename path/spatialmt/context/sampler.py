"""
spatialmt.context.sampler — ContextSampler

Context window sampler for ICL training supporting three independent
anchor-selection strategies that can be combined in any proportion:

  Day-stratified       — ``cells_per_bin`` anchors per active collection day.
  Pseudotime-stratified — ``n_pseudotime_anchors`` anchors drawn from equal-width
                          pseudotime bins across [0, 1], covering developmental
                          progression independent of collection day.
  Composition-stratified — ``n_composition_anchors`` anchors distributed evenly
                           across dominant-class pools, ensuring each cell-type
                           archetype is represented in every context window.

Day-11 cells are excluded from all anchor pools; they are the withheld test set.
"""
from __future__ import annotations

import warnings

import numpy as np

from spatialmt.config.experiment import ContextConfig
from spatialmt.data_preparation.dataset import ProcessedDataset

_WITHHELD_DAY: int = 11


class ContextSampler:
    """Sample a context window for a query cell.

    Parameters
    ----------
    dataset : ProcessedDataset
        Schema-validated dataset.
    config : ContextConfig
        Context window layout — controls which sampling components are active
        and how many anchors each contributes.
    bin_edges : np.ndarray | None
        Optional custom pseudotime bin edges (n_pseudotime_bins + 1 values).
        If None, equal-width bins over [0, 1] are used.
    """

    def __init__(
        self,
        dataset: ProcessedDataset,
        config: ContextConfig,
        bin_edges: np.ndarray | None = None,
    ) -> None:
        self._dataset = dataset
        self._config = config

        cell_ids = dataset.cell_ids
        days = dataset.collection_day  # np.ndarray[int32]

        # --- Day-stratified index lists (excluding withheld day) ---
        self._day_to_indices: dict[int, list[int]] = {}
        for idx, (cid, day) in enumerate(zip(cell_ids, days)):
            d = int(day)
            if d == _WITHHELD_DAY:
                continue
            self._day_to_indices.setdefault(d, []).append(idx)

        self._active_days: list[int] = sorted(self._day_to_indices.keys())

        # --- Pseudotime-stratified index lists ---
        # Equal-width bins over [0, 1]; day-11 cells excluded.
        n_pt_bins = config.n_pseudotime_bins
        if bin_edges is not None:
            self._pt_bin_edges = bin_edges
        else:
            self._pt_bin_edges = np.linspace(0.0, 1.0, n_pt_bins + 1)

        self._pt_bin_to_indices: dict[int, list[int]] = {b: [] for b in range(n_pt_bins)}
        for idx, day in enumerate(days):
            if int(day) == _WITHHELD_DAY:
                continue
            pt = float(dataset.pseudotime[idx])
            # np.searchsorted on right edges; clip so pt==1.0 lands in last bin
            bin_idx = int(np.searchsorted(self._pt_bin_edges[1:], pt, side="left"))
            bin_idx = min(bin_idx, n_pt_bins - 1)
            self._pt_bin_to_indices[bin_idx].append(idx)

        # --- Composition-stratified index lists ---
        # Dominant class = argmax of soft_labels row; day-11 cells excluded.
        dominant = dataset.soft_labels.argmax(axis=1)
        n_classes = dataset.soft_labels.shape[1]
        self._class_to_indices: dict[int, list[int]] = {c: [] for c in range(n_classes)}
        for idx, day in enumerate(days):
            if int(day) == _WITHHELD_DAY:
                continue
            self._class_to_indices[int(dominant[idx])].append(idx)

    # ------------------------------------------------------------------

    def sample(
        self,
        query_cell_id: str,
        rng: np.random.Generator | int | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Return a context window for *query_cell_id*.

        Parameters
        ----------
        query_cell_id : str
            Cell whose context window is being built.
        rng : Generator | int | None
            Random source.  An ``int`` is wrapped in
            ``np.random.default_rng(int)``.  ``None`` uses a fresh
            non-seeded generator.

        Returns
        -------
        cell_ids : list[str]
            Anchor cell identifiers (day-stratified block first, then
            pseudotime-stratified, then composition-stratified).
        pseudotimes : np.ndarray[float32]
            Pseudotime for each anchor, aligned to *cell_ids*.
        """
        if isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(int(rng))
        elif rng is None:
            rng = np.random.default_rng()

        dataset = self._dataset
        cfg = self._config
        allow_replacement = cfg.allow_replacement
        query_idx = dataset.cell_ids.index(query_cell_id)

        selected_indices: list[int] = []

        # --- Day-stratified component ---
        cpb = cfg.cells_per_bin
        if cpb > 0:
            for day in self._active_days:
                candidates = [i for i in self._day_to_indices[day] if i != query_idx]
                if len(candidates) < cpb:
                    if not allow_replacement:
                        raise ValueError(
                            f"Bin for day {day} has only {len(candidates)} eligible cells "
                            f"(need {cpb}) and allow_replacement=False."
                        )
                    warnings.warn(
                        f"Sparse bin: day {day} has only {len(candidates)} eligible cells "
                        f"(< cells_per_bin={cpb}). Sampling with replacement.",
                        UserWarning,
                        stacklevel=2,
                    )
                    chosen = rng.choice(candidates, size=cpb, replace=True).tolist()
                else:
                    chosen = rng.choice(candidates, size=cpb, replace=False).tolist()
                selected_indices.extend(chosen)

        # --- Pseudotime-stratified component ---
        n_pt = cfg.n_pseudotime_anchors
        if n_pt > 0:
            n_bins = cfg.n_pseudotime_bins
            base = n_pt // n_bins
            remainder = n_pt % n_bins
            for b in range(n_bins):
                n_draw = base + (1 if b < remainder else 0)
                if n_draw == 0:
                    continue
                candidates = [i for i in self._pt_bin_to_indices[b] if i != query_idx]
                if not candidates:
                    continue
                if len(candidates) < n_draw:
                    chosen = rng.choice(candidates, size=n_draw, replace=True).tolist()
                else:
                    chosen = rng.choice(candidates, size=n_draw, replace=False).tolist()
                selected_indices.extend(chosen)

        # --- Composition-stratified component ---
        n_comp = cfg.n_composition_anchors
        if n_comp > 0:
            n_classes = len(self._class_to_indices)
            base = n_comp // n_classes
            remainder = n_comp % n_classes
            class_order = list(range(n_classes))
            rng.shuffle(class_order)
            for rank, cls in enumerate(class_order):
                n_draw = base + (1 if rank < remainder else 0)
                if n_draw == 0:
                    continue
                candidates = [i for i in self._class_to_indices[cls] if i != query_idx]
                if not candidates:
                    continue
                if len(candidates) < n_draw:
                    chosen = rng.choice(candidates, size=n_draw, replace=True).tolist()
                else:
                    chosen = rng.choice(candidates, size=n_draw, replace=False).tolist()
                selected_indices.extend(chosen)

        cell_ids = [dataset.cell_ids[i] for i in selected_indices]
        pseudotimes = np.array(
            [dataset.pseudotime[i] for i in selected_indices], dtype=np.float32
        )
        return cell_ids, pseudotimes
