"""
spatialmt.context.sampler — ContextSampler

Pseudotime-stratified context window sampler for ICL training.

Groups cells by collection day, excludes the withheld test day (11), and
samples `cells_per_bin` anchors from each active day-bin per query cell.
"""
from __future__ import annotations

import warnings

import numpy as np

from spatialmt.config.experiment import ContextConfig
from spatialmt.data_preparation.dataset import ProcessedDataset

_WITHHELD_DAY: int = 11


class ContextSampler:
    """Sample a pseudotime-stratified context window for a query cell.

    Parameters
    ----------
    dataset : ProcessedDataset
        Schema-validated dataset.
    config : ContextConfig
        Context window layout (n_bins, cells_per_bin, allow_replacement).
    bin_edges : np.ndarray | None
        Optional pseudotime bin edges (n_bins + 1 values).  Stored but not
        used when day-based grouping is unambiguous; reserved for future
        pseudotime-only binning mode.
    """

    def __init__(
        self,
        dataset: ProcessedDataset,
        config: ContextConfig,
        bin_edges: np.ndarray | None = None,
    ) -> None:
        self._dataset = dataset
        self._config = config
        self._bin_edges = bin_edges

        # Pre-build per-day cell index lists (excluding withheld day)
        cell_ids = dataset.cell_ids
        days = dataset.collection_day  # np.ndarray[int32]

        self._day_to_indices: dict[int, list[int]] = {}
        for idx, (cid, day) in enumerate(zip(cell_ids, days)):
            d = int(day)
            if d == _WITHHELD_DAY:
                continue
            self._day_to_indices.setdefault(d, []).append(idx)

        self._active_days: list[int] = sorted(self._day_to_indices.keys())

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
            Anchor cell identifiers (5 active bins × cells_per_bin).
        pseudotimes : np.ndarray[float32]
            Pseudotime for each anchor, aligned to *cell_ids*.
        """
        if isinstance(rng, (int, np.integer)):
            rng = np.random.default_rng(int(rng))
        elif rng is None:
            rng = np.random.default_rng()

        dataset = self._dataset
        cpb = self._config.cells_per_bin
        allow_replacement = self._config.allow_replacement

        # Build query index lookup once
        query_idx = dataset.cell_ids.index(query_cell_id)

        selected_indices: list[int] = []

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

        cell_ids = [dataset.cell_ids[i] for i in selected_indices]
        pseudotimes = np.array(
            [dataset.pseudotime[i] for i in selected_indices], dtype=np.float32
        )
        return cell_ids, pseudotimes
