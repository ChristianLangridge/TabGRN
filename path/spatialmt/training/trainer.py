"""
spatialmt.training.trainer — Fine-tuning loop for TabGRN-ICL.

Design note — amortized inference / ICL
----------------------------------------
TabGRN uses a pretrained in-context learning backbone. Each training step
constructs a fresh (query_cell, context_cells) pair via ContextSampler and
CellTableBuilder; the model's forward pass *is* the inference step.

"Training" here means gradient-descent fine-tuning:
  • The dual-head stack (PseudotimeHead, CompositionHead) is always trainable.
  • col_embedder (gene×gene attention) is frozen for `warmup_col_steps` steps;
    its random init would perturb the pretrained backbone before the embeddings
    stabilise.
  • tf_icl (pretrained ICL transformer) is frozen for `warmup_icl_steps` steps.
  • row_interactor and anchor_label_embedder are unfrozen from step 0.

Day 11 cells are the held-out test set. They are excluded from the query pool;
ContextSampler already excludes them from the anchor pool.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from spatialmt.context.collate import ICLBatch, icl_collate
from spatialmt.training.muon import Muon

if TYPE_CHECKING:
    from spatialmt.config.experiment import ExperimentConfig
    from spatialmt.context.builder import CellTableBuilder
    from spatialmt.context.sampler import ContextSampler
    from spatialmt.data_preparation.dataset import ProcessedDataset
    from spatialmt.model.loss import DualHeadLoss
    from spatialmt.model.tabgrn import TabICLRegressor

_WITHHELD_DAY: int = 11


class MuonAdamW:
    """Combined Muon + AdamW optimizer.

    Routes each parameter to the appropriate sub-optimizer based on ndim:
      ndim >= 2  →  Muon  (weight matrices: attention projections, FC weights)
      ndim <  2  →  AdamW (biases, layer-norm scales, scalars)

    The ``param_groups`` attribute exposes the original *logical* groups
    (col, row, icl, head, loss) for LR inspection and scheduling.  The
    internal sub-optimizers each have their own split groups.

    Parameters
    ----------
    param_groups : list[dict]
        Each dict must have 'params' (list of tensors) and 'lr' (float).
        An optional 'name' key is preserved for test introspection.
    """

    def __init__(self, param_groups: list[dict]) -> None:
        self.param_groups = param_groups

        muon_groups: list[dict] = []
        adamw_groups: list[dict] = []

        for g in param_groups:
            name = g.get("name", "")
            lr   = g["lr"]
            matrix_params = [p for p in g["params"] if p.ndim >= 2]
            vector_params  = [p for p in g["params"] if p.ndim < 2]
            if matrix_params:
                muon_groups.append({"name": name, "params": matrix_params, "lr": lr})
            if vector_params:
                adamw_groups.append({"name": name, "params": vector_params, "lr": lr})

        self._muon  = Muon(muon_groups)  if muon_groups  else None
        self._adamw = torch.optim.AdamW(adamw_groups) if adamw_groups else None

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._muon  is not None:
            self._muon.zero_grad(set_to_none=set_to_none)
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> None:
        if self._muon  is not None:
            self._muon.step(closure)
        if self._adamw is not None:
            self._adamw.step(closure)


def _batch_to_device(batch: ICLBatch, device: torch.device) -> ICLBatch:
    return ICLBatch(
        context_expression=batch.context_expression.to(device),
        context_pseudotime=batch.context_pseudotime.to(device),
        context_soft_labels=batch.context_soft_labels.to(device),
        query_expression=batch.query_expression.to(device),
        query_pseudotime=batch.query_pseudotime.to(device),
        query_soft_labels=batch.query_soft_labels.to(device),
    )


class Trainer:
    """Fine-tunes TabICLRegressor on the pseudotime + composition dual-head task.

    Parameters
    ----------
    model : TabICLRegressor
    dataset : ProcessedDataset
    sampler : ContextSampler
    builder : CellTableBuilder
    loss_fn : DualHeadLoss
    config : ExperimentConfig
    n_epochs : int
        Number of passes through n_steps_per_epoch optimiser steps.
    n_steps_per_epoch : int
        Optimiser steps per epoch.  Each step draws one fresh ICL sample.
    callbacks : list
        Objects with an ``on_epoch_end(model, dataset, epoch)`` method called
        at the end of every epoch.
    """

    def __init__(
        self,
        model: "TabICLRegressor",
        dataset: "ProcessedDataset",
        sampler: "ContextSampler",
        builder: "CellTableBuilder",
        loss_fn: "DualHeadLoss",
        config: "ExperimentConfig",
        n_epochs: int = 50,
        n_steps_per_epoch: int = 100,
        callbacks: list | None = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.sampler = sampler
        self.builder = builder
        self.loss_fn = loss_fn
        self.config = config
        self.n_epochs = n_epochs
        self.n_steps_per_epoch = n_steps_per_epoch
        self.callbacks: list = callbacks if callbacks is not None else []
        self.global_step: int = 0

        # Pre-filter query pool: exclude day 11 (held-out test set)
        self._query_pool: list[str] = [
            cid
            for cid, day in zip(dataset.cell_ids, dataset.collection_day)
            if int(day) != _WITHHELD_DAY
        ]

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------

    def _make_dataloader(self) -> DataLoader:
        """Return a DataLoader backed by a dynamically-sampled IterableDataset.

        Each iteration draws a fresh (query_cell, context) pair from the
        Trainer's ContextSampler and CellTableBuilder.  The dataset yields
        n_steps_per_epoch individual (CellTable, TrainingTargets) items;
        icl_collate wraps each into a batch of size 1.
        """
        sampler = self.sampler
        builder = self.builder
        query_pool = self._query_pool
        n_steps = self.n_steps_per_epoch

        class _ICLDataset(IterableDataset):
            def __iter__(self):
                rng = np.random.default_rng()
                for _ in range(n_steps):
                    query_id = str(rng.choice(query_pool))
                    anchor_ids, _ = sampler.sample(query_id)
                    table, targets = builder.build(query_id, anchor_ids)
                    yield table, targets

        return DataLoader(
            _ICLDataset(),
            batch_size=1,
            collate_fn=icl_collate,
        )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _make_optimizer(self) -> "MuonAdamW":
        """Muon (weight matrices) + AdamW (biases / norms / scalars).

        The 4 model parameter groups and 1 loss group are preserved as logical
        groups on the returned MuonAdamW object.  Within each group, ndim >= 2
        parameters go to the Muon sub-optimizer; ndim < 2 parameters go to
        AdamW.  The Kendall uncertainty parameters (log_sigma_sq_*) are 0-dim
        scalars and always route to AdamW.
        """
        model_groups = self.model.parameter_groups()
        loss_group = {
            "name": "loss",
            "params": list(self.loss_fn.parameters()),
            "lr": self.config.model.lr_head,
        }
        return MuonAdamW(model_groups + [loss_group])

    # ------------------------------------------------------------------
    # Warmup / freeze scheduling
    # ------------------------------------------------------------------

    def _apply_warmup_freeze(self, global_step: int) -> None:
        """Freeze or unfreeze col_embedder and tf_icl based on global_step.

        col_embedder : frozen while global_step < warmup_col_steps
        tf_icl       : frozen while global_step < warmup_icl_steps
        All other parameters (row_interactor, heads) are always trainable.
        """
        cfg = self.config.model

        col_frozen = global_step < cfg.warmup_col_steps
        for p in self.model.col_embedder.parameters():
            p.requires_grad_(not col_frozen)

        icl_frozen = global_step < cfg.warmup_icl_steps
        for p in self.model.tf_icl.parameters():
            p.requires_grad_(not icl_frozen)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(self) -> dict:
        """Run fine-tuning and return averaged metrics.

        Returns
        -------
        dict with keys:
            "train_loss"  — Kendall uncertainty-weighted total loss
            "pt_loss"     — raw MSE component
            "comp_loss"   — raw KL divergence component
        """
        optimizer = self._make_optimizer()
        device = next(self.model.parameters()).device

        sum_loss = sum_pt = sum_comp = 0.0
        total_steps = 0

        for epoch in range(self.n_epochs):
            dataloader = self._make_dataloader()

            for batch in dataloader:
                self._apply_warmup_freeze(self.global_step)

                batch = _batch_to_device(batch, device)

                optimizer.zero_grad()

                pt_pred, comp_pred = self.model(batch)

                total_loss, pt_loss, comp_loss = self.loss_fn(
                    pt_pred,
                    batch.query_pseudotime,
                    comp_pred,
                    batch.query_soft_labels,
                )

                total_loss.backward()
                optimizer.step()

                self.global_step += 1
                sum_loss += total_loss.item()
                sum_pt += pt_loss.item()
                sum_comp += comp_loss.item()
                total_steps += 1

            for cb in self.callbacks:
                cb.on_epoch_end(self.model, self.dataset, epoch)

        n = max(total_steps, 1)
        return {
            "train_loss": sum_loss / n,
            "pt_loss": sum_pt / n,
            "comp_loss": sum_comp / n,
        }
