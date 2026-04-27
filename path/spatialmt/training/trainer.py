"""
spatialmt.training.trainer — Fine-tuning loop for TabGRN-ICL.

Design note — amortized inference / ICL
----------------------------------------
TabGRN uses a pretrained in-context learning backbone. Each training step
constructs a fresh (query_cell, context_cells) pair via ContextSampler and
CellTableBuilder; the model's forward pass *is* the inference step.

There is no epoch concept here — the training data is not a fixed dataset being
iterated. Instead, each step draws a stochastically sampled ICL pair (query cell
+ context), so the loop is simply a flat step budget (`n_steps`). Callbacks for
periodic evaluation or checkpointing fire every `eval_every` steps.

"Training" here means gradient-descent fine-tuning:
  • The dual-head stack (PseudotimeHead, CompositionHead) is always trainable.
  • col_embedder (gene×gene attention) is frozen for `warmup_col_steps` steps;
    its random init would perturb the pretrained backbone before the embeddings
    stabilise.
  • tf_icl (pretrained ICL transformer) is frozen for `warmup_icl_steps` steps.
  • row_interactor and anchor_label_embedder are unfrozen from step 0.

Day 11 cells are the held-out test set. They are excluded from the fine-tuning query pool;
ContextSampler already excludes them from the anchor pool.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

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
    n_steps : int
        Total number of gradient update steps.  Each step draws a fresh ICL
        sample (query cell + stochastic context).  There is no epoch concept —
        the training data is not a fixed dataset.
    eval_every : int
        Callbacks are invoked every ``eval_every`` steps for periodic
        evaluation or checkpointing.
    callbacks : list
        Objects with an ``on_epoch_end(model, dataset, step)`` method.
    """

    def __init__(
        self,
        model: "TabICLRegressor",
        dataset: "ProcessedDataset",
        sampler: "ContextSampler",
        builder: "CellTableBuilder",
        loss_fn: "DualHeadLoss",
        config: "ExperimentConfig",
        n_steps: int = 5000,
        eval_every: int = 100,
        callbacks: list | None = None,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.sampler = sampler
        self.builder = builder
        self.loss_fn = loss_fn
        self.config = config
        self.n_steps = n_steps
        self.eval_every = eval_every
        self.callbacks: list = callbacks if callbacks is not None else []
        self.global_step: int = 0
        self.seed = seed

        # Pre-filter query pool: exclude day 11 (held-out test set)
        self._query_pool: list[str] = [
            cid
            for cid, day in zip(dataset.cell_ids, dataset.collection_day)
            if int(day) != _WITHHELD_DAY
        ]

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

        Each step draws a fresh (query_cell, context_cells) ICL pair —
        there is no epoch boundary.  Callbacks fire every ``eval_every`` steps.

        Returns
        -------
        dict with keys:
            "train_loss"  — Kendall uncertainty-weighted total loss
            "pt_loss"     — raw MSE component
            "comp_loss"   — raw KL divergence component
        """
        optimizer = self._make_optimizer()
        device = next(self.model.parameters()).device
        rng = np.random.default_rng(self.seed)

        sum_loss = sum_pt = sum_comp = 0.0

        for _ in range(self.n_steps):
            self._apply_warmup_freeze(self.global_step)

            # Draw a fresh ICL sample — stochastic context, no fixed dataset
            query_id = str(rng.choice(self._query_pool))
            anchor_ids, _ = self.sampler.sample(query_id)
            table, targets = self.builder.build(query_id, anchor_ids)
            batch = icl_collate([(table, targets)])
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

            if self.global_step % self.eval_every == 0:
                for cb in self.callbacks:
                    cb.on_epoch_end(self.model, self.dataset, self.global_step)

        n = max(self.n_steps, 1)
        return {
            "train_loss": sum_loss / n,
            "pt_loss": sum_pt / n,
            "comp_loss": sum_comp / n,
        }
