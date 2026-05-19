"""spatialmt.training.trainer — ICL fine-tuning (Trainer) and supervised fine-tuning (SupervisedTrainer)."""
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
    from spatialmt.model.trace import TabICLRegressor

_WITHHELD_DAY: int = 11


class MuonAdamW:
    """Combined Muon + AdamW: ndim >= 2 → Muon, ndim < 2 → AdamW."""

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
        self._adamw = torch.optim.AdamW(adamw_groups, weight_decay=0.0) if adamw_groups else None

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

    def state_dict(self) -> dict:
        return {
            "muon":  self._muon.state_dict()  if self._muon  is not None else None,
            "adamw": self._adamw.state_dict() if self._adamw is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self._muon  is not None and state["muon"]  is not None:
            self._muon.load_state_dict(state["muon"])
        if self._adamw is not None and state["adamw"] is not None:
            self._adamw.load_state_dict(state["adamw"])


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
    """ICL fine-tuning loop. Each step draws a fresh stochastic (query, context) pair.

    Day-11 cells are excluded from the query pool — they are the held-out test set.
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
        model_groups = self.model.parameter_groups(self.config.model)
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
        """Run fine-tuning. Returns dict: train_loss, pt_loss, comp_loss, loss_history."""
        self.optimizer = self._make_optimizer()
        optimizer = self.optimizer
        device = next(self.model.parameters()).device
        rng = np.random.default_rng(self.seed)

        sum_loss = sum_pt = sum_comp = 0.0
        interval_loss = interval_pt = interval_comp = 0.0
        loss_history: list[dict] = []

        for _ in range(self.n_steps):
            self._apply_warmup_freeze(self.global_step)

            query_id = str(rng.choice(self._query_pool))
            anchor_ids, _ = self.sampler.sample(query_id)
            table, targets = self.builder.build(query_id, anchor_ids)
            batch = icl_collate([(table, targets)])
            batch = _batch_to_device(batch, device)

            optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
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
            tl, pl, cl = total_loss.item(), pt_loss.item(), comp_loss.item()
            sum_loss += tl;  interval_loss += tl
            sum_pt   += pl;  interval_pt   += pl
            sum_comp += cl;  interval_comp += cl

            if self.global_step % self.eval_every == 0:
                loss_history.append({
                    "step":       self.global_step,
                    "train_loss": interval_loss / self.eval_every,
                    "pt_loss":    interval_pt   / self.eval_every,
                    "comp_loss":  interval_comp / self.eval_every,
                })
                interval_loss = interval_pt = interval_comp = 0.0
                for cb in self.callbacks:
                    cb.on_epoch_end(self.model, self.dataset, self.global_step)

        n = max(self.n_steps, 1)
        return {
            "train_loss":   sum_loss / n,
            "pt_loss":      sum_pt   / n,
            "comp_loss":    sum_comp / n,
            "loss_history": loss_history,
        }


# ---------------------------------------------------------------------------
# SupervisedTrainer
# ---------------------------------------------------------------------------

class SupervisedTrainer:
    """Batched supervised fine-tuning on all non-day-11 cells.

    tf_icl is permanently frozen and excluded from the optimizer.
    """

    def __init__(
        self,
        model: "TabICLRegressor",
        dataset: "ProcessedDataset",
        loss_fn: "DualHeadLoss",
        config: "ExperimentConfig",
        n_epochs: int = 3,
        eval_every: int = 100,
        callbacks: list | None = None,
        seed: int | None = None,
    ) -> None:
        self.model      = model
        self.dataset    = dataset
        self.loss_fn    = loss_fn
        self.config     = config
        self.n_epochs   = n_epochs
        self.eval_every = eval_every
        self.callbacks  = callbacks if callbacks is not None else []
        self.global_step: int = 0
        self.seed       = seed

        # tf_icl permanently frozen — excluded from optimizer and never updated
        for p in model.tf_icl.parameters():
            p.requires_grad_(False)

        self._train_indices: np.ndarray = np.array([
            i for i, day in enumerate(dataset.collection_day)
            if int(day) != _WITHHELD_DAY
        ])

        # Pre-compute population anchor once — mean expression of all non-day-11
        # cells. Stored as CPU float32; moved to device once at fit() start.
        self._population_anchor: torch.Tensor = torch.tensor(
            dataset.expression[self._train_indices].mean(axis=0).astype(np.float32)
        )

    def _make_optimizer(self) -> "MuonAdamW":
        model_groups = [g for g in self.model.parameter_groups(self.config.model) if g["name"] != "icl"]
        loss_group = {
            "name":   "loss",
            "params": list(self.loss_fn.parameters()),
            "lr":     self.config.model.lr_head,
        }
        return MuonAdamW(model_groups + [loss_group])

    def _apply_warmup_freeze(self, global_step: int) -> None:
        col_frozen = global_step < self.config.model.warmup_col_steps
        for p in self.model.col_embedder.parameters():
            p.requires_grad_(not col_frozen)
        # tf_icl always frozen — never re-enabled

    def fit(self) -> dict:
        """Run supervised fine-tuning. Returns dict: train_loss, pt_loss, comp_loss, loss_history."""
        self.optimizer = self._make_optimizer()
        optimizer = self.optimizer
        device = next(self.model.parameters()).device
        rng = np.random.default_rng(self.seed)
        batch_size = self.config.model.supervised_batch_size
        anchor = self._population_anchor.to(device)

        sum_loss = sum_pt = sum_comp = 0.0
        interval_loss = interval_pt = interval_comp = 0.0
        loss_history: list[dict] = []

        for _ in range(self.n_epochs):
            shuffled = rng.permutation(self._train_indices)
            for start in range(0, len(shuffled), batch_size):
                self._apply_warmup_freeze(self.global_step)
                batch_idx = shuffled[start : start + batch_size]

                expr = torch.tensor(
                    self.dataset.expression[batch_idx].astype(np.float32),
                    device=device,
                )
                pt_target = torch.tensor(
                    self.dataset.pseudotime[batch_idx].astype(np.float32),
                    device=device,
                )
                sl_target = torch.tensor(
                    self.dataset.soft_labels[batch_idx].astype(np.float32),
                    device=device,
                )

                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pt_pred, comp_pred = self.model.forward_supervised(expr, anchor)
                    total_loss, pt_loss, comp_loss = self.loss_fn(
                        pt_pred, pt_target, comp_pred, sl_target,
                    )
                total_loss.backward()
                optimizer.step()

                self.global_step += 1
                tl, pl, cl = total_loss.item(), pt_loss.item(), comp_loss.item()
                sum_loss += tl;  interval_loss += tl
                sum_pt   += pl;  interval_pt   += pl
                sum_comp += cl;  interval_comp += cl

                if self.global_step % self.eval_every == 0:
                    loss_history.append({
                        "step":       self.global_step,
                        "train_loss": interval_loss / self.eval_every,
                        "pt_loss":    interval_pt   / self.eval_every,
                        "comp_loss":  interval_comp / self.eval_every,
                    })
                    interval_loss = interval_pt = interval_comp = 0.0
                    for cb in self.callbacks:
                        cb.on_epoch_end(self.model, self.dataset, self.global_step)

        n = max(self.global_step, 1)
        return {
            "train_loss":   sum_loss / n,
            "pt_loss":      sum_pt   / n,
            "comp_loss":    sum_comp / n,
            "loss_history": loss_history,
        }
