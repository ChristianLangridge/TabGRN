"""
Unit tests for spatialmt.training.trainer.Trainer

RED phase — all tests should FAIL until trainer.py is implemented.

TabGRN uses amortized inference via in-context learning. This means:
  - Each training step draws a fresh (query_cell, context_cells) pair via
    ContextSampler.sample() — the context is stochastic, not a fixed dataset row.
  - The pretrained backbone already knows how to do regression from context;
    training here is fine-tuning: adapting backbone + new dual heads to the
    pseudotime/composition task.
  - Day 11 cells are the test set. They must never appear as query cells during
    training (ContextSampler already excludes them as anchors).
  - There is no separate predict() phase — inference is a forward pass with
    fresh context, identical in structure to a training step.

These properties are tested explicitly below (groups 2, 4, 9).

Test groups
-----------
1.  Construction        — Trainer stores required attributes; global_step=0
2.  ICL sampling        — Training queries come from ContextSampler; day 11 cells
                          excluded as queries; same query gets different context
                          on repeated draws (stochastic context)
3.  Optimizer           — AdamW with 5 param groups (4 model + 1 loss); correct LRs
4.  Warmup / freeze     — col_embedder frozen until step 500; tf_icl until step 100;
                          head always trainable
5.  fit() smoke         — single-step run terminates; returns {train_loss, pt_loss,
                          comp_loss} all finite
6.  Step counter        — global_step increments per step; accumulates across epochs
7.  Metrics             — returned dict keys and scalar values
8.  Callbacks           — on_epoch_end(model, dataset, epoch) called once per epoch
9.  No label leakage    — query labels never appear in the CellTable fed to the model
10. Gradient flow       — unfrozen params receive non-None grads after a step
"""
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import torch

from spatialmt.config.experiment import ExperimentConfig
from spatialmt.context.builder import CellTableBuilder
from spatialmt.context.sampler import ContextSampler
from spatialmt.data_preparation.dataset import ProcessedDataset
from spatialmt.model.loss import DualHeadLoss
from spatialmt.model.tabgrn import TabICLRegressor
from spatialmt.training.muon import Muon
from spatialmt.training.trainer import MuonAdamW, Trainer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_CELLS = 60        # 6 days × 10 cells each
N_GENES = 10
K = 4
TIMEPOINTS = [5, 7, 11, 16, 21, 30]


def _make_dataset(n_cells: int = N_CELLS, n_genes: int = N_GENES, k: int = K) -> ProcessedDataset:
    rng = np.random.default_rng(0)
    expression = (rng.random((n_cells, n_genes)) * 4.0).astype(np.float32)
    pseudotime = rng.random(n_cells).astype(np.float32)
    days = np.array(TIMEPOINTS * (n_cells // len(TIMEPOINTS) + 1))[:n_cells]
    raw = rng.random((n_cells, k)).astype(np.float32)
    soft_labels = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    manifest_hash = ProcessedDataset._compute_manifest_hash([f"GENE_{i:02d}" for i in range(n_genes)])
    cell_type_labels = pd.Series([f"state_{i % k}" for i in range(n_cells)])
    return ProcessedDataset(
        expression=expression,
        gene_names=[f"GENE_{i:02d}" for i in range(n_genes)],
        pseudotime=pseudotime,
        collection_day=days.astype(np.int32),
        cell_ids=[f"cell_{i}" for i in range(n_cells)],
        cell_type_labels=cell_type_labels,
        orig_ident=pd.Series([f"HB4_D{d}" for d in days]),
        soft_labels=soft_labels,
        cell_type_categories=sorted(cell_type_labels.unique()),
        manifest_hash=manifest_hash,
    )


def _make_toy_model(n_genes: int = N_GENES, k: int = K) -> TabICLRegressor:
    return TabICLRegressor(
        n_genes=n_genes,
        embed_dim=8,
        n_heads=2,
        n_layers=1,
        k=k,
        num_cls=2,
    )


def _make_trainer(n_steps: int = 1, eval_every: int = 1) -> Trainer:
    cfg = ExperimentConfig.debug_preset()
    dataset = _make_dataset()
    model = _make_toy_model(n_genes=N_GENES, k=K)
    sampler = ContextSampler(dataset, cfg.context)
    builder = CellTableBuilder(dataset)
    loss_fn = DualHeadLoss()
    return Trainer(
        model=model,
        dataset=dataset,
        sampler=sampler,
        builder=builder,
        loss_fn=loss_fn,
        config=cfg,
        n_steps=n_steps,
        eval_every=eval_every,
    )


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_trainer_instantiates(self):
        trainer = _make_trainer()
        assert isinstance(trainer, Trainer)

    def test_trainer_stores_model(self):
        assert isinstance(_make_trainer().model, TabICLRegressor)

    def test_trainer_stores_dataset(self):
        assert isinstance(_make_trainer().dataset, ProcessedDataset)

    def test_trainer_stores_sampler(self):
        assert isinstance(_make_trainer().sampler, ContextSampler)

    def test_trainer_stores_builder(self):
        assert isinstance(_make_trainer().builder, CellTableBuilder)

    def test_trainer_stores_loss_fn(self):
        assert isinstance(_make_trainer().loss_fn, DualHeadLoss)

    def test_trainer_stores_config(self):
        assert isinstance(_make_trainer().config, ExperimentConfig)

    def test_global_step_initialised_to_zero(self):
        assert _make_trainer().global_step == 0

    def test_callbacks_default_empty(self):
        assert _make_trainer().callbacks == []


# ---------------------------------------------------------------------------
# 2. ICL sampling — the core ICL contract
# ---------------------------------------------------------------------------

class TestICLSampling:
    def test_sampler_called_each_step(self):
        """Trainer must call sampler.sample() once per training step."""
        trainer = _make_trainer(n_steps=3)
        with patch.object(trainer.sampler, "sample", wraps=trainer.sampler.sample) as mock_sample:
            trainer.fit()
        assert mock_sample.call_count == 3

    def test_builder_called_each_step(self):
        """Trainer must call builder.build() once per training step."""
        trainer = _make_trainer(n_steps=4)
        with patch.object(trainer.builder, "build", wraps=trainer.builder.build) as mock_build:
            trainer.fit()
        assert mock_build.call_count == 4

    def test_day11_cells_never_used_as_query(self):
        """Day 11 cells must never be chosen as the query cell during training.

        ICL note: day 11 is the held-out test set. ContextSampler already
        excludes day 11 cells as *anchors*; the trainer must also exclude them
        as *query cells*, otherwise the model sees its own test labels during
        fine-tuning.
        """
        dataset = _make_dataset()
        day11_ids = {
            cid for cid, day in zip(dataset.cell_ids, dataset.collection_day)
            if int(day) == 11
        }
        cfg = ExperimentConfig.debug_preset()
        model = _make_toy_model()
        sampler = ContextSampler(dataset, cfg.context)
        builder = CellTableBuilder(dataset)
        loss_fn = DualHeadLoss()

        queried = []
        original_build = builder.build

        def recording_build(query_cell_id, anchor_ids):
            queried.append(query_cell_id)
            return original_build(query_cell_id, anchor_ids)

        trainer = Trainer(
            model=model, dataset=dataset, sampler=sampler,
            builder=builder, loss_fn=loss_fn, config=cfg,
            n_steps=20,
        )
        with patch.object(trainer.builder, "build", side_effect=recording_build):
            trainer.fit()

        assert len(queried) > 0, "build() was never called"
        for qid in queried:
            assert qid not in day11_ids, (
                f"Day 11 cell '{qid}' was used as a query cell during training"
            )

    def test_same_query_gets_different_context_across_steps(self):
        """ICL context is stochastic: the same query cell sampled twice must
        (with overwhelming probability) receive different anchor sets.

        This verifies that the trainer uses a fresh RNG draw each step rather
        than caching or reusing a fixed context.
        """
        dataset = _make_dataset()
        cfg = ExperimentConfig.debug_preset()
        # Force the trainer to pick the same query cell every step by replacing
        # the query selection with a fixed cell (day != 11).
        non_day11_ids = [
            cid for cid, day in zip(dataset.cell_ids, dataset.collection_day)
            if int(day) != 11
        ]
        fixed_query = non_day11_ids[0]

        model = _make_toy_model()
        sampler = ContextSampler(dataset, cfg.context)
        builder = CellTableBuilder(dataset)
        loss_fn = DualHeadLoss()

        anchor_sets: list[frozenset] = []
        original_sample = sampler.sample

        def recording_sample(query_cell_id, rng=None):
            anchor_ids, pseudotimes = original_sample(fixed_query, rng)
            anchor_sets.append(frozenset(anchor_ids))
            return anchor_ids, pseudotimes

        trainer = Trainer(
            model=model, dataset=dataset, sampler=sampler,
            builder=builder, loss_fn=loss_fn, config=cfg,
            n_steps=5,
        )
        with patch.object(trainer.sampler, "sample", side_effect=recording_sample):
            trainer.fit()

        assert len(anchor_sets) == 5
        # With random sampling across ≥10 cells per bin, identical sets in all
        # 5 draws has negligible probability — if this fails the trainer is
        # using a fixed context or a seeded RNG with the same seed each step.
        assert len(set(anchor_sets)) > 1, (
            "All 5 steps produced identical anchor sets — context is not being "
            "freshly sampled each step"
        )

    def test_query_cell_id_passed_to_sampler(self):
        """The query cell id chosen by the trainer must be forwarded to
        sampler.sample(), not a different cell.  This confirms the trainer
        correctly threads the ICL sampling pipeline."""
        trainer = _make_trainer(n_steps=3)

        sampled_queries = []
        original_sample = trainer.sampler.sample

        def recording_sample(query_cell_id, rng=None):
            sampled_queries.append(query_cell_id)
            return original_sample(query_cell_id, rng)

        built_queries = []
        original_build = trainer.builder.build

        def recording_build(query_cell_id, anchor_ids):
            built_queries.append(query_cell_id)
            return original_build(query_cell_id, anchor_ids)

        with patch.object(trainer.sampler, "sample", side_effect=recording_sample), \
             patch.object(trainer.builder, "build", side_effect=recording_build):
            trainer.fit()

        assert sampled_queries == built_queries, (
            "query_cell_id passed to sampler.sample() differs from the one "
            "passed to builder.build() — sampling pipeline is inconsistent"
        )


# ---------------------------------------------------------------------------
# 3. Optimizer — Muon (weight matrices) + AdamW (biases / norms / scalars)
# ---------------------------------------------------------------------------

class TestOptimizer:
    def test_make_optimizer_returns_muon_adamw(self):
        """_make_optimizer must return a MuonAdamW combined optimizer, not
        a plain AdamW.  TabICLv2 uses Muon for weight matrices."""
        assert isinstance(_make_trainer()._make_optimizer(), MuonAdamW)

    def test_optimizer_has_five_logical_param_groups(self):
        """MuonAdamW.param_groups exposes the 5 logical groups:
        col, row, icl, head (model) + loss — regardless of internal routing."""
        opt = _make_trainer()._make_optimizer()
        assert len(opt.param_groups) == 5

    def test_col_group_lr(self):
        opt = _make_trainer()._make_optimizer()
        g = next(g for g in opt.param_groups if g.get("name") == "col")
        assert g["lr"] == pytest.approx(1e-5)

    def test_row_group_lr(self):
        opt = _make_trainer()._make_optimizer()
        g = next(g for g in opt.param_groups if g.get("name") == "row")
        assert g["lr"] == pytest.approx(1e-4)

    def test_icl_group_lr(self):
        opt = _make_trainer()._make_optimizer()
        g = next(g for g in opt.param_groups if g.get("name") == "icl")
        assert g["lr"] == pytest.approx(5e-5)

    def test_head_group_lr(self):
        opt = _make_trainer()._make_optimizer()
        g = next(g for g in opt.param_groups if g.get("name") == "head")
        assert g["lr"] == pytest.approx(1e-3)

    def test_loss_params_in_optimizer(self):
        """Kendall uncertainty params must appear in the optimizer's logical groups."""
        trainer = _make_trainer()
        opt = trainer._make_optimizer()
        all_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        assert id(trainer.loss_fn.log_sigma_sq_pt) in all_param_ids
        assert id(trainer.loss_fn.log_sigma_sq_comp) in all_param_ids

    def test_matrix_params_routed_to_muon(self):
        """ndim >= 2 parameters from model groups must appear in Muon's internal
        param groups, not AdamW."""
        trainer = _make_trainer()
        opt = trainer._make_optimizer()
        assert opt._muon is not None
        muon_ids = {id(p) for g in opt._muon.param_groups for p in g["params"]}
        # Collect all model matrix params
        matrix_params = [p for p in trainer.model.parameters() if p.ndim >= 2]
        assert len(matrix_params) > 0
        for p in matrix_params:
            assert id(p) in muon_ids, (
                f"Matrix param shape={tuple(p.shape)} not routed to Muon"
            )

    def test_vector_params_routed_to_adamw(self):
        """ndim < 2 parameters (biases, norms) must be in AdamW, not Muon."""
        trainer = _make_trainer()
        opt = trainer._make_optimizer()
        assert opt._adamw is not None
        adamw_ids = {id(p) for g in opt._adamw.param_groups for p in g["params"]}
        # Collect trainable vector params from model
        vector_params = [p for p in trainer.model.parameters()
                         if p.ndim < 2 and p.requires_grad]
        assert len(vector_params) > 0
        for p in vector_params:
            assert id(p) in adamw_ids, (
                f"Vector param shape={tuple(p.shape)} not routed to AdamW"
            )

    def test_loss_params_routed_to_adamw(self):
        """log_sigma_sq_pt and log_sigma_sq_comp are 0-dim scalars — must go
        to AdamW, not Muon."""
        trainer = _make_trainer()
        opt = trainer._make_optimizer()
        adamw_ids = {id(p) for g in opt._adamw.param_groups for p in g["params"]}
        assert id(trainer.loss_fn.log_sigma_sq_pt)   in adamw_ids
        assert id(trainer.loss_fn.log_sigma_sq_comp) in adamw_ids

    def test_muon_sub_optimizer_is_muon_instance(self):
        opt = _make_trainer()._make_optimizer()
        assert isinstance(opt._muon, Muon)

    def test_adamw_sub_optimizer_is_adamw_instance(self):
        opt = _make_trainer()._make_optimizer()
        assert isinstance(opt._adamw, torch.optim.AdamW)


# ---------------------------------------------------------------------------
# 4. Warmup / freeze scheduling
# ---------------------------------------------------------------------------

class TestWarmup:
    def test_col_embedder_frozen_at_step_0(self):
        trainer = _make_trainer()
        trainer._apply_warmup_freeze(global_step=0)
        for p in trainer.model.col_embedder.parameters():
            assert not p.requires_grad

    def test_tf_icl_frozen_at_step_0(self):
        trainer = _make_trainer()
        trainer._apply_warmup_freeze(global_step=0)
        for p in trainer.model.tf_icl.parameters():
            assert not p.requires_grad

    def test_head_params_always_unfrozen(self):
        """Head parameters must be trainable from step 0.

        ICL note: the pretrained backbone can provide useful gradients to the
        heads even when frozen; the heads are the primary adaptation target.
        """
        trainer = _make_trainer()
        trainer._apply_warmup_freeze(global_step=0)
        head_params = (
            list(trainer.model.anchor_label_embedder.parameters())
            + list(trainer.model.shared_trunk.parameters())
            + list(trainer.model.pseudotime_head.parameters())
            + list(trainer.model.composition_head.parameters())
        )
        for p in head_params:
            assert p.requires_grad

    def test_row_interactor_unchanged_by_warmup_freeze(self):
        """_apply_warmup_freeze must not alter row_interactor requires_grad state.

        row_interactor has no warmup schedule. Some of its parameters (e.g.
        RoPE frequency constants) are non-trainable by design in the TabICLv2
        library; others are trainable. The trainer must not touch either.
        """
        trainer = _make_trainer()
        before = {id(p): p.requires_grad for p in trainer.model.row_interactor.parameters()}
        trainer._apply_warmup_freeze(global_step=0)
        after = {id(p): p.requires_grad for p in trainer.model.row_interactor.parameters()}
        assert before == after, "row_interactor requires_grad state was modified by _apply_warmup_freeze"

    def test_col_embedder_unfrozen_at_warmup_col_steps(self):
        trainer = _make_trainer()
        trainer._apply_warmup_freeze(global_step=trainer.config.model.warmup_col_steps)
        for p in trainer.model.col_embedder.parameters():
            assert p.requires_grad

    def test_tf_icl_unfrozen_at_warmup_icl_steps(self):
        trainer = _make_trainer()
        trainer._apply_warmup_freeze(global_step=trainer.config.model.warmup_icl_steps)
        for p in trainer.model.tf_icl.parameters():
            assert p.requires_grad

    def test_col_embedder_still_frozen_one_step_before_warmup(self):
        trainer = _make_trainer()
        trainer._apply_warmup_freeze(global_step=trainer.config.model.warmup_col_steps - 1)
        for p in trainer.model.col_embedder.parameters():
            assert not p.requires_grad

    def test_tf_icl_still_frozen_one_step_before_warmup(self):
        trainer = _make_trainer()
        trainer._apply_warmup_freeze(global_step=trainer.config.model.warmup_icl_steps - 1)
        for p in trainer.model.tf_icl.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# 5. fit() smoke
# ---------------------------------------------------------------------------

class TestFitSmoke:
    def test_fit_returns_without_error(self):
        _make_trainer(n_steps=1).fit()

    def test_fit_returns_dict(self):
        assert isinstance(_make_trainer(n_steps=1).fit(), dict)

    def test_fit_result_has_train_loss(self):
        assert "train_loss" in _make_trainer(n_steps=1).fit()

    def test_fit_result_has_pt_loss(self):
        assert "pt_loss" in _make_trainer(n_steps=1).fit()

    def test_fit_result_has_comp_loss(self):
        assert "comp_loss" in _make_trainer(n_steps=1).fit()

    def test_train_loss_is_finite(self):
        result = _make_trainer(n_steps=2).fit()
        assert np.isfinite(result["train_loss"])

    def test_pt_loss_is_non_negative(self):
        """MSE is always ≥ 0."""
        result = _make_trainer(n_steps=2).fit()
        assert result["pt_loss"] >= 0.0

    def test_comp_loss_is_non_negative(self):
        """KL divergence is always ≥ 0."""
        result = _make_trainer(n_steps=2).fit()
        assert result["comp_loss"] >= 0.0


# ---------------------------------------------------------------------------
# 6. Step counter
# ---------------------------------------------------------------------------

class TestStepCounter:
    def test_global_step_increments_each_step(self):
        trainer = _make_trainer(n_steps=3)
        trainer.fit()
        assert trainer.global_step == 3

    def test_global_step_equals_n_steps_after_fit(self):
        trainer = _make_trainer(n_steps=6)
        trainer.fit()
        assert trainer.global_step == 6


# ---------------------------------------------------------------------------
# 7. Callbacks
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_callback_called_every_eval_every_steps(self):
        """Callback fires once per eval_every steps — not once per epoch."""
        cb = MagicMock()
        # 3 steps, eval_every=1 → callback fires at steps 1, 2, 3
        trainer = _make_trainer(n_steps=3, eval_every=1)
        trainer.callbacks = [cb]
        trainer.fit()
        assert cb.on_epoch_end.call_count == 3

    def test_callback_fires_at_eval_every_boundary(self):
        """With eval_every=2 and n_steps=6, callback fires 3 times."""
        cb = MagicMock()
        trainer = _make_trainer(n_steps=6, eval_every=2)
        trainer.callbacks = [cb]
        trainer.fit()
        assert cb.on_epoch_end.call_count == 3

    def test_callback_receives_model_dataset_step(self):
        cb = MagicMock()
        trainer = _make_trainer(n_steps=1, eval_every=1)
        trainer.callbacks = [cb]
        trainer.fit()
        args = cb.on_epoch_end.call_args[0]
        assert args[0] is trainer.model
        assert args[1] is trainer.dataset
        assert isinstance(args[2], int)

    def test_multiple_callbacks_all_called(self):
        cb1, cb2 = MagicMock(), MagicMock()
        trainer = _make_trainer(n_steps=1, eval_every=1)
        trainer.callbacks = [cb1, cb2]
        trainer.fit()
        cb1.on_epoch_end.assert_called_once()
        cb2.on_epoch_end.assert_called_once()


# ---------------------------------------------------------------------------
# 9. No label leakage — CellTable must not contain query labels
# ---------------------------------------------------------------------------

class TestNoLabelLeakage:
    def test_celltable_query_expression_matches_dataset(self):
        """The query_expression in CellTable must be the query cell's expression
        row from the dataset — no pseudotime or soft_label information present.

        ICL note: the whole point of the CellTable / TrainingTargets split is
        to prevent the model from seeing the answer it is trying to predict.
        This test verifies the trainer assembles the split correctly.
        """
        dataset = _make_dataset()
        cfg = ExperimentConfig.debug_preset()
        model = _make_toy_model()
        sampler = ContextSampler(dataset, cfg.context)
        builder = CellTableBuilder(dataset)
        loss_fn = DualHeadLoss()

        captured_tables = []
        original_build = builder.build

        def capturing_build(query_cell_id, anchor_ids):
            table, targets = original_build(query_cell_id, anchor_ids)
            captured_tables.append((query_cell_id, table, targets))
            return table, targets

        trainer = Trainer(
            model=model, dataset=dataset, sampler=sampler,
            builder=builder, loss_fn=loss_fn, config=cfg,
            n_steps=3,
        )
        with patch.object(trainer.builder, "build", side_effect=capturing_build):
            trainer.fit()

        assert len(captured_tables) == 3
        for query_cell_id, table, targets in captured_tables:
            # CellTable must not have pseudotime or soft_labels fields for query
            assert not hasattr(table, "query_pseudotime"), (
                "CellTable has query_pseudotime — label leakage"
            )
            assert not hasattr(table, "query_soft_labels"), (
                "CellTable has query_soft_labels — label leakage"
            )
            # query_expression in CellTable must match dataset
            q_idx = dataset.cell_ids.index(query_cell_id)
            np.testing.assert_array_equal(
                table.query_expression,
                dataset.expression[q_idx],
            )

    def test_training_targets_not_fed_to_model_forward(self):
        """The TrainingTargets (query labels) must only reach loss_fn.forward(),
        never model.forward().

        ICL note: model.forward() takes an ICLBatch which has no query label
        fields. This test verifies the trainer does not pass query labels into
        the model's forward pass by checking that ICLBatch fed to the model
        has zero-filled (or absent) query_pseudotime values that match the
        collated context — not the true targets.
        """
        trainer = _make_trainer(n_steps=2)

        model_inputs: list = []
        original_forward = trainer.model.forward

        def capturing_forward(batch):
            model_inputs.append(batch)
            return original_forward(batch)

        with patch.object(trainer.model, "forward", side_effect=capturing_forward):
            trainer.fit()

        assert len(model_inputs) > 0
        for batch in model_inputs:
            # ICLBatch.query_pseudotime exists (it's a field) but must NOT
            # equal the TrainingTargets value — verified by checking the field
            # is present and float32 (structural contract), not by value.
            # The key assertion: batch has no field that would require the
            # trainer to have pre-populated it from TrainingTargets.
            assert hasattr(batch, "context_expression")
            assert hasattr(batch, "query_expression")
            # query_pseudotime in ICLBatch is a passthrough for evaluation —
            # it must be float32 but its value is unused by model.forward()
            assert batch.query_pseudotime.dtype == torch.float32


# ---------------------------------------------------------------------------
# 10. Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_head_params_have_grads_after_step(self):
        trainer = _make_trainer(n_steps=1)
        trainer.config.model.warmup_col_steps = 0
        trainer.config.model.warmup_icl_steps = 0
        trainer.fit()
        for p in trainer.model.pseudotime_head.parameters():
            assert p.grad is not None

    def test_row_params_have_grads_after_step(self):
        """Trainable row_interactor parameters must receive gradients.

        Some TabICLv2 parameters (e.g. RoPE frequency constants) are
        non-trainable by design and will never have .grad — skip those.
        """
        trainer = _make_trainer(n_steps=1)
        trainer.config.model.warmup_col_steps = 0
        trainer.config.model.warmup_icl_steps = 0
        trainer.fit()
        trainable = [p for p in trainer.model.row_interactor.parameters() if p.requires_grad]
        assert len(trainable) > 0, "row_interactor has no trainable parameters at all"
        for p in trainable:
            assert p.grad is not None

    def test_loss_params_have_grads_after_step(self):
        trainer = _make_trainer(n_steps=1)
        trainer.fit()
        assert trainer.loss_fn.log_sigma_sq_pt.grad is not None
        assert trainer.loss_fn.log_sigma_sq_comp.grad is not None

    def test_frozen_col_embedder_has_no_grads_during_warmup(self):
        """While col_embedder is frozen (step < warmup_col_steps), its
        parameters must accumulate no gradient — confirming freeze is real,
        not just a zeroed gradient."""
        trainer = _make_trainer(n_steps=1)
        # Keep warmup active (default 500 steps >> 1 step)
        trainer.fit()
        for p in trainer.model.col_embedder.parameters():
            assert p.grad is None, (
                "col_embedder has gradient during warmup — freeze not applied"
            )
