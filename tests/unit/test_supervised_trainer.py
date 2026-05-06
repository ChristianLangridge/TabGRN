"""
Unit tests for spatialmt.training.trainer.SupervisedTrainer.

SupervisedTrainer fine-tunes col_embedder, row_interactor, and the output
heads via standard batched supervised learning on all non-day-11 cells.
tf_icl is permanently frozen at construction and excluded from the optimizer.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import _make_valid_dataset

# ---------------------------------------------------------------------------
# Toy model constants (small enough for CPU tests)
# ---------------------------------------------------------------------------

N_GENES   = 10
EMBED_DIM = 32
N_HEADS   = 2
N_LAYERS  = 1
NUM_CLS   = 2
COL_NUM_INDS = 4
K         = 8


def _make_model(composition_loss_type: str = "kl"):
    from spatialmt.model.tabgrn import TabICLRegressor
    return TabICLRegressor(
        n_genes=N_GENES, k=K,
        embed_dim=EMBED_DIM, n_heads=N_HEADS, num_cls=NUM_CLS,
        col_num_inds=COL_NUM_INDS,
        n_layers_col=N_LAYERS, n_layers_row=N_LAYERS, n_layers_icl=N_LAYERS,
        composition_loss_type=composition_loss_type,
    )


def _make_loss(composition_loss_type: str = "kl"):
    from spatialmt.model.loss import DirichletDualHeadLoss, DualHeadLoss
    return DirichletDualHeadLoss() if composition_loss_type == "dirichlet" else DualHeadLoss()


def _make_config(supervised_batch_size: int = 8):
    from spatialmt.config.experiment import ExperimentConfig
    cfg = ExperimentConfig.debug_preset()
    cfg.model.supervised_batch_size = supervised_batch_size
    return cfg


def _make_dataset():
    # 60 cells, days [5,7,11,16,21,30] × 10 — day-11 cells = 10 of 60
    return _make_valid_dataset(n_cells=60, n_genes=N_GENES, k=K)


# ---------------------------------------------------------------------------
# Freeze / optimizer contract
# ---------------------------------------------------------------------------

def test_supervised_trainer_freezes_tf_icl_at_init():
    from spatialmt.training.trainer import SupervisedTrainer
    model   = _make_model()
    dataset = _make_dataset()
    loss_fn = _make_loss()
    config  = _make_config()
    SupervisedTrainer(model=model, dataset=dataset, loss_fn=loss_fn, config=config,
                      n_epochs=1, eval_every=100, callbacks=None, seed=0)
    assert all(not p.requires_grad for p in model.tf_icl.parameters())


def test_supervised_trainer_icl_excluded_from_optimizer():
    """The 'icl' parameter group must not appear in the MuonAdamW instance."""
    from spatialmt.training.trainer import SupervisedTrainer
    model   = _make_model()
    dataset = _make_dataset()
    loss_fn = _make_loss()
    config  = _make_config()
    trainer = SupervisedTrainer(model=model, dataset=dataset, loss_fn=loss_fn,
                                config=config, n_epochs=1, eval_every=100,
                                callbacks=None, seed=0)
    trainer.fit()
    group_names = {g["name"] for g in trainer.optimizer.param_groups}
    assert "icl" not in group_names


def test_supervised_trainer_non_day11_only():
    """_train_indices must not contain any day-11 cell."""
    from spatialmt.training.trainer import SupervisedTrainer
    dataset = _make_dataset()
    trainer = SupervisedTrainer(
        model=_make_model(), dataset=dataset, loss_fn=_make_loss(),
        config=_make_config(), n_epochs=1, eval_every=100, callbacks=None, seed=0,
    )
    for idx in trainer._train_indices:
        assert int(dataset.collection_day[idx]) != 11


# ---------------------------------------------------------------------------
# fit() return contract
# ---------------------------------------------------------------------------

def test_supervised_trainer_fit_returns_required_keys():
    from spatialmt.training.trainer import SupervisedTrainer
    trainer = SupervisedTrainer(
        model=_make_model(), dataset=_make_dataset(), loss_fn=_make_loss(),
        config=_make_config(), n_epochs=1, eval_every=100, callbacks=None, seed=0,
    )
    metrics = trainer.fit()
    assert {"train_loss", "pt_loss", "comp_loss", "loss_history"} <= set(metrics)


def test_supervised_trainer_fit_loss_is_finite():
    from spatialmt.training.trainer import SupervisedTrainer
    trainer = SupervisedTrainer(
        model=_make_model(), dataset=_make_dataset(), loss_fn=_make_loss(),
        config=_make_config(), n_epochs=1, eval_every=100, callbacks=None, seed=0,
    )
    metrics = trainer.fit()
    assert np.isfinite(metrics["train_loss"])


def test_supervised_trainer_global_step_increments():
    from spatialmt.training.trainer import SupervisedTrainer
    dataset = _make_dataset()
    config  = _make_config(supervised_batch_size=8)
    trainer = SupervisedTrainer(
        model=_make_model(), dataset=dataset, loss_fn=_make_loss(),
        config=config, n_epochs=2, eval_every=100, callbacks=None, seed=0,
    )
    assert trainer.global_step == 0
    trainer.fit()
    n_train = len(trainer._train_indices)
    batch_size = config.model.supervised_batch_size
    expected_steps = 2 * int(np.ceil(n_train / batch_size))
    assert trainer.global_step == expected_steps


def test_supervised_trainer_dirichlet_variant():
    """SupervisedTrainer works with DirichletDualHeadLoss."""
    from spatialmt.training.trainer import SupervisedTrainer
    trainer = SupervisedTrainer(
        model=_make_model("dirichlet"), dataset=_make_dataset(),
        loss_fn=_make_loss("dirichlet"), config=_make_config(),
        n_epochs=1, eval_every=100, callbacks=None, seed=0,
    )
    metrics = trainer.fit()
    assert np.isfinite(metrics["train_loss"])


def test_supervised_trainer_tf_icl_weights_unchanged():
    """tf_icl weights must be identical before and after fit()."""
    from spatialmt.training.trainer import SupervisedTrainer
    model = _make_model()
    before = {k: v.clone() for k, v in model.tf_icl.state_dict().items()}
    trainer = SupervisedTrainer(
        model=model, dataset=_make_dataset(), loss_fn=_make_loss(),
        config=_make_config(), n_epochs=1, eval_every=100, callbacks=None, seed=0,
    )
    trainer.fit()
    after = model.tf_icl.state_dict()
    for k in before:
        assert torch.allclose(before[k], after[k]), f"tf_icl weight '{k}' changed"
