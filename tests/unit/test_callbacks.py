"""
Unit tests for spatialmt.training.callbacks.CheckpointCallback
"""
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
from spatialmt.training.callbacks import CheckpointCallback
from spatialmt.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Shared helpers (minimal toy setup identical to test_trainer.py)
# ---------------------------------------------------------------------------

N_CELLS = 60
N_GENES = 10
K = 4
TIMEPOINTS = [5, 7, 11, 16, 21, 30]


def _make_dataset() -> ProcessedDataset:
    rng = np.random.default_rng(0)
    expression = (rng.random((N_CELLS, N_GENES)) * 4.0).astype(np.float32)
    pseudotime = rng.random(N_CELLS).astype(np.float32)
    days = np.array(TIMEPOINTS * (N_CELLS // len(TIMEPOINTS) + 1))[:N_CELLS]
    raw = rng.random((N_CELLS, K)).astype(np.float32)
    soft_labels = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    cell_type_labels = pd.Series([f"state_{i % K}" for i in range(N_CELLS)])
    return ProcessedDataset(
        expression=expression,
        gene_names=[f"GENE_{i:02d}" for i in range(N_GENES)],
        pseudotime=pseudotime,
        collection_day=days.astype(np.int32),
        cell_ids=[f"cell_{i}" for i in range(N_CELLS)],
        cell_type_labels=cell_type_labels,
        orig_ident=pd.Series([f"HB4_D{d}" for d in days]),
        soft_labels=soft_labels,
        cell_type_categories=sorted(cell_type_labels.unique()),
        manifest_hash=ProcessedDataset._compute_manifest_hash(
            [f"GENE_{i:02d}" for i in range(N_GENES)]
        ),
    )


def _make_trainer(n_steps: int = 2, eval_every: int = 1) -> Trainer:
    cfg = ExperimentConfig.debug_preset()
    dataset = _make_dataset()
    model = TabICLRegressor(
        n_genes=N_GENES, k=K, embed_dim=8, n_heads=2, num_cls=2,
        col_num_inds=4, n_layers_col=1, n_layers_row=1, n_layers_icl=1,
    )
    return Trainer(
        model=model,
        dataset=dataset,
        sampler=ContextSampler(dataset, cfg.context),
        builder=CellTableBuilder(dataset),
        loss_fn=DualHeadLoss(),
        config=cfg,
        n_steps=n_steps,
        eval_every=eval_every,
    )


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestCheckpointCallbackConstruction:
    def test_instantiates(self, tmp_path):
        trainer = _make_trainer()
        cb = CheckpointCallback(trainer, trainer.loss_fn, tmp_path, every=1)
        assert isinstance(cb, CheckpointCallback)

    def test_stores_every(self, tmp_path):
        trainer = _make_trainer()
        cb = CheckpointCallback(trainer, trainer.loss_fn, tmp_path, every=500)
        assert cb.every == 500

    def test_out_dir_stored_as_path(self, tmp_path):
        from pathlib import Path
        trainer = _make_trainer()
        cb = CheckpointCallback(trainer, trainer.loss_fn, str(tmp_path), every=1)
        assert isinstance(cb.out_dir, Path)


# ---------------------------------------------------------------------------
# 2. File creation
# ---------------------------------------------------------------------------

class TestCheckpointFileCreation:
    def test_creates_checkpoint_file(self, tmp_path):
        trainer = _make_trainer(n_steps=1, eval_every=1)
        cb = CheckpointCallback(trainer, trainer.loss_fn, tmp_path, every=1)
        trainer.callbacks = [cb]
        trainer.fit()
        files = list(tmp_path.glob("step_*.pt"))
        assert len(files) == 1

    def test_filename_contains_step(self, tmp_path):
        trainer = _make_trainer(n_steps=1, eval_every=1)
        cb = CheckpointCallback(trainer, trainer.loss_fn, tmp_path, every=1)
        trainer.callbacks = [cb]
        trainer.fit()
        assert (tmp_path / "step_000001.pt").exists()

    def test_creates_output_dir_if_absent(self, tmp_path):
        out_dir = tmp_path / "nested" / "checkpoints"
        trainer = _make_trainer(n_steps=1, eval_every=1)
        cb = CheckpointCallback(trainer, trainer.loss_fn, out_dir, every=1)
        trainer.callbacks = [cb]
        trainer.fit()
        assert out_dir.exists()

    def test_no_file_when_step_does_not_divide_evenly(self, tmp_path):
        # eval_every=1 so callback fires at step 1, but every=2 so no write
        trainer = _make_trainer(n_steps=1, eval_every=1)
        cb = CheckpointCallback(trainer, trainer.loss_fn, tmp_path, every=2)
        trainer.callbacks = [cb]
        trainer.fit()
        assert list(tmp_path.glob("step_*.pt")) == []

    def test_multiple_checkpoints_written(self, tmp_path):
        trainer = _make_trainer(n_steps=2, eval_every=1)
        cb = CheckpointCallback(trainer, trainer.loss_fn, tmp_path, every=1)
        trainer.callbacks = [cb]
        trainer.fit()
        files = sorted(tmp_path.glob("step_*.pt"))
        assert len(files) == 2


# ---------------------------------------------------------------------------
# 3. Checkpoint contents
# ---------------------------------------------------------------------------

class TestCheckpointContents:
    @pytest.fixture
    def ckpt(self, tmp_path):
        trainer = _make_trainer(n_steps=1, eval_every=1)
        cb = CheckpointCallback(trainer, trainer.loss_fn, tmp_path, every=1)
        trainer.callbacks = [cb]
        trainer.fit()
        return torch.load(tmp_path / "step_000001.pt", map_location="cpu", weights_only=True)

    def test_has_global_step(self, ckpt):
        assert "global_step" in ckpt

    def test_global_step_value(self, ckpt):
        assert ckpt["global_step"] == 1

    def test_has_model_state(self, ckpt):
        assert "model_state" in ckpt

    def test_model_state_is_dict(self, ckpt):
        assert isinstance(ckpt["model_state"], dict)

    def test_has_optimizer_state(self, ckpt):
        assert "optimizer_state" in ckpt

    def test_optimizer_state_is_dict(self, ckpt):
        assert isinstance(ckpt["optimizer_state"], dict)

    def test_has_loss_fn_state(self, ckpt):
        assert "loss_fn_state" in ckpt

    def test_loss_fn_state_contains_kendall_sigmas(self, ckpt):
        keys = ckpt["loss_fn_state"].keys()
        assert "log_sigma_sq_pt" in keys
        assert "log_sigma_sq_comp" in keys

    def test_model_state_keys_nonempty(self, ckpt):
        assert len(ckpt["model_state"]) > 0


# ---------------------------------------------------------------------------
# 4. Restore round-trip
# ---------------------------------------------------------------------------

class TestCheckpointRestore:
    def test_model_state_restores_cleanly(self, tmp_path):
        trainer = _make_trainer(n_steps=1, eval_every=1)
        cb = CheckpointCallback(trainer, trainer.loss_fn, tmp_path, every=1)
        trainer.callbacks = [cb]
        trainer.fit()

        ckpt = torch.load(tmp_path / "step_000001.pt", map_location="cpu", weights_only=True)
        fresh_model = TabICLRegressor(
            n_genes=N_GENES, k=K, embed_dim=8, n_heads=2, num_cls=2,
            col_num_inds=4, n_layers_col=1, n_layers_row=1, n_layers_icl=1,
        )
        missing, unexpected = fresh_model.load_state_dict(
            ckpt["model_state"], strict=True
        ).missing_keys, fresh_model.load_state_dict(
            ckpt["model_state"], strict=True
        ).unexpected_keys
        assert missing == []
        assert unexpected == []

    def test_loss_fn_state_restores_kendall_sigmas(self, tmp_path):
        trainer = _make_trainer(n_steps=1, eval_every=1)
        cb = CheckpointCallback(trainer, trainer.loss_fn, tmp_path, every=1)
        trainer.callbacks = [cb]
        trainer.fit()

        ckpt = torch.load(tmp_path / "step_000001.pt", map_location="cpu", weights_only=True)
        fresh_loss = DualHeadLoss()
        fresh_loss.load_state_dict(ckpt["loss_fn_state"])
        assert torch.isfinite(fresh_loss.log_sigma_sq_pt)
        assert torch.isfinite(fresh_loss.log_sigma_sq_comp)
