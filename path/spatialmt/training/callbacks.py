"""spatialmt.training.callbacks — Training callbacks for TabGRN."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Optional, Union
    from spatialmt.data_preparation.dataset import ProcessedDataset
    from spatialmt.model.loss import DualHeadLoss
    from spatialmt.model.tabgrn import TabICLRegressor
    from spatialmt.training.trainer import Trainer, SupervisedTrainer


class CheckpointCallback:
    """Save model, optimizer, loss_fn, and global_step every ``every`` steps.

    Parameters
    ----------
    trainer : Trainer | SupervisedTrainer | None
        The active trainer — provides optimizer and global_step.
        May be set to None at construction and assigned before fit() runs.
    loss_fn : DualHeadLoss
        The loss function whose Kendall sigma parameters are also saved.
    out_dir : Path
        Directory to write checkpoint files into (created if absent).
    every : int
        Checkpoint is written when ``step % every == 0``.
    """

    def __init__(
        self,
        trainer: "Optional[Union[Trainer, SupervisedTrainer]]",
        loss_fn: "DualHeadLoss",
        out_dir: Path,
        every: int = 1000,
    ) -> None:
        self.trainer = trainer
        self.loss_fn = loss_fn
        self.out_dir = Path(out_dir)
        self.every   = every

    def on_epoch_end(
        self,
        model: "TabICLRegressor",
        _dataset: "ProcessedDataset",
        step: int,
    ) -> None:
        if step % self.every != 0:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"step_{step:06d}.pt"
        torch.save(
            {
                "global_step":      step,
                "model_state":      model.state_dict(),
                "optimizer_state":  self.trainer.optimizer.state_dict(),
                "loss_fn_state":    self.loss_fn.state_dict(),
            },
            path,
        )
        print(f"  [ckpt] step {step:>6d} → {path}")


class WarmupBoundaryCallback:
    """Save a named checkpoint once at exactly ``save_step``.

    Intended for saving ``warmup_final.pt`` at the ICL warm-up boundary —
    the canonical GRN extraction point where tf_icl has not yet trained.

    Parameters
    ----------
    save_step : int
        The training step at which to save. Fires on the first
        ``on_epoch_end`` call where ``step >= save_step``.
    out_path : Path
        Full file path for the checkpoint (not a directory).
    loss_fn :
        Loss function whose state is included in the checkpoint.
    run_id : str
        Stored in the checkpoint for provenance.
    composition_loss_type : str
        Stored in the checkpoint for provenance.
    """

    def __init__(
        self,
        save_step: int,
        out_path: Path,
        loss_fn,
        run_id: str,
        composition_loss_type: str,
    ) -> None:
        self.save_step             = save_step
        self.out_path              = Path(out_path)
        self.loss_fn               = loss_fn
        self.run_id                = run_id
        self.composition_loss_type = composition_loss_type
        self._saved                = False

    def on_epoch_end(
        self,
        model: "TabICLRegressor",
        _dataset: "ProcessedDataset",
        step: int,
    ) -> None:
        if self._saved or step < self.save_step:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state":           model.state_dict(),
                "loss_fn_state":         self.loss_fn.state_dict(),
                "run_id":                self.run_id,
                "composition_loss_type": self.composition_loss_type,
                "global_step":           step,
            },
            self.out_path,
        )
        print(f"  [warmup_final] GRN extraction checkpoint saved → {self.out_path}")
        self._saved = True
