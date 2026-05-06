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
