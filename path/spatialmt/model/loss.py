"""
spatialmt.model.loss — DualHeadLoss

Uncertainty-weighted dual-head loss (Kendall, Gal & Cipolla, NeurIPS 2018).

Pseudotime head
---------------
    L_pt = MSE(pt_pred, pt_target)

Composition head
----------------
    L_comp = KL(comp_target ∥ comp_pred)
           = Σ_k  π_k · (log π_k – log p_k)

    where π = comp_target (distance-to-centroid softmax from ProcessedDataset)
    and   p = comp_pred   (CompositionHead softmax output).

    Both distributions live on the K-simplex. KL = 0 iff p = π exactly.
    No concentration hyperparameter is required — KL divergence is the natural
    parameter-free measure between two probability distributions.

Uncertainty normalisation
-------------------------
Let s_pt = log σ²_pt  and  s_comp = log σ²_comp  (learnable nn.Parameters).

    L_total = exp(–s_pt)  · L_pt   + ½ · s_pt
            + exp(–s_comp) · L_comp + ½ · s_comp

At initialisation s = 0 → σ = 1 → equal unit weighting of both tasks.
The ½·s penalty term prevents σ → ∞ (which would trivially zero both losses).
During training the model learns the appropriate relative scale of each task
without manual weight tuning, and the learned σ values can be inspected as
a diagnostic of relative task difficulty.

Usage
-----
    loss_fn = DualHeadLoss()
    total, L_pt, L_comp = loss_fn(pt_pred, pt_target, comp_pred, comp_target)
    total.backward()

The loss_fn parameters (log_sigma_sq_pt, log_sigma_sq_comp) must be included
in the optimiser. Add them to the "head" parameter group:

    head_params = (
        list(model.label_injector.parameters())
        + list(model.shared_trunk.parameters())
        + list(model.pseudotime_head.parameters())
        + list(model.composition_head.parameters())
        + list(loss_fn.parameters())
    )
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHeadLoss(nn.Module):
    """Uncertainty-weighted MSE + KL dual-head loss.

    Parameters
    ----------
    eps : float
        Small constant added before log to guard against log(0).
        Applied to both comp_pred and comp_target inside composition_loss.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        # Initialised to 0 → σ² = 1 → equal weighting at start of training
        # Shape () keeps the weighted total a scalar throughout.
        self.log_sigma_sq_pt   = nn.Parameter(torch.zeros(()))
        self.log_sigma_sq_comp = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------
    # Component losses
    # ------------------------------------------------------------------

    def pseudotime_loss(
        self,
        pt_pred: torch.Tensor,
        pt_target: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared error between predicted and target pseudotime.

        Parameters
        ----------
        pt_pred   : (B,) sigmoid output of PseudotimeHead, ∈ (0, 1)
        pt_target : (B,) rank-transformed pseudotime from ProcessedDataset, ∈ [0, 1]
        """
        return F.mse_loss(pt_pred, pt_target)

    def composition_loss(
        self,
        comp_pred: torch.Tensor,
        comp_target: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence KL(comp_target ∥ comp_pred), averaged over the batch.

        Parameters
        ----------
        comp_pred   : (B, K) softmax output of CompositionHead
        comp_target : (B, K) distance-to-centroid soft labels from ProcessedDataset

        Returns
        -------
        Scalar mean KL divergence ≥ 0.  Zero iff comp_pred == comp_target.
        """
        log_pred   = torch.log(comp_pred.clamp(min=self.eps))
        log_target = torch.log(comp_target.clamp(min=self.eps))
        # KL(target ∥ pred) = Σ_k target_k · (log target_k – log pred_k)
        kl = (comp_target * (log_target - log_pred)).sum(dim=-1)  # (B,)
        return kl.mean()

    # ------------------------------------------------------------------
    # Combined uncertainty-weighted forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pt_pred: torch.Tensor,
        pt_target: torch.Tensor,
        comp_pred: torch.Tensor,
        comp_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the uncertainty-weighted dual-head loss.

        Parameters
        ----------
        pt_pred     : (B,)    PseudotimeHead output
        pt_target   : (B,)    ground-truth pseudotime
        comp_pred   : (B, K)  CompositionHead output
        comp_target : (B, K)  soft label targets

        Returns
        -------
        total    : scalar — uncertainty-weighted combined loss (use for backward)
        L_pt     : scalar — raw MSE component (log only)
        L_comp   : scalar — raw KL component  (log only)
        """
        L_pt   = self.pseudotime_loss(pt_pred, pt_target)
        L_comp = self.composition_loss(comp_pred, comp_target)

        s_pt   = self.log_sigma_sq_pt
        s_comp = self.log_sigma_sq_comp

        # Kendall uncertainty weighting
        total = (
            torch.exp(-s_pt)   * L_pt   + 0.5 * s_pt
            + torch.exp(-s_comp) * L_comp + 0.5 * s_comp
        )
        return total, L_pt, L_comp
