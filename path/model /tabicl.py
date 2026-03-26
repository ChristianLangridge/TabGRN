"""
spatialmt.model.tabicl
=======================
TabICLv2 wrapper for dual-head pseudotime regression + cell state composition.

ROTATION SCOPE (active now):
    - PseudotimeHead only (sigmoid, scalar output)
    - TabICLv2 column + row + ICL attention
    - Differential LR with warmup
    - forward() returns pseudotime scalar only

FULL PROJECT SCOPE (July 3rd onwards):
    - CompositionHead added (Dirichlet, K=5 outputs)
    - forward() returns (pseudotime, composition) tuple
    - Normalised dual loss in Trainer
    - Dual-head biological plausibility gate

The CompositionHead class is implemented as a tested skeleton now.
It is wired into the model in Phase 5A (July onwards).

Warmup schedule
---------------
Column attention:  frozen for warmup_col_steps (default 500)
ICL attention:     frozen for warmup_icl_steps  (default 100)
Row attention:     no freeze — adapts quickly
Both heads:        no freeze — always trained from scratch

Differential learning rates
----------------------------
column_attention:   lr_col   (default 1e-5)
row_attention:      lr_row   (default 1e-4)
icl_attention:      lr_icl   (default 5e-5)
column_embeddings:  lr_emb   (default 1e-3)  — always from scratch
pseudotime_head:    lr_head  (default 1e-3)
composition_head:   lr_head  (default 1e-3)  — Phase 5A only
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from spatialmt.config.experiment import ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

class PseudotimeHead(nn.Module):
    """
    Regression head for scalar pseudotime prediction.

    Output: sigmoid(linear(x)) ∈ (0, 1)
    Maps directly onto normalised DC1 range without post-hoc scaling.

    Initialisation
    --------------
    weight ~ N(0, 0.01)  — near-zero prevents large initial gradients
    bias = 0.5            — trajectory midpoint prior
    Forces model to learn deviation from midpoint = the GRN signal.

    ROTATION SCOPE: active.
    FULL PROJECT:   unchanged — pseudotime head is stable across both scopes.
    """

    def __init__(
        self,
        d_model: int,
        init_std: float = 0.01,
        init_bias: float = 0.5,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        nn.init.normal_(self.linear.weight, mean=0.0, std=init_std)
        nn.init.constant_(self.linear.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, d_model)

        Returns
        -------
        pseudotime : (batch,) in (0, 1)
        """
        return torch.sigmoid(self.linear(x)).squeeze(-1)


class CompositionHead(nn.Module):
    """
    Dirichlet head for cell state composition prediction.

    Output: K concentration parameters α_k > 0 via softplus.
    Models uncertainty over the K-simplex — genuinely transitional cells
    produce a flat Dirichlet (low Σα_k), not a spuriously confident assignment.

    Loss: negative Dirichlet log-likelihood
        L = -log Dir(y | α)

    K = 5 for Matrigel condition:
        0: neuroectodermal progenitor
        1: neural tube neuroepithelial
        2: prosencephalic progenitor
        3: telencephalic progenitor
        4: early neuron

    ROTATION SCOPE: skeleton tested on synthetic data, NOT wired into training.
    FULL PROJECT:   wired into forward() in Phase 5A (July onwards).

    Implementation note
    -------------------
    Do not implement Phase 5A wiring until:
      1. Pseudotime-only model passes biological plausibility gate
      2. soft_labels are validated in ProcessedDataset
      3. Normalised loss balancer is implemented in Trainer
    """

    def __init__(self, d_model: int, n_cell_states: int = 5) -> None:
        super().__init__()
        self.n_cell_states = n_cell_states
        self.linear = nn.Linear(d_model, n_cell_states)
        # Softplus ensures α_k > 0 — required for valid Dirichlet parameters
        self.softplus = nn.Softplus()
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.linear.bias, 1.0)  # initialise near uniform Dir(1,...,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, d_model)

        Returns
        -------
        alpha : (batch, K)  — Dirichlet concentration parameters, all > 0
        """
        return self.softplus(self.linear(x)) + 1e-6  # numerical stability

    def dirichlet_nll(
        self,
        alpha: torch.Tensor,
        soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Negative Dirichlet log-likelihood.

        Parameters
        ----------
        alpha       : (batch, K) — predicted concentration parameters
        soft_labels : (batch, K) — target soft label proportions, rows sum to 1

        Returns
        -------
        loss : scalar
        """
        raise NotImplementedError(
            "Implement in Phase 5A.\n"
            "torch.distributions.Dirichlet(alpha).log_prob(soft_labels).mean()\n"
            "Note: soft_labels must be clamped to (eps, 1-eps) before log_prob.\n"
            "Dirichlet is undefined at 0 or 1."
        )


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TabICLRegressor(nn.Module):
    """
    TabICLv2 adapted for pseudotime regression with optional composition head.

    ROTATION SCOPE
    --------------
    forward() returns pseudotime scalar only.
    CompositionHead exists but is not called.
    configure_optimizers() returns 5 parameter groups (no composition head LR).

    FULL PROJECT (Phase 5A)
    -----------------------
    Set model.enable_composition_head() to wire in the CompositionHead.
    forward() then returns (pseudotime, alpha) tuple.
    Trainer must use NormalisedDualLoss.
    Do not enable before pseudotime plausibility gate passes.

    Parameters
    ----------
    config : ModelConfig
    n_genes : int — must equal ProcessedDataset.n_genes
    """

    def __init__(self, config: "ModelConfig", n_genes: int) -> None:
        super().__init__()
        self.config = config
        self.n_genes = n_genes
        self._composition_enabled = False  # Phase 5A gate

        # Column embeddings — always re-initialised (pre-trained do not
        # generalise to this gene count)
        self.column_embeddings = nn.Embedding(n_genes, config.d_model)
        nn.init.normal_(self.column_embeddings.weight, std=0.02)

        # TabICLv2 three-stage attention — loaded from pre-trained checkpoint
        # Stage 1: column-wise (gene × gene) — GRN signal lives here
        self.column_attention_layers = nn.ModuleList()

        # Stage 2: row-wise (feature → cell representation)
        self.row_attention_layers = nn.ModuleList()

        # Stage 3: dataset-wise ICL (cell × cell, target-aware)
        self.icl_attention_layers = nn.ModuleList()

        # Output heads
        self.pseudotime_head = PseudotimeHead(
            d_model=config.d_model,
            init_std=config.output_head_init_std,
            init_bias=config.output_head_init_bias,
        )

        # Composition head — exists but is not called until Phase 5A
        self.composition_head = CompositionHead(
            d_model=config.d_model,
            n_cell_states=config.n_cell_states,
        )

        # Warmup state
        self._col_frozen = config.warmup_col_steps > 0
        self._icl_frozen = config.warmup_icl_steps > 0

        if self._col_frozen:
            self._freeze(self.column_attention_layers)
            logger.info(
                f"Column attention frozen for {config.warmup_col_steps} steps."
            )
        if self._icl_frozen:
            self._freeze(self.icl_attention_layers)
            logger.info(
                f"ICL attention frozen for {config.warmup_icl_steps} steps."
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        context_expression: torch.Tensor,
        context_pseudotime: torch.Tensor,
        query_expression: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        ROTATION SCOPE
        --------------
        Returns pseudotime: (batch,) in (0, 1)

        FULL PROJECT (after enable_composition_head())
        ----------------------------------------------
        Returns (pseudotime, alpha):
            pseudotime : (batch,)     in (0, 1)
            alpha      : (batch, K)   Dirichlet concentration parameters

        Parameters
        ----------
        context_expression : (batch, n_context, n_genes)
        context_pseudotime : (batch, n_context)  — DC1 labels for anchor cells
        query_expression   : (batch, n_genes)
        """
        raise NotImplementedError(
            "Implement in GREEN phase (Phase 2, Week 5).\n\n"
            "ROTATION SCOPE steps:\n"
            "  1. Embed genes via column_embeddings\n"
            "  2. Stage 1: column attention (gene × gene)\n"
            "     — this is where AttentionScorer hooks in\n"
            "  3. Stage 2: row attention (build cell representations)\n"
            "  4. Stage 3: ICL attention with context_pseudotime as target embedding\n"
            "     — anchor cells attend to query, query attends to anchors\n"
            "  5. Extract query cell representation from ICL stage\n"
            "  6. pseudotime_head(query_repr) → scalar\n"
            "  7. Return pseudotime\n\n"
            "PHASE 5A addition (do not implement now):\n"
            "  8. If self._composition_enabled:\n"
            "       alpha = composition_head(query_repr)\n"
            "       return pseudotime, alpha\n"
            "  9. Else: return pseudotime"
        )

    # ------------------------------------------------------------------
    # Phase gate
    # ------------------------------------------------------------------

    def enable_composition_head(self) -> None:
        """
        Wire in the CompositionHead for dual-head training.

        CALL ONLY AFTER:
          1. Pseudotime-only model passes biological plausibility gate
          2. soft_labels validated in ProcessedDataset
          3. NormalisedDualLoss implemented in Trainer

        This is a Phase 5A operation. Calling it before those conditions
        are met will produce a model that trains without error but produces
        biologically uninterpretable composition outputs.
        """
        self._composition_enabled = True
        logger.info(
            "CompositionHead enabled. forward() now returns "
            "(pseudotime, alpha). Switch Trainer to NormalisedDualLoss."
        )

    # ------------------------------------------------------------------
    # Training step hook
    # ------------------------------------------------------------------

    def on_training_step(self, step: int) -> None:
        """Unfreeze attention layers after warmup steps."""
        if self._col_frozen and step >= self.config.warmup_col_steps:
            self._unfreeze(self.column_attention_layers)
            self._col_frozen = False
            logger.info(
                f"Step {step}: Column attention unfrozen "
                f"(lr={self.config.lr_col})"
            )
        if self._icl_frozen and step >= self.config.warmup_icl_steps:
            self._unfreeze(self.icl_attention_layers)
            self._icl_frozen = False
            logger.info(
                f"Step {step}: ICL attention unfrozen "
                f"(lr={self.config.lr_icl})"
            )

    # ------------------------------------------------------------------
    # Optimiser configuration
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> list[dict]:
        """
        Differential learning rate parameter groups.

        ROTATION SCOPE: 5 groups (no composition head).
        FULL PROJECT:   6 groups (add composition head group in Phase 5A).
        """
        groups = [
            {
                "params": list(self.column_attention_layers.parameters()),
                "lr": self.config.lr_col,
                "name": "column_attention",
            },
            {
                "params": list(self.row_attention_layers.parameters()),
                "lr": self.config.lr_row,
                "name": "row_attention",
            },
            {
                "params": list(self.icl_attention_layers.parameters()),
                "lr": self.config.lr_icl,
                "name": "icl_attention",
            },
            {
                "params": list(self.column_embeddings.parameters()),
                "lr": self.config.lr_emb,
                "name": "column_embeddings",
            },
            {
                "params": list(self.pseudotime_head.parameters()),
                "lr": self.config.lr_head,
                "name": "pseudotime_head",
            },
        ]
        # Phase 5A: uncomment when enable_composition_head() is called
        # if self._composition_enabled:
        #     groups.append({
        #         "params": list(self.composition_head.parameters()),
        #         "lr": self.config.lr_head,
        #         "name": "composition_head",
        #     })
        return groups

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze(module: nn.ModuleList) -> None:
        for param in module.parameters():
            param.requires_grad = False

    @staticmethod
    def _unfreeze(module: nn.ModuleList) -> None:
        for param in module.parameters():
            param.requires_grad = True

    @classmethod
    def load_pretrained(
        cls,
        config: "ModelConfig",
        n_genes: int,
        checkpoint_path: str,
    ) -> "TabICLRegressor":
        """
        Load TabICLv2 pre-trained weights into attention layers.

        Column embeddings are NOT loaded — always re-initialised for n_genes.
        Output heads are NOT loaded — always trained from scratch.
        """
        raise NotImplementedError(
            "Implement in Phase 2 (Week 5).\n"
            "1. Load TabICLv2 checkpoint\n"
            "2. Filter keys to column/row/icl attention only\n"
            "3. model.load_state_dict(filtered, strict=False)\n"
            "4. Assert column_embeddings are NOT loaded from checkpoint\n"
            "5. Assert pseudotime_head weights are randomly initialised\n"
            "6. Return model"
        )