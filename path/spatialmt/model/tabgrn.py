"""spatialmt.model.tabgrn — TabICLRegressor built on the pretrained TabICLv2 backbone."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabicl.model.embedding import ColEmbedding
from tabicl.model.interaction import RowInteraction
from tabicl.model.encoders import Encoder


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _init_linear(layer: nn.Linear, weight_std: float = 0.01, bias_value: float = 0.0) -> None:
    nn.init.normal_(layer.weight, std=weight_std)
    nn.init.constant_(layer.bias, bias_value)


# ---------------------------------------------------------------------------
# AnchorLabelEmbedder
# ---------------------------------------------------------------------------

class AnchorLabelEmbedder(nn.Module):
    """Adds pseudotime + soft_label projections to anchor rows; query row is untouched."""

    def __init__(self, d_model: int, k: int) -> None:
        super().__init__()
        self.pt_proj = nn.Linear(1, d_model)
        self.sl_proj = nn.Linear(k, d_model)

    def forward(
        self,
        x: torch.Tensor,
        anchor_pseudotime: torch.Tensor,
        anchor_soft_labels: torch.Tensor,
    ) -> torch.Tensor:
        B, n_cells, d_model = x.shape
        n_anchors = anchor_pseudotime.shape[1]

        pt_emb  = self.pt_proj(anchor_pseudotime.unsqueeze(-1))  # (B, n_anchors, d_model)
        sl_emb  = self.sl_proj(anchor_soft_labels)               # (B, n_anchors, d_model)
        label_emb = pt_emb + sl_emb                             # (B, n_anchors, d_model)

        n_query  = n_cells - n_anchors
        zero_pad = torch.zeros(B, n_query, d_model, device=x.device, dtype=x.dtype)
        injection = torch.cat([label_emb, zero_pad], dim=1)     # (B, n_cells, d_model)

        return x + injection   # query rows receive +0 exactly — no label leakage


# ---------------------------------------------------------------------------
# SharedTrunk
# ---------------------------------------------------------------------------

class SharedTrunk(nn.Module):
    """LayerNorm → Linear → GELU → Linear."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, d_model)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


# ---------------------------------------------------------------------------
# PseudotimeHead
# ---------------------------------------------------------------------------

class PseudotimeHead(nn.Module):
    """Linear → sigmoid → scalar ∈ (0, 1)."""

    def __init__(self, d_model: int, init_bias: float = 0.5, init_std: float = 0.01) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        _init_linear(self.linear, weight_std=init_std, bias_value=init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x).squeeze(-1))


# ---------------------------------------------------------------------------
# CompositionHead
# ---------------------------------------------------------------------------

class CompositionHead(nn.Module):
    """Linear → softmax → (B, K) cell-state probabilities."""

    def __init__(self, d_model: int, k: int, init_std: float = 0.01) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, k)
        _init_linear(self.linear, weight_std=init_std, bias_value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x), dim=-1)


# ---------------------------------------------------------------------------
# DirichletCompositionHead
# ---------------------------------------------------------------------------

class DirichletCompositionHead(nn.Module):
    """Linear → softplus → Dirichlet concentration parameters α ∈ (0, ∞)^K.

    Output is NOT a probability vector. Mean: α_k/Σα_k. Precision: Σα_k.
    """

    def __init__(self, d_model: int, k: int, init_std: float = 0.01) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, k)
        _init_linear(self.linear, weight_std=init_std, bias_value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.linear(x))


# ---------------------------------------------------------------------------
# TabICLRegressor
# ---------------------------------------------------------------------------

class TabICLRegressor(nn.Module):
    """ICL regression model on the pretrained TabICLv2 backbone.

    Architecture dims (embed_dim, n_heads, num_cls, col_num_inds, n_layers_*)
    must match the pretrained checkpoint — see TABICL_V2_ARCH in experiment.py.
    """

    def __init__(
        self,
        n_genes: int,
        k: int,
        embed_dim: int,
        n_heads: int,
        num_cls: int,
        col_num_inds: int,
        n_layers_col: int,
        n_layers_row: int,
        n_layers_icl: int,
        composition_loss_type: str = "kl",
    ) -> None:
        super().__init__()
        d_model = num_cls * embed_dim   # output dim of row_interactor

        _VALID_COMP_TYPES = {"kl", "dirichlet"}
        if composition_loss_type not in _VALID_COMP_TYPES:
            raise ValueError(
                f"composition_loss_type must be one of {_VALID_COMP_TYPES!r}, "
                f"got {composition_loss_type!r}"
            )
        self.composition_loss_type = composition_loss_type

        # --- TabICLv2 backbone (pretrained weights loaded via load_backbone) ---
        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=n_layers_col,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            num_inds=col_num_inds,
            target_aware=False,   # pseudotime is continuous — no class conditioning
        )
        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=n_layers_row,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            num_cls=num_cls,
        )

        # Pretrained ICL transformer from TabICLv2's icl_predictor.tf_icl.
        # Loaded via load_backbone with key remapping icl_predictor.tf_icl.* → tf_icl.*
        self.tf_icl = Encoder(
            num_blocks=n_layers_icl,
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
        )

        # --- Custom regression head stack (always reinitialised) ---
        self.anchor_label_embedder = AnchorLabelEmbedder(d_model=d_model, k=k)
        self.shared_trunk    = SharedTrunk(d_model=d_model)
        self.pseudotime_head = PseudotimeHead(d_model=d_model)
        self.composition_head = (
            CompositionHead(d_model=d_model, k=k)
            if composition_loss_type == "kl"
            else DirichletCompositionHead(d_model=d_model, k=k)
        )

    def forward(self, batch: "ICLBatch") -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pt_pred (B,), comp_pred (B, K)). query_pseudotime/soft_labels unused."""
        n_anchors = batch.context_expression.shape[1]

        query_expr = batch.query_expression.unsqueeze(1)
        all_expr   = torch.cat([batch.context_expression, query_expr], dim=1)

        # Pass context_pseudotime so ColEmbedding's train_size == n_anchors.
        emb = self.col_embedder(all_expr, batch.context_pseudotime)

        # clone() needed: RowInteraction._train_forward writes CLS slots in-place,
        # which would corrupt emb's version counter and break autograd through col_embedder.
        x = self.row_interactor(emb.clone())

        # Label injection deferred to post-row-interaction; col_embedder sees clean
        # gene features, preserving GRN signal quality.
        x = self.anchor_label_embedder(
            x,
            anchor_pseudotime=batch.context_pseudotime,
            anchor_soft_labels=batch.context_soft_labels,
        )

        x = self.tf_icl(x, train_size=n_anchors)
        x = x[:, -1, :]   # query row

        x         = self.shared_trunk(x)
        pt_pred   = self.pseudotime_head(x)
        comp_pred = self.composition_head(x)
        return pt_pred, comp_pred

    def forward_supervised(
        self,
        gene_expression: torch.Tensor,
        population_anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Supervised forward bypassing tf_icl. Uses a single population anchor so
        col_embedder has train_size >= 1. Returns (pt_pred (B,), comp_pred (B, K))."""
        B      = gene_expression.shape[0]
        anchor = population_anchor.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x      = torch.cat([anchor, gene_expression.unsqueeze(1)], dim=1)
        dummy_pt = torch.zeros(B, 1, device=gene_expression.device, dtype=gene_expression.dtype)
        emb = self.col_embedder(x, dummy_pt)
        x   = self.row_interactor(emb.clone())
        x   = x[:, -1, :]
        x   = self.shared_trunk(x)
        return self.pseudotime_head(x), self.composition_head(x)

    def load_backbone(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load row_interactor and tf_icl weights from a TabICLv2 .ckpt file.

        col_embedder is excluded — pretrained position-based weights don't transfer
        to our gene-indexed input space. icl_predictor.tf_icl.* is remapped to tf_icl.*.
        """
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]

        backbone_state: dict = {}
        for k, v in state.items():
            if k.startswith("row_interactor."):
                backbone_state[k] = v
            elif k.startswith("icl_predictor.tf_icl."):
                backbone_state[k[len("icl_predictor."):]] = v

        self.load_state_dict(backbone_state, strict=strict)

    def parameter_groups(self) -> list[dict]:
        """Return four optimizer param groups: col, row, icl, head."""
        from spatialmt.config.experiment import ModelConfig
        cfg = ModelConfig()
        return [
            {
                "name":   "col",
                "params": list(self.col_embedder.parameters()),
                "lr":     cfg.lr_col,
            },
            {
                "name":   "row",
                "params": list(self.row_interactor.parameters()),
                "lr":     cfg.lr_row,
            },
            {
                "name":   "icl",
                "params": list(self.tf_icl.parameters()),
                "lr":     cfg.lr_icl,
            },
            {
                "name":   "head",
                "params": (
                    list(self.anchor_label_embedder.parameters())
                    + list(self.shared_trunk.parameters())
                    + list(self.pseudotime_head.parameters())
                    + list(self.composition_head.parameters())
                ),
                "lr":     cfg.lr_head,
            },
        ]


# ---------------------------------------------------------------------------
# AttentionScorer
# ---------------------------------------------------------------------------

class AttentionScorer:
    """Fast GRN diagnostic: cosine similarity of col_embedder gene embeddings.

    Returns (B, n_cells, n_genes, n_genes) after a forward pass.
    Fast diagnostic only — use Integrated Gradients for publication-grade edges.

    Warning: only valid when feature_group=False (our default). With feature
    grouping each token represents a gene triplet, not a single gene.
    """

    def __init__(self, model: TabICLRegressor) -> None:
        self._model = model
        self._gene_embeddings: torch.Tensor | None = None
        self._hook = model.col_embedder.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, input, output: torch.Tensor) -> None:
        n_genes = input[0].shape[-1]
        self._gene_embeddings = output[:, :, -n_genes:, :].detach()

    def extract(self) -> torch.Tensor:
        """Return cosine similarity (B, n_cells, n_genes, n_genes). Call after forward()."""
        if self._gene_embeddings is None:
            raise RuntimeError(
                "No forward pass recorded. Call model(batch) before extract()."
            )
        # emb: (B, n_cells, n_genes, embed_dim)
        emb = self._gene_embeddings
        # L2-normalise over embed_dim
        emb_norm = F.normalize(emb, dim=-1)  # (B, n_cells, n_genes, embed_dim)
        # Cosine similarity: (B, n_cells, n_genes, n_genes)
        scores = torch.matmul(emb_norm, emb_norm.transpose(-1, -2))
        return scores.detach()

    def remove_hook(self) -> None:
        """Detach the forward hook when the scorer is no longer needed."""
        self._hook.remove()
