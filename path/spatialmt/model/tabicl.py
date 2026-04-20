"""
spatialmt.model.tabicl — TabICLRegressor and sub-modules.

Backbone
--------
Uses the pretrained TabICLv2 backbone (soda-inria/tabicl on GitHub, v2.0+)
for the two feature-interaction stages:

  col_embedder    tabicl.model.embedding.ColEmbedding
                  Distribution-aware column-wise embedding via shared Set
                  Transformer (ISAB). Processes each gene independently across
                  all cells in the ICL window.
                  Input:  (B, n_cells, n_genes)  + y_train (B, n_anchors)
                  Output: (B, n_cells, seq_len, embed_dim)

  row_interactor  tabicl.model.interaction.RowInteraction
                  Transformer encoder with rotary positional encoding over the
                  feature (gene) dimension within each cell.
                  Input:  (B, n_cells, seq_len, embed_dim)
                  Output: (B, n_cells, d_model)   where d_model = num_cls × embed_dim

Custom regression head stack (not present in TabICLv2)
------------------------------------------------------
  LabelInjector   Adds pseudotime + soft_label projections to anchor rows
                  only. Query row is never modified (no label leakage).

  ICLAttention    Query (last row) cross-attends all anchor representations.
                  Output: (B, d_model)

  SharedTrunk     LayerNorm → Linear → GELU → Linear.

  PseudotimeHead  Linear → sigmoid scalar ∈ (0, 1)
  CompositionHead Linear → softmax over K cell states

Supporting
----------
  AttentionScorer Uses a forward hook on col_embedder to capture per-cell
                  gene embeddings after the column-wise attention pass,
                  then computes a (B, n_cells, n_genes, n_genes) similarity
                  matrix for GRN explainability.

Parameter groups (for torch.optim)
-----------------------------------
  col   col_embedder     lr_col = 1e-5  [pretrained; frozen for warmup_col_steps]
  row   row_interactor   lr_row = 1e-4  [pretrained]
  icl   icl_attention    lr_icl = 5e-5  [pretrained; frozen for warmup_icl_steps]
  head  label_injector + shared_trunk + pseudotime_head + composition_head
                         lr_head = 1e-3 [always reinitialised]

Checkpoint loading
------------------
Load a pretrained TabICL checkpoint with strict=False — col_embedder and
row_interactor weight names match; head layers are skipped automatically.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tabicl.model.embedding import ColEmbedding
from tabicl.model.interaction import RowInteraction


# ---------------------------------------------------------------------------
# LabelInjector
# ---------------------------------------------------------------------------

class LabelInjector(nn.Module):
    """Injects anchor pseudotime and soft_labels into the cell representation.

    Anchor rows (positions 0 … n_anchors-1) receive the summed projection of
    pseudotime and soft_labels. The query row (last position) is left unchanged
    — injecting query labels here would constitute data leakage.

    Input:  x (B, n_cells, d_model),
            anchor_pseudotime (B, n_anchors),
            anchor_soft_labels (B, n_anchors, K)
    Output: (B, n_cells, d_model)
    """

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
# ICLAttention
# ---------------------------------------------------------------------------

class ICLAttention(nn.Module):
    """Query cross-attends all anchor representations.

    The query cell is at the last position; all preceding positions are anchors.

    Input:  (B, n_cells, d_model)
    Output: (B, d_model)
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query   = x[:, -1:, :]  # (B, 1, d_model)
        anchors = x[:, :-1, :]  # (B, n_anchors, d_model)
        out, _  = self.attn(query, anchors, anchors)
        return out.squeeze(1)   # (B, d_model)


# ---------------------------------------------------------------------------
# SharedTrunk
# ---------------------------------------------------------------------------

class SharedTrunk(nn.Module):
    """LayerNorm → Linear → GELU → Linear.

    Input/output: (B, d_model)
    """

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
    """Linear → sigmoid scalar.

    Bias initialised to init_bias so sigmoid(init_bias) ≈ 0.62 at start of
    training — a modest positive pseudotime prior.

    Input:  (B, d_model)
    Output: (B,)  ∈ (0, 1)
    """

    def __init__(self, d_model: int, init_bias: float = 0.5, init_std: float = 0.01) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        nn.init.normal_(self.linear.weight, std=init_std)
        nn.init.constant_(self.linear.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x).squeeze(-1))


# ---------------------------------------------------------------------------
# CompositionHead
# ---------------------------------------------------------------------------

class CompositionHead(nn.Module):
    """Linear → softmax over K cell states.

    Input:  (B, d_model)
    Output: (B, K)  rows sum to 1.0
    """

    def __init__(self, d_model: int, k: int, init_std: float = 0.01) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, k)
        nn.init.normal_(self.linear.weight, std=init_std)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x), dim=-1)


# ---------------------------------------------------------------------------
# TabICLRegressor
# ---------------------------------------------------------------------------

class TabICLRegressor(nn.Module):
    """ICL regression model built on the pretrained TabICLv2 backbone.

    The forward pass maps an ICLBatch to (pseudotime, composition) predictions
    for the query cells (day-11 holdout) given the anchor context window.

    Parameters
    ----------
    n_genes : int
        Number of highly variable genes (HVG count from DataConfig.max_genes).
    embed_dim : int
        Token embedding dimension inside ColEmbedding / RowInteraction.
    n_heads : int
        Number of attention heads in backbone and ICLAttention.
    n_layers : int
        Number of transformer blocks in ColEmbedding and RowInteraction.
    k : int
        Number of cell states (DataConfig.n_cell_states).
    num_cls : int
        Number of CLS tokens in RowInteraction. d_model = num_cls × embed_dim.
    col_num_inds : int
        Number of inducing points in ColEmbedding's ISAB blocks.
    """

    def __init__(
        self,
        n_genes: int,
        embed_dim: int,
        n_heads: int,
        n_layers: int,
        k: int,
        num_cls: int = 2,
        col_num_inds: int = 64,
    ) -> None:
        super().__init__()
        d_model = num_cls * embed_dim   # output dim of row_interactor

        # --- TabICLv2 backbone (pretrained weights loaded via load_backbone) ---
        self.col_embedder = ColEmbedding(
            embed_dim=embed_dim,
            num_blocks=n_layers,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            num_inds=col_num_inds,
            target_aware=False,   # pseudotime is continuous — no class conditioning
        )
        self.row_interactor = RowInteraction(
            embed_dim=embed_dim,
            num_blocks=n_layers,
            nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            num_cls=num_cls,
        )

        # --- Custom regression head stack (always reinitialised) ---
        self.label_injector   = LabelInjector(d_model=d_model, k=k)
        self.icl_attention    = ICLAttention(d_model=d_model, n_heads=n_heads)
        self.shared_trunk     = SharedTrunk(d_model=d_model)
        self.pseudotime_head  = PseudotimeHead(d_model=d_model)
        self.composition_head = CompositionHead(d_model=d_model, k=k)

        self._embed_dim = embed_dim
        self._d_model   = d_model
        self._k         = k

    def forward(self, batch: "ICLBatch") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch : ICLBatch
            context_expression  (B, n_anchors, n_genes)
            context_pseudotime  (B, n_anchors)
            context_soft_labels (B, n_anchors, K)
            query_expression    (B, n_genes)
            query_pseudotime    (B,)          — NOT used in forward (evaluation only)
            query_soft_labels   (B, K)        — NOT used in forward (evaluation only)

        Returns
        -------
        pt_pred   : (B,)   pseudotime predictions ∈ (0, 1)
        comp_pred : (B, K) composition predictions, rows sum to 1.0
        """
        # 1. Stack query after anchors → (B, n_cells, n_genes)
        query_expr = batch.query_expression.unsqueeze(1)
        all_expr   = torch.cat([batch.context_expression, query_expr], dim=1)

        # 2. Column-wise embedding — anchor pseudotime passed as y_train so
        #    col_embedder can distinguish anchor vs query positions internally.
        #    target_aware=False so y_train values are not used for conditioning.
        emb = self.col_embedder(all_expr, batch.context_pseudotime)
        # emb: (B, n_cells, seq_len, embed_dim)

        # 3. Row interaction — transformer over gene dimension within each cell
        x = self.row_interactor(emb)
        # x: (B, n_cells, d_model)

        # 4. Label injection — add pseudotime + soft_labels to anchor rows only
        x = self.label_injector(
            x,
            anchor_pseudotime=batch.context_pseudotime,
            anchor_soft_labels=batch.context_soft_labels,
        )

        # 5. ICL attention — query cross-attends all anchors jointly
        x = self.icl_attention(x)    # (B, d_model)

        # 6. Shared trunk
        x = self.shared_trunk(x)     # (B, d_model)

        # 7. Dual output heads
        pt_pred   = self.pseudotime_head(x)   # (B,)
        comp_pred = self.composition_head(x)  # (B, K)

        return pt_pred, comp_pred

    def load_backbone(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load pretrained TabICLv2 weights into col_embedder and row_interactor.

        Parameters
        ----------
        checkpoint_path : str
            Path to a .ckpt file from the TabICLv2 training run.
        strict : bool
            Passed to load_state_dict. False (default) skips head layers that
            are absent from the checkpoint.
        """
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        # Filter to backbone keys only
        backbone_state = {
            k: v for k, v in state.items()
            if k.startswith("col_embedder.") or k.startswith("row_interactor.")
        }
        self.load_state_dict(backbone_state, strict=strict)

    def parameter_groups(self) -> list[dict]:
        """Four parameter groups keyed by learning-rate schedule.

        Returns a list of dicts compatible with torch.optim constructors:
          {"name": str, "params": list[Parameter], "lr": float}
        """
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
                "params": list(self.icl_attention.parameters()),
                "lr":     cfg.lr_icl,
            },
            {
                "name":   "head",
                "params": (
                    list(self.label_injector.parameters())
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
    """Extracts gene-level similarity scores from col_embedder after a forward pass.

    TabICLv2's ColEmbedding processes each gene as a separate token via a Set
    Transformer (ISAB). After col_embedder, we have per-cell, per-gene embeddings
    of shape (B, n_cells, n_genes, embed_dim). A cosine similarity matrix over
    the gene dimension gives a (B, n_cells, n_genes, n_genes) relationship matrix
    that serves as the gene-gene interaction proxy for GRN explainability.

    This is complementary to (not a replacement for) Integrated Hessians — the
    ISAB weights are not directly accessible as a clean gene×gene matrix, but the
    cosine similarity of final gene embeddings captures effective co-attention.

    Usage
    -----
    scorer = AttentionScorer(model)
    model(batch)
    scores = scorer.extract()   # (B, n_cells, n_genes, n_genes)
    """

    def __init__(self, model: TabICLRegressor) -> None:
        self._model = model
        self._gene_embeddings: torch.Tensor | None = None
        self._hook = model.col_embedder.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, input, output: torch.Tensor) -> None:
        # output: (B, n_cells, seq_len, embed_dim)
        # seq_len may include CLS tokens prepended by ColEmbedding;
        # the last n_genes positions correspond to gene tokens.
        n_genes = input[0].shape[-1]   # X has shape (B, n_cells, n_genes)
        # Take the last n_genes tokens (gene positions), detach for post-hoc use
        self._gene_embeddings = output[:, :, -n_genes:, :].detach()

    def extract(self) -> torch.Tensor:
        """Return cosine similarity matrix over gene embeddings (detached).

        Returns
        -------
        scores : (B, n_cells, n_genes, n_genes)
            Entry [b, c, i, j] is the cosine similarity between the embeddings
            of gene i and gene j in cell c of batch item b. Values ∈ [-1, 1].
        """
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
