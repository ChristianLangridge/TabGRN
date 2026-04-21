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
                  Input:  (B, n_cells, n_genes)
                  Output: (B, n_cells, seq_len, embed_dim)
                  Note: target_aware=False — y_train arg is accepted but ignored.

  row_interactor  tabicl.model.interaction.RowInteraction
                  Transformer encoder with rotary positional encoding over the
                  feature (gene) dimension within each cell.
                  Input:  (B, n_cells, seq_len, embed_dim)
                  Output: (B, n_cells, d_model)  where d_model = num_cls × embed_dim

Custom regression head stack (not present in TabICLv2)
------------------------------------------------------
  AnchorLabelEmbedder
                  Adds pseudotime + soft_label projections to anchor rows only.
                  Query row is never modified (no label leakage).
                  Injection deferred to post-row-interaction to preserve clean
                  feature-feature attention for GRN extraction via AttentionScorer.

  tf_icl          tabicl.model.encoders.Encoder — the pretrained ICL transformer
                  from TabICLv2's icl_predictor.tf_icl sub-module.
                  Runs bidirectional attention within anchors; query position
                  attends to anchor context only (train_size masking).
                  Input:  (B, n_cells, d_model), train_size=n_anchors
                  Output: (B, n_cells, d_model)  → we take [:, -1, :] (query row)

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
  icl   tf_icl           lr_icl = 5e-5  [pretrained; frozen for warmup_icl_steps]
  head  anchor_label_embedder + shared_trunk + pseudotime_head + composition_head
                         lr_head = 1e-3 [always reinitialised]

Checkpoint loading
------------------
Load a pretrained TabICL checkpoint with strict=False:
  - col_embedder.* and row_interactor.* keys match directly.
  - icl_predictor.tf_icl.* keys are remapped to tf_icl.* for our tf_icl Encoder.
  - Head layers are absent from the checkpoint and are skipped automatically.
"""
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
    """Initialise a Linear layer with small-normal weights and a constant bias.

    Used by PseudotimeHead and CompositionHead to keep both output heads at a
    near-uniform prior at the start of training, avoiding premature gradient
    dominance by either task.
    """
    nn.init.normal_(layer.weight, std=weight_std)
    nn.init.constant_(layer.bias, bias_value)


# ---------------------------------------------------------------------------
# AnchorLabelEmbedder
# ---------------------------------------------------------------------------

class AnchorLabelEmbedder(nn.Module):
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
        _init_linear(self.linear, weight_std=init_std, bias_value=init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x).squeeze(-1))


# ---------------------------------------------------------------------------
# CompositionHead
# ---------------------------------------------------------------------------

class CompositionHead(nn.Module):
    """Linear → softmax over K cell states.

    Input:  (B, d_model)
    Output: (B, K)  rows sum to 1.0

    Loss: KL(target ∥ pred) where target is the distance-to-centroid softmax
    from ProcessedDataset. Dirichlet NLL planned for a later phase once the
    rotation fine-tune baseline is established (TDD v1.3.0 interim).
    """

    def __init__(self, d_model: int, k: int, init_std: float = 0.01) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, k)
        _init_linear(self.linear, weight_std=init_std, bias_value=0.0)

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
        Number of attention heads in backbone and tf_icl.
    n_layers : int
        Number of transformer blocks in ColEmbedding, RowInteraction, and tf_icl.
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

        # Pretrained ICL transformer from TabICLv2's icl_predictor.tf_icl.
        # Loaded via load_backbone with key remapping icl_predictor.tf_icl.* → tf_icl.*
        self.tf_icl = Encoder(
            num_blocks=n_layers,
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
        )

        # --- Custom regression head stack (always reinitialised) ---
        self.anchor_label_embedder   = AnchorLabelEmbedder(d_model=d_model, k=k)
        self.shared_trunk     = SharedTrunk(d_model=d_model)
        self.pseudotime_head  = PseudotimeHead(d_model=d_model)
        self.composition_head = CompositionHead(d_model=d_model, k=k)

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
        n_anchors = batch.context_expression.shape[1]

        # 1. Stack query after anchors → (B, n_cells, n_genes)
        query_expr = batch.query_expression.unsqueeze(1)
        all_expr   = torch.cat([batch.context_expression, query_expr], dim=1)

        # 2. Column-wise embedding — target_aware=False so y_train values are
        #    never used for class conditioning. However ColEmbedding always reads
        #    y_train.shape[1] to determine train_size for inducing-point attention
        #    masking. We pass context_pseudotime so shape[1] == n_anchors.
        emb = self.col_embedder(all_expr, batch.context_pseudotime)
        # emb: (B, n_cells, seq_len, embed_dim)

        # 3. Row interaction — transformer over gene dimension within each cell.
        #    RowInteraction._train_forward does an in-place write to replace the
        #    reserved CLS-token slots with learned CLS embeddings.  Passing a
        #    clone ensures that write lands on the clone's version counter, not
        #    on emb's, so autograd can still backprop through col_embedder.
        x = self.row_interactor(emb.clone())
        # x: (B, n_cells, d_model)

        # 4. Label injection — add pseudotime + soft_labels to anchor rows only.
        #    Deferred to post-row-interaction so col_embedder sees clean gene
        #    features without label contamination (preserves GRN signal quality).
        x = self.anchor_label_embedder(
            x,
            anchor_pseudotime=batch.context_pseudotime,
            anchor_soft_labels=batch.context_soft_labels,
        )

        # 5. ICL transformer — pretrained Encoder from TabICLv2's icl_predictor.
        #    train_size masking: query attends to anchors only (ICL protocol).
        x = self.tf_icl(x, train_size=n_anchors)  # (B, n_cells, d_model)
        x = x[:, -1, :]                            # query row → (B, d_model)

        # 6. Shared trunk
        x = self.shared_trunk(x)     # (B, d_model)

        # 7. Dual output heads
        pt_pred   = self.pseudotime_head(x)   # (B,)
        comp_pred = self.composition_head(x)  # (B, K)

        return pt_pred, comp_pred

    def load_backbone(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load pretrained TabICLv2 weights into col_embedder, row_interactor, and tf_icl.

        Parameters
        ----------
        checkpoint_path : str
            Path to a .ckpt file from the TabICLv2 training run.
        strict : bool
            Passed to load_state_dict. False (default) skips head layers that
            are absent from the checkpoint.

        Key remapping
        -------------
        col_embedder.*           → col_embedder.*       (direct match)
        row_interactor.*         → row_interactor.*     (direct match)
        icl_predictor.tf_icl.*   → tf_icl.*             (strip icl_predictor. prefix)
        """
        state = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]

        backbone_state: dict = {}
        for k, v in state.items():
            if k.startswith("col_embedder.") or k.startswith("row_interactor."):
                backbone_state[k] = v
            elif k.startswith("icl_predictor.tf_icl."):
                # Remap pretrained ICL transformer into our tf_icl sub-module
                backbone_state[k[len("icl_predictor."):]] = v

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
    """Extracts gene-level similarity scores from col_embedder after a forward pass.

    TabICLv2's ColEmbedding processes each gene as a separate token via a Set
    Transformer (ISAB). After col_embedder, we have per-cell, per-gene embeddings
    of shape (B, n_cells, n_genes, embed_dim). A cosine similarity matrix over
    the gene dimension gives a (B, n_cells, n_genes, n_genes) relationship matrix
    that serves as the gene-gene interaction proxy for GRN explainability.

    This is complementary to (not a replacement for) Integrated Hessians — the
    ISAB weights are not directly accessible as a clean gene×gene matrix, but the
    cosine similarity of final gene embeddings captures effective co-attention.

    Warning
    -------
    With TabICLv2's feature grouping enabled (feature_group != False), each token
    position j represents a linear projection of a triplet of genes (j, j+1, j+2)
    mod n_genes (circular permutation). In that setting cosine similarity at
    positions (i, j) reflects triplet-context similarity, not atomic gene-gene
    similarity. Our model uses feature_group=False (default), so each token maps
    1:1 to one gene — this warning applies if you enable feature grouping downstream.
    For publication-grade GRN edges use Integrated Gradients (primary) and
    in-silico perturbation (causal validation); this scorer is a fast diagnostic.

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
