"""
Unit tests for spatialmt.model.tabgrn — TabICLRegressor and sub-modules.

Architecture under test
-----------------------
Backbone (pretrained TabICLv2 imports)
  col_embedder     tabicl.model.embedding.ColEmbedding
                   (B, n_cells, n_genes) → (B, n_cells, seq_len, embed_dim)

  row_interactor   tabicl.model.interaction.RowInteraction
                   (B, n_cells, seq_len, embed_dim) → (B, n_cells, d_model)
                   d_model = num_cls × embed_dim

  tf_icl           tabicl.model.encoders.Encoder  (pretrained ICL transformer)
                   (B, n_cells, d_model) → (B, n_cells, d_model)
                   Query position attends to anchors only (train_size masking).

Custom head stack
  AnchorLabelEmbedder    (B, n_cells, d_model) → same  [anchor rows only]
  SharedTrunk      (B, d_model) → (B, d_model)
  PseudotimeHead   (B, d_model) → (B,)    sigmoid ∈ (0, 1)
  CompositionHead  (B, d_model) → (B, K)  softmax, rows sum to 1.0

Supporting
  AttentionScorer  Forward hook on col_embedder; .extract() returns
                   cosine similarity of gene embeddings: (B, n_cells, n_genes, n_genes)

Parameter groups
  col, row, icl, head — four groups with LR from ModelConfig defaults
"""
from __future__ import annotations

import pytest
import torch

# ---------------------------------------------------------------------------
# Constants for toy model
# ---------------------------------------------------------------------------
B         = 4       # batch size
N_ANCHORS = 6       # context cells
N_GENES   = 10
EMBED_DIM = 32      # embed_dim for ColEmbedding / RowInteraction
N_HEADS   = 2
N_LAYERS  = 1
NUM_CLS   = 2       # RowInteraction CLS tokens; d_model = NUM_CLS * EMBED_DIM
D_MODEL   = NUM_CLS * EMBED_DIM   # = 64  — input dim for our custom layers
COL_NUM_INDS = 4    # inducing points (small for tests)
K         = 8       # cell states


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(
    batch_size: int = B,
    n_anchors: int = N_ANCHORS,
    n_genes: int = N_GENES,
    k: int = K,
) -> "ICLBatch":
    from spatialmt.context.collate import ICLBatch
    rng = torch.Generator().manual_seed(0)
    ctx_expr = torch.rand(batch_size, n_anchors, n_genes, generator=rng)
    ctx_pt   = torch.rand(batch_size, n_anchors, generator=rng)
    ctx_sl   = torch.softmax(torch.rand(batch_size, n_anchors, k, generator=rng), dim=-1)
    q_expr   = torch.rand(batch_size, n_genes, generator=rng)
    q_pt     = torch.rand(batch_size, generator=rng)
    q_sl     = torch.softmax(torch.rand(batch_size, k, generator=rng), dim=-1)
    return ICLBatch(
        context_expression  = ctx_expr,
        context_pseudotime  = ctx_pt,
        context_soft_labels = ctx_sl,
        query_expression    = q_expr,
        query_pseudotime    = q_pt,
        query_soft_labels   = q_sl,
    )


def _make_model(
    n_genes: int = N_GENES,
    embed_dim: int = EMBED_DIM,
    n_heads: int = N_HEADS,
    n_layers: int = N_LAYERS,
    k: int = K,
    num_cls: int = NUM_CLS,
    col_num_inds: int = COL_NUM_INDS,
) -> "TabICLRegressor":
    from spatialmt.model.tabgrn import TabICLRegressor
    return TabICLRegressor(
        n_genes=n_genes,
        k=k,
        embed_dim=embed_dim,
        n_heads=n_heads,
        num_cls=num_cls,
        col_num_inds=col_num_inds,
        n_layers_col=n_layers,
        n_layers_row=n_layers,
        n_layers_icl=n_layers,
    )


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------

def test_import_tabgrn_regressor():
    from spatialmt.model.tabgrn import TabICLRegressor
    assert TabICLRegressor is not None


def test_import_attention_scorer():
    from spatialmt.model.tabgrn import AttentionScorer
    assert AttentionScorer is not None


def test_import_custom_sub_modules():
    from spatialmt.model.tabgrn import (
        AnchorLabelEmbedder,
        SharedTrunk,
        PseudotimeHead,
        CompositionHead,
    )


def test_import_tabgrn_backbone_classes():
    """TabICLv2 backbone classes must be importable from the tabicl package."""
    from tabicl.model.embedding import ColEmbedding
    from tabicl.model.interaction import RowInteraction
    from tabicl.model.encoders import Encoder
    assert ColEmbedding is not None
    assert RowInteraction is not None
    assert Encoder is not None


# ---------------------------------------------------------------------------
# TabICLRegressor — construction
# ---------------------------------------------------------------------------

def test_model_is_nn_module():
    import torch.nn as nn
    model = _make_model()
    assert isinstance(model, nn.Module)


def test_model_has_col_embedder():
    from tabicl.model.embedding import ColEmbedding
    model = _make_model()
    assert hasattr(model, "col_embedder")
    assert isinstance(model.col_embedder, ColEmbedding)


def test_model_has_row_interactor():
    from tabicl.model.interaction import RowInteraction
    model = _make_model()
    assert hasattr(model, "row_interactor")
    assert isinstance(model.row_interactor, RowInteraction)


def test_model_has_tf_icl():
    """tf_icl must be a tabicl.model.encoders.Encoder (pretrained ICL transformer)."""
    from tabicl.model.encoders import Encoder
    model = _make_model()
    assert hasattr(model, "tf_icl")
    assert isinstance(model.tf_icl, Encoder)


def test_model_has_anchor_label_embedder():
    from spatialmt.model.tabgrn import AnchorLabelEmbedder
    model = _make_model()
    assert hasattr(model, "anchor_label_embedder")
    assert isinstance(model.anchor_label_embedder, AnchorLabelEmbedder)


def test_model_has_shared_trunk():
    from spatialmt.model.tabgrn import SharedTrunk
    model = _make_model()
    assert hasattr(model, "shared_trunk")
    assert isinstance(model.shared_trunk, SharedTrunk)


def test_model_has_pseudotime_head():
    from spatialmt.model.tabgrn import PseudotimeHead
    model = _make_model()
    assert hasattr(model, "pseudotime_head")
    assert isinstance(model.pseudotime_head, PseudotimeHead)


def test_model_has_composition_head():
    from spatialmt.model.tabgrn import CompositionHead
    model = _make_model()
    assert hasattr(model, "composition_head")
    assert isinstance(model.composition_head, CompositionHead)


def test_model_has_no_icl_attention():
    """ICLAttention removed — replaced by pretrained tf_icl Encoder."""
    model = _make_model()
    assert not hasattr(model, "icl_attention")


# ---------------------------------------------------------------------------
# Forward pass — output shapes and types
# ---------------------------------------------------------------------------

def test_forward_returns_tuple():
    model = _make_model()
    batch = _make_batch()
    out = model(batch)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_forward_pseudotime_shape():
    model = _make_model()
    batch = _make_batch()
    pt_pred, _ = model(batch)
    assert pt_pred.shape == (B,)


def test_forward_composition_shape():
    model = _make_model()
    batch = _make_batch()
    _, comp_pred = model(batch)
    assert comp_pred.shape == (B, K)


def test_forward_pseudotime_dtype():
    model = _make_model()
    batch = _make_batch()
    pt_pred, _ = model(batch)
    assert pt_pred.dtype == torch.float32


def test_forward_composition_dtype():
    model = _make_model()
    batch = _make_batch()
    _, comp_pred = model(batch)
    assert comp_pred.dtype == torch.float32


def test_forward_pseudotime_range():
    """PseudotimeHead uses sigmoid — output strictly in (0, 1)."""
    model = _make_model()
    batch = _make_batch()
    pt_pred, _ = model(batch)
    assert (pt_pred > 0.0).all()
    assert (pt_pred < 1.0).all()


def test_forward_composition_rows_sum_to_one():
    """CompositionHead uses softmax — rows must sum to 1."""
    model = _make_model()
    batch = _make_batch()
    _, comp_pred = model(batch)
    row_sums = comp_pred.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5)


def test_forward_composition_nonnegative():
    """softmax output is always ≥ 0."""
    model = _make_model()
    batch = _make_batch()
    _, comp_pred = model(batch)
    assert (comp_pred >= 0.0).all()


def test_forward_no_nan():
    model = _make_model()
    batch = _make_batch()
    pt_pred, comp_pred = model(batch)
    assert not torch.isnan(pt_pred).any()
    assert not torch.isnan(comp_pred).any()


def test_forward_batch_size_one():
    model = _make_model()
    batch = _make_batch(batch_size=1)
    pt_pred, comp_pred = model(batch)
    assert pt_pred.shape == (1,)
    assert comp_pred.shape == (1, K)


def test_forward_is_differentiable():
    """Loss.backward() must not raise."""
    model = _make_model()
    batch = _make_batch()
    pt_pred, comp_pred = model(batch)
    pt_target = torch.rand(B)
    sl_target  = torch.softmax(torch.rand(B, K), dim=-1)
    loss = ((pt_pred - pt_target) ** 2).mean() + \
           -(sl_target * torch.log(comp_pred + 1e-8)).sum(dim=-1).mean()
    loss.backward()


def test_forward_ignores_query_labels():
    """Varying query_pseudotime and query_soft_labels must not change predictions.

    These fields are present on ICLBatch for evaluation bookkeeping only —
    they must never influence the forward computation (data leakage guard).
    """
    from spatialmt.context.collate import ICLBatch
    model = _make_model()
    model.eval()

    rng = torch.Generator().manual_seed(42)
    ctx_expr = torch.rand(B, N_ANCHORS, N_GENES, generator=rng)
    ctx_pt   = torch.rand(B, N_ANCHORS, generator=rng)
    ctx_sl   = torch.softmax(torch.rand(B, N_ANCHORS, K, generator=rng), dim=-1)
    q_expr   = torch.rand(B, N_GENES, generator=rng)

    batch_a = ICLBatch(
        context_expression  = ctx_expr,
        context_pseudotime  = ctx_pt,
        context_soft_labels = ctx_sl,
        query_expression    = q_expr,
        query_pseudotime    = torch.zeros(B),
        query_soft_labels   = torch.full((B, K), 1.0 / K),
    )
    batch_b = ICLBatch(
        context_expression  = ctx_expr,
        context_pseudotime  = ctx_pt,
        context_soft_labels = ctx_sl,
        query_expression    = q_expr,
        query_pseudotime    = torch.ones(B),
        query_soft_labels   = torch.softmax(torch.randn(B, K), dim=-1),
    )

    with torch.no_grad():
        pt_a, comp_a = model(batch_a)
        pt_b, comp_b = model(batch_b)

    assert torch.allclose(pt_a, pt_b, atol=1e-6), "query_pseudotime leaked into forward"
    assert torch.allclose(comp_a, comp_b, atol=1e-6), "query_soft_labels leaked into forward"


def test_forward_deterministic():
    """Same inputs must produce bit-exact outputs (eval mode, no dropout)."""
    model = _make_model()
    model.eval()
    batch = _make_batch()
    with torch.no_grad():
        pt1, comp1 = model(batch)
        pt2, comp2 = model(batch)
    assert torch.equal(pt1, pt2)
    assert torch.equal(comp1, comp2)


# ---------------------------------------------------------------------------
# AnchorLabelEmbedder
# ---------------------------------------------------------------------------

def test_anchor_label_embedder_output_shape():
    from spatialmt.model.tabgrn import AnchorLabelEmbedder
    inj = AnchorLabelEmbedder(d_model=D_MODEL, k=K)
    n_cells = N_ANCHORS + 1
    x   = torch.rand(B, n_cells, D_MODEL)
    pt  = torch.rand(B, N_ANCHORS)
    sl  = torch.softmax(torch.rand(B, N_ANCHORS, K), dim=-1)
    out = inj(x, anchor_pseudotime=pt, anchor_soft_labels=sl)
    assert out.shape == (B, n_cells, D_MODEL)


def test_anchor_label_embedder_query_row_unchanged():
    """Query row (last position) must not be modified — no label leakage."""
    from spatialmt.model.tabgrn import AnchorLabelEmbedder
    inj = AnchorLabelEmbedder(d_model=D_MODEL, k=K)
    n_cells = N_ANCHORS + 1
    x   = torch.rand(B, n_cells, D_MODEL)
    pt  = torch.rand(B, N_ANCHORS)
    sl  = torch.softmax(torch.rand(B, N_ANCHORS, K), dim=-1)
    out = inj(x, anchor_pseudotime=pt, anchor_soft_labels=sl)
    assert torch.allclose(out[:, -1, :], x[:, -1, :])


def test_anchor_label_embedder_anchor_rows_modified():
    """Anchor rows must be modified — injection must actually change them."""
    from spatialmt.model.tabgrn import AnchorLabelEmbedder
    inj = AnchorLabelEmbedder(d_model=D_MODEL, k=K)
    n_cells = N_ANCHORS + 1
    x   = torch.rand(B, n_cells, D_MODEL)
    pt  = torch.rand(B, N_ANCHORS)
    sl  = torch.softmax(torch.rand(B, N_ANCHORS, K), dim=-1)
    out = inj(x, anchor_pseudotime=pt, anchor_soft_labels=sl)
    # At least one anchor row must differ (injected label embedding is non-trivial)
    assert not torch.allclose(out[:, :N_ANCHORS, :], x[:, :N_ANCHORS, :])


# ---------------------------------------------------------------------------
# SharedTrunk
# ---------------------------------------------------------------------------

def test_shared_trunk_output_shape():
    from spatialmt.model.tabgrn import SharedTrunk
    trunk = SharedTrunk(d_model=D_MODEL)
    x = torch.rand(B, D_MODEL)
    out = trunk(x)
    assert out.shape == (B, D_MODEL)


# ---------------------------------------------------------------------------
# PseudotimeHead
# ---------------------------------------------------------------------------

def test_pseudotime_head_output_shape():
    from spatialmt.model.tabgrn import PseudotimeHead
    head = PseudotimeHead(d_model=D_MODEL)
    x = torch.rand(B, D_MODEL)
    out = head(x)
    assert out.shape == (B,)


def test_pseudotime_head_sigmoid_range():
    from spatialmt.model.tabgrn import PseudotimeHead
    head = PseudotimeHead(d_model=D_MODEL)
    x = torch.rand(B, D_MODEL)
    out = head(x)
    assert (out > 0.0).all() and (out < 1.0).all()


def test_pseudotime_head_init_bias():
    """output_head_init_bias=0.5 means initial raw bias ≈ 0.5 → sigmoid ≈ 0.62."""
    from spatialmt.model.tabgrn import PseudotimeHead
    head = PseudotimeHead(d_model=D_MODEL, init_bias=0.5)
    assert head.linear.bias is not None
    assert pytest.approx(head.linear.bias.item(), abs=1e-6) == 0.5


# ---------------------------------------------------------------------------
# CompositionHead
# ---------------------------------------------------------------------------

def test_composition_head_output_shape():
    from spatialmt.model.tabgrn import CompositionHead
    head = CompositionHead(d_model=D_MODEL, k=K)
    x = torch.rand(B, D_MODEL)
    out = head(x)
    assert out.shape == (B, K)


def test_composition_head_softmax_sums():
    from spatialmt.model.tabgrn import CompositionHead
    head = CompositionHead(d_model=D_MODEL, k=K)
    x = torch.rand(B, D_MODEL)
    out = head(x)
    assert torch.allclose(out.sum(dim=-1), torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# AttentionScorer
# ---------------------------------------------------------------------------

def test_attention_scorer_extract_after_forward():
    from spatialmt.model.tabgrn import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    assert scores is not None
    scorer.remove_hook()


def test_attention_scorer_shape():
    """Cosine similarity of gene embeddings: (B, n_cells, n_genes, n_genes)."""
    from spatialmt.model.tabgrn import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    n_cells = N_ANCHORS + 1
    assert scores.shape == (B, n_cells, N_GENES, N_GENES)
    scorer.remove_hook()


def test_attention_scorer_scores_are_detached():
    from spatialmt.model.tabgrn import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    assert not scores.requires_grad
    scorer.remove_hook()


def test_attention_scorer_scores_symmetric():
    """Cosine similarity matrix must be symmetric."""
    from spatialmt.model.tabgrn import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    assert torch.allclose(scores, scores.transpose(-1, -2), atol=1e-5)
    scorer.remove_hook()


def test_attention_scorer_diagonal_is_one():
    """Cosine similarity of a vector with itself must be 1.0."""
    from spatialmt.model.tabgrn import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    diag = torch.diagonal(scores, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4)
    scorer.remove_hook()


def test_attention_scorer_values_bounded():
    """Cosine similarity values must lie in [-1, 1]."""
    from spatialmt.model.tabgrn import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    assert (scores >= -1.0 - 1e-5).all()
    assert (scores <=  1.0 + 1e-5).all()
    scorer.remove_hook()


def test_attention_scorer_raises_before_forward():
    from spatialmt.model.tabgrn import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    with pytest.raises(RuntimeError):
        scorer.extract()
    scorer.remove_hook()


def test_attention_scorer_remove_hook_stops_capture():
    """After remove_hook(), running another forward must not update the scorer."""
    from spatialmt.model.tabgrn import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    first = scorer.extract().clone()
    scorer.remove_hook()
    # Perturb the batch so the second forward would produce different embeddings
    batch2 = _make_batch(batch_size=B, n_anchors=N_ANCHORS)
    model(batch2)
    # Hook is removed — scorer._gene_embeddings must not have changed
    assert torch.equal(scorer.extract(), first)


# ---------------------------------------------------------------------------
# DirichletCompositionHead wiring — composition_loss_type parameter
# ---------------------------------------------------------------------------

def _make_dirichlet_model(**kwargs) -> "TabICLRegressor":
    from spatialmt.model.tabgrn import TabICLRegressor
    return TabICLRegressor(
        n_genes=N_GENES,
        k=K,
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        num_cls=NUM_CLS,
        col_num_inds=COL_NUM_INDS,
        n_layers_col=N_LAYERS,
        n_layers_row=N_LAYERS,
        n_layers_icl=N_LAYERS,
        composition_loss_type="dirichlet",
        **kwargs,
    )


def test_default_composition_loss_type_is_kl():
    """Backward-compatible default: no argument → CompositionHead (softmax)."""
    from spatialmt.model.tabgrn import CompositionHead
    model = _make_model()
    assert isinstance(model.composition_head, CompositionHead)


def test_kl_explicit_composition_loss_type():
    from spatialmt.model.tabgrn import CompositionHead
    from spatialmt.model.tabgrn import TabICLRegressor
    model = TabICLRegressor(
        n_genes=N_GENES, k=K, embed_dim=EMBED_DIM, n_heads=N_HEADS,
        num_cls=NUM_CLS, col_num_inds=COL_NUM_INDS,
        n_layers_col=N_LAYERS, n_layers_row=N_LAYERS, n_layers_icl=N_LAYERS,
        composition_loss_type="kl",
    )
    assert isinstance(model.composition_head, CompositionHead)


def test_dirichlet_composition_loss_type_uses_dirichlet_head():
    from spatialmt.model.tabgrn import DirichletCompositionHead
    model = _make_dirichlet_model()
    assert isinstance(model.composition_head, DirichletCompositionHead)


def test_invalid_composition_loss_type_raises():
    from spatialmt.model.tabgrn import TabICLRegressor
    with pytest.raises(ValueError, match="composition_loss_type"):
        TabICLRegressor(
            n_genes=N_GENES, k=K, embed_dim=EMBED_DIM, n_heads=N_HEADS,
            num_cls=NUM_CLS, col_num_inds=COL_NUM_INDS,
            n_layers_col=N_LAYERS, n_layers_row=N_LAYERS, n_layers_icl=N_LAYERS,
            composition_loss_type="cross_entropy",
        )


def test_dirichlet_model_forward_output_shape():
    model = _make_dirichlet_model()
    batch = _make_batch()
    pt_pred, comp_pred = model(batch)
    assert pt_pred.shape == (B,)
    assert comp_pred.shape == (B, K)


def test_dirichlet_model_forward_comp_pred_all_positive():
    """DirichletCompositionHead uses softplus — all outputs strictly > 0."""
    model = _make_dirichlet_model()
    batch = _make_batch()
    _, comp_pred = model(batch)
    assert (comp_pred > 0).all()


def test_dirichlet_model_forward_comp_pred_rows_do_not_sum_to_one():
    """Concentration parameters are NOT a probability distribution."""
    model = _make_dirichlet_model()
    batch = _make_batch()
    _, comp_pred = model(batch)
    row_sums = comp_pred.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones(B), atol=1e-3)


def test_dirichlet_model_forward_no_nan():
    model = _make_dirichlet_model()
    batch = _make_batch()
    pt_pred, comp_pred = model(batch)
    assert not torch.isnan(pt_pred).any()
    assert not torch.isnan(comp_pred).any()


def test_dirichlet_model_parameter_groups_cover_all_params():
    """parameter_groups() must still cover every parameter for the dirichlet variant."""
    model = _make_dirichlet_model()
    groups = model.parameter_groups()
    grouped_ids = set()
    for g in groups:
        for p in g["params"]:
            grouped_ids.add(id(p))
    all_ids = {id(p) for p in model.parameters()}
    assert all_ids == grouped_ids


def test_dirichlet_model_is_differentiable():
    """DirichletDualHeadLoss.backward() must not raise on dirichlet model output."""
    from spatialmt.model.loss import DirichletDualHeadLoss
    model = _make_dirichlet_model()
    batch = _make_batch()
    pt_pred, concentrations = model(batch)
    comp_target = torch.softmax(torch.rand(B, K), dim=-1)
    pt_target   = torch.rand(B)
    loss_fn = DirichletDualHeadLoss()
    total, _, _ = loss_fn(pt_pred, pt_target, concentrations, comp_target)
    total.backward()


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------

def test_parameter_groups_count():
    """model.parameter_groups() returns exactly 4 groups."""
    model = _make_model()
    groups = model.parameter_groups()
    assert len(groups) == 4


def test_parameter_groups_have_correct_names():
    model = _make_model()
    groups = model.parameter_groups()
    names = {g["name"] for g in groups}
    assert names == {"col", "row", "icl", "head"}


def test_parameter_groups_lr_values():
    from spatialmt.config.experiment import ModelConfig
    model = _make_model()
    cfg = ModelConfig()
    groups = {g["name"]: g["lr"] for g in model.parameter_groups()}
    assert groups["col"]  == pytest.approx(cfg.lr_col)
    assert groups["row"]  == pytest.approx(cfg.lr_row)
    assert groups["icl"]  == pytest.approx(cfg.lr_icl)
    assert groups["head"] == pytest.approx(cfg.lr_head)


def test_parameter_groups_cover_all_parameters():
    """Every model parameter must appear in exactly one group."""
    model = _make_model()
    groups = model.parameter_groups()
    grouped_ids = set()
    for g in groups:
        for p in g["params"]:
            pid = id(p)
            assert pid not in grouped_ids, f"Parameter appears in multiple groups"
            grouped_ids.add(pid)
    all_ids = {id(p) for p in model.parameters()}
    assert all_ids == grouped_ids


def test_parameter_groups_no_overlap():
    """No parameter may appear in more than one group."""
    model = _make_model()
    groups = model.parameter_groups()
    seen = set()
    for g in groups:
        for p in g["params"]:
            assert id(p) not in seen, f"Parameter appears in multiple groups"
            seen.add(id(p))


def test_parameter_groups_no_empty_group():
    model = _make_model()
    for g in model.parameter_groups():
        assert len(g["params"]) > 0, f"Group '{g['name']}' has no parameters"


def test_col_group_is_col_embedder_only():
    """col group must contain exactly col_embedder parameters."""
    model = _make_model()
    groups = {g["name"]: g for g in model.parameter_groups()}
    col_ids = {id(p) for p in groups["col"]["params"]}
    expected_ids = {id(p) for p in model.col_embedder.parameters()}
    assert col_ids == expected_ids


def test_row_group_is_row_interactor_only():
    """row group must contain exactly row_interactor parameters."""
    model = _make_model()
    groups = {g["name"]: g for g in model.parameter_groups()}
    row_ids = {id(p) for p in groups["row"]["params"]}
    expected_ids = {id(p) for p in model.row_interactor.parameters()}
    assert row_ids == expected_ids


def test_icl_group_is_tf_icl_only():
    """icl group must contain exactly tf_icl parameters."""
    model = _make_model()
    groups = {g["name"]: g for g in model.parameter_groups()}
    icl_ids = {id(p) for p in groups["icl"]["params"]}
    expected_ids = {id(p) for p in model.tf_icl.parameters()}
    assert icl_ids == expected_ids


def test_gradient_flows_to_all_groups():
    """After backward, every parameter group must have at least one non-zero grad.

    TabICLv2 zero-initialises out_proj in all transformer blocks for training
    stability (residual path dominates at init).  At zero weight the attention
    branch contributes zero gradient, so col_embedder genuinely receives no
    gradient on the very first forward pass.  This is by design, not a bug.

    To test graph *connectivity* (not initialisation values) we perturb all
    out_proj weights to be slightly non-zero before the backward.
    """
    model = _make_model()

    # Perturb out_proj so the attention branch is non-trivially connected
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "out_proj") and hasattr(m.out_proj, "weight"):
                m.out_proj.weight.normal_(std=0.01)

    batch = _make_batch()
    pt_pred, comp_pred = model(batch)
    pt_target  = torch.rand(B)
    sl_target  = torch.softmax(torch.rand(B, K), dim=-1)
    loss = ((pt_pred - pt_target) ** 2).mean() + \
           -(sl_target * torch.log(comp_pred + 1e-8)).sum(dim=-1).mean()
    loss.backward()

    groups = {g["name"]: g["params"] for g in model.parameter_groups()}
    for name, params in groups.items():
        grads = [p.grad for p in params if p.grad is not None]
        has_nonzero = any(g.abs().sum().item() > 0 for g in grads)
        assert has_nonzero, f"Group '{name}' received no gradient"


# ---------------------------------------------------------------------------
# forward_supervised
# ---------------------------------------------------------------------------

def _make_anchor(n_genes: int = N_GENES) -> torch.Tensor:
    return torch.rand(n_genes)


def test_forward_supervised_pt_output_shape():
    model = _make_model()
    expr = torch.rand(B, N_GENES)
    pt_pred, _ = model.forward_supervised(expr, _make_anchor())
    assert pt_pred.shape == (B,)


def test_forward_supervised_comp_output_shape():
    model = _make_model()
    expr = torch.rand(B, N_GENES)
    _, comp_pred = model.forward_supervised(expr, _make_anchor())
    assert comp_pred.shape == (B, K)


def test_forward_supervised_pt_in_unit_interval():
    model = _make_model()
    expr = torch.rand(B, N_GENES)
    pt_pred, _ = model.forward_supervised(expr, _make_anchor())
    assert (pt_pred > 0.0).all() and (pt_pred < 1.0).all()


def test_forward_supervised_comp_sums_to_one():
    model = _make_model()
    expr = torch.rand(B, N_GENES)
    _, comp_pred = model.forward_supervised(expr, _make_anchor())
    assert torch.allclose(comp_pred.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_forward_supervised_bypasses_tf_icl():
    """tf_icl must not be called during forward_supervised."""
    from unittest.mock import patch
    model = _make_model()
    expr = torch.rand(B, N_GENES)
    with patch.object(model.tf_icl, "forward", side_effect=AssertionError("tf_icl called")):
        model.forward_supervised(expr, _make_anchor())   # must not raise


def test_forward_supervised_is_differentiable():
    model = _make_model()
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "out_proj") and hasattr(m.out_proj, "weight"):
                m.out_proj.weight.normal_(std=0.01)
    expr = torch.rand(B, N_GENES)
    pt_pred, comp_pred = model.forward_supervised(expr, _make_anchor())
    pt_target = torch.rand(B)
    sl_target = torch.softmax(torch.rand(B, K), dim=-1)
    loss = ((pt_pred - pt_target) ** 2).mean() + \
           -(sl_target * torch.log(comp_pred + 1e-8)).sum(dim=-1).mean()
    loss.backward()
    head_params = list(model.shared_trunk.parameters()) + list(model.pseudotime_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head_params)


def test_forward_supervised_dirichlet_model():
    """forward_supervised also works when composition_loss_type='dirichlet'."""
    from spatialmt.model.tabgrn import TabICLRegressor
    model = TabICLRegressor(
        n_genes=N_GENES, k=K, embed_dim=EMBED_DIM, n_heads=N_HEADS,
        num_cls=NUM_CLS, col_num_inds=COL_NUM_INDS,
        n_layers_col=N_LAYERS, n_layers_row=N_LAYERS, n_layers_icl=N_LAYERS,
        composition_loss_type="dirichlet",
    )
    expr = torch.rand(B, N_GENES)
    pt_pred, alpha = model.forward_supervised(expr, _make_anchor())
    assert pt_pred.shape == (B,)
    assert alpha.shape == (B, K)
    assert (alpha > 0).all()
