"""
Unit tests for spatialmt.model.tabicl — TabICLRegressor and sub-modules.

Architecture under test
-----------------------
Backbone (pretrained TabICLv2 imports)
  col_embedder     tabicl.model.embedding.ColEmbedding
                   (B, n_cells, n_genes) → (B, n_cells, seq_len, embed_dim)

  row_interactor   tabicl.model.interaction.RowInteraction
                   (B, n_cells, seq_len, embed_dim) → (B, n_cells, d_model)
                   d_model = num_cls × embed_dim

Custom head stack
  LabelInjector    (B, n_cells, d_model) → same  [anchor rows only]
  ICLAttention     (B, n_cells, d_model) → (B, d_model)
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
import numpy as np
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
    from spatialmt.model.tabicl import TabICLRegressor
    return TabICLRegressor(
        n_genes=n_genes,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        k=k,
        num_cls=num_cls,
        col_num_inds=col_num_inds,
    )


# ---------------------------------------------------------------------------
# Import sanity
# ---------------------------------------------------------------------------

def test_import_tabicl_regressor():
    from spatialmt.model.tabicl import TabICLRegressor
    assert TabICLRegressor is not None


def test_import_attention_scorer():
    from spatialmt.model.tabicl import AttentionScorer
    assert AttentionScorer is not None


def test_import_custom_sub_modules():
    from spatialmt.model.tabicl import (
        LabelInjector,
        ICLAttention,
        SharedTrunk,
        PseudotimeHead,
        CompositionHead,
    )


def test_import_tabicl_backbone_classes():
    """TabICLv2 backbone classes must be importable from the tabicl package."""
    from tabicl.model.embedding import ColEmbedding
    from tabicl.model.interaction import RowInteraction
    assert ColEmbedding is not None
    assert RowInteraction is not None


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


def test_model_has_label_injector():
    from spatialmt.model.tabicl import LabelInjector
    model = _make_model()
    assert hasattr(model, "label_injector")
    assert isinstance(model.label_injector, LabelInjector)


def test_model_has_icl_attention():
    from spatialmt.model.tabicl import ICLAttention
    model = _make_model()
    assert hasattr(model, "icl_attention")
    assert isinstance(model.icl_attention, ICLAttention)


def test_model_has_shared_trunk():
    from spatialmt.model.tabicl import SharedTrunk
    model = _make_model()
    assert hasattr(model, "shared_trunk")
    assert isinstance(model.shared_trunk, SharedTrunk)


def test_model_has_pseudotime_head():
    from spatialmt.model.tabicl import PseudotimeHead
    model = _make_model()
    assert hasattr(model, "pseudotime_head")
    assert isinstance(model.pseudotime_head, PseudotimeHead)


def test_model_has_composition_head():
    from spatialmt.model.tabicl import CompositionHead
    model = _make_model()
    assert hasattr(model, "composition_head")
    assert isinstance(model.composition_head, CompositionHead)


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


# ---------------------------------------------------------------------------
# LabelInjector
# ---------------------------------------------------------------------------

def test_label_injector_output_shape():
    from spatialmt.model.tabicl import LabelInjector
    inj = LabelInjector(d_model=D_MODEL, k=K)
    n_cells = N_ANCHORS + 1
    x   = torch.rand(B, n_cells, D_MODEL)
    pt  = torch.rand(B, N_ANCHORS)
    sl  = torch.softmax(torch.rand(B, N_ANCHORS, K), dim=-1)
    out = inj(x, anchor_pseudotime=pt, anchor_soft_labels=sl)
    assert out.shape == (B, n_cells, D_MODEL)


def test_label_injector_query_row_unchanged():
    """Query row (last position) must not be modified — no label leakage."""
    from spatialmt.model.tabicl import LabelInjector
    inj = LabelInjector(d_model=D_MODEL, k=K)
    n_cells = N_ANCHORS + 1
    x   = torch.rand(B, n_cells, D_MODEL)
    pt  = torch.rand(B, N_ANCHORS)
    sl  = torch.softmax(torch.rand(B, N_ANCHORS, K), dim=-1)
    out = inj(x, anchor_pseudotime=pt, anchor_soft_labels=sl)
    assert torch.allclose(out[:, -1, :], x[:, -1, :])


# ---------------------------------------------------------------------------
# ICLAttention
# ---------------------------------------------------------------------------

def test_icl_attention_output_shape():
    from spatialmt.model.tabicl import ICLAttention
    attn = ICLAttention(d_model=D_MODEL, n_heads=N_HEADS)
    n_cells = N_ANCHORS + 1
    x = torch.rand(B, n_cells, D_MODEL)
    out = attn(x)
    assert out.shape == (B, D_MODEL)


# ---------------------------------------------------------------------------
# SharedTrunk
# ---------------------------------------------------------------------------

def test_shared_trunk_output_shape():
    from spatialmt.model.tabicl import SharedTrunk
    trunk = SharedTrunk(d_model=D_MODEL)
    x = torch.rand(B, D_MODEL)
    out = trunk(x)
    assert out.shape == (B, D_MODEL)


# ---------------------------------------------------------------------------
# PseudotimeHead
# ---------------------------------------------------------------------------

def test_pseudotime_head_output_shape():
    from spatialmt.model.tabicl import PseudotimeHead
    head = PseudotimeHead(d_model=D_MODEL)
    x = torch.rand(B, D_MODEL)
    out = head(x)
    assert out.shape == (B,)


def test_pseudotime_head_sigmoid_range():
    from spatialmt.model.tabicl import PseudotimeHead
    head = PseudotimeHead(d_model=D_MODEL)
    x = torch.rand(B, D_MODEL)
    out = head(x)
    assert (out > 0.0).all() and (out < 1.0).all()


def test_pseudotime_head_init_bias():
    """output_head_init_bias=0.5 means initial raw bias ≈ 0.5 → sigmoid ≈ 0.62."""
    from spatialmt.model.tabicl import PseudotimeHead
    head = PseudotimeHead(d_model=D_MODEL, init_bias=0.5)
    assert head.linear.bias is not None
    assert pytest.approx(head.linear.bias.item(), abs=1e-6) == 0.5


# ---------------------------------------------------------------------------
# CompositionHead
# ---------------------------------------------------------------------------

def test_composition_head_output_shape():
    from spatialmt.model.tabicl import CompositionHead
    head = CompositionHead(d_model=D_MODEL, k=K)
    x = torch.rand(B, D_MODEL)
    out = head(x)
    assert out.shape == (B, K)


def test_composition_head_softmax_sums():
    from spatialmt.model.tabicl import CompositionHead
    head = CompositionHead(d_model=D_MODEL, k=K)
    x = torch.rand(B, D_MODEL)
    out = head(x)
    assert torch.allclose(out.sum(dim=-1), torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# AttentionScorer
# ---------------------------------------------------------------------------

def test_attention_scorer_extract_after_forward():
    from spatialmt.model.tabicl import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    assert scores is not None
    scorer.remove_hook()


def test_attention_scorer_shape():
    """Cosine similarity of gene embeddings: (B, n_cells, n_genes, n_genes)."""
    from spatialmt.model.tabicl import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    n_cells = N_ANCHORS + 1
    assert scores.shape == (B, n_cells, N_GENES, N_GENES)
    scorer.remove_hook()


def test_attention_scorer_scores_are_detached():
    from spatialmt.model.tabicl import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    assert not scores.requires_grad
    scorer.remove_hook()


def test_attention_scorer_scores_symmetric():
    """Cosine similarity matrix must be symmetric."""
    from spatialmt.model.tabicl import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    assert torch.allclose(scores, scores.transpose(-1, -2), atol=1e-5)
    scorer.remove_hook()


def test_attention_scorer_diagonal_is_one():
    """Cosine similarity of a vector with itself must be 1.0."""
    from spatialmt.model.tabicl import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    batch = _make_batch()
    model(batch)
    scores = scorer.extract()
    diag = torch.diagonal(scores, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4)
    scorer.remove_hook()


def test_attention_scorer_raises_before_forward():
    from spatialmt.model.tabicl import AttentionScorer
    model = _make_model()
    scorer = AttentionScorer(model)
    with pytest.raises(RuntimeError):
        scorer.extract()
    scorer.remove_hook()


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
