"""
Unit tests for spatialmt.model.tabgrn.DirichletCompositionHead.

DirichletCompositionHead replaces the softmax CompositionHead for the
Dirichlet NLL training run (rotation_002). Instead of outputting probabilities
on the K-simplex, it outputs Dirichlet concentration parameters α_k > 0 for
each of the K cell states.

Architecture
------------
  Linear(d_model, K) → softplus → α ∈ (0, ∞)^K

Key distinctions from CompositionHead
--------------------------------------
- Output is NOT a probability vector (rows do not sum to 1)
- All outputs are strictly positive (softplus, not softmax)
- Mean cell-state probability is α_k / Σα_k (derived post-hoc, not the output)
- Total precision α₀ = Σα_k encodes prediction confidence

Tests are organised into:
  - Import / construction
  - Forward contract (shapes, dtypes, positivity)
  - Distinguishability from CompositionHead
  - Backward pass (gradients flow)
  - Numerical stability
"""
from __future__ import annotations

import pytest
import torch

B = 8
K = 6
D_MODEL = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_head(d_model: int = D_MODEL, k: int = K):
    from spatialmt.model.tabgrn import DirichletCompositionHead
    return DirichletCompositionHead(d_model=d_model, k=k)


def _make_input(batch_size: int = B, d_model: int = D_MODEL, seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, d_model, generator=rng)


# ---------------------------------------------------------------------------
# Import / construction
# ---------------------------------------------------------------------------

def test_import_dirichlet_composition_head():
    from spatialmt.model.tabgrn import DirichletCompositionHead
    assert DirichletCompositionHead is not None


def test_dirichlet_composition_head_is_nn_module():
    import torch.nn as nn
    assert isinstance(_make_head(), nn.Module)


def test_constructor_accepts_d_model_and_k():
    head = _make_head(d_model=64, k=10)
    assert head is not None


def test_has_linear_layer():
    head = _make_head()
    assert hasattr(head, "linear")
    import torch.nn as nn
    assert isinstance(head.linear, nn.Linear)


def test_linear_in_features_matches_d_model():
    head = _make_head(d_model=64, k=K)
    assert head.linear.in_features == 64


def test_linear_out_features_matches_k():
    head = _make_head(d_model=D_MODEL, k=10)
    assert head.linear.out_features == 10


# ---------------------------------------------------------------------------
# Forward contract — shape and dtype
# ---------------------------------------------------------------------------

def test_forward_output_shape():
    head = _make_head()
    x = _make_input()
    out = head(x)
    assert out.shape == (B, K)


def test_forward_output_dtype_float32():
    head = _make_head()
    out = head(_make_input())
    assert out.dtype == torch.float32


def test_forward_no_nan():
    head = _make_head()
    out = head(_make_input())
    assert not torch.isnan(out).any()


def test_forward_no_inf():
    head = _make_head()
    out = head(_make_input())
    assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# Forward contract — concentration semantics (strictly positive, not simplex)
# ---------------------------------------------------------------------------

def test_all_outputs_strictly_positive():
    """softplus guarantees α_k > 0 for every element."""
    head = _make_head()
    out = head(_make_input())
    assert (out > 0).all()


def test_outputs_do_not_sum_to_one():
    """Concentrations are NOT a probability distribution — rows should not sum to 1."""
    head = _make_head()
    out = head(_make_input())
    row_sums = out.sum(dim=-1)
    # With randomly initialised weights this will not be 1 for all rows
    assert not torch.allclose(row_sums, torch.ones(B), atol=1e-3)


def test_outputs_remain_positive_for_large_negative_input():
    """softplus maps very negative logits to small-but-positive values (not 0)."""
    head = _make_head()
    x = torch.full((B, D_MODEL), -100.0)
    out = head(x)
    assert (out > 0).all()


def test_large_positive_input_gives_large_concentrations():
    """softplus is monotone — large logits → large α₀ → high-precision prediction."""
    head = _make_head()
    x_small = torch.zeros(1, D_MODEL)
    x_large = torch.full((1, D_MODEL), 10.0)
    # Replace linear weights with all-ones so output magnitude tracks input
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.zero_()
    out_small = head(x_small).sum()
    out_large = head(x_large).sum()
    assert out_large > out_small


# ---------------------------------------------------------------------------
# Distinguishability from CompositionHead
# ---------------------------------------------------------------------------

def test_dirichlet_head_is_not_composition_head():
    from spatialmt.model.tabgrn import DirichletCompositionHead, CompositionHead
    assert DirichletCompositionHead is not CompositionHead


def test_composition_head_outputs_sum_to_one():
    """Baseline sanity: CompositionHead rows sum to 1 (softmax) — DirichletHead does not."""
    from spatialmt.model.tabgrn import CompositionHead
    kl_head = CompositionHead(d_model=D_MODEL, k=K)
    x = _make_input()
    out = kl_head(x)
    assert torch.allclose(out.sum(dim=-1), torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def test_backward_does_not_raise():
    head = _make_head()
    x = _make_input().requires_grad_(True)
    out = head(x)
    loss = out.sum()
    loss.backward()


def test_gradients_flow_to_linear_weight():
    head = _make_head()
    x = _make_input().requires_grad_(True)
    out = head(x)
    out.sum().backward()
    assert head.linear.weight.grad is not None
    assert not torch.isnan(head.linear.weight.grad).any()


def test_gradients_flow_to_input():
    head = _make_head()
    x = _make_input().requires_grad_(True)
    head(x).sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

def test_stable_for_very_large_negative_logits():
    """softplus must not underflow to zero for very negative pre-activations."""
    head = _make_head()
    with torch.no_grad():
        head.linear.weight.fill_(-1.0)
        head.linear.bias.fill_(-50.0)
    x = _make_input()
    out = head(x)
    assert (out > 0).all()
    assert not torch.isnan(out).any()


def test_stable_for_zero_input():
    head = _make_head()
    x = torch.zeros(B, D_MODEL)
    out = head(x)
    assert not torch.isnan(out).any()
    assert (out > 0).all()
