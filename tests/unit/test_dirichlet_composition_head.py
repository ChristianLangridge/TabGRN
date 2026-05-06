"""Unit tests for DirichletCompositionHead — unique coverage not in test_tabgrn_model.py.

Shape, positivity, differentiability, and wiring are tested as part of the full
model in test_tabgrn_model.py. This file covers: constructor dimensions,
softplus semantics (monotonicity, edge cases), numerical stability, and
fine-grained gradient flow to the Linear layer.
"""
from __future__ import annotations

import torch

B = 8
K = 6
D_MODEL = 32


def _make_head(d_model: int = D_MODEL, k: int = K):
    from spatialmt.model.tabgrn import DirichletCompositionHead
    return DirichletCompositionHead(d_model=d_model, k=k)


def _make_input(batch_size: int = B, d_model: int = D_MODEL, seed: int = 0):
    return torch.randn(batch_size, d_model, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def test_constructor_dimensions():
    head = _make_head(d_model=64, k=10)
    assert head.linear.in_features == 64
    assert head.linear.out_features == 10


# ---------------------------------------------------------------------------
# Softplus semantics
# ---------------------------------------------------------------------------

def test_outputs_remain_positive_for_large_negative_input():
    """softplus maps very negative logits to small-but-positive values, not zero."""
    head = _make_head()
    out = head(torch.full((B, D_MODEL), -100.0))
    assert (out > 0).all()


def test_large_positive_input_gives_large_concentrations():
    """softplus is monotone — larger logits → larger α₀."""
    head = _make_head()
    with torch.no_grad():
        head.linear.weight.fill_(1.0)
        head.linear.bias.zero_()
    assert head(torch.full((1, D_MODEL), 10.0)).sum() > head(torch.zeros(1, D_MODEL)).sum()


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

def test_stable_for_very_large_negative_logits():
    """softplus must not underflow to zero with extreme negative bias."""
    head = _make_head()
    with torch.no_grad():
        head.linear.weight.fill_(-1.0)
        head.linear.bias.fill_(-50.0)
    out = head(_make_input())
    assert (out > 0).all()
    assert not torch.isnan(out).any()


def test_stable_for_zero_input():
    out = _make_head()(torch.zeros(B, D_MODEL))
    assert not torch.isnan(out).any()
    assert (out > 0).all()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_flow_to_linear_weight():
    head = _make_head()
    x = _make_input().requires_grad_(True)
    head(x).sum().backward()
    assert head.linear.weight.grad is not None
    assert not torch.isnan(head.linear.weight.grad).any()


def test_gradients_flow_to_input():
    head = _make_head()
    x = _make_input().requires_grad_(True)
    head(x).sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
