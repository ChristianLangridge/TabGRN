"""Unit tests for DirichletDualHeadLoss — Dirichlet-specific coverage.

Construction, Kendall uncertainty mechanics, pseudotime MSE, forward contract,
and backward are identical to DualHeadLoss and tested in test_dual_head_loss.py.
This file tests only what is unique to the Dirichlet NLL composition term.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

B = 8
K = 6


def _make_loss():
    from spatialmt.model.loss import DirichletDualHeadLoss
    return DirichletDualHeadLoss()


def _make_concentrations(batch_size: int = B, k: int = K, seed: int = 0) -> torch.Tensor:
    return F.softplus(torch.randn(batch_size, k, generator=torch.Generator().manual_seed(seed))) + 1e-4


def _make_soft_labels(batch_size: int = B, k: int = K, seed: int = 1) -> torch.Tensor:
    return torch.softmax(torch.randn(batch_size, k, generator=torch.Generator().manual_seed(seed)), dim=-1)


def _make_inputs(batch_size: int = B, k: int = K, seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    pt_pred        = torch.sigmoid(torch.randn(batch_size, generator=rng))
    pt_target      = torch.rand(batch_size, generator=rng)
    concentrations = _make_concentrations(batch_size, k, seed=seed + 10)
    comp_target    = _make_soft_labels(batch_size, k, seed=seed + 20)
    return pt_pred, pt_target, concentrations, comp_target


# ---------------------------------------------------------------------------
# Composition loss — correctness and semantics
# ---------------------------------------------------------------------------

def test_composition_loss_matches_torch_dirichlet():
    loss_fn = _make_loss()
    conc, target = _make_concentrations(), _make_soft_labels()
    expected = -torch.distributions.Dirichlet(conc).log_prob(target).mean()
    assert torch.allclose(loss_fn.composition_loss(conc, target), expected, atol=1e-5)


def test_composition_loss_decreases_as_concentrations_sharpen_toward_target():
    loss_fn = _make_loss()
    target    = torch.tensor([[0.7, 0.2, 0.1]])
    loss_low  = loss_fn.composition_loss(torch.tensor([[1.0, 1.0, 1.0]]), target)
    loss_high = loss_fn.composition_loss(torch.tensor([[70.0, 20.0, 10.0]]), target)
    assert loss_high < loss_low


def test_composition_loss_increases_when_concentrations_misaligned():
    loss_fn = _make_loss()
    target          = torch.tensor([[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]])
    conc_aligned    = torch.tensor([[90.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    conc_misaligned = torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0, 90.0]])
    assert loss_fn.composition_loss(conc_misaligned, target) > loss_fn.composition_loss(conc_aligned, target)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

def test_composition_loss_stable_for_small_concentrations():
    loss_fn = _make_loss()
    result = loss_fn.composition_loss(torch.full((B, K), 1e-4), _make_soft_labels())
    assert not torch.isnan(result)


def test_composition_loss_stable_for_target_near_zero():
    loss_fn = _make_loss()
    target = torch.full((B, K), 1e-6)
    target[:, 0] = 1.0 - (K - 1) * 1e-6
    target = target / target.sum(-1, keepdim=True)
    result = loss_fn.composition_loss(_make_concentrations(), target)
    assert not torch.isnan(result) and torch.isfinite(result)


# ---------------------------------------------------------------------------
# Gradient flow to concentrations
# ---------------------------------------------------------------------------

def test_gradients_flow_to_concentrations():
    loss_fn = _make_loss()
    _, pt_target, conc, comp_target = _make_inputs()
    conc    = conc.detach().requires_grad_(True)
    pt_pred = torch.rand(B).requires_grad_(True)
    total, _, _ = loss_fn(pt_pred, pt_target, conc, comp_target)
    total.backward()
    assert conc.grad is not None
    assert not torch.isnan(conc.grad).any()
