"""
Unit tests for spatialmt.model.loss.DualHeadLoss.

Loss design
-----------
PseudotimeHead   : MSE(pt_pred, pt_target)
                   Both are scalars in [0, 1].

CompositionHead  : KL(comp_target ∥ comp_pred)
                   = Σ_k  π_k · (log π_k – log p_k)
                   where π = comp_target (soft labels, distance-to-centroid softmax)
                   and   p = comp_pred   (CompositionHead softmax output).
                   Both live on the K-simplex. KL = 0 ↔ p = π (perfect prediction).
                   No extra hyperparameters required.

Normalisation (Kendall et al. 2018 uncertainty weighting)
----------------------------------------------------------
  L = exp(–s_pt)  · L_MSE + ½·s_pt
    + exp(–s_comp) · L_KL  + ½·s_comp

where s_pt = log σ²_pt and s_comp = log σ²_comp are learnable nn.Parameters.
Initialised to 0 → σ = 1 → equal unscaled weighting at start of training.
The ½·s penalty prevents the model from driving σ → ∞ to zero both tasks.

Tests are organised into:
  - Import / construction
  - Forward contract (shapes, dtypes, differentiability)
  - Pseudotime MSE component
  - KL divergence component (correctness, stability, ordering)
  - Uncertainty weighting mechanics
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

B = 8
K = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loss():
    from spatialmt.model.loss import DualHeadLoss
    return DualHeadLoss()


def _make_inputs(batch_size: int = B, k: int = K, seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    pt_pred     = torch.sigmoid(torch.randn(batch_size, generator=rng))
    pt_target   = torch.rand(batch_size, generator=rng)
    comp_pred   = torch.softmax(torch.randn(batch_size, k, generator=rng), dim=-1)
    comp_target = torch.softmax(torch.randn(batch_size, k, generator=rng), dim=-1)
    return pt_pred, pt_target, comp_pred, comp_target


# ---------------------------------------------------------------------------
# Import / construction
# ---------------------------------------------------------------------------

def test_import_dual_head_loss():
    from spatialmt.model.loss import DualHeadLoss
    assert DualHeadLoss is not None


def test_dual_head_loss_is_nn_module():
    import torch.nn as nn
    assert isinstance(_make_loss(), nn.Module)


def test_has_log_sigma_sq_pt():
    loss_fn = _make_loss()
    assert hasattr(loss_fn, "log_sigma_sq_pt")
    assert isinstance(loss_fn.log_sigma_sq_pt, torch.nn.Parameter)


def test_has_log_sigma_sq_comp():
    loss_fn = _make_loss()
    assert hasattr(loss_fn, "log_sigma_sq_comp")
    assert isinstance(loss_fn.log_sigma_sq_comp, torch.nn.Parameter)


def test_log_sigma_sq_initialised_to_zero():
    """Equal weighting at start of training — both σ² = 1."""
    loss_fn = _make_loss()
    assert loss_fn.log_sigma_sq_pt.item()   == pytest.approx(0.0)
    assert loss_fn.log_sigma_sq_comp.item() == pytest.approx(0.0)


def test_no_concentration_parameter():
    """KL divergence requires no extra hyperparameter — DualHeadLoss must not have one."""
    loss_fn = _make_loss()
    assert not hasattr(loss_fn, "concentration")


# ---------------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------------

def test_forward_returns_three_tuple():
    loss_fn = _make_loss()
    out = loss_fn(*_make_inputs())
    assert isinstance(out, tuple) and len(out) == 3


def test_forward_total_is_scalar():
    total, _, _ = _make_loss()(*_make_inputs())
    assert total.shape == ()


def test_forward_pt_loss_is_scalar():
    _, pt_loss, _ = _make_loss()(*_make_inputs())
    assert pt_loss.shape == ()


def test_forward_comp_loss_is_scalar():
    _, _, comp_loss = _make_loss()(*_make_inputs())
    assert comp_loss.shape == ()


def test_forward_all_float32():
    total, pt_loss, comp_loss = _make_loss()(*_make_inputs())
    assert total.dtype    == torch.float32
    assert pt_loss.dtype  == torch.float32
    assert comp_loss.dtype == torch.float32


def test_forward_no_nan():
    total, pt_loss, comp_loss = _make_loss()(*_make_inputs())
    assert not torch.isnan(total)
    assert not torch.isnan(pt_loss)
    assert not torch.isnan(comp_loss)


def test_forward_is_differentiable():
    """backward() must not raise and gradients must flow to both σ² parameters."""
    loss_fn = _make_loss()
    pt_pred, pt_target, comp_pred, comp_target = _make_inputs()
    pt_pred   = pt_pred.detach().requires_grad_(True)
    comp_pred = comp_pred.detach().requires_grad_(True)
    total, _, _ = loss_fn(pt_pred, pt_target, comp_pred, comp_target)
    total.backward()
    assert loss_fn.log_sigma_sq_pt.grad   is not None
    assert loss_fn.log_sigma_sq_comp.grad is not None


# ---------------------------------------------------------------------------
# Pseudotime MSE component
# ---------------------------------------------------------------------------

def test_pseudotime_loss_matches_mse():
    loss_fn = _make_loss()
    pt_pred, pt_target, _, _ = _make_inputs()
    assert torch.allclose(
        loss_fn.pseudotime_loss(pt_pred, pt_target),
        F.mse_loss(pt_pred, pt_target),
    )


def test_pseudotime_loss_zero_for_perfect_prediction():
    loss_fn = _make_loss()
    pt = torch.rand(B)
    assert loss_fn.pseudotime_loss(pt, pt).item() == pytest.approx(0.0, abs=1e-7)


def test_pseudotime_loss_increases_with_error():
    loss_fn = _make_loss()
    target = torch.full((B,), 0.5)
    assert (
        loss_fn.pseudotime_loss(torch.full((B,), 0.51), target)
        < loss_fn.pseudotime_loss(torch.full((B,), 0.9),  target)
    )


# ---------------------------------------------------------------------------
# KL divergence component
# ---------------------------------------------------------------------------

def test_composition_loss_is_finite():
    _, _, comp_pred, comp_target = _make_inputs()
    assert torch.isfinite(_make_loss().composition_loss(comp_pred, comp_target))


def test_composition_loss_zero_for_perfect_prediction():
    """KL(π ∥ π) = 0 — a closed-form minimum that Dirichlet NLL cannot offer."""
    loss_fn = _make_loss()
    comp = torch.softmax(torch.randn(B, K), dim=-1)
    assert loss_fn.composition_loss(comp, comp).item() == pytest.approx(0.0, abs=1e-5)


def test_composition_loss_lower_when_prediction_correct():
    """KL must be strictly lower when pred = target than when pred is random."""
    loss_fn = _make_loss()
    comp_target = torch.softmax(torch.randn(B, K), dim=-1)
    loss_correct = loss_fn.composition_loss(comp_target.clone(), comp_target)
    loss_wrong   = loss_fn.composition_loss(torch.full((B, K), 1.0 / K), comp_target)
    assert loss_correct.item() < loss_wrong.item()


def test_composition_loss_nonnegative():
    """KL divergence is always ≥ 0 by Gibbs' inequality."""
    loss_fn = _make_loss()
    _, _, comp_pred, comp_target = _make_inputs()
    assert _make_loss().composition_loss(comp_pred, comp_target).item() >= 0.0


def test_composition_loss_no_nan_near_zero_inputs():
    """Near-zero entries in comp_pred (after clamping) must not produce NaN."""
    loss_fn = _make_loss()
    comp_pred = torch.full((B, K), 1e-8)
    comp_pred[:, 0] = 1.0 - (K - 1) * 1e-8
    comp_pred = comp_pred / comp_pred.sum(-1, keepdim=True)
    comp_target = torch.softmax(torch.randn(B, K), dim=-1)
    assert not torch.isnan(loss_fn.composition_loss(comp_pred, comp_target))


def test_composition_loss_matches_manual_kl():
    """Cross-check against hand-computed KL(target ∥ pred)."""
    loss_fn = _make_loss()
    comp_pred   = torch.tensor([[0.4, 0.3, 0.3]])
    comp_target = torch.tensor([[0.5, 0.3, 0.2]])
    eps = 1e-8
    # KL(target ∥ pred) = Σ target_k * (log target_k – log pred_k)
    manual = (comp_target * (
        torch.log(comp_target.clamp(min=eps)) - torch.log(comp_pred.clamp(min=eps))
    )).sum(-1).mean()
    assert torch.allclose(loss_fn.composition_loss(comp_pred, comp_target), manual, atol=1e-5)


# ---------------------------------------------------------------------------
# Uncertainty weighting mechanics
# ---------------------------------------------------------------------------

def test_total_equals_sum_at_init():
    """At init (s=0): exp(–0)=1 and ½·0=0, so total = L_pt + L_comp exactly."""
    loss_fn = _make_loss()
    total, pt_loss, comp_loss = loss_fn(*_make_inputs())
    assert torch.allclose(total, pt_loss + comp_loss, atol=1e-5)


def test_total_differs_from_sum_after_sigma_change():
    """Non-zero s → weighted total diverges from the naive sum."""
    loss_fn = _make_loss()
    with torch.no_grad():
        loss_fn.log_sigma_sq_pt.fill_(1.0)
        loss_fn.log_sigma_sq_comp.fill_(-1.0)
    total, pt_loss, comp_loss = loss_fn(*_make_inputs())
    assert not torch.allclose(total, pt_loss + comp_loss, atol=1e-4)


def test_sigma_params_receive_gradient():
    loss_fn = _make_loss()
    pt_pred, pt_target, comp_pred, comp_target = _make_inputs()
    pt_pred   = pt_pred.detach().requires_grad_(True)
    comp_pred = comp_pred.detach().requires_grad_(True)
    total, _, _ = loss_fn(pt_pred, pt_target, comp_pred, comp_target)
    total.backward()
    assert not torch.isnan(loss_fn.log_sigma_sq_pt.grad)
    assert not torch.isnan(loss_fn.log_sigma_sq_comp.grad)


def test_penalty_prevents_trivial_zero_loss():
    """With large σ² (s >> 0): penalty ½s > 0 prevents total from collapsing to zero."""
    loss_fn = _make_loss()
    with torch.no_grad():
        loss_fn.log_sigma_sq_pt.fill_(20.0)
        loss_fn.log_sigma_sq_comp.fill_(20.0)
    # At s=20: exp(-20)*L ≈ 0, but ½*20 = 10 per task → total ≈ 20
    total, _, _ = loss_fn(*_make_inputs())
    assert total.item() > 1.0
