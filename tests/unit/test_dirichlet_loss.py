"""
Unit tests for spatialmt.model.loss.DirichletDualHeadLoss.

DirichletDualHeadLoss is a drop-in replacement for DualHeadLoss in the
rotation_002 training run. It retains the same Kendall uncertainty weighting
structure but replaces the KL divergence composition loss with Dirichlet NLL:

Pseudotime head (unchanged)
----------------------------
    L_pt = MSE(pt_pred, pt_target)

Composition head (Dirichlet NLL)
---------------------------------
    L_comp = -mean_B [ log Dir(y_b ; α_b) ]
           = mean_B [ log Γ(Σ_k α_k) − Σ_k log Γ(α_k) − Σ_k (α_k−1) log y_k ]

    where α = DirichletCompositionHead output (concentrations, all > 0)
    and   y = comp_target (soft labels, rows sum to 1, all > 0)

    Uses torch.distributions.Dirichlet for numerical stability.

Uncertainty normalisation (same as DualHeadLoss)
-------------------------------------------------
    L_total = exp(−s_pt)  · L_pt   + ½ · s_pt
            + exp(−s_comp) · L_comp + ½ · s_comp

Uncertainty output (new)
------------------------
    At inference: mean prediction = α / α₀  (α₀ = Σα)
                  total precision  = α₀
    These are NOT part of the loss forward — computed externally by the caller.

Tests are organised into:
  - Import / construction
  - Forward contract (shapes, dtypes, finiteness)
  - composition_loss correctness (Dirichlet NLL semantics)
  - pseudotime_loss (unchanged from DualHeadLoss — MSE)
  - Uncertainty weighting mechanics (same as DualHeadLoss)
  - Backward pass
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

B = 8
K = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_loss():
    from spatialmt.model.loss import DirichletDualHeadLoss
    return DirichletDualHeadLoss()


def _make_concentrations(batch_size: int = B, k: int = K, seed: int = 0) -> torch.Tensor:
    """Random positive concentrations (softplus of random logits)."""
    rng = torch.Generator().manual_seed(seed)
    return F.softplus(torch.randn(batch_size, k, generator=rng)) + 1e-4


def _make_soft_labels(batch_size: int = B, k: int = K, seed: int = 1) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(batch_size, k, generator=rng), dim=-1)


def _make_inputs(batch_size: int = B, k: int = K, seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    pt_pred       = torch.sigmoid(torch.randn(batch_size, generator=rng))
    pt_target     = torch.rand(batch_size, generator=rng)
    concentrations = _make_concentrations(batch_size, k, seed=seed + 10)
    comp_target   = _make_soft_labels(batch_size, k, seed=seed + 20)
    return pt_pred, pt_target, concentrations, comp_target


# ---------------------------------------------------------------------------
# Import / construction
# ---------------------------------------------------------------------------

def test_import_dirichlet_dual_head_loss():
    from spatialmt.model.loss import DirichletDualHeadLoss
    assert DirichletDualHeadLoss is not None


def test_dirichlet_dual_head_loss_is_nn_module():
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
    """Equal weighting at start of training."""
    loss_fn = _make_loss()
    assert loss_fn.log_sigma_sq_pt.item()   == pytest.approx(0.0)
    assert loss_fn.log_sigma_sq_comp.item() == pytest.approx(0.0)


def test_dirichlet_loss_is_separate_class_from_dual_head_loss():
    from spatialmt.model.loss import DirichletDualHeadLoss, DualHeadLoss
    assert DirichletDualHeadLoss is not DualHeadLoss


# ---------------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------------

def test_forward_returns_three_tuple():
    out = _make_loss()(*_make_inputs())
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
    assert total.dtype     == torch.float32
    assert pt_loss.dtype   == torch.float32
    assert comp_loss.dtype == torch.float32


def test_forward_no_nan():
    total, pt_loss, comp_loss = _make_loss()(*_make_inputs())
    assert not torch.isnan(total)
    assert not torch.isnan(pt_loss)
    assert not torch.isnan(comp_loss)


def test_forward_all_finite():
    total, pt_loss, comp_loss = _make_loss()(*_make_inputs())
    assert torch.isfinite(total)
    assert torch.isfinite(pt_loss)
    assert torch.isfinite(comp_loss)


# ---------------------------------------------------------------------------
# Pseudotime MSE component (unchanged from DualHeadLoss)
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
    assert loss_fn.pseudotime_loss(pt, pt).item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Dirichlet NLL composition component
# ---------------------------------------------------------------------------

def test_composition_loss_is_finite():
    loss_fn = _make_loss()
    conc = _make_concentrations()
    target = _make_soft_labels()
    result = loss_fn.composition_loss(conc, target)
    assert torch.isfinite(result)


def test_composition_loss_no_nan():
    loss_fn = _make_loss()
    result = loss_fn.composition_loss(_make_concentrations(), _make_soft_labels())
    assert not torch.isnan(result)


def test_composition_loss_matches_torch_dirichlet():
    """Cross-check against torch.distributions.Dirichlet.log_prob."""
    loss_fn = _make_loss()
    conc   = _make_concentrations()
    target = _make_soft_labels()
    expected = -torch.distributions.Dirichlet(conc).log_prob(target).mean()
    assert torch.allclose(loss_fn.composition_loss(conc, target), expected, atol=1e-5)


def test_composition_loss_decreases_as_concentrations_sharpen_toward_target():
    """Higher-precision concentrations aligned to target → lower NLL."""
    loss_fn = _make_loss()
    # Use a single sample with known target
    target = torch.tensor([[0.7, 0.2, 0.1]])   # shape (1, 3)
    # Low precision: uniform-ish concentrations
    conc_low  = torch.tensor([[1.0, 1.0, 1.0]])
    # High precision: concentrations peaked at target direction
    conc_high = torch.tensor([[70.0, 20.0, 10.0]])
    loss_low  = loss_fn.composition_loss(conc_low,  target)
    loss_high = loss_fn.composition_loss(conc_high, target)
    assert loss_high < loss_low


def test_composition_loss_increases_when_concentrations_misaligned_with_target():
    """Peaked concentrations pointing AWAY from the target direction incur higher NLL
    than a diffuse (near-uniform) distribution, confirming the loss is informative."""
    loss_fn = _make_loss()
    # Target: strongly loaded on class 0
    target = torch.tensor([[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]])
    # Aligned: high precision toward class 0
    conc_aligned    = torch.tensor([[90.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    # Misaligned: high precision toward class 5 (wrong class)
    conc_misaligned = torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0, 90.0]])
    loss_aligned    = loss_fn.composition_loss(conc_aligned,    target)
    loss_misaligned = loss_fn.composition_loss(conc_misaligned, target)
    assert loss_misaligned > loss_aligned


def test_composition_loss_scalar_output():
    loss_fn = _make_loss()
    result = loss_fn.composition_loss(_make_concentrations(), _make_soft_labels())
    assert result.shape == ()


def test_composition_loss_stable_for_small_concentrations():
    """Near-zero concentrations (softplus output ≈ 1e-4) must not produce NaN."""
    loss_fn = _make_loss()
    conc   = torch.full((B, K), 1e-4)
    target = _make_soft_labels()
    result = loss_fn.composition_loss(conc, target)
    assert not torch.isnan(result)


def test_composition_loss_stable_for_target_near_zero():
    """Near-zero target entries must not cause log(0) in the Dirichlet NLL."""
    loss_fn = _make_loss()
    target = torch.full((B, K), 1e-6)
    target[:, 0] = 1.0 - (K - 1) * 1e-6
    target = target / target.sum(-1, keepdim=True)
    conc   = _make_concentrations()
    result = loss_fn.composition_loss(conc, target)
    assert not torch.isnan(result)
    assert torch.isfinite(result)


# ---------------------------------------------------------------------------
# Uncertainty weighting mechanics (same structure as DualHeadLoss)
# ---------------------------------------------------------------------------

def test_total_equals_sum_at_init():
    """At init (s=0): exp(−0)=1, ½·0=0, so total = L_pt + L_comp."""
    loss_fn = _make_loss()
    total, pt_loss, comp_loss = loss_fn(*_make_inputs())
    assert torch.allclose(total, pt_loss + comp_loss, atol=1e-5)


def test_total_differs_from_sum_after_sigma_change():
    loss_fn = _make_loss()
    with torch.no_grad():
        loss_fn.log_sigma_sq_pt.fill_(1.0)
        loss_fn.log_sigma_sq_comp.fill_(-1.0)
    total, pt_loss, comp_loss = loss_fn(*_make_inputs())
    assert not torch.allclose(total, pt_loss + comp_loss, atol=1e-4)


def test_sigma_params_receive_gradient():
    loss_fn = _make_loss()
    pt_pred, pt_target, conc, comp_target = _make_inputs()
    pt_pred = pt_pred.detach().requires_grad_(True)
    conc    = conc.detach().requires_grad_(True)
    total, _, _ = loss_fn(pt_pred, pt_target, conc, comp_target)
    total.backward()
    assert not torch.isnan(loss_fn.log_sigma_sq_pt.grad)
    assert not torch.isnan(loss_fn.log_sigma_sq_comp.grad)


def test_penalty_prevents_trivial_zero_loss():
    loss_fn = _make_loss()
    with torch.no_grad():
        loss_fn.log_sigma_sq_pt.fill_(20.0)
        loss_fn.log_sigma_sq_comp.fill_(20.0)
    total, _, _ = loss_fn(*_make_inputs())
    assert total.item() > 1.0


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

def test_backward_does_not_raise():
    loss_fn = _make_loss()
    pt_pred, pt_target, conc, comp_target = _make_inputs()
    pt_pred = pt_pred.requires_grad_(True)
    conc    = conc.requires_grad_(True)
    total, _, _ = loss_fn(pt_pred, pt_target, conc, comp_target)
    total.backward()


def test_gradients_flow_to_concentrations():
    loss_fn = _make_loss()
    _, pt_target, conc, comp_target = _make_inputs()
    conc = conc.detach().requires_grad_(True)
    pt_pred = torch.rand(B).requires_grad_(True)
    total, _, _ = loss_fn(pt_pred, pt_target, conc, comp_target)
    total.backward()
    assert conc.grad is not None
    assert not torch.isnan(conc.grad).any()
