"""
spatialmt.training.muon — Muon optimizer.

Muon (Momentum + Orthogonal Updates via Nesterov) applies Newton-Schulz
orthogonalization to the Nesterov momentum estimate of weight matrices before
applying the gradient update.  This replaces the Adam adaptive scaling with a
per-parameter-matrix orthogonal projection, giving better conditioning for
transformer weight matrices.

Reference
---------
Bernstein & Newhouse (2024) "Old Optimizer, New Norm"
Implementation follows Keller Jordan's reference code
(https://github.com/KellerJordan/Muon).

Usage
-----
Muon should only receive *weight matrices* (ndim >= 2).  Biases, layer-norm
scales, and embedding tables should use AdamW.  The MuonAdamW wrapper in
trainer.py handles the split automatically.

Parameters
----------
params : param_groups
    Each group must have a 'params' list of ndim-2 weight tensors and an
    optional 'lr' float.
lr : float
    Default learning rate (overridden per-group).
momentum : float
    Nesterov momentum coefficient (default 0.95).
nesterov : bool
    Use Nesterov gradient estimate (default True).
ns_steps : int
    Newton-Schulz iteration count (default 5).  5 steps is sufficient for
    float32; increase to 10 for bfloat16.
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


def _ns5_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal polar factor of G via the degree-5 Chebyshev-
    Halley Newton-Schulz iteration.

    Parameters
    ----------
    G : (m, n) float32 tensor, m <= n required (caller must transpose if not).
    steps : int

    Returns
    -------
    (m, n) tensor with orthonormal rows (approximate).
    """
    assert G.ndim == 2, f"_ns5_orthogonalize expects a 2-D matrix, got shape {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315

    # Normalize so spectral norm ≈ 1 before iterating
    X = G / (G.norm() + 1e-7)

    for _ in range(steps):
        A = X @ X.T               # (m, m)
        B = b * A + c * (A @ A)   # degree-3 contribution
        X = a * X + B @ X         # rank-preserving update

    return X


def _muon_update(
    G: torch.Tensor,
    momentum_buf: torch.Tensor,
    beta: float,
    nesterov: bool,
    ns_steps: int,
) -> torch.Tensor:
    """Compute the Muon parameter update direction for a single weight matrix.

    Parameters
    ----------
    G : gradient tensor for a single parameter (arbitrary shape, ndim >= 2)
    momentum_buf : running momentum buffer (same shape as G)
    beta : momentum coefficient
    nesterov : whether to use Nesterov estimate
    ns_steps : NS iteration count

    Returns
    -------
    update : orthogonalized direction tensor (same shape as G)
    """
    momentum_buf.mul_(beta).add_(G)

    if nesterov:
        g_eff = G + beta * momentum_buf
    else:
        g_eff = momentum_buf

    # Flatten to 2-D: (rows, cols) where rows = shape[0], cols = prod(shape[1:])
    rows = g_eff.shape[0]
    g_2d = g_eff.reshape(rows, -1)           # (rows, cols)

    # Newton-Schulz requires m <= n; transpose if taller than wide
    transposed = g_2d.shape[0] > g_2d.shape[1]
    if transposed:
        g_2d = g_2d.T

    ortho = _ns5_orthogonalize(g_2d, steps=ns_steps)

    if transposed:
        ortho = ortho.T

    return ortho.reshape(G.shape)


class Muon(Optimizer):
    """Muon optimizer for weight matrices (ndim >= 2).

    Only pass weight matrices — not biases, embeddings, or layer-norm scales.
    Those should go to AdamW; the MuonAdamW wrapper handles routing.

    Parameters
    ----------
    params : iterable of parameter groups or parameters
    lr : float, default 1e-3
    momentum : float, default 0.95
    nesterov : bool, default True
    ns_steps : int, default 5
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            beta     = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                G = p.grad

                state = self.state[p]
                if "momentum_buf" not in state:
                    state["momentum_buf"] = torch.zeros_like(G)

                update = _muon_update(
                    G,
                    state["momentum_buf"],
                    beta=beta,
                    nesterov=nesterov,
                    ns_steps=ns_steps,
                )

                # Scale by sqrt(max_dim) so effective lr is invariant to
                # matrix shape (a common Muon convention).
                scale = float(max(p.shape)) ** 0.5
                p.add_(update, alpha=-lr * scale)

        return loss
