"""Unit tests for TRiGS SE(3) math helpers.

Equations referenced are from the TRiGS paper main text
(sections/3_Method.tex) and supplement (sections/Suppl/3_MathMaticalFormulation.tex).
"""

import math

import pytest
import torch

from trigs.math.se3 import (
    bezier_quadratic,
    exp_se3,
    exp_se3_quat,
    gauge_fix_omega,
    tau_from_window,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_exp_se3_identity_at_zero():
    """exp(0) = (I, 0) — T(µ_t) = I by construction (Eq. S1)."""
    xi = torch.zeros(4, 6)
    R, p = exp_se3(xi)
    assert torch.allclose(R, torch.eye(3).expand(4, 3, 3), atol=1e-8)
    assert torch.allclose(p, torch.zeros(4, 3), atol=1e-8)


def test_exp_se3_rotation_matches_scipy():
    """R via Rodrigues (Eq. S2) must match scipy reference."""
    scipy = pytest.importorskip("scipy.spatial.transform")
    omega = torch.randn(16, 3) * 0.5
    xi = torch.cat([torch.zeros(16, 3), omega], dim=-1)
    R, p = exp_se3(xi)
    R_ref = torch.from_numpy(
        scipy.Rotation.from_rotvec(omega.numpy()).as_matrix()
    ).to(R.dtype)
    assert torch.allclose(R, R_ref, atol=1e-6)
    # No translation component input → p = J·0 = 0
    assert torch.allclose(p, torch.zeros(16, 3), atol=1e-6)


def test_exp_se3_translation_only():
    """ω = 0 ⇒ J = I ⇒ p = ρ."""
    rho = torch.randn(8, 3)
    xi = torch.cat([rho, torch.zeros(8, 3)], dim=-1)
    R, p = exp_se3(xi)
    assert torch.allclose(R, torch.eye(3).expand(8, 3, 3), atol=1e-7)
    assert torch.allclose(p, rho, atol=1e-7)


@pytest.mark.parametrize("theta", [1e-10, 1e-6, 1e-3, 1.0, math.pi - 1e-4])
def test_exp_se3_small_angle_stability(theta: float):
    """Value and gradient stable across the full θ range (Eq. S11–S13)."""
    axis = torch.tensor([0.0, 0.0, 1.0])
    omega = axis * theta
    rho = torch.tensor([0.1, 0.2, 0.3])
    xi = torch.cat([rho, omega], dim=-1).requires_grad_(True)
    R, p = exp_se3(xi)
    assert torch.isfinite(R).all()
    assert torch.isfinite(p).all()
    (R.sum() + p.sum()).backward()
    assert torch.isfinite(xi.grad).all()


def test_exp_se3_matches_exp_se3_quat():
    """Matrix-form and quaternion-form exp maps agree."""
    from trigs.math.quaternion import quaternion_to_rotmat

    xi = torch.randn(32, 6) * 0.3
    R_mat, p_mat = exp_se3(xi)
    q, p_quat = exp_se3_quat(xi)
    R_from_quat = quaternion_to_rotmat(q)
    assert torch.allclose(R_mat, R_from_quat, atol=1e-6)
    assert torch.allclose(p_mat, p_quat, atol=1e-6)


def test_gauge_fix_small_omega_passthrough():
    """‖ω‖² ≤ eps ⇒ a_⊥ = a, ν_∥ = ν (Eq. S10)."""
    N = 5
    omega = torch.full((N, 3), 1e-6)  # ‖ω‖² = 3e-12 < 1e-8
    nu = torch.randn(N, 3)
    anchor = torch.randn(N, 3)
    a_perp, nu_par = gauge_fix_omega(omega, nu, anchor, eps=1e-8)
    assert torch.allclose(a_perp, anchor)
    assert torch.allclose(nu_par, nu)


def test_gauge_fix_orthogonal_to_axis():
    """After gauge fix, a_⊥ is orthogonal to ω and ν_∥ is parallel to ω."""
    omega = torch.randn(16, 3) * 0.5 + 0.1  # keep magnitude away from zero
    nu = torch.randn(16, 3)
    anchor = torch.randn(16, 3)
    a_perp, nu_par = gauge_fix_omega(omega, nu, anchor)

    # a_⊥ · ω = 0
    a_dot = (a_perp * omega).sum(dim=-1)
    assert torch.allclose(a_dot, torch.zeros_like(a_dot), atol=1e-5)

    # ν_∥ parallel to ω: ν_∥ × ω = 0
    cross = torch.cross(nu_par, omega, dim=-1)
    assert torch.allclose(cross, torch.zeros_like(cross), atol=1e-5)


def test_gauge_fix_invariance_under_axis_aligned_shift():
    """Shifting a by any δ along ω direction must not change a_⊥ (Eq. S8)."""
    omega = torch.randn(8, 3) + 0.5
    nu = torch.randn(8, 3)
    anchor = torch.randn(8, 3)

    axis_shift = omega * torch.randn(8, 1)  # δ ∥ ω
    a_perp_1, _ = gauge_fix_omega(omega, nu, anchor)
    a_perp_2, _ = gauge_fix_omega(omega, nu, anchor + axis_shift)
    assert torch.allclose(a_perp_1, a_perp_2, atol=1e-5)


def _trigs_forward(omega, nu, mu, anchor, dt):
    """Apply the TRiGS deformation µ(t) = R(µ−a_⊥) + a_⊥ + p_∥ (Eq. 9)."""
    a_perp, nu_par = gauge_fix_omega(omega, nu, anchor)
    u = torch.cat([nu_par * dt, omega * dt], dim=-1)
    R, p_par = exp_se3(u)
    centered = (mu - a_perp).unsqueeze(-1)
    rotated = (R @ centered).squeeze(-1)
    return rotated + a_perp + p_par


def test_full_forward_identity_at_central_time():
    """At t = µ_t (Δt = 0) the deformation reduces to identity."""
    N = 12
    mu = torch.randn(N, 3)
    anchor = torch.randn(N, 3)
    omega = torch.randn(N, 3)
    nu = torch.randn(N, 3)
    dt = torch.zeros(N, 1)
    mu_t = _trigs_forward(omega, nu, mu, anchor, dt)
    assert torch.allclose(mu_t, mu, atol=1e-6)


def test_full_forward_axis_aligned_anchor_invariance():
    """Axis-aligned (δ ∥ ω) anchor shift leaves the deformation unchanged.

    The paper (Eq. S5) shows a_∥ lies in the null space of (I−R), so the
    gauge-fix drops it. Confirms the chosen gauge is the canonical one the
    full forward respects. Orthogonal shifts legitimately change a_⊥ and
    the deformation — not an invariance after gauge fix.
    """
    N = 8
    mu = torch.randn(N, 3)
    anchor = torch.randn(N, 3)
    omega = torch.randn(N, 3) + 0.4
    nu = torch.randn(N, 3)
    dt = torch.full((N, 1), 0.2)

    out_a = _trigs_forward(omega, nu, mu, anchor, dt)
    axis_shift = omega * torch.randn(N, 1)
    out_b = _trigs_forward(omega, nu, mu, anchor + axis_shift, dt)
    assert torch.allclose(out_a, out_b, atol=1e-4)


def test_tau_from_window_bounds():
    """τ clamps to [0, 1] outside W (Eq. 5)."""
    mu_t = torch.tensor([[0.5]])
    log_sigma = torch.log(torch.tensor([[0.1]]))
    # t at center
    assert torch.allclose(
        tau_from_window(torch.tensor([[0.5]]), mu_t, log_sigma),
        torch.tensor([[0.5]]),
        atol=1e-6,
    )
    # Below W
    assert torch.allclose(
        tau_from_window(torch.tensor([[0.0]]), mu_t, log_sigma),
        torch.tensor([[0.0]]),
    )
    # Above W
    assert torch.allclose(
        tau_from_window(torch.tensor([[1.0]]), mu_t, log_sigma),
        torch.tensor([[1.0]]),
    )
    # W boundary: t = µ_t − 2σ ⇒ τ = 0; t = µ_t + 2σ ⇒ τ = 1
    assert torch.allclose(
        tau_from_window(torch.tensor([[0.3]]), mu_t, log_sigma),
        torch.tensor([[0.0]]),
        atol=1e-6,
    )
    assert torch.allclose(
        tau_from_window(torch.tensor([[0.7]]), mu_t, log_sigma),
        torch.tensor([[1.0]]),
        atol=1e-6,
    )


def test_bezier_endpoints_and_acceleration():
    """Bézier evaluates to c0 at τ=0 and c2 at τ=1 (Eq. 6).

    For a quadratic Bézier ζ_res(τ) = (1−τ)² c0 + 2(1−τ)τ c1 + τ² c2,
    d²ζ/dτ² = 2·(c0 − 2 c1 + c2) is constant.
    """
    c0, c1, c2 = torch.randn(6), torch.randn(6), torch.randn(6)
    assert torch.allclose(
        bezier_quadratic(torch.tensor(0.0), c0, c1, c2), c0, atol=1e-7
    )
    assert torch.allclose(
        bezier_quadratic(torch.tensor(1.0), c0, c1, c2), c2, atol=1e-7
    )
    # Midpoint value closed-form
    mid_expected = 0.25 * c0 + 0.5 * c1 + 0.25 * c2
    assert torch.allclose(
        bezier_quadratic(torch.tensor(0.5), c0, c1, c2), mid_expected, atol=1e-7
    )

    # Acceleration closed-form
    tau = torch.tensor(0.3, requires_grad=True)
    val = bezier_quadratic(tau, c0, c1, c2).sum()
    (grad_tau,) = torch.autograd.grad(val, tau, create_graph=True)
    (accel,) = torch.autograd.grad(grad_tau.sum(), tau)
    expected_accel = 2.0 * (c0 - 2 * c1 + c2).sum()
    assert torch.allclose(accel, expected_accel, atol=1e-5)
