"""SE(3) Lie algebra math utilities for continuous rigid-body parameterization."""

import torch
from torch import Tensor


def exp_se3(xi: Tensor) -> tuple[Tensor, Tensor]:
    """Exponential map from se(3) Lie algebra to SE(3) transformation.

    Computes the rotation matrix R and translation vector p from a 6D
    Lie algebra vector xi = (rho, omega), where:
      - rho is the translation component (first 3 elements)
      - omega is the rotation component (last 3 elements)

    Uses numerically stable small-angle approximations (sinc) for omega -> 0.

    Args:
        xi: [..., 6] Lie algebra vectors

    Returns:
        R: [..., 3, 3] rotation matrices
        p: [..., 3] translation vectors
    """
    rho = xi[..., :3]
    omega = xi[..., 3:]

    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)  # [..., 1]
    theta_sq = theta ** 2

    # Avoid division by zero with small angle approximations
    mask_small = (theta_sq < 1e-8).squeeze(-1)

    # Sinc formulation:
    # A = sin(theta) / theta
    # B = (1 - cos(theta)) / theta^2
    # C = (theta - sin(theta)) / theta^3

    # Safe division masks (prevent NaN in backward pass)
    theta_safe = torch.where(mask_small.unsqueeze(-1), torch.ones_like(theta), theta)
    theta_sq_safe = theta_safe ** 2
    theta_cb_safe = theta_sq_safe * theta_safe

    # Standard
    A_std = torch.sin(theta_safe) / theta_safe
    B_std = (1.0 - torch.cos(theta_safe)) / theta_sq_safe
    C_std = (theta_safe - torch.sin(theta_safe)) / theta_cb_safe

    # Small angle (Taylor expansion)
    A_small = 1.0 - theta_sq / 6.0
    B_small = 0.5 - theta_sq / 24.0
    C_small = 1.0 / 6.0 - theta_sq / 120.0

    A = torch.where(mask_small.unsqueeze(-1), A_small, A_std)
    B = torch.where(mask_small.unsqueeze(-1), B_small, B_std)
    C = torch.where(mask_small.unsqueeze(-1), C_small, C_std)

    # Hat operator for omega
    O = torch.zeros(*omega.shape[:-1], 3, 3, device=omega.device, dtype=omega.dtype)
    O[..., 0, 1] = -omega[..., 2]
    O[..., 0, 2] = omega[..., 1]
    O[..., 1, 0] = omega[..., 2]
    O[..., 1, 2] = -omega[..., 0]
    O[..., 2, 0] = -omega[..., 1]
    O[..., 2, 1] = omega[..., 0]

    O_sq = O @ O

    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(O)

    # Rodrigues' formula for rotation
    R = I + A.unsqueeze(-1) * O + B.unsqueeze(-1) * O_sq

    # Left Jacobian for SE(3) translation
    J = I + B.unsqueeze(-1) * O + C.unsqueeze(-1) * O_sq
    p = (J @ rho.unsqueeze(-1)).squeeze(-1)

    return R, p


def compose_se3(R1: Tensor, p1: Tensor, R2: Tensor, p2: Tensor) -> tuple[Tensor, Tensor]:
    """Compose two SE(3) transforms: T1 * T2.

    T1 = [R1 p1]
         [0  1 ]
    T2 = [R2 p2]
         [0  1 ]

    T1 * T2 = [R1*R2  R1*p2 + p1]
              [0      1         ]

    Args:
        R1: [..., 3, 3]
        p1: [..., 3]
        R2: [..., 3, 3]
        p2: [..., 3]

    Returns:
        R_out: [..., 3, 3] = R1 @ R2
        p_out: [..., 3] = (R1 @ p2) + p1
    """
    R_out = R1 @ R2
    p_out = (R1 @ p2.unsqueeze(-1)).squeeze(-1) + p1
    return R_out, p_out


def log_se3_from_quat_p(q: Tensor, p: Tensor) -> Tensor:
    """Logarithm map from relative quaternion and translation to se(3).

    Args:
        q: [..., 4] relative unit quaternions in WXYZ format.
        p: [..., 3] relative translation vectors.

    Returns:
        xi: [..., 6] Lie algebra vectors (rho, omega).
    """
    # 1. log_so3: from q to omega
    q = torch.where(q[..., 0:1] < 0, -q, q)
    w = q[..., 0:1]
    v = q[..., 1:]

    sin_half_theta = torch.linalg.norm(v, dim=-1, keepdim=True)
    half_theta = torch.atan2(sin_half_theta, w)

    mask_small = (sin_half_theta < 1e-8).squeeze(-1)

    sin_half_theta_safe = torch.where(
        mask_small.unsqueeze(-1),
        torch.ones_like(sin_half_theta),
        sin_half_theta
    )

    scale = torch.where(
        mask_small.unsqueeze(-1),
        torch.ones_like(sin_half_theta),
        half_theta / sin_half_theta_safe
    )
    omega = 2.0 * scale * v

    # 2. inverse left Jacobian of SO(3)
    theta = 2.0 * half_theta
    theta_sq = theta ** 2

    theta_safe = torch.where(mask_small.unsqueeze(-1), torch.ones_like(theta), theta)
    theta_sq_safe = theta_safe ** 2

    cos_theta = torch.cos(theta_safe)
    sin_theta = torch.sin(theta_safe)

    A_std = 1.0 / theta_sq_safe - (1.0 + cos_theta) / (2.0 * theta_safe * sin_theta)
    A_small = 1.0 / 12.0 + theta_sq / 720.0

    A = torch.where(mask_small.unsqueeze(-1), A_small, A_std)

    # J^{-1} p = p - 0.5 * (omega x p) + A * (omega x (omega x p))
    omega_x_p = torch.cross(omega, p, dim=-1)
    omega_x_omega_x_p = torch.cross(omega, omega_x_p, dim=-1)

    rho = p - 0.5 * omega_x_p + A * omega_x_omega_x_p

    xi = torch.cat([rho, omega], dim=-1)
    return xi


def exp_se3_quat(xi: Tensor) -> tuple[Tensor, Tensor]:
    """Vectorized Exponential map directly to Quaternion and Translation.

    Bypasses all 3x3 matrix construction.
    Returns:
        q: [..., 4] quaternions (WXYZ)
        p: [..., 3] translation vectors
    """
    rho = xi[..., :3]
    omega = xi[..., 3:]

    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
    theta_sq = theta ** 2
    mask_small = (theta_sq < 1e-8).squeeze(-1)

    theta_safe = torch.where(mask_small.unsqueeze(-1), torch.ones_like(theta), theta)
    theta_sq_safe = theta_safe ** 2
    theta_cb_safe = theta_sq_safe * theta_safe

    # 1. Direct Quaternion computation (WXYZ)
    half_theta = theta_safe * 0.5
    sin_half = torch.sin(half_theta)

    # q_w = cos(theta/2)
    q_w = torch.where(mask_small.unsqueeze(-1), 1.0 - theta_sq / 8.0, torch.cos(half_theta))
    # q_xyz = omega * sin(theta/2) / theta
    q_xyz_factor = torch.where(mask_small.unsqueeze(-1), 0.5 - theta_sq / 48.0, sin_half / theta_safe)
    q_xyz = omega * q_xyz_factor
    q = torch.cat([q_w, q_xyz], dim=-1)

    # 2. Direct Translation computation via Cross Products (No Jacobian matrix!)
    # p = rho + B*(omega x rho) + C*(omega x (omega x rho))
    B_std = (1.0 - torch.cos(theta_safe)) / theta_sq_safe
    C_std = (theta_safe - torch.sin(theta_safe)) / theta_cb_safe

    B_small = 0.5 - theta_sq / 24.0
    C_small = 1.0 / 6.0 - theta_sq / 120.0

    B = torch.where(mask_small.unsqueeze(-1), B_small, B_std)
    C = torch.where(mask_small.unsqueeze(-1), C_small, C_std)

    omega_x_rho = torch.cross(omega, rho, dim=-1)
    omega_x_omega_x_rho = torch.cross(omega, omega_x_rho, dim=-1)

    p = rho + B * omega_x_rho + C * omega_x_omega_x_rho

    # Normalize q to prevent drift
    q = torch.nn.functional.normalize(q, dim=-1)
    return q, p


def gauge_fix_omega(
    omega: Tensor,
    nu: Tensor,
    anchor: Tensor,
    eps: float = 1e-8,
    denom_clamp: float = 1e-10,
) -> tuple[Tensor, Tensor]:
    """Gauge fixing for local-anchor SE(3) (TRiGS supplement Eq. S9/S10).

    Projects the anchor onto the plane orthogonal to the rotation axis and
    retains only the axis-aligned component of the translation coefficient.
    Uses the instantaneous angular coefficient ``omega`` directly to avoid
    forming a unit rotation axis in the near-zero angular regime.

    For ``‖omega‖² > eps``:
        a_⊥ = a − (a·ω / ‖ω‖²) ω
        ν_∥ = (ν·ω / ‖ω‖²) ω
    Otherwise (translation-dominant regime):
        a_⊥ = a,  ν_∥ = ν.

    Args:
        omega: [..., 3] instantaneous angular coefficient ω (not ω·Δt).
        nu: [..., 3] instantaneous translation coefficient ν.
        anchor: [..., 3] per-primitive anchor a.
        eps: Threshold on ‖ω‖² below which we treat rotation as zero.
        denom_clamp: Minimum denominator for numerical safety.

    Returns:
        a_perp: [..., 3] gauge-fixed anchor (orthogonal component).
        nu_parallel: [..., 3] axis-aligned translation coefficient.
    """
    omega_sq = (omega * omega).sum(dim=-1, keepdim=True)  # [..., 1]
    mask_small = (omega_sq < eps)  # [..., 1]
    omega_sq_safe = omega_sq.clamp(min=denom_clamp)

    a_dot_omega = (anchor * omega).sum(dim=-1, keepdim=True)
    nu_dot_omega = (nu * omega).sum(dim=-1, keepdim=True)

    a_perp_proj = anchor - (a_dot_omega / omega_sq_safe) * omega
    nu_par_proj = (nu_dot_omega / omega_sq_safe) * omega

    a_perp = torch.where(mask_small, anchor, a_perp_proj)
    nu_parallel = torch.where(mask_small, nu, nu_par_proj)
    return a_perp, nu_parallel


def tau_from_window(
    t: Tensor,
    mu_t: Tensor,
    log_sigma_t: Tensor,
) -> Tensor:
    """Normalized local time inside the effective visibility window
    ``W = [µ_t − 2 σ_t, µ_t + 2 σ_t]`` (TRiGS Eq. 4/5).

    ``t_norm = (t − (µ_t − 2 σ_t)) / (4 σ_t)``
    ``τ     = clamp(t_norm, 0, 1)``

    Args:
        t: [..., 1] query time (scalar broadcast is fine).
        mu_t: [..., 1] per-primitive central time.
        log_sigma_t: [..., 1] per-primitive log temporal scale.

    Returns:
        tau: [..., 1] normalized local time in [0, 1].
    """
    sigma_t = torch.exp(log_sigma_t)
    t_norm = (t - (mu_t - 2.0 * sigma_t)) / (4.0 * sigma_t)
    return torch.clamp(t_norm, 0.0, 1.0)


def bezier_quadratic(
    tau: Tensor,
    c0: Tensor,
    c1: Tensor,
    c2: Tensor,
) -> Tensor:
    """Quadratic Bézier evaluation (TRiGS Eq. 6).

    ``ζ_res(τ) = (1−τ)² c0 + 2(1−τ)τ c1 + τ² c2``

    Args:
        tau: [..., 1] normalized time in [0, 1].
        c0, c1, c2: [..., D] control points (D=6 for se(3)).

    Returns:
        [..., D] Bézier value at ``τ``.
    """
    one_minus = 1.0 - tau
    b0 = one_minus * one_minus
    b1 = 2.0 * one_minus * tau
    b2 = tau * tau
    return b0 * c0 + b1 * c1 + b2 * c2


def bezier_rewindow(
    c0: Tensor,
    c1: Tensor,
    c2: Tensor,
    sigma_ratio: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Control points that keep a residual's timing when σ_t changes.

    The window ``[µ−2σ, µ+2σ]`` scales with σ, so a new σ' = r σ maps new τ'
    to old ``τ = r τ' + (1 − r) / 2``. The returned controls evaluate at τ' to
    what the old ones gave at that τ (the old quadratic, extended past its old
    window when it grows), i.e. the old curve restricted to [(1−r)/2, (1+r)/2]
    by blossoming. ``sigma_ratio`` is [..., 1].
    """
    u = 0.5 * (1.0 - sigma_ratio)
    v = 0.5 * (1.0 + sigma_ratio)

    def blossom(a: Tensor, b: Tensor) -> Tensor:
        return (1 - a) * (1 - b) * c0 + ((1 - a) * b + a * (1 - b)) * c1 + a * b * c2

    return blossom(u, u), blossom(u, v), blossom(v, v)
