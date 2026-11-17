"""Quaternion math utilities for motion computation."""

import torch
import torch.nn.functional as F
from torch import Tensor


def quaternion_conjugate(q: Tensor) -> Tensor:
    """Compute quaternion conjugate.

    Args:
        q: (N, 4) quaternions in WXYZ format [w, x, y, z]

    Returns:
        (N, 4) conjugate quaternions [w, -x, -y, -z]
    """
    sign = torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)
    return q * sign


def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Multiply quaternions (batched).

    Both quaternions should be in WXYZ format (w, x, y, z).
    Computes q1 * q2.

    Args:
        q1: (N, 4) quaternion(s)
        q2: (N, 4) quaternion(s)

    Returns:
        (N, 4) result quaternions
    """
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_relative(q_from: Tensor, q_to: Tensor) -> Tensor:
    """Compute relative quaternion: q_delta such that q_to ≈ q_delta ⊗ q_from.

    Args:
        q_from: (N, 4) source quaternions in WXYZ format
        q_to: (N, 4) target quaternions in WXYZ format

    Returns:
        (N, 4) relative quaternions
    """
    # Normalize inputs
    q_from_norm = F.normalize(q_from, dim=-1)
    q_to_norm = F.normalize(q_to, dim=-1)

    # Align hemisphere (handle q ≡ -q ambiguity)
    dot = (q_from_norm * q_to_norm).sum(dim=-1, keepdim=True)
    q_to_norm = torch.where(dot < 0, -q_to_norm, q_to_norm)

    # q_delta = q_to ⊗ conjugate(q_from)
    q_from_conj = quaternion_conjugate(q_from_norm)
    q_delta = quaternion_multiply(q_to_norm, q_from_conj)

    return F.normalize(q_delta, dim=-1)


def quaternion_slerp(q1: Tensor, q2: Tensor, t: Tensor) -> Tensor:
    """Spherical linear interpolation between quaternions.

    Args:
        q1: (N, 4) start quaternions
        q2: (N, 4) end quaternions
        t: (N,) or scalar interpolation parameter in [0, 1]

    Returns:
        (N, 4) interpolated quaternions
    """
    # Normalize inputs
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)

    # Compute dot product
    dot = (q1 * q2).sum(dim=-1, keepdim=True)

    # If q1 and q2 are in opposite hemispheres, flip q2
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.abs(dot)

    # If quaternions are very close, use linear interpolation
    close_mask = dot > 0.9995

    if isinstance(t, float):
        t = torch.full((q1.shape[0], 1), t, device=q1.device, dtype=q1.dtype)
    elif t.dim() == 1:
        t = t.unsqueeze(-1)

    # Linear interpolation for close quaternions
    result_linear = F.normalize(q1 + t * (q2 - q1), dim=-1)

    # Slerp for general case
    theta = torch.acos(dot.clamp(-1, 1))
    sin_theta = torch.sin(theta)

    # Avoid division by zero
    sin_theta = torch.where(sin_theta.abs() < 1e-6, torch.ones_like(sin_theta), sin_theta)

    s1 = torch.sin((1 - t) * theta) / sin_theta
    s2 = torch.sin(t * theta) / sin_theta
    result_slerp = s1 * q1 + s2 * q2

    # Use linear interpolation where quaternions are close
    result = torch.where(close_mask, result_linear, result_slerp)

    return F.normalize(result, dim=-1)


def axis_angle_to_quaternion(rotvec: Tensor) -> Tensor:
    """Convert axis-angle (rotation vector) to unit quaternion via exponential map.

    For small angles, uses first-order Taylor expansion for numerical stability.
    Output is always a unit quaternion by construction.

    Args:
        rotvec: [N, 3] rotation vectors (axis * angle in radians).

    Returns:
        [N, 4] unit quaternions in WXYZ format.
    """
    theta = rotvec.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # [N, 1]
    half_theta = theta * 0.5

    # For small angles: sin(θ/2)/θ ≈ 0.5, cos(θ/2) ≈ 1
    small = (theta < 1e-6).squeeze(-1)  # [N]

    w = torch.cos(half_theta)                          # [N, 1]
    xyz = rotvec * (torch.sin(half_theta) / theta)     # [N, 3]

    if small.any():
        w[small] = 1.0
        xyz[small] = rotvec[small] * 0.5

    return F.normalize(torch.cat([w, xyz], dim=-1), dim=-1)  # [N, 4]


def quaternion_to_rotmat(q: Tensor) -> Tensor:
    """Convert unit quaternion to 3x3 rotation matrix.

    Args:
        q: [N, 4] unit quaternions in WXYZ format.

    Returns:
        [N, 3, 3] rotation matrices.
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Pre-compute products
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (xx + zz),  2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),       1 - 2 * (xx + yy),
    ], dim=-1).reshape(-1, 3, 3)

    return R


def quaternion_weighted_average(quats: Tensor, weights: Tensor) -> Tensor:
    """Compute weighted average of quaternions.

    Uses weighted sum followed by normalization, which is a good approximation
    for quaternions that are relatively close to each other.

    Args:
        quats: (N, K, 4) quaternions to average
        weights: (N, K) weights (should sum to 1 along K dimension)

    Returns:
        (N, 4) averaged quaternions
    """
    # Align quaternions to first one in each group to handle hemisphere issues
    reference = quats[:, 0:1, :]  # (N, 1, 4)
    dots = (quats * reference).sum(dim=-1, keepdim=True)  # (N, K, 1)
    quats_aligned = torch.where(dots < 0, -quats, quats)

    # Weighted average
    weights_expanded = weights.unsqueeze(-1)  # (N, K, 1)
    avg = (weights_expanded * quats_aligned).sum(dim=1)  # (N, 4)

    return F.normalize(avg, dim=-1)
