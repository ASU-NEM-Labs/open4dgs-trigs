"""Utility functions for Gaussian splat transformations."""

import math
from typing import TYPE_CHECKING, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from gsply import GSTensor


def euler_to_rotation_matrix(
    rx: float, ry: float, rz: float, device, dtype
) -> Tensor:
    """Convert Euler angles (degrees) to rotation matrix.

    Uses PlayCanvas convention: extrinsic XYZ (equivalent to intrinsic ZYX).
    This means rotations are applied in fixed world axes: first X, then Y, then Z.

    Args:
        rx: Rotation around X axis in degrees
        ry: Rotation around Y axis in degrees
        rz: Rotation around Z axis in degrees
        device: Torch device
        dtype: Torch dtype

    Returns:
        (3, 3) rotation matrix
    """
    rx_rad = math.radians(rx)
    ry_rad = math.radians(ry)
    rz_rad = math.radians(rz)

    cos_x, sin_x = math.cos(rx_rad), math.sin(rx_rad)
    cos_y, sin_y = math.cos(ry_rad), math.sin(ry_rad)
    cos_z, sin_z = math.cos(rz_rad), math.sin(rz_rad)

    Rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, cos_x, -sin_x],
        [0.0, sin_x, cos_x],
    ], device=device, dtype=dtype)

    Ry = torch.tensor([
        [cos_y, 0.0, sin_y],
        [0.0, 1.0, 0.0],
        [-sin_y, 0.0, cos_y],
    ], device=device, dtype=dtype)

    Rz = torch.tensor([
        [cos_z, -sin_z, 0.0],
        [sin_z, cos_z, 0.0],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=dtype)

    # Extrinsic XYZ (PlayCanvas convention) = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def rotation_matrix_to_quaternion(R: Tensor) -> Tensor:
    """Convert rotation matrix to quaternion.

    Uses the Shepperd method for numerical stability.
    Returns quaternion in WXYZ format (w, x, y, z).

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (4,) quaternion tensor [w, x, y, z]
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = torch.stack([w, x, y, z])
    q = q / torch.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def quat_inv(q: Tensor) -> Tensor:
    inv = q.clone()
    inv[..., 1:] = -inv[..., 1:]
    return inv


def quat_apply(q: Tensor, v: Tensor) -> Tensor:
    """Fast rotation of vector v by quaternion q (WXYZ)."""
    if q.dim() == 1 and v.dim() > 1:
        q = q.unsqueeze(0).expand(v.shape[0], -1)
    q_w = q[..., 0:1]
    q_xyz = q[..., 1:]
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    if q1.dim() == 1 and q2.dim() > 1:
        q1 = q1.unsqueeze(0).expand(q2.shape[0], -1)
    if q2.dim() == 1 and q1.dim() > 1:
        q2 = q2.unsqueeze(0).expand(q1.shape[0], -1)
        
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def rotate_sh(sh: Tensor, R: Tensor) -> Tensor:
    """Rotate spherical harmonics coefficients.

    Only rotates bands 1-3 (band 0 is rotation-invariant).

    Args:
        sh: (N, K, 3) SH coefficients where K is number of bands
        R: (3, 3) rotation matrix

    Returns:
        (N, K, 3) rotated SH coefficients
    """
    if sh is None or sh.numel() == 0:
        return sh

    n, k, c = sh.shape
    device = sh.device
    dtype = sh.dtype

    r = R.flatten()
    result = sh.clone()

    # Band 1 (indices 0-2): transforms like a vector
    if k >= 3:
        sh1 = sh[:, 0:3, :]
        for c_idx in range(c):
            v = sh1[:, :, c_idx]
            # Map gsplat SH basis to Cartesian [X, Y, Z]
            # gsplat basis: Y_1^-1 ~ -y,  Y_1^0 ~ z,  Y_1^1 ~ -x
            v_xyz = torch.stack([-v[:, 2], -v[:, 0], v[:, 1]], dim=-1)
            # Apply standard rotation
            v_rot = (R @ v_xyz.T).T
            # Map back to gsplat SH basis
            result[:, 0, c_idx] = -v_rot[:, 1]
            result[:, 1, c_idx] = v_rot[:, 2]
            result[:, 2, c_idx] = -v_rot[:, 0]

    # Band 2 (indices 3-7): 5x5 rotation matrix
    if k >= 8:
        sh2_rot = torch.zeros(5, 5, device=device, dtype=dtype)

        r00, r01, r02 = r[0], r[1], r[2]
        r10, r11, r12 = r[3], r[4], r[5]
        r20, r21, r22 = r[6], r[7], r[8]

        sh2_rot[0, 0] = 1.0*r00*r11 + 1.0*r01*r10
        sh2_rot[0, 1] = -2.0*r01*r12 - 2.0*r02*r11
        sh2_rot[0, 2] = (2.0/3.0)*r02*r12
        sh2_rot[0, 3] = -2.0*r00*r12 - 2.0*r02*r10
        sh2_rot[0, 4] = 1.0*r00*r10 - 1.0*r01*r11
        sh2_rot[1, 0] = -0.5*r10*r21 - 0.5*r11*r20
        sh2_rot[1, 1] = 1.0*r11*r22 + 1.0*r12*r21
        sh2_rot[1, 2] = -(1.0/3.0)*r12*r22
        sh2_rot[1, 3] = 1.0*r10*r22 + 1.0*r12*r20
        sh2_rot[1, 4] = -0.5*r10*r20 + 0.5*r11*r21
        sh2_rot[2, 0] = 3.0*r20*r21
        sh2_rot[2, 1] = -6.0*r21*r22
        sh2_rot[2, 2] = 1.0*r22**2
        sh2_rot[2, 3] = -6.0*r20*r22
        sh2_rot[2, 4] = 1.5*r20**2 - 1.5*r21**2
        sh2_rot[3, 0] = -0.5*r00*r21 - 0.5*r01*r20
        sh2_rot[3, 1] = 1.0*r01*r22 + 1.0*r02*r21
        sh2_rot[3, 2] = -(1.0/3.0)*r02*r22
        sh2_rot[3, 3] = 1.0*r00*r22 + 1.0*r02*r20
        sh2_rot[3, 4] = -0.5*r00*r20 + 0.5*r01*r21
        sh2_rot[4, 0] = 1.0*r00*r01 - 1.0*r10*r11
        sh2_rot[4, 1] = -2.0*r01*r02 + 2.0*r11*r12
        sh2_rot[4, 2] = (1.0/3.0)*r02**2 - (1.0/3.0)*r12**2
        sh2_rot[4, 3] = -2.0*r00*r02 + 2.0*r10*r12
        sh2_rot[4, 4] = 0.5*r00**2 - 0.5*r01**2 - 0.5*r10**2 + 0.5*r11**2

        for c_idx in range(c):
            v = sh[:, 3:8, c_idx]
            result[:, 3:8, c_idx] = (sh2_rot @ v.T).T

    # Band 3 (indices 8-14): 7x7 rotation matrix
    if k >= 15:
        sh3_rot = torch.zeros(7, 7, device=device, dtype=dtype)
        sh3_rot[0, 0] = 9*r00**2*r11/10 + 9*r00*r01*r10/5 - 3*r01**2*r11/10 - 9*r10**2*r11/10 + r11**3/10
        sh3_rot[0, 1] = -3*r00*r01*r12 - 3*r00*r02*r11 - 3*r01*r02*r10 + 3*r10*r11*r12
        sh3_rot[0, 2] = 6*r01*r02*r12/5 + 3*r02**2*r11/5 - 3*r11*r12**2/5
        sh3_rot[0, 3] = r12*(-3*r02**2 + r12**2)/5
        sh3_rot[0, 4] = 6*r00*r02*r12/5 + 3*r02**2*r10/5 - 3*r10*r12**2/5
        sh3_rot[0, 5] = -3*r00**2*r12/2 - 3*r00*r02*r10 + 3*r01**2*r12/2 + 3*r01*r02*r11 + 3*r10**2*r12/2 - 3*r11**2*r12/2
        sh3_rot[0, 6] = 3*r00**2*r10/10 - 9*r00*r01*r11/5 - 9*r01**2*r10/10 - r10**3/10 + 9*r10*r11**2/10
        sh3_rot[1, 0] = -3*r00*r10*r21/5 - 3*r00*r11*r20/5 - 3*r01*r10*r20/5 + r01*r11*r21/5
        sh3_rot[1, 1] = r00*r11*r22 + r00*r12*r21 + r01*r10*r22 + r01*r12*r20 + r02*r10*r21 + r02*r11*r20
        sh3_rot[1, 2] = -2*r01*r12*r22/5 - 2*r02*r11*r22/5 - 2*r02*r12*r21/5
        sh3_rot[1, 3] = 2*r02*r12*r22/5
        sh3_rot[1, 4] = -2*r00*r12*r22/5 - 2*r02*r10*r22/5 - 2*r02*r12*r20/5
        sh3_rot[1, 5] = r00*r10*r22 + r00*r12*r20 - r01*r11*r22 - r01*r12*r21 + r02*r10*r20 - r02*r11*r21
        sh3_rot[1, 6] = -r00*r10*r20/5 + 3*r00*r11*r21/5 + 3*r01*r10*r21/5 + 3*r01*r11*r20/5
        sh3_rot[2, 0] = 3*r10*r20*r21 + 3*r11*r20**2/2 - r11*r21**2/2
        sh3_rot[2, 1] = -5*r10*r21*r22 - 5*r11*r20*r22 - 5*r12*r20*r21
        sh3_rot[2, 2] = r22*(r11*r22 + 2*r12*r21)
        sh3_rot[2, 3] = -r12*r22**2
        sh3_rot[2, 4] = r22*(r10*r22 + 2*r12*r20)
        sh3_rot[2, 5] = -5*r10*r20*r22 + 5*r11*r21*r22 - 5*r12*r20**2/2 + 5*r12*r21**2/2
        sh3_rot[2, 6] = r10*r20**2/2 - 3*r10*r21**2/2 - 3*r11*r20*r21
        sh3_rot[3, 0] = r21*(-9*r20**2 + r21**2)/2
        sh3_rot[3, 1] = 15*r20*r21*r22
        sh3_rot[3, 2] = -3*r21*r22**2
        sh3_rot[3, 3] = r22**3
        sh3_rot[3, 4] = -3*r20*r22**2
        sh3_rot[3, 5] = 15*r22*(r20**2 - r21**2)/2
        sh3_rot[3, 6] = r20*(-r20**2 + 9*r21**2)/2
        sh3_rot[4, 0] = 3*r00*r20*r21 + 3*r01*r20**2/2 - r01*r21**2/2
        sh3_rot[4, 1] = -5*r00*r21*r22 - 5*r01*r20*r22 - 5*r02*r20*r21
        sh3_rot[4, 2] = r22*(r01*r22 + 2*r02*r21)
        sh3_rot[4, 3] = -r02*r22**2
        sh3_rot[4, 4] = r22*(r00*r22 + 2*r02*r20)
        sh3_rot[4, 5] = -5*r00*r20*r22 + 5*r01*r21*r22 - 5*r02*r20**2/2 + 5*r02*r21**2/2
        sh3_rot[4, 6] = r00*r20**2/2 - 3*r00*r21**2/2 - 3*r01*r20*r21
        sh3_rot[5, 0] = -3*r00**2*r21/10 - 3*r00*r01*r20/5 + r01**2*r21/10 + 3*r10**2*r21/10 + 3*r10*r11*r20/5 - r11**2*r21/10
        sh3_rot[5, 1] = r00*r01*r22 + r00*r02*r21 + r01*r02*r20 - r10*r11*r22 - r10*r12*r21 - r11*r12*r20
        sh3_rot[5, 2] = -2*r01*r02*r22/5 - r02**2*r21/5 + 2*r11*r12*r22/5 + r12**2*r21/5
        sh3_rot[5, 3] = r22*(r02**2 - r12**2)/5
        sh3_rot[5, 4] = -2*r00*r02*r22/5 - r02**2*r20/5 + 2*r10*r12*r22/5 + r12**2*r20/5
        sh3_rot[5, 5] = r00**2*r22/2 + r00*r02*r20 - r01**2*r22/2 - r01*r02*r21 - r10**2*r22/2 - r10*r12*r20 + r11**2*r22/2 + r11*r12*r21
        sh3_rot[5, 6] = -r00**2*r20/10 + 3*r00*r01*r21/5 + 3*r01**2*r20/10 + r10**2*r20/10 - 3*r10*r11*r21/5 - 3*r11**2*r20/10
        sh3_rot[6, 0] = 9*r00**2*r01/10 - 9*r00*r10*r11/5 - r01**3/10 - 9*r01*r10**2/10 + 3*r01*r11**2/10
        sh3_rot[6, 1] = -3*r00*r01*r02 + 3*r00*r11*r12 + 3*r01*r10*r12 + 3*r02*r10*r11
        sh3_rot[6, 2] = 3*r01*r02**2/5 - 3*r01*r12**2/5 - 6*r02*r11*r12/5
        sh3_rot[6, 3] = r02*(-r02**2 + 3*r12**2)/5
        sh3_rot[6, 4] = 3*r00*r02**2/5 - 3*r00*r12**2/5 - 6*r02*r10*r12/5
        sh3_rot[6, 5] = -3*r00**2*r02/2 + 3*r00*r10*r12 + 3*r01**2*r02/2 - 3*r01*r11*r12 + 3*r02*r10**2/2 - 3*r02*r11**2/2
        sh3_rot[6, 6] = r00**3/10 - 9*r00*r01**2/10 - 3*r00*r10**2/10 + 9*r00*r11**2/10 + 9*r01*r10*r11/5

        for c_idx in range(c):
            v = sh[:, 8:15, c_idx]
            result[:, 8:15, c_idx] = (sh3_rot @ v.T).T

    return result


def rotate_sh_batched(sh: Tensor, R: Tensor) -> Tensor:
    """Rotate SH coefficients with per-Gaussian rotation matrices.

    Only rotates bands 1-3 (band 0 is rotation-invariant).
    Rotates bands 1-3.

    Args:
        sh: (N, K, 3) SH coefficients
        R: (N, 3, 3) per-Gaussian rotation matrices

    Returns:
        (N, K, 3) rotated SH coefficients
    """
    if sh is None or sh.numel() == 0:
        return sh

    n, k, c = sh.shape
    result = sh.clone()

    # Band 1 (indices 0-2): transforms like a vector
    # sh1[:, :, c] is a 3-vector per Gaussian, rotated by R
    if k >= 3:
        for c_idx in range(c):
            v = sh[:, 0:3, c_idx]  # [N, 3]
            # Map gsplat SH basis to Cartesian [X, Y, Z]
            # gsplat basis: Y_1^-1 ~ -y,  Y_1^0 ~ z,  Y_1^1 ~ -x
            v_xyz = torch.stack([-v[:, 2], -v[:, 0], v[:, 1]], dim=-1)
            # Apply per-Gaussian rotation
            v_rot = (R @ v_xyz.unsqueeze(-1)).squeeze(-1)
            # Map back to gsplat SH basis
            result[:, 0, c_idx] = -v_rot[:, 1]
            result[:, 1, c_idx] = v_rot[:, 2]
            result[:, 2, c_idx] = -v_rot[:, 0]

    # Band 2 (indices 3-7): 5x5 rotation per Gaussian
    if k >= 8:
        r00 = R[:, 0, 0]; r01 = R[:, 0, 1]; r02 = R[:, 0, 2]
        r10 = R[:, 1, 0]; r11 = R[:, 1, 1]; r12 = R[:, 1, 2]
        r20 = R[:, 2, 0]; r21 = R[:, 2, 1]; r22 = R[:, 2, 2]

        # Build [N, 5, 5] rotation matrices for band 2
        sh2_rot = torch.zeros(n, 5, 5, device=sh.device, dtype=sh.dtype)
        sh2_rot[:, 0, 0] = 1.0*r00*r11 + 1.0*r01*r10
        sh2_rot[:, 0, 1] = -2.0*r01*r12 - 2.0*r02*r11
        sh2_rot[:, 0, 2] = (2.0/3.0)*r02*r12
        sh2_rot[:, 0, 3] = -2.0*r00*r12 - 2.0*r02*r10
        sh2_rot[:, 0, 4] = 1.0*r00*r10 - 1.0*r01*r11
        sh2_rot[:, 1, 0] = -0.5*r10*r21 - 0.5*r11*r20
        sh2_rot[:, 1, 1] = 1.0*r11*r22 + 1.0*r12*r21
        sh2_rot[:, 1, 2] = -(1.0/3.0)*r12*r22
        sh2_rot[:, 1, 3] = 1.0*r10*r22 + 1.0*r12*r20
        sh2_rot[:, 1, 4] = -0.5*r10*r20 + 0.5*r11*r21
        sh2_rot[:, 2, 0] = 3.0*r20*r21
        sh2_rot[:, 2, 1] = -6.0*r21*r22
        sh2_rot[:, 2, 2] = 1.0*r22**2
        sh2_rot[:, 2, 3] = -6.0*r20*r22
        sh2_rot[:, 2, 4] = 1.5*r20**2 - 1.5*r21**2
        sh2_rot[:, 3, 0] = -0.5*r00*r21 - 0.5*r01*r20
        sh2_rot[:, 3, 1] = 1.0*r01*r22 + 1.0*r02*r21
        sh2_rot[:, 3, 2] = -(1.0/3.0)*r02*r22
        sh2_rot[:, 3, 3] = 1.0*r00*r22 + 1.0*r02*r20
        sh2_rot[:, 3, 4] = -0.5*r00*r20 + 0.5*r01*r21
        sh2_rot[:, 4, 0] = 1.0*r00*r01 - 1.0*r10*r11
        sh2_rot[:, 4, 1] = -2.0*r01*r02 + 2.0*r11*r12
        sh2_rot[:, 4, 2] = (1.0/3.0)*r02**2 - (1.0/3.0)*r12**2
        sh2_rot[:, 4, 3] = -2.0*r00*r02 + 2.0*r10*r12
        sh2_rot[:, 4, 4] = 0.5*r00**2 - 0.5*r01**2 - 0.5*r10**2 + 0.5*r11**2

        for c_idx in range(c):
            v = sh[:, 3:8, c_idx]  # [N, 5]
            result[:, 3:8, c_idx] = (sh2_rot @ v.unsqueeze(-1)).squeeze(-1)

    # Band 3 (indices 8-14): 7x7 rotation per Gaussian
    if k >= 15:
        sh3_rot = torch.zeros(n, 7, 7, device=sh.device, dtype=sh.dtype)
        sh3_rot[:, 0, 0] = 9*r00**2*r11/10 + 9*r00*r01*r10/5 - 3*r01**2*r11/10 - 9*r10**2*r11/10 + r11**3/10
        sh3_rot[:, 0, 1] = -3*r00*r01*r12 - 3*r00*r02*r11 - 3*r01*r02*r10 + 3*r10*r11*r12
        sh3_rot[:, 0, 2] = 6*r01*r02*r12/5 + 3*r02**2*r11/5 - 3*r11*r12**2/5
        sh3_rot[:, 0, 3] = r12*(-3*r02**2 + r12**2)/5
        sh3_rot[:, 0, 4] = 6*r00*r02*r12/5 + 3*r02**2*r10/5 - 3*r10*r12**2/5
        sh3_rot[:, 0, 5] = -3*r00**2*r12/2 - 3*r00*r02*r10 + 3*r01**2*r12/2 + 3*r01*r02*r11 + 3*r10**2*r12/2 - 3*r11**2*r12/2
        sh3_rot[:, 0, 6] = 3*r00**2*r10/10 - 9*r00*r01*r11/5 - 9*r01**2*r10/10 - r10**3/10 + 9*r10*r11**2/10
        sh3_rot[:, 1, 0] = -3*r00*r10*r21/5 - 3*r00*r11*r20/5 - 3*r01*r10*r20/5 + r01*r11*r21/5
        sh3_rot[:, 1, 1] = r00*r11*r22 + r00*r12*r21 + r01*r10*r22 + r01*r12*r20 + r02*r10*r21 + r02*r11*r20
        sh3_rot[:, 1, 2] = -2*r01*r12*r22/5 - 2*r02*r11*r22/5 - 2*r02*r12*r21/5
        sh3_rot[:, 1, 3] = 2*r02*r12*r22/5
        sh3_rot[:, 1, 4] = -2*r00*r12*r22/5 - 2*r02*r10*r22/5 - 2*r02*r12*r20/5
        sh3_rot[:, 1, 5] = r00*r10*r22 + r00*r12*r20 - r01*r11*r22 - r01*r12*r21 + r02*r10*r20 - r02*r11*r21
        sh3_rot[:, 1, 6] = -r00*r10*r20/5 + 3*r00*r11*r21/5 + 3*r01*r10*r21/5 + 3*r01*r11*r20/5
        sh3_rot[:, 2, 0] = 3*r10*r20*r21 + 3*r11*r20**2/2 - r11*r21**2/2
        sh3_rot[:, 2, 1] = -5*r10*r21*r22 - 5*r11*r20*r22 - 5*r12*r20*r21
        sh3_rot[:, 2, 2] = r22*(r11*r22 + 2*r12*r21)
        sh3_rot[:, 2, 3] = -r12*r22**2
        sh3_rot[:, 2, 4] = r22*(r10*r22 + 2*r12*r20)
        sh3_rot[:, 2, 5] = -5*r10*r20*r22 + 5*r11*r21*r22 - 5*r12*r20**2/2 + 5*r12*r21**2/2
        sh3_rot[:, 2, 6] = r10*r20**2/2 - 3*r10*r21**2/2 - 3*r11*r20*r21
        sh3_rot[:, 3, 0] = r21*(-9*r20**2 + r21**2)/2
        sh3_rot[:, 3, 1] = 15*r20*r21*r22
        sh3_rot[:, 3, 2] = -3*r21*r22**2
        sh3_rot[:, 3, 3] = r22**3
        sh3_rot[:, 3, 4] = -3*r20*r22**2
        sh3_rot[:, 3, 5] = 15*r22*(r20**2 - r21**2)/2
        sh3_rot[:, 3, 6] = r20*(-r20**2 + 9*r21**2)/2
        sh3_rot[:, 4, 0] = 3*r00*r20*r21 + 3*r01*r20**2/2 - r01*r21**2/2
        sh3_rot[:, 4, 1] = -5*r00*r21*r22 - 5*r01*r20*r22 - 5*r02*r20*r21
        sh3_rot[:, 4, 2] = r22*(r01*r22 + 2*r02*r21)
        sh3_rot[:, 4, 3] = -r02*r22**2
        sh3_rot[:, 4, 4] = r22*(r00*r22 + 2*r02*r20)
        sh3_rot[:, 4, 5] = -5*r00*r20*r22 + 5*r01*r21*r22 - 5*r02*r20**2/2 + 5*r02*r21**2/2
        sh3_rot[:, 4, 6] = r00*r20**2/2 - 3*r00*r21**2/2 - 3*r01*r20*r21
        sh3_rot[:, 5, 0] = -3*r00**2*r21/10 - 3*r00*r01*r20/5 + r01**2*r21/10 + 3*r10**2*r21/10 + 3*r10*r11*r20/5 - r11**2*r21/10
        sh3_rot[:, 5, 1] = r00*r01*r22 + r00*r02*r21 + r01*r02*r20 - r10*r11*r22 - r10*r12*r21 - r11*r12*r20
        sh3_rot[:, 5, 2] = -2*r01*r02*r22/5 - r02**2*r21/5 + 2*r11*r12*r22/5 + r12**2*r21/5
        sh3_rot[:, 5, 3] = r22*(r02**2 - r12**2)/5
        sh3_rot[:, 5, 4] = -2*r00*r02*r22/5 - r02**2*r20/5 + 2*r10*r12*r22/5 + r12**2*r20/5
        sh3_rot[:, 5, 5] = r00**2*r22/2 + r00*r02*r20 - r01**2*r22/2 - r01*r02*r21 - r10**2*r22/2 - r10*r12*r20 + r11**2*r22/2 + r11*r12*r21
        sh3_rot[:, 5, 6] = -r00**2*r20/10 + 3*r00*r01*r21/5 + 3*r01**2*r20/10 + r10**2*r20/10 - 3*r10*r11*r21/5 - 3*r11**2*r20/10
        sh3_rot[:, 6, 0] = 9*r00**2*r01/10 - 9*r00*r10*r11/5 - r01**3/10 - 9*r01*r10**2/10 + 3*r01*r11**2/10
        sh3_rot[:, 6, 1] = -3*r00*r01*r02 + 3*r00*r11*r12 + 3*r01*r10*r12 + 3*r02*r10*r11
        sh3_rot[:, 6, 2] = 3*r01*r02**2/5 - 3*r01*r12**2/5 - 6*r02*r11*r12/5
        sh3_rot[:, 6, 3] = r02*(-r02**2 + 3*r12**2)/5
        sh3_rot[:, 6, 4] = 3*r00*r02**2/5 - 3*r00*r12**2/5 - 6*r02*r10*r12/5
        sh3_rot[:, 6, 5] = -3*r00**2*r02/2 + 3*r00*r10*r12 + 3*r01**2*r02/2 - 3*r01*r11*r12 + 3*r02*r10**2/2 - 3*r02*r11**2/2
        sh3_rot[:, 6, 6] = r00**3/10 - 9*r00*r01**2/10 - 3*r00*r10**2/10 + 9*r00*r11**2/10 + 9*r01*r10*r11/5

        for c_idx in range(c):
            v = sh[:, 8:15, c_idx]  # [N, 7]
            result[:, 8:15, c_idx] = (sh3_rot @ v.unsqueeze(-1)).squeeze(-1)

    return result


def rotate_gaussians(
    means: Tensor,
    quats: Tensor,
    shN: Tensor | None,
    rotation: Tuple[float, float, float],
) -> Tuple[Tensor, Tensor, Tensor | None]:
    """Rotate Gaussian splat data around the origin.

    Args:
        means: (N, 3) Gaussian positions
        quats: (N, 4) Gaussian quaternions in WXYZ format
        shN: (N, K, 3) Higher-order SH coefficients or None
        rotation: (rx, ry, rz) Euler angles in degrees (intrinsic XYZ)

    Returns:
        Tuple of (rotated_means, rotated_quats, rotated_shN)
    """
    rx, ry, rz = rotation

    if rx == 0 and ry == 0 and rz == 0:
        return means, quats, shN

    device = means.device
    dtype = means.dtype

    R = euler_to_rotation_matrix(rx, ry, rz, device, dtype)
    q_rot = rotation_matrix_to_quaternion(R)

    # Rotate positions around origin
    rotated_means = (R @ means.T).T

    # Rotate quaternions: q_new = q_rot * q_old
    rotated_quats = quat_multiply(q_rot, quats)
    rotated_quats = rotated_quats / torch.linalg.norm(rotated_quats, dim=1, keepdim=True)

    # Rotate SH coefficients (band 0/sh0 is rotation-invariant, only shN needs rotation)
    rotated_shN = rotate_sh(shN, R) if shN is not None else None

    return rotated_means, rotated_quats, rotated_shN


def transform_gaussians(
    means: Tensor,
    quats: Tensor,
    shN: Tensor | None,
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Tensor, Tensor, Tensor | None]:
    """Transform Gaussians: apply rotation (around origin) then translation.

    Args:
        means: (N, 3) Gaussian positions
        quats: (N, 4) Gaussian quaternions in WXYZ format
        shN: (N, K, 3) Higher-order SH coefficients or None
        rotation: (rx, ry, rz) Euler angles in degrees (PlayCanvas convention)
        translation: (tx, ty, tz) Translation vector

    Returns:
        Tuple of (means, quats, shN) - transformed
    """
    # Apply rotation first
    means, quats, shN = rotate_gaussians(means, quats, shN, rotation)

    # Apply translation
    tx, ty, tz = translation
    if tx != 0 or ty != 0 or tz != 0:
        offset = torch.tensor(
            [tx, ty, tz],
            device=means.device,
            dtype=means.dtype,
        )
        means = means + offset

    return means, quats, shN


def transform(
    gs: "GSTensor",
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> "GSTensor":
    """Transform a GSTensor: apply rotation (around origin) then translation.

    Args:
        gs: GSTensor to transform
        rotation: (rx, ry, rz) Euler angles in degrees (PlayCanvas convention)
        translation: (tx, ty, tz) Translation vector

    Returns:
        New GSTensor with transformed positions and orientations
    """
    from gsply import GSTensor

    # Skip if no transform needed
    if rotation == (0, 0, 0) and translation == (0, 0, 0):
        return gs

    # Transform the tensors
    means, quats, shN = transform_gaussians(
        gs.means, gs.quats, gs.shN,
        rotation=rotation,
        translation=translation,
    )

    # Create new GSTensor with transformed data
    return GSTensor.from_arrays(
        means=means,
        scales=gs.scales,
        quats=quats,
        opacities=gs.opacities,
        sh0=gs.sh0,
        shN=shN,
        format="ply",
        device=gs.device,
    )
