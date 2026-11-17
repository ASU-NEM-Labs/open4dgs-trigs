"""PLY-sequence initialization path.

The dense-PLY flow is handled by :meth:`TrigsTrainer._build_canonical_set`.
This module owns PLY-specific discovery and early capacity selection helpers.
Capacity must be reduced per keyframe before velocity/anchor initialization;
concatenating every dense keyframe first makes low-stride sequences too large
to initialize or load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch import Tensor


@dataclass
class PlyMotionInitConfig:
    """Motion-adaptive canonical selection for the PLY path only.

    The default PLY init gives every row the same temporal window and
    re-instantiates a decimated copy of the whole surface at every keyframe.
    A stationary point therefore gets one canonical row per keyframe -- 261
    copies of the same surface for a stride-3 sequence at 30 fps -- while a
    fast limb gets no more coverage than the torso.  One global sigma cannot
    serve both: wide enough for the torso means fast rows are transported far
    beyond where the constant-twist model stays valid (ghosting), narrow
    enough for a foot means the torso leaves temporal gaps (sparse popping).

    Here sigma is derived per row from how far the row actually moves, and
    rows are then kept in proportion to how many cohorts they do NOT already
    span.  Slow rows survive in a small fraction of keyframes with long
    windows; fast rows stay at full density with short ones.
    """

    enabled: bool = False
    """Enable motion-adaptive sigma + coverage dedup (PLY path only)."""

    travel_radii: float = 1.5
    """Row travel tolerated within its own window, in multiples of its size.

    A row is a valid stand-in for a surface patch only while its motion has
    not carried it far off that patch. Larger values keep more rows alive
    longer and shrink the canonical set at the cost of motion fidelity.
    """

    sigma_floor_s: float = 0.0
    """Lower bound on derived sigma. ``0`` uses half the cohort interval."""

    sigma_ceiling_s: float = 0.0
    """Upper bound on derived sigma. ``0`` uses 8x the cohort interval.

    The trainer additionally clamps sigma to ``config.max_sigma_t``; set that
    at least this high or long windows are silently clipped back.
    """

    redundancy: float = 1.5
    """Surviving copies per row per span. 1.0 is bare coverage, no overlap."""

    support_sigmas: float = 2.0
    """Half-width of a row's usable support, in sigmas."""

    seed: int = 12_345
    """Base seed; each keyframe offsets it so selection is reproducible."""


@torch.no_grad()
def motion_adaptive_sigma(
    accels: Tensor,
    radii: Tensor,
    *,
    sigma_floor: float,
    sigma_ceiling: float,
    travel_radii: float,
) -> Tensor:
    """Per-row temporal window from how badly constant twist fits the row.

    A row is carried by ``exp(zeta * dt)`` -- constant translational and
    angular velocity -- so uniform motion is represented *exactly* however
    fast it is. A torso translating at walking pace needs no shorter a window
    than a stationary one. What the model cannot follow is acceleration, whose
    unmodelled displacement grows as ``0.5 * |a| * dt^2``.

    Solving ``0.5 * |a| * sigma^2 = travel_radii * radius`` gives the window
    over which a row stays within ``travel_radii`` of its own size:

        sigma = sqrt(2 * travel_radii * radius / |a|)

    Uniformly moving rows saturate at the ceiling; a swinging foot reversing
    direction lands on the floor.
    """
    if sigma_floor <= 0.0 or sigma_ceiling < sigma_floor:
        raise ValueError("require 0 < sigma_floor <= sigma_ceiling")
    accel = accels.clamp_min(1e-8)
    sigma = torch.sqrt(
        (2.0 * float(travel_radii) * radii.clamp_min(1e-8)) / accel
    )
    return sigma.clamp(float(sigma_floor), float(sigma_ceiling))


@torch.no_grad()
def coverage_keep_prob(
    sigma: Tensor,
    *,
    cohort_dt: float,
    redundancy: float,
    support_sigmas: float,
) -> Tensor:
    """Fraction of spanned keyframes in which a row needs to be instantiated.

    A row with support half-width ``support_sigmas * sigma`` already covers
    ``2 * support_sigmas * sigma / cohort_dt`` keyframes, so instantiating it
    in every one of them is redundant. Rows spanning at most one cohort keep
    probability 1 and are never thinned.
    """
    if cohort_dt <= 0.0:
        raise ValueError("cohort_dt must be positive")
    span = (2.0 * float(support_sigmas) * sigma).clamp_min(float(cohort_dt))
    return (float(redundancy) * float(cohort_dt) / span).clamp(0.0, 1.0)


@torch.no_grad()
def coverage_keep_mask(
    sigma: Tensor,
    *,
    cohort_dt: float,
    redundancy: float,
    support_sigmas: float,
    seed: int,
) -> Tensor:
    """Reproducible per-row keep mask from :func:`coverage_keep_prob`.

    Independent Bernoulli draws spread survivors evenly over time without the
    banding a strided rule would produce, and without a cross-keyframe kNN.
    """
    prob = coverage_keep_prob(
        sigma,
        cohort_dt=cohort_dt,
        redundancy=redundancy,
        support_sigmas=support_sigmas,
    )
    generator = torch.Generator(device=sigma.device)
    generator.manual_seed(int(seed))
    draw = torch.rand(
        sigma.shape, device=sigma.device, dtype=torch.float32, generator=generator,
    )
    keep = draw < prob
    if not bool(keep.any()):
        # Never let a keyframe vanish; retain its most motion-critical row.
        keep[int(sigma.argmin().item())] = True
    return keep


def read_ply_vertex_count(path: str | Path) -> int:
    """Read the declared vertex count without loading a PLY payload."""
    path = Path(path)
    vertex_count: int | None = None
    header_bytes = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            header_bytes += len(line)
            if header_bytes > 1_048_576:
                raise ValueError(f"PLY header exceeds 1 MiB: {path}")
            text = line.decode("ascii", errors="strict").strip()
            fields = text.split()
            if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            if text == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"PLY header has no vertex element: {path}")
    if vertex_count < 0:
        raise ValueError(f"PLY vertex count must be non-negative: {path}")
    return vertex_count


def count_keyframe_gaussians(
    ply_files: List[Path],
    keyframe_stride: int,
) -> int:
    """Count source rows used by the PLY canonical keyframes."""
    stride = max(int(keyframe_stride), 1)
    return sum(read_ply_vertex_count(path) for path in ply_files[::stride])


def discover_ply_files(input_dir: str | Path) -> List[Path]:
    """Return PLY files in ``input_dir`` sorted by filename.

    Args:
        input_dir: Directory containing per-frame dense Gaussian PLYs.

    Returns:
        Sorted list of PLY paths.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"PLY input directory does not exist: {input_dir}")
    files = sorted(input_dir.glob("*.ply"))
    if not files:
        raise FileNotFoundError(f"No PLY files found in {input_dir}")
    return files


def allocate_keyframe_quotas(target_count: int, n_keyframes: int) -> List[int]:
    """Divide an exact total row budget evenly across temporal keyframes."""
    target_count = int(target_count)
    n_keyframes = int(n_keyframes)
    if n_keyframes <= 0:
        raise ValueError("n_keyframes must be positive")
    if target_count < n_keyframes:
        raise ValueError(
            f"target_count={target_count} must be at least the number of "
            f"keyframes ({n_keyframes})"
        )
    base, remainder = divmod(target_count, n_keyframes)
    return [base + (1 if i < remainder else 0) for i in range(n_keyframes)]


@torch.no_grad()
def select_keyframe_gaussians(
    means: Tensor,
    quota: int,
    *,
    seed: int,
) -> Tensor:
    """Select a reproducible subset from one dense PLY.

    Dense per-frame PLYs already contain optimized opacity and scale values.
    Using a uniform random subset preserves the underlying density distribution
    of the original point cloud without deleting dense structures.
    """
    n = int(means.shape[0])
    quota = int(quota)
    if quota <= 0:
        raise ValueError("quota must be positive")
    if quota >= n:
        return torch.arange(n, device=means.device, dtype=torch.long)
    if means.ndim != 2 or int(means.shape[1]) != 3:
        raise ValueError("means must have shape [N, 3]")

    try:
        gen = torch.Generator(device=means.device)
        gen.manual_seed(int(seed))
    except Exception:
        gen = None

    if gen is not None:
        perm = torch.randperm(n, device=means.device, generator=gen)[:quota]
    else:
        perm = torch.randperm(n, device=means.device)[:quota]
        
    return perm.sort().values
