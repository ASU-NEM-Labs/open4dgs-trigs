"""Thin wrappers around ``gsply.GSTensor`` for TriGS I/O conveniences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gsply import GSTensor


def _load_original_3dgs_ply(path: Path, device: str | None) -> GSTensor:
    """Load the original 3DGS binary PLY layout, including dummy normals.

    Original INRIA 3DGS exports place ``nx``, ``ny``, and ``nz`` between the
    positions and SH coefficients.  SuperSplat accepts those placeholder
    fields, while gsply 0.2.x only accepts its compact layout without them.
    """
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"PLY header is not ASCII: {path}") from exc
            header_lines.append(line)
            if line == "end_header":
                data_offset = f.tell()
                break

    if "format binary_little_endian 1.0" not in header_lines:
        raise ValueError("Compatibility loader only supports binary_little_endian PLYs")

    vertex_count: int | None = None
    property_names: list[str] = []
    in_vertex = False
    for line in header_lines:
        if line.startswith("element "):
            parts = line.split()
            in_vertex = len(parts) == 3 and parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and line.startswith("property "):
            parts = line.split()
            if len(parts) != 3 or parts[1] not in {"float", "float32"}:
                raise ValueError(
                    "Compatibility loader requires scalar float32 vertex properties"
                )
            property_names.append(parts[2])

    if vertex_count is None:
        raise ValueError(f"PLY has no vertex element: {path}")

    prefix = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    if property_names[: len(prefix)] != prefix:
        raise ValueError("PLY is not the supported original 3DGS-with-normals layout")

    required_tail = [
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    if property_names[-len(required_tail) :] != required_tail:
        raise ValueError("PLY has an unsupported opacity/scale/rotation property layout")

    rest_names = property_names[len(prefix) : -len(required_tail)]
    expected_rest = [f"f_rest_{i}" for i in range(len(rest_names))]
    if rest_names != expected_rest or len(rest_names) not in {0, 9, 24, 45}:
        raise ValueError("PLY has an unsupported spherical-harmonics property layout")

    property_count = len(property_names)
    flat = np.fromfile(
        path,
        dtype="<f4",
        count=vertex_count * property_count,
        offset=data_offset,
    )
    if flat.size != vertex_count * property_count:
        raise ValueError(
            f"Truncated PLY payload: expected {vertex_count * property_count} "
            f"float32 values, found {flat.size}"
        )
    rows = flat.reshape(vertex_count, property_count)
    index = {name: i for i, name in enumerate(property_names)}

    means = torch.from_numpy(rows[:, 0:3].copy())
    sh0 = torch.from_numpy(rows[:, 6:9].copy())
    opacities = torch.from_numpy(rows[:, index["opacity"]].copy())
    scales = torch.from_numpy(
        rows[:, [index[f"scale_{i}"] for i in range(3)]].copy()
    )
    quats = torch.from_numpy(
        rows[:, [index[f"rot_{i}"] for i in range(4)]].copy()
    )

    shN = None
    if rest_names:
        rest = rows[:, len(prefix) : -len(required_tail)]
        # Original 3DGS stores channel-major (3, K); gsply uses (K, 3).
        shN = torch.from_numpy(
            rest.reshape(vertex_count, 3, len(rest_names) // 3)
            .transpose(0, 2, 1)
            .copy()
        )

    target_device = device or "cpu"
    return GSTensor.from_arrays(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="ply",
        device=target_device,
    )


def load_ply(path: str | Path, device: str | None = None) -> GSTensor:
    """Load a Gaussian Splatting PLY into a :class:`GSTensor`.

    Args:
        path: Path to the PLY file.
        device: Target device (e.g. ``"cuda"``). If None, keeps the gsply default.

    Returns:
        Loaded :class:`GSTensor`.
    """
    path = Path(path)
    try:
        if device is None:
            return GSTensor.load(str(path))
        return GSTensor.load(str(path), device=device)
    except ValueError as gsply_error:
        try:
            return _load_original_3dgs_ply(path, device)
        except ValueError as compatibility_error:
            raise ValueError(
                f"Unable to load Gaussian PLY {path}. gsply rejected it, and "
                f"the original-3DGS compatibility loader reported: "
                f"{compatibility_error}"
            ) from gsply_error


def save_ply(gs: GSTensor, path: str | Path) -> None:
    """Save a :class:`GSTensor` to a PLY file.

    Args:
        gs: Gaussian tensor to serialize.
        path: Output PLY path.
    """
    gs.save(str(path))
