#!/usr/bin/env python3
"""
Convert a TriGSVanilla checkpoint into a stripped `.trigs` runtime package.

The source checkpoint is expected to contain `checkpoint["params"]`, matching
the viewer.py layout:

    means, scales, quats, opacities, sh0, shN, times, log_durations,
    anchors, zeta_base_r/t, zeta_c0_r/t, zeta_c1_r/t, zeta_c2_r/t

Tensor payloads are little-endian mixed precision by default. Temporal fields
stay float32 because TRiGS evaluates visibility in seconds; storing those in
float16 can shift visibility windows enough to show neighboring frames.

Usage:
    python tools/convert_trigs_checkpoint.py trigs_ckpt.pt scene.trigs --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

SUPPORTED_OUTPUT_DTYPES = ("mixed", "float16", "float32")
TEMPORAL_FLOAT32_KEYS = {"times", "log_durations"}

REQUIRED_KEYS = (
    "means",
    "scales",
    "quats",
    "opacities",
    "sh0",
    "shN",
    "times",
    "log_durations",
    "anchors",
    "zeta_base_r",
    "zeta_base_t",
    "zeta_c0_r",
    "zeta_c0_t",
    "zeta_c1_r",
    "zeta_c1_t",
    "zeta_c2_r",
    "zeta_c2_t",
)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: list[int]
    dtype: str
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a TriGSVanilla PyTorch checkpoint into a stripped .trigs runtime package."
    )
    parser.add_argument("input", type=Path, help="Path to the source checkpoint (.pt/.ckpt)")
    parser.add_argument("output", type=Path, help="Path to the output runtime package (.trigs)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
    parser.add_argument("--frames", type=int, help="Playback frame count to store in the .trigs manifest")
    parser.add_argument("--frame-rate", type=float, help="Playback frame rate to store in the .trigs manifest")
    parser.add_argument("--t-start", type=float, help="Override sequence start time in seconds")
    parser.add_argument("--t-end", type=float, help="Override sequence end time in seconds")
    parser.add_argument(
        "--tensor-dtype",
        choices=SUPPORTED_OUTPUT_DTYPES,
        default="mixed",
        help=(
            "Tensor payload dtype to store in the .trigs package. "
            "mixed keeps temporal fields as float32 and other tensors as float16."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.is_file():
        print(f"Input checkpoint does not exist: {args.input}", file=sys.stderr)
        return 1

    if args.output.exists() and not args.overwrite:
        print(f"Output already exists: {args.output}. Pass --overwrite to replace it.", file=sys.stderr)
        return 1

    if (args.frames is None) != (args.frame_rate is None):
        print("Pass both --frames and --frame-rate together.", file=sys.stderr)
        return 1

    print(f"Loading checkpoint: {args.input}")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint to be a dict-like object")

    params = checkpoint.get("params")
    if not isinstance(params, dict):
        raise KeyError('Checkpoint is missing top-level "params"')

    tensor_specs: list[TensorSpec] = []
    tensor_bytes: dict[str, bytes] = {}
    arrays: dict[str, np.ndarray] = {}
    count = None
    sh_coeff_count = None

    for key in REQUIRED_KEYS:
        tensor = params.get(key)
        if tensor is None:
            if key != "shN" or count is None:
                raise KeyError(f'Missing required tensor in checkpoint["params"]: {key}')
            array = np.zeros((count, 0, 3), dtype=np.float32)
        else:
            array = to_float32_numpy(tensor, key)
        array = normalize_tensor_shape(key, array)
        validate_shape(key, array)

        if key == "means":
            count = int(array.shape[0])
        elif array.shape[0] != count:
            raise ValueError(f'Leading dimension mismatch for "{key}": expected {count}, got {array.shape[0]}')

        if key == "shN":
            sh_coeff_count = int(array.shape[1])

        arrays[key] = array
        output_dtype = dtype_for_tensor(key, args.tensor_dtype)
        path = f"dynamic/{key}.{dtype_file_extension(output_dtype)}"
        tensor_specs.append(
            TensorSpec(name=key, shape=[int(v) for v in array.shape], dtype=output_dtype, path=path)
        )
        tensor_bytes[path] = encode_tensor_bytes(array, output_dtype)

    if count is None or sh_coeff_count is None:
        raise ValueError("Failed to derive runtime metadata from checkpoint tensors")

    t_start, t_end = infer_time_range(arrays["times"], arrays["log_durations"])
    if args.t_start is not None:
        t_start = float(args.t_start)
    if args.t_end is not None:
        t_end = float(args.t_end)
    if t_end <= t_start:
        raise ValueError(f"Invalid time range: t_start={t_start}, t_end={t_end}")

    fps = args.frame_rate
    frames = args.frames
    meta = checkpoint.get("selfcap_eval_meta", {})
    if fps is None:
        fps = as_positive_float(meta.get("fps", checkpoint.get("fps")))
    if frames is None:
        frames = as_positive_int(meta.get("n_frames", checkpoint.get("n_frames")))

    playback: dict[str, Any] = {
        "tStart": t_start,
        "tEnd": t_end,
        "durationSeconds": t_end - t_start,
    }
    if frames is not None:
        playback["frames"] = frames
    if fps is not None:
        playback["frameRate"] = fps

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_package(
        output_path=args.output,
        source_path=args.input,
        count=count,
        sh_coeff_count=sh_coeff_count,
        step=checkpoint.get("step"),
        playback=playback,
        tensor_specs=tensor_specs,
        tensor_bytes=tensor_bytes,
    )

    output_size = args.output.stat().st_size
    print(f"Wrote runtime package: {args.output} ({format_size(output_size)})")
    print(f"Splat count: {count}")
    print(f"SH coeff count: {sh_coeff_count}")
    print(f"Time range: {t_start:.6f}s -> {t_end:.6f}s")
    return 0


def to_float32_numpy(value: Any, name: str) -> np.ndarray:
    if isinstance(value, torch.nn.Parameter):
        value = value.detach()

    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise TypeError(f'Tensor "{name}" must be strided, got {value.layout}')
        value = value.detach().cpu().contiguous().to(torch.float32).numpy()
    elif isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value, dtype=np.float32)
    else:
        raise TypeError(f'Unsupported value for tensor "{name}": {type(value).__name__}')

    return np.asarray(value, dtype=np.float32, order="C")


def validate_shape(name: str, array: np.ndarray) -> None:
    if name in (
        "means",
        "scales",
        "anchors",
        "zeta_base_r",
        "zeta_base_t",
        "zeta_c0_r",
        "zeta_c0_t",
        "zeta_c1_r",
        "zeta_c1_t",
        "zeta_c2_r",
        "zeta_c2_t",
    ):
        expect_shape(name, array, ndim=2, trailing=(3,))
    elif name == "quats":
        expect_shape(name, array, ndim=2, trailing=(4,))
    elif name in ("opacities", "times", "log_durations"):
        if array.ndim == 2 and array.shape[1] == 1:
            return
        if array.ndim != 1:
            raise ValueError(f'Tensor "{name}" must be shape [N] or [N,1], got {list(array.shape)}')
    elif name == "sh0":
        expect_shape(name, array, ndim=3, trailing=(1, 3))
    elif name == "shN":
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f'Tensor "{name}" must be shape [N,K,3], got {list(array.shape)}')


def normalize_tensor_shape(name: str, array: np.ndarray) -> np.ndarray:
    if name == "sh0" and array.ndim == 2 and array.shape[1] == 3:
        return np.ascontiguousarray(array[:, None, :], dtype=np.float32)
    if name in ("opacities", "times", "log_durations") and array.ndim == 2 and array.shape[1] == 1:
        return np.ascontiguousarray(array.reshape(array.shape[0]), dtype=np.float32)
    return array


def expect_shape(name: str, array: np.ndarray, ndim: int, trailing: tuple[int, ...]) -> None:
    if array.ndim != ndim:
        raise ValueError(f'Tensor "{name}" must be rank {ndim}, got shape {list(array.shape)}')
    if tuple(array.shape[-len(trailing):]) != trailing:
        raise ValueError(f'Tensor "{name}" must end with shape {list(trailing)}, got {list(array.shape)}')


def infer_time_range(times: np.ndarray, log_durations: np.ndarray) -> tuple[float, float]:
    mu = times.reshape(-1)
    sigma = np.exp(log_durations.reshape(-1))
    return float(np.min(mu - 2.0 * sigma)), float(np.max(mu + 2.0 * sigma))


def as_positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def as_positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def ensure_little_endian(array: np.ndarray) -> np.ndarray:
    if array.dtype.byteorder in ("<", "=") and sys.byteorder == "little":
        return array
    return array.byteswap().newbyteorder("<")


def encode_tensor_bytes(array: np.ndarray, dtype: str) -> bytes:
    if dtype == "float32":
        return ensure_little_endian(np.asarray(array, dtype=np.float32)).tobytes(order="C")
    if dtype == "float16":
        return ensure_little_endian(np.asarray(array, dtype=np.float16)).tobytes(order="C")
    raise ValueError(f"Unsupported output dtype: {dtype}")


def dtype_for_tensor(name: str, requested: str) -> str:
    if requested == "mixed":
        return "float32" if name in TEMPORAL_FLOAT32_KEYS else "float16"
    return requested


def dtype_file_extension(dtype: str) -> str:
    if dtype == "float32":
        return "f32"
    if dtype == "float16":
        return "f16"
    raise ValueError(f"Unsupported output dtype: {dtype}")


def write_package(
    output_path: Path,
    source_path: Path,
    count: int,
    sh_coeff_count: int,
    step: Any,
    playback: dict[str, Any],
    tensor_specs: list[TensorSpec],
    tensor_bytes: dict[str, bytes],
) -> None:
    manifest = {
        "format": "trigs",
        "version": 1,
        "source": {
            "checkpoint": os.fspath(source_path),
        },
        "count": count,
        "shCoeffCount": sh_coeff_count,
        "step": int(step) if isinstance(step, (int, np.integer)) else None,
        "playback": playback,
        "dynamic": {
            "count": count,
            "shCoeffCount": sh_coeff_count,
            "tensors": {
                spec.name: {
                    "path": spec.path,
                    "dtype": spec.dtype,
                    "shape": spec.shape,
                }
                for spec in tensor_specs
            },
        },
    }

    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        for spec in tensor_specs:
            zf.writestr(spec.path, tensor_bytes[spec.path])


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


if __name__ == "__main__":
    raise SystemExit(main())
