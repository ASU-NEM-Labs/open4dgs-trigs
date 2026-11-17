#!/usr/bin/env python3
"""
Build a chunked TRiGS timeline container from an existing single-chunk .trigs.

First test use case:
    - You have one 360-frame .trigs export.
    - You want a 720-frame test file containing two copies of that same chunk.

The output is still a ZIP archive, but manifest.json uses:
    format = "trigs.timeline"
    version = 2

Archive layout:
    manifest.json
    chunks/chunk_0000/dynamic/...
    chunks/chunk_0001/dynamic/...

Renderer rule:
    global_frame -> owning chunk by global.frameStart/global.frameEndExclusive
    local_frame  -> global_frame - chunk.global.frameStart + chunk.local.frameStart
    local_time   -> local_frame / timeline.frameRate
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


MANIFEST_NAME = "manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap a single .trigs into a multi-chunk TRiGS timeline container."
    )
    parser.add_argument("input", type=Path, help="Input single-chunk .trigs file")
    parser.add_argument("output", type=Path, help="Output timeline .trigs file")
    parser.add_argument("--chunks", type=int, default=2, help="Number of timeline chunks to create")
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=None,
        help="Frames each duplicated chunk contributes. Defaults to input playback.frames.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Timeline frame rate. Defaults to input playback.frameRate.",
    )
    parser.add_argument(
        "--transition-mode",
        choices=("hard_cut", "opacity_fade"),
        default="hard_cut",
        help="Metadata hint for the renderer.",
    )
    parser.add_argument(
        "--fade-frames",
        type=int,
        default=0,
        help="Optional opacity fade in/out frames to store per chunk.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    return parser.parse_args()


def read_manifest(zf: zipfile.ZipFile) -> dict[str, Any]:
    if MANIFEST_NAME not in zf.namelist():
        raise ValueError(f"Input archive is missing {MANIFEST_NAME}")
    with zf.open(MANIFEST_NAME, "r") as f:
        manifest = json.loads(f.read().decode("utf-8"))
    if not isinstance(manifest, dict):
        raise TypeError("manifest.json must decode to a JSON object")
    return manifest


def require_single_trigs_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("format") != "trigs":
        raise ValueError(f"Expected input format='trigs', got {manifest.get('format')!r}")
    if "dynamic" not in manifest or "tensors" not in manifest["dynamic"]:
        raise ValueError("Input manifest is missing dynamic.tensors")
    if not isinstance(manifest["dynamic"]["tensors"], dict):
        raise TypeError("Input manifest dynamic.tensors must be an object")


def positive_number(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Could not infer {name}; pass it explicitly") from None
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {out}")
    return out


def positive_int(value: Any, name: str) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Could not infer {name}; pass it explicitly") from None
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {out}")
    return out


def rewrite_tensor_paths(
    tensor_specs: dict[str, Any],
    chunk_prefix: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in tensor_specs.items():
        if not isinstance(spec, dict) or "path" not in spec:
            raise ValueError(f"Tensor spec for {name!r} is invalid")
        new_spec = copy.deepcopy(spec)
        old_path = str(spec["path"]).lstrip("/")
        new_spec["path"] = f"{chunk_prefix}/{old_path}"
        out[name] = new_spec
    return out


def make_chunk_record(
    *,
    chunk_index: int,
    input_manifest: dict[str, Any],
    frames_per_chunk: int,
    frame_rate: float,
    transition_mode: str,
    fade_frames: int,
) -> dict[str, Any]:
    chunk_id = f"chunk_{chunk_index:04d}"
    chunk_prefix = f"chunks/{chunk_id}"

    global_start = chunk_index * frames_per_chunk
    global_end = global_start + frames_per_chunk
    global_time_start = global_start / frame_rate
    global_time_end = global_end / frame_rate

    playback = input_manifest.get("playback", {}) or {}
    local_time_start = float(playback.get("tStart", 0.0))
    local_time_end = float(playback.get("tEnd", frames_per_chunk / frame_rate))

    tensors = rewrite_tensor_paths(input_manifest["dynamic"]["tensors"], chunk_prefix)

    sh_coeff_count = int(input_manifest.get("shCoeffCount", input_manifest["dynamic"].get("shCoeffCount", 0)))
    sh_degree = None
    # shCoeffCount is the non-DC count, e.g. 15 -> degree 3, 0 -> degree 0.
    if sh_coeff_count >= 0:
        import math
        sh_degree = int(math.isqrt(sh_coeff_count + 1) - 1)

    return {
        "id": chunk_id,
        "path": chunk_prefix,
        "source": {
            "input": str(input_manifest.get("source", {}).get("checkpoint", "")),
            "copyOf": None if chunk_index == 0 else "chunk_0000",
        },
        "global": {
            "frameStart": global_start,
            "frameEndExclusive": global_end,
            "timeStart": global_time_start,
            "timeEnd": global_time_end,
        },
        "local": {
            "frameStart": 0,
            "frameEndExclusive": frames_per_chunk,
            "timeStart": local_time_start,
            "timeEnd": local_time_end,
        },
        "training": {
            "globalFrameStart": global_start,
            "globalFrameEndExclusive": global_end,
            "localFrameStart": 0,
            "localFrameEndExclusive": frames_per_chunk,
            "hasGuardFrames": False,
            "guardBeforeFrames": 0,
            "guardAfterFrames": 0,
        },
        "render": {
            "frameCount": frames_per_chunk,
            "localFrameOffset": -global_start,
            "localTimeOffset": -global_time_start,
            "transitionMode": transition_mode,
            "opacityFadeInFrames": fade_frames if chunk_index > 0 else 0,
            "opacityFadeOutFrames": fade_frames if chunk_index >= 0 else 0,
        },
        "model": {
            "count": int(input_manifest.get("count", input_manifest["dynamic"].get("count"))),
            "shCoeffCount": sh_coeff_count,
            "shDegree": sh_degree,
            "motionModel": "trigs.se3_bezier_v1",
            "tensorLayout": "split_dynamic",
        },
        "dynamic": {
            "count": int(input_manifest["dynamic"].get("count", input_manifest.get("count"))),
            "shCoeffCount": sh_coeff_count,
            "tensors": tensors,
        },
    }


def build_timeline_manifest(
    *,
    input_manifest: dict[str, Any],
    chunks: int,
    frames_per_chunk: int,
    frame_rate: float,
    transition_mode: str,
    fade_frames: int,
) -> dict[str, Any]:
    total_frames = chunks * frames_per_chunk
    return {
        "format": "trigs.timeline",
        "version": 2,
        "sourceFormat": {
            "format": input_manifest.get("format"),
            "version": input_manifest.get("version"),
        },
        "timeline": {
            "frameRate": frame_rate,
            "totalFrames": total_frames,
            "durationSeconds": total_frames / frame_rate,
            "timeUnit": "seconds",
            "frameIndexing": "zero_based_end_exclusive",
        },
        "playback": {
            "loop": False,
            "chunkSelection": "globalFrameRange",
            "transitionMode": transition_mode,
        },
        "chunks": [
            make_chunk_record(
                chunk_index=i,
                input_manifest=input_manifest,
                frames_per_chunk=frames_per_chunk,
                frame_rate=frame_rate,
                transition_mode=transition_mode,
                fade_frames=fade_frames,
            )
            for i in range(chunks)
        ],
    }


def main() -> int:
    args = parse_args()

    if not args.input.is_file():
        print(f"Input does not exist: {args.input}", file=sys.stderr)
        return 1
    if args.output.exists() and not args.overwrite:
        print(f"Output already exists: {args.output}. Use --overwrite.", file=sys.stderr)
        return 1
    if args.chunks <= 0:
        print("--chunks must be positive", file=sys.stderr)
        return 1
    if args.fade_frames < 0:
        print("--fade-frames cannot be negative", file=sys.stderr)
        return 1

    with zipfile.ZipFile(args.input, "r") as zin:
        input_manifest = read_manifest(zin)
        require_single_trigs_manifest(input_manifest)

        playback = input_manifest.get("playback", {}) or {}
        frames_per_chunk = (
            positive_int(args.frames_per_chunk, "frames_per_chunk")
            if args.frames_per_chunk is not None
            else positive_int(playback.get("frames"), "frames_per_chunk")
        )
        frame_rate = (
            positive_number(args.frame_rate, "frame_rate")
            if args.frame_rate is not None
            else positive_number(playback.get("frameRate"), "frame_rate")
        )

        output_manifest = build_timeline_manifest(
            input_manifest=input_manifest,
            chunks=args.chunks,
            frames_per_chunk=frames_per_chunk,
            frame_rate=frame_rate,
            transition_mode=args.transition_mode,
            fade_frames=args.fade_frames,
        )

        source_names = [n for n in zin.namelist() if n != MANIFEST_NAME]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zout:
            zout.writestr(MANIFEST_NAME, json.dumps(output_manifest, indent=2, sort_keys=True))
            for i in range(args.chunks):
                chunk_id = f"chunk_{i:04d}"
                for src_name in source_names:
                    dst_name = f"chunks/{chunk_id}/{src_name.lstrip('/')}"
                    zout.writestr(dst_name, zin.read(src_name))

    print(f"Wrote timeline container: {args.output}")
    print(f"  chunks: {args.chunks}")
    print(f"  frames/chunk: {frames_per_chunk}")
    print(f"  total frames: {args.chunks * frames_per_chunk}")
    print(f"  fps: {frame_rate}")
    print(f"  duration: {(args.chunks * frames_per_chunk) / frame_rate:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())