#!/usr/bin/env python3
"""Inspect and validate a chunked TRiGS timeline .trigs container."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a trigs.timeline archive")
    parser.add_argument("input", type=Path, help="Timeline .trigs file")
    parser.add_argument("--show-tensors", action="store_true", help="Print tensor specs for first chunk")
    return parser.parse_args()


def find_chunk(manifest: dict[str, Any], global_frame: int) -> dict[str, Any] | None:
    for chunk in manifest["chunks"]:
        g = chunk["global"]
        if int(g["frameStart"]) <= global_frame < int(g["frameEndExclusive"]):
            return chunk
    return None


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"Input does not exist: {args.input}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(args.input, "r") as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            print("Missing manifest.json", file=sys.stderr)
            return 1
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

        if manifest.get("format") != "trigs.timeline":
            print(f"Unexpected format: {manifest.get('format')!r}", file=sys.stderr)
            return 1

        fps = float(manifest["timeline"]["frameRate"])
        total_frames = int(manifest["timeline"]["totalFrames"])
        chunks = manifest["chunks"]

        print("TRiGS timeline")
        print(f"  file: {args.input}")
        print(f"  chunks: {len(chunks)}")
        print(f"  totalFrames: {total_frames}")
        print(f"  frameRate: {fps}")
        print(f"  durationSeconds: {total_frames / fps:.3f}")

        errors: list[str] = []
        for chunk in chunks:
            cid = chunk["id"]
            g = chunk["global"]
            l = chunk["local"]
            tensors = chunk["dynamic"]["tensors"]
            print(
                f"  {cid}: global [{g['frameStart']}, {g['frameEndExclusive']}) "
                f"-> local [{l['frameStart']}, {l['frameEndExclusive']})"
            )
            for tensor_name, spec in tensors.items():
                path = spec["path"]
                if path not in names:
                    errors.append(f"{cid}: missing tensor {tensor_name}: {path}")

        probe_frames = [0, max(0, total_frames // 2), max(0, total_frames - 1)]
        print("\nFrame mapping probes:")
        for frame in probe_frames:
            chunk = find_chunk(manifest, frame)
            if chunk is None:
                print(f"  frame {frame}: no chunk")
                continue
            local_frame = frame - int(chunk["global"]["frameStart"]) + int(chunk["local"]["frameStart"])
            print(
                f"  global {frame} -> {chunk['id']} local {local_frame} "
                f"localTime={local_frame / fps:.6f}s"
            )

        if args.show_tensors and chunks:
            print("\nFirst chunk tensor specs:")
            for name, spec in chunks[0]["dynamic"]["tensors"].items():
                print(f"  {name}: {spec['dtype']} {spec['shape']} {spec['path']}")

        if errors:
            print("\nERRORS:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return 2

        print("\nOK: all tensor paths exist.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())