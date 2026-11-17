#!/usr/bin/env python3
"""
camera_offsets.py — pull learned per-camera time offsets out of a TriGS
checkpoint, name the physical cameras, and find the one railed at the clamp.

TensorBoard only gives you motion/time_offset_max_ms and a histogram. This
tells you WHICH camera folder is at the rail, whether the rail is a stable
camera property or optimizer noise, and how the learned residual compares to
what sync.json already supplied.

Note the trainer mean-subtracts the offsets every step
(trainer.py:3074), so these are zero-mean relative values. A single camera at
-8 ms is an outlier against the other 43, not an absolute clock error.

Usage
-----
  # single checkpoint
  uv run python camera_offsets.py \
      --ckpt results-f300-k3-v3/trigs_ckpt.pt --data-dir .

  # stability across two runs / two snapshots (the useful test)
  uv run python camera_offsets.py \
      --ckpt results-f300-k3-v3/trigs_ckpt.pt \
      --compare results-f300-k3-v2/trigs_ckpt.pt \
      --data-dir . --clamp-s 0.0083

If the same camera folder rails in both, it is that camera's sync, and
re-solving (or excluding it) is worth more than any training-side knob. If the
rail moves, the offsets are absorbing motion-model error and the clamp is
doing its job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_offsets(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for holder in ("params", "temporal_params"):
        d = ckpt.get(holder) or {}
        if "camera_time_offsets" in d:
            v = d["camera_time_offsets"]
            v = v.detach().cpu().numpy().reshape(-1).astype(np.float64)
            return v, ckpt
    raise SystemExit(
        f"{ckpt_path}: no 'camera_time_offsets'. Was the run started with "
        "--config.learn-camera-time-offsets?"
    )


def camera_names(data_dir: Path, start_frame: int, end_frame: int,
                 data_factor: int, test_every: int, n_expected: int):
    """training-position -> (orig parser id, image folder). Best effort."""
    try:
        from trigs.cli.train_from_colmap import _infer_camera_folders
        from trigs.io.dataset import FreeTimeDataset, FreeTimeParser

        parser = FreeTimeParser(
            data_dir=str(data_dir), factor=data_factor, test_every=test_every,
            start_frame=start_frame, end_frame=end_frame + 1,
        )
        dataset = FreeTimeDataset(parser, split="train", test_set=[])
        ids = list(dataset.camera_indices)
        folders = _infer_camera_folders(parser, Path(data_dir))
        out = [(int(o), folders[int(o)] if int(o) < len(folders) else "?")
               for o in ids]
        if len(out) != n_expected:
            print(f"  WARNING: parser gave {len(out)} training cameras but the "
                  f"checkpoint has {n_expected} offsets; names may be misaligned")
        return out, parser
    except ModuleNotFoundError as exc:  # noqa: BLE001
        print(f"  (could not resolve camera folders: {exc})")
        print("   -> the `trigs` package is not importable here. Re-run with"
              " --project so folder names resolve:")
        print("      uv run --no-sync --project /path/to/open4dgs-trigs \\")
        print("          python /path/to/open4dgs-trigs/scripts/camera_offsets.py ...")
        return [(i, "?") for i in range(n_expected)], None
    except Exception as exc:  # noqa: BLE001 - diagnostics must not hard-fail
        print(f"  (could not resolve camera folders: {exc})")
        return [(i, "?") for i in range(n_expected)], None


def load_sync(data_dir: Path, parser, cam_info):
    """sync.json offset per training position, in seconds."""
    sync_file = Path(data_dir) / "sync.json"
    if not sync_file.exists() or parser is None:
        return None
    try:
        from trigs.cli.train_from_colmap import _lookup_sync_offset
        raw = json.loads(sync_file.read_text(encoding="utf-8"))
        sync_data = {str(k): float(v) for k, v in raw.items()}
        out = np.zeros(len(cam_info))
        for pos, (orig, folder) in enumerate(cam_info):
            off, _ = _lookup_sync_offset(
                sync_data, cam_pos=pos, orig_cam_id=orig, folder=folder,
            )
            out[pos] = off
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"  (could not read sync.json: {exc})")
        return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None,
                    help="second checkpoint for the stability test")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="enables camera folder names and sync.json comparison")
    ap.add_argument("--start-frame", type=int, default=1)
    ap.add_argument("--end-frame", type=int, default=300)
    ap.add_argument("--data-factor", type=int, default=1)
    ap.add_argument("--test-every", type=int, default=8)
    ap.add_argument("--clamp-s", type=float, default=0.0083,
                    help="the --config.camera-time-offset-clamp-s you trained with")
    ap.add_argument("--rail-frac", type=float, default=0.9,
                    help="flag cameras beyond this fraction of the clamp")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    off, ckpt = load_offsets(args.ckpt)
    ms = off * 1000.0
    clamp_ms = args.clamp_s * 1000.0
    step = ckpt.get("step_completed", ckpt.get("step", "?"))
    print(f"{args.ckpt}  (step {step})")
    print(f"  {off.size} cameras   mean={ms.mean():+.3f}  std={ms.std():.3f}  "
          f"min={ms.min():+.3f}  max={ms.max():+.3f} ms")
    print(f"  clamp = +/-{clamp_ms:.2f} ms")

    cam_info, parser = ([(i, "?") for i in range(off.size)], None)
    sync_ms = None
    if args.data_dir is not None:
        cam_info, parser = camera_names(
            args.data_dir, args.start_frame, args.end_frame,
            args.data_factor, args.test_every, off.size,
        )
        s = load_sync(args.data_dir, parser, cam_info)
        if s is not None:
            sync_ms = s * 1000.0
            print(f"  sync.json: mean={sync_ms.mean():+.3f}  "
                  f"std={sync_ms.std():.3f}  "
                  f"range {sync_ms.min():+.3f}..{sync_ms.max():+.3f} ms")

    cmp_ms = None
    if args.compare is not None:
        c, _ = load_offsets(args.compare)
        if c.size != off.size:
            print(f"  WARNING: --compare has {c.size} cameras vs {off.size}; "
                  "skipping stability test")
        else:
            cmp_ms = c * 1000.0

    rail = clamp_ms * args.rail_frac
    railed = np.where(np.abs(ms) >= rail)[0]

    print(f"\n{'pos':>4} {'orig':>5} {'folder':>8} {'learned_ms':>11}", end="")
    if sync_ms is not None:
        print(f" {'sync_ms':>9} {'total_ms':>9}", end="")
    if cmp_ms is not None:
        print(f" {'compare_ms':>11}", end="")
    print("  flag")

    order = np.argsort(-np.abs(ms))
    for pos in order:
        orig, folder = cam_info[pos]
        print(f"{pos:>4} {orig:>5} {folder:>8} {ms[pos]:>+11.3f}", end="")
        if sync_ms is not None:
            print(f" {sync_ms[pos]:>+9.3f} {sync_ms[pos]+ms[pos]:>+9.3f}", end="")
        if cmp_ms is not None:
            print(f" {cmp_ms[pos]:>+11.3f}", end="")
        flag = ""
        if abs(ms[pos]) >= rail:
            flag = "  <-- RAILED AT CLAMP"
        elif abs(ms[pos]) > 3.0 * ms.std():
            flag = "  <- outlier (>3 sigma)"
        print(flag)

    print("\n---------------------------------------------------------------")
    if railed.size:
        names = ", ".join(
            f"pos {p} (folder {cam_info[p][1]}, {ms[p]:+.2f} ms)" for p in railed
        )
        print(f"RAILED: {railed.size} camera(s) at >= {args.rail_frac:.0%} of "
              f"the clamp: {names}")
        print("  The optimizer wants more time correction than you allow. "
              "Re-run with --config.camera-time-offset-clamp-s 0.0166 to see "
              "where it actually wants to sit, then fix that camera's sync.")
    else:
        print(f"No camera within {args.rail_frac:.0%} of the clamp — the clamp "
              "is not binding at this snapshot.")

    if cmp_ms is not None:
        r = float(np.corrcoef(ms, cmp_ms)[0, 1])
        rms = float(np.sqrt(((ms - cmp_ms) ** 2).mean()))
        print(f"\nStability vs {args.compare.name}: r={r:+.3f}  "
              f"rms diff={rms:.3f} ms")
        both = set(railed.tolist()) & set(
            np.where(np.abs(cmp_ms) >= rail)[0].tolist())
        if both:
            print(f"  SAME camera(s) railed in both: "
                  f"{[cam_info[p][1] for p in sorted(both)]}")
            print("  -> stable camera property. This is a real sync error, not "
                  "optimizer noise. Re-solve or drop that camera.")
        elif r > 0.5:
            print("  -> offsets are largely a stable camera property "
                  "(r > 0.5), but the rail moved; part sync, part model error.")
        else:
            print("  -> offsets do NOT reproduce across runs (r <= 0.5). They "
                  "are absorbing motion-model error, not measuring sync. "
                  "Do not chase individual cameras.")

    if sync_ms is not None:
        ratio = np.abs(ms).mean() / max(np.abs(sync_ms).std(), 1e-9)
        print(f"\nLearned residual mean |offset| = {np.abs(ms).mean():.3f} ms "
              f"vs sync.json spread std = {sync_ms.std():.3f} ms  "
              f"(ratio {ratio:.2f})")
        print("  Residuals comparable to or larger than the sync spread mean "
              "sync.json is not carrying the timing — check whether the solve "
              "for this dataset was written trusted:false.")

    if args.json_out:
        payload = {
            "ckpt": str(args.ckpt), "step": step, "clamp_ms": clamp_ms,
            "cameras": [
                {
                    "pos": int(p), "orig_id": int(cam_info[p][0]),
                    "folder": cam_info[p][1],
                    "learned_ms": float(ms[p]),
                    "sync_ms": (None if sync_ms is None else float(sync_ms[p])),
                    "compare_ms": (None if cmp_ms is None else float(cmp_ms[p])),
                    "railed": bool(abs(ms[p]) >= rail),
                }
                for p in range(off.size)
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
