"""Params-only snapshots for the live-reload viewer.

The trainer writes a small snapshot of the model every N global steps; the
viewer (``viewer.py --watch DIR``) follows ``latest.json`` and swaps the new
snapshot in without restarting. Nothing here imports gsplat, nerfview or viser,
so the trainer, the viewer and CPU-only tests can all share it.

Layout of a snapshot directory::

    snap_s0001000.pt   params-only payload (see ``write_viewer_snapshot``)
    snap_s0002000.pt
    latest.json        atomic pointer to the newest snapshot + progress info

Both the snapshot and the pointer are written to a ``.tmp`` sibling and moved
into place with ``os.replace``, so a reader never sees a half-written file.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torch

SNAPSHOT_FORMAT = "trigs_viewer_snapshot"
SNAPSHOT_VERSION = 1
POINTER_NAME = "latest.json"
_SNAP_RE = re.compile(r"^snap_s(\d+)\.pt$")
# The trainer's resume-oriented safety checkpoints (config.checkpoint_dir).
_LEGACY_RE = re.compile(r"^trigs_step_(\d+)_pct\d+\.pt$")
LEGACY_POINTER_NAME = "latest_checkpoint.txt"

# Parameters the viewer reads. Optional keys are skipped when absent.
VIEW_KEYS: Tuple[str, ...] = (
    "means", "quats", "scales", "opacities", "sh0", "shN",
    "times", "log_durations",
    "anchors", "zeta_base_r", "zeta_base_t",
    "zeta_c0_r", "zeta_c0_t", "zeta_c1_r", "zeta_c1_t", "zeta_c2_r", "zeta_c2_t",
    "is_static", "dynamic_protect",
)

# Per-camera learned values (one row per training camera, not per primitive).
# The viewer ignores them; evaluation needs them to render each camera at the
# time and colour training used. Stored under ``camera_params`` so ``params``
# stays row-aligned. Optional keys are skipped when absent.
CAMERA_KEYS: Tuple[str, ...] = (
    "camera_time_offsets", "camera_time_offset_scale", "cam_gain", "cam_bias",
)


def apply_default_snapshot_dir(config: Any, output_path: Path | str) -> Optional[Path]:
    """CLI helper: when snapshots are enabled and no directory was given, use
    ``<output>/viewer_snapshots``. Returns the directory (None if disabled).
    """
    if int(getattr(config, "viewer_snapshot_every", 0)) <= 0:
        return None
    if not getattr(config, "viewer_snapshot_dir", None):
        config.viewer_snapshot_dir = str(Path(output_path) / "viewer_snapshots")
    return Path(config.viewer_snapshot_dir)


def should_snapshot(
    step_completed: int, every: int, last_written_step: int, *, final: bool = False,
) -> bool:
    """True once ``step_completed`` has crossed a multiple of ``every`` since
    the last write. Chunked training advances in jumps of a whole chunk call,
    so "crossed a multiple" (not "is a multiple") is the right test.
    """
    if int(every) <= 0:
        return False
    if final:
        return int(step_completed) > int(last_written_step)
    return int(step_completed) // int(every) > int(last_written_step) // int(every)


def _as_cpu(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    t = tensor.detach().to("cpu")
    if dtype is not None and t.is_floating_point() and t.dtype != dtype:
        t = t.to(dtype)
    # torch.save writes a tensor's whole backing storage, so a small view into
    # a big buffer would balloon the file.
    if t.untyped_storage().nbytes() != t.numel() * t.element_size():
        t = t.clone()
    return t


def list_snapshots(
    out_dir: Path, *, include_legacy: bool = False,
) -> List[Tuple[int, Path]]:
    """``(step, path)`` of every snapshot in ``out_dir``, oldest first.
    ``include_legacy`` also lists safety checkpoints (never pruned by us)."""
    found: List[Tuple[int, Path]] = []
    if not Path(out_dir).is_dir():
        return found
    for p in Path(out_dir).iterdir():
        m = _SNAP_RE.match(p.name) or (include_legacy and _LEGACY_RE.match(p.name))
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda sp: sp[0])
    return found


def _prune(out_dir: Path, keep_last: int, keep_every: int) -> None:
    snaps = list_snapshots(out_dir)
    keep_last = max(int(keep_last), 1)
    survivors = {step for step, _ in snaps[-keep_last:]}
    if int(keep_every) > 0:
        survivors |= {step for step, _ in snaps if step % int(keep_every) == 0}
    for step, path in snaps:
        if step not in survivors:
            try:
                path.unlink()
            except OSError:
                pass


def write_viewer_snapshot(
    out_dir: Path | str,
    *,
    params: Mapping[str, torch.Tensor],
    step: int,
    n_steps: int,
    fps: float,
    n_frames: int,
    sh_degree: int,
    sh_degree_active: Optional[int] = None,
    config: Any = None,
    extra: Optional[Dict[str, Any]] = None,
    keep_last: int = 3,
    keep_every: int = 0,
    dtype: str = "float32",
) -> Tuple[Path, int, float]:
    """Write a params-only snapshot and move ``latest.json`` onto it.

    ``sh_degree`` is the model's maximum SH degree; ``sh_degree_active`` is the
    degree training renders at this step (it ramps one band per 1000 steps).
    ``dtype="float16"`` stores ``shN`` (the bulk of the bytes) as half.

    Returns ``(path, nbytes, seconds)``.
    """
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dtype not in ("float32", "float16"):
        raise ValueError(f"viewer snapshot dtype must be float32 or float16, got {dtype!r}")

    snap_params: Dict[str, torch.Tensor] = {}
    for key in VIEW_KEYS:
        if key not in params:
            continue
        cast = torch.float16 if (dtype == "float16" and key == "shN") else None
        snap_params[key] = _as_cpu(params[key], cast)
    camera_params = {
        key: _as_cpu(params[key]) for key in CAMERA_KEYS if key in params
    }

    extra = dict(extra or {})
    written_unix = time.time()
    extra["written_unix"] = written_unix
    progress = float(step) / max(float(n_steps), 1.0)

    payload: Dict[str, Any] = {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "step": int(step),
        "n_steps": int(n_steps),
        "progress": progress,
        "fps": float(fps),
        "n_frames": int(n_frames),
        "sh_degree": int(sh_degree),
        "sh_degree_active": int(sh_degree if sh_degree_active is None else sh_degree_active),
        "params": snap_params,
        "camera_params": camera_params,
        "config": dict(vars(config)) if config is not None else {},
        "extra": extra,
    }

    path = out_dir / f"snap_s{int(step):07d}.pt"
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    nbytes = tmp.stat().st_size
    os.replace(tmp, path)

    pointer = {
        "file": path.name,
        "step": int(step),
        "n_steps": int(n_steps),
        "progress": progress,
        "written_unix": written_unix,
        "extra": {k: v for k, v in extra.items() if _json_ok(v)},
    }
    ptmp = out_dir / (POINTER_NAME + ".tmp")
    ptmp.write_text(json.dumps(pointer))
    os.replace(ptmp, out_dir / POINTER_NAME)

    _prune(out_dir, keep_last, keep_every)
    return path, int(nbytes), time.perf_counter() - t0


def _json_ok(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def read_latest_pointer(
    out_dir: Path | str, *, allow_legacy: bool = False,
) -> Optional[Dict[str, Any]]:
    """The parsed ``latest.json``, or None if it is missing or unreadable.
    Callers just retry on their next poll.

    ``allow_legacy`` falls back to ``latest_checkpoint.txt`` (safety
    checkpoints from ``config.checkpoint_dir``), so the viewer can also follow
    runs that were not started with viewer snapshots.
    """
    out_dir = Path(out_dir)
    try:
        data = json.loads((out_dir / POINTER_NAME).read_text())
        if isinstance(data, dict) and "file" in data:
            return data
    except (OSError, ValueError):
        pass
    if not allow_legacy:
        return None
    try:
        name = (out_dir / LEGACY_POINTER_NAME).read_text().strip()
        path = out_dir / name
        mtime = path.stat().st_mtime
    except OSError:
        return None
    m = _LEGACY_RE.match(name)
    if not m:
        return None
    return {"file": name, "step": int(m.group(1)), "written_unix": mtime, "extra": {}}


def load_snapshot(path: Path | str) -> Dict[str, Any]:
    """Load a snapshot (or a full training checkpoint) with its tensors
    memory-mapped, so bytes the viewer never touches (optimizer state in a
    safety checkpoint) are not read.
    """
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def snapshot_paths(out_dir: Path | str) -> Iterable[Path]:
    return [p for _, p in list_snapshots(Path(out_dir))]
