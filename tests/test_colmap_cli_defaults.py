"""Production defaults resolved by trigs-train-colmap from the NPZ (frame range, keyframe step, fps,
per-keyframe splat budget) and the run_args.json record trigs-eval-colmap reads back."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trigs.cli.train_from_colmap import (
    TrigsColmapArgs,
    resolve_npz_defaults,
    resolve_target_count,
    write_run_args,
)


def _npz(path: Path, *, step: int = 5, fps: float = 60.0) -> Path:
    np.savez(path, positions=np.zeros((2, 3), np.float32), source_frame_start=np.int32(66),
             source_frame_end=np.int32(126), keyframe_step=np.int32(step), fps=np.float32(fps))
    return path


def _args(tmp_path: Path, **kw) -> TrigsColmapArgs:
    return TrigsColmapArgs(data_dir=tmp_path, output_dir=tmp_path / "out",
                           npz_path=_npz(tmp_path / "k.npz"), **kw)


def test_frame_range_keyframe_step_and_fps_come_from_the_npz(tmp_path):
    a = _args(tmp_path)
    resolve_npz_defaults(a)
    assert (a.start_frame, a.end_frame, a.keyframe_step) == (66, 126, 5)
    assert a.config.keyframe_stride == 5 and a.config.fps == 60.0


def test_explicit_window_is_kept(tmp_path):
    a = _args(tmp_path, start_frame=76, end_frame=106)
    resolve_npz_defaults(a)
    assert (a.start_frame, a.end_frame) == (76, 106)


def test_mismatched_keyframe_step_or_fps_stops(tmp_path):
    with pytest.raises(SystemExit, match="keyframe_step"):
        resolve_npz_defaults(_args(tmp_path, keyframe_step=3))
    a = _args(tmp_path)
    a.config.fps = 24.0
    with pytest.raises(SystemExit, match="fps"):
        resolve_npz_defaults(a)


def test_target_count_is_per_keyframe_times_keyframes_in_the_window(tmp_path):
    a = _args(tmp_path)
    resolve_npz_defaults(a)
    fps = 60.0
    # 3 keyframes (offsets 0, 5, 10) plus a sub-keyframe row at offset 7 and a static row.
    frames = torch.tensor([0, 0, 5, 10, 7, 3], dtype=torch.float64)
    params = {"times": (frames / fps).float().reshape(-1, 1),
              "is_static": torch.tensor([0, 0, 0, 0, 0, 1], dtype=torch.float32).reshape(-1, 1)}
    resolve_target_count(a, params)
    assert a.config.target_count == 3 * a.config.target_count_per_keyframe
    b = _args(tmp_path)
    b.config.target_count = 1234
    resolve_target_count(b, params)
    assert b.config.target_count == 1234


def test_run_args_record_the_resolved_window(tmp_path):
    a = _args(tmp_path)
    resolve_npz_defaults(a)
    (tmp_path / "out").mkdir()
    write_run_args(a, tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "run_args.json").read_text())
    assert (saved["start_frame"], saved["end_frame"], saved["keyframe_step"]) == (66, 126, 5)
    assert saved["npz_path"].endswith("k.npz")
