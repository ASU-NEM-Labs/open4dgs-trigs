from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from trigs.init.from_colmap import build_canonical_from_npz


def _write_npz(path: Path, is_static: np.ndarray, source_frames: np.ndarray) -> None:
    n = len(is_static)
    positions = np.stack(
        [
            np.arange(n, dtype=np.float32) * 0.01,
            np.zeros(n, dtype=np.float32),
            np.ones(n, dtype=np.float32),
        ],
        axis=1,
    )
    np.savez_compressed(
        path,
        positions=positions,
        colors=np.full((n, 3), 0.5, dtype=np.float32),
        velocities=np.zeros((n, 3), dtype=np.float32),
        times=np.zeros((n, 1), dtype=np.float32),
        durations=np.full((n, 1), 1.0 / 60.0, dtype=np.float32),
        has_velocity=~is_static,
        is_static=is_static,
        dynamic_protect=np.zeros(n, dtype=bool),
        dynamic_allowed=~is_static,
        stage_keep=np.ones(n, dtype=bool),
        source_frames=source_frames.reshape(-1, 1).astype(np.int32),
        source_frame_start=np.asarray(1, dtype=np.int32),
        source_frame_end=np.asarray(26, dtype=np.int32),
        keyframe_step=np.asarray(2, dtype=np.int32),
        velocity_units=np.asarray("world_per_video_frame"),
    )


@pytest.mark.parametrize("end_frame", [26, 60])
def test_all_static_npz_is_valid_with_source_range_filter(tmp_path: Path, end_frame: int) -> None:
    path = tmp_path / "all-static.npz"
    _write_npz(
        path,
        is_static=np.ones(4, dtype=bool),
        source_frames=np.asarray([1, 3, 5, 25]),
    )

    params, _, _ = build_canonical_from_npz(
        path,
        fps=60.0,
        keyframe_stride=2,
        device="cpu",
        source_frame_start=1,
        source_frame_end=end_frame,
    )

    assert params["means"].shape[0] == 4
    assert torch.all(params["is_static"] > 0.5)
    assert torch.count_nonzero(params["zeta_base_t"]) == 0
    assert torch.count_nonzero(params["zeta_base_r"]) == 0


@pytest.mark.parametrize("start_frame,end_frame", [(1, 60), (-1, 26)])
def test_dynamic_npz_rejects_training_beyond_declared_source_coverage(
    tmp_path: Path, start_frame: int, end_frame: int,
) -> None:
    path = tmp_path / "misleading-f60.npz"
    _write_npz(
        path, is_static=np.asarray([False, False]),
        source_frames=np.asarray([1, 25]),
    )
    with pytest.raises(ValueError, match="exceed NPZ source coverage 1..26"):
        build_canonical_from_npz(
            path, fps=60.0, keyframe_stride=2, device="cpu",
            source_frame_start=start_frame, source_frame_end=end_frame,
        )


def test_dynamic_npz_still_rejects_misaligned_empty_dynamic_range(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dynamic-outside-range.npz"
    _write_npz(
        path,
        is_static=np.asarray([True, False]),
        source_frames=np.asarray([1, 27]),
    )

    with pytest.raises(ValueError, match="No dynamic rows remain"):
        build_canonical_from_npz(
            path,
            fps=60.0,
            keyframe_stride=2,
            device="cpu",
            source_frame_start=1,
            source_frame_end=26,
        )


def test_static_rows_released_to_dynamic_keep_static_init(tmp_path: Path) -> None:
    """--static-as-dynamic: whole-clip, motionless start; never demoted back."""
    from trigs.core.trainer import _enforce_relocation_class_invariants
    from trigs.init.from_colmap import release_static_rows_to_dynamic

    path = tmp_path / "mixed.npz"
    _write_npz(
        path,
        is_static=np.asarray([True, True, False, False]),
        source_frames=np.asarray([1, 1, 3, 5]),
    )
    params, _, _ = build_canonical_from_npz(
        path, fps=60.0, keyframe_stride=2, device="cpu",
        source_frame_start=1, source_frame_end=26,
    )
    before = {k: v.detach().clone() for k, v in params.items()}

    assert release_static_rows_to_dynamic(params) == 2
    assert release_static_rows_to_dynamic(params) == 0
    released = before["is_static"].reshape(-1) > 0.5
    assert torch.all(params["is_static"] <= 0.5)
    assert torch.all(params["dynamic_allowed"].reshape(-1)[released] > 0.5)

    total_sec = 25 / 60.0
    assert torch.allclose(params["times"][released], torch.full((2, 1), total_sec / 2))
    assert torch.allclose(params["log_durations"][released].exp(), torch.full((2, 1), total_sec))
    assert torch.count_nonzero(params["zeta_base_t"][released]) == 0
    assert torch.count_nonzero(params["zeta_base_r"][released]) == 0
    assert torch.equal(params["anchors"][released], params["means"][released])

    assert _enforce_relocation_class_invariants(params) == 0
    assert torch.all(params["is_static"] <= 0.5)
