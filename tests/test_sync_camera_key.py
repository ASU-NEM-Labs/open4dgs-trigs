"""Regression tests for matching training cameras to per-camera sync keys."""

from __future__ import annotations

from trigs.cli.train_from_colmap import _lookup_sync_offset, _resolve_camera_key


def test_folder_key_is_used() -> None:
    data = {"01": 0.001, "02": 0.002}
    assert _resolve_camera_key(data, cam_pos=0, orig_cam_id=0, folder="01") == "01"
    assert _resolve_camera_key(data, cam_pos=1, orig_cam_id=1, folder="02") == "02"


def test_padded_folder_key_is_used() -> None:
    data = {"0001": 0.001}
    assert _resolve_camera_key(data, cam_pos=0, orig_cam_id=0, folder="01") == "0001"


def test_missing_one_based_folder_does_not_borrow_neighbor() -> None:
    # Folders are one-based ("01", "02", "03"); the file has no entry for "02".
    # Position 1 must not resolve to key "1"/"0001", which is folder "01".
    data = {"1": 0.001, "3": 0.003, "0001": 0.001, "0003": 0.003}
    assert _resolve_camera_key(data, cam_pos=1, orig_cam_id=1, folder="02") is None
    offset, key = _lookup_sync_offset(data, cam_pos=1, orig_cam_id=1, folder="02")
    assert offset == 0.0
    assert key is None


def test_missing_zero_based_folder_after_held_out_camera_does_not_borrow() -> None:
    # Zero-based folders; camera id 19 is held out, so camera id 20 trains at
    # position 19. Its folder "0020" is absent and must not match "0019".
    data = {"0019": 0.002}
    assert _resolve_camera_key(data, cam_pos=19, orig_cam_id=20, folder="0020") is None


def test_index_fallback_only_without_folder() -> None:
    data = {"0001": 0.001}
    assert _resolve_camera_key(data, cam_pos=1, orig_cam_id=5, folder=None) == "0001"
    assert _resolve_camera_key(data, cam_pos=1, orig_cam_id=5, folder="camB") is None
