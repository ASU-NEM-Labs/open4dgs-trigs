"""Viewer snapshot writer/reader (params-only, atomic, pruned)."""

import json
import types

import pytest
import torch

from trigs.core.viewer_snapshot import (
    POINTER_NAME,
    apply_default_snapshot_dir,
    VIEW_KEYS,
    LEGACY_POINTER_NAME,
    list_snapshots,
    load_snapshot,
    read_latest_pointer,
    should_snapshot,
    write_viewer_snapshot,
)


def _params(n: int = 8, sh_bands: int = 3):
    g = torch.Generator().manual_seed(0)
    p = {k: torch.randn(n, 3, generator=g) for k in VIEW_KEYS}
    p["sh0"] = torch.randn(n, 1, 3, generator=g)
    p["shN"] = torch.randn(n, sh_bands, 3, generator=g)
    p["times"] = torch.rand(n, 1, generator=g)
    p["log_durations"] = torch.randn(n, 1, generator=g)
    p["opacities"] = torch.randn(n, generator=g)
    p["is_static"] = (torch.rand(n, 1, generator=g) > 0.5).float()
    p["optimizer_junk"] = torch.randn(n, 3, generator=g)  # must not be saved
    return p


def _write(d, step, params=None, **kw):
    cfg = types.SimpleNamespace(relocate=True, fps=30.0)
    return write_viewer_snapshot(
        d, params=params or _params(), step=step, n_steps=10_000, fps=30.0,
        n_frames=90, sh_degree=3, sh_degree_active=1, config=cfg, **kw,
    )


@pytest.mark.parametrize(
    "step,every,last,final,expected",
    [
        (500, 1000, 0, False, False),
        (1000, 1000, 0, False, True),
        (1000, 1000, 1000, False, False),
        (1500, 1000, 0, False, True),      # jumped past a multiple
        (3000, 1000, 1500, False, True),   # skipped over 2000 but crossed it
        (1900, 1000, 1500, False, False),
        (1900, 0, 0, False, False),        # disabled
        (1900, 1000, 1500, True, True),    # final always writes new progress
        (1900, 1000, 1900, True, False),   # ...but not a duplicate
    ],
)
def test_should_snapshot(step, every, last, final, expected):
    assert should_snapshot(step, every, last, final=final) is expected


def test_payload_round_trips_and_is_params_only(tmp_path):
    params = _params()
    path, nbytes, secs = _write(tmp_path, 1000, params)
    assert nbytes == path.stat().st_size > 0 and secs >= 0.0

    snap = load_snapshot(path)
    assert snap["format"] == "trigs_viewer_snapshot"
    assert snap["step"] == 1000 and snap["n_frames"] == 90
    assert snap["fps"] == 30.0 and snap["sh_degree"] == 3
    assert snap["sh_degree_active"] == 1
    assert snap["config"]["relocate"] is True
    assert "optimizer_junk" not in snap["params"]
    assert set(snap["params"]) == set(VIEW_KEYS)
    for key, value in snap["params"].items():
        assert torch.equal(value, params[key])


def test_optional_keys_are_skipped_when_absent(tmp_path):
    params = _params()
    for k in ("shN", "is_static", "dynamic_protect"):
        del params[k]
    path, _, _ = _write(tmp_path, 10, params)
    keys = set(load_snapshot(path)["params"])
    assert not keys & {"shN", "is_static", "dynamic_protect"}
    assert "means" in keys


def test_camera_params_are_saved_apart_from_rows(tmp_path):
    params = _params()
    path, _, _ = _write(tmp_path, 10, params)
    assert load_snapshot(path)["camera_params"] == {}

    params["camera_time_offsets"] = torch.tensor([0.001, -0.002, 0.0])
    params["cam_gain"] = torch.ones(3, 3)
    path, _, _ = _write(tmp_path, 11, params)
    snap = load_snapshot(path)
    assert set(snap["camera_params"]) == {"camera_time_offsets", "cam_gain"}
    assert torch.equal(snap["camera_params"]["camera_time_offsets"], params["camera_time_offsets"])
    assert set(snap["params"]) == set(VIEW_KEYS)


def test_float16_only_casts_shN(tmp_path):
    params = _params()
    path, _, _ = _write(tmp_path, 10, params, dtype="float16")
    snap = load_snapshot(path)["params"]
    assert snap["shN"].dtype == torch.float16
    assert snap["means"].dtype == torch.float32
    assert torch.allclose(snap["shN"].float(), params["shN"], atol=2e-2, rtol=2e-3)

    with pytest.raises(ValueError):
        _write(tmp_path, 11, dtype="bfloat16")


def test_view_into_big_storage_is_not_saved_whole(tmp_path):
    big = torch.randn(10_000, 3)
    params = _params()
    params["means"] = big[:8]
    path, nbytes, _ = _write(tmp_path, 10, params)
    assert nbytes < 20_000
    assert torch.equal(load_snapshot(path)["params"]["means"], big[:8])


def test_pointer_is_valid_json_and_names_an_existing_file(tmp_path):
    assert read_latest_pointer(tmp_path) is None
    _write(tmp_path, 1000, extra={"mode": "chunked", "trained_window": [0.0, 1.5],
                                  "not_json": object()})
    ptr = read_latest_pointer(tmp_path)
    assert ptr["step"] == 1000 and (tmp_path / ptr["file"]).is_file()
    assert ptr["extra"]["trained_window"] == [0.0, 1.5]
    assert "not_json" not in ptr["extra"]
    assert not list(tmp_path.glob("*.tmp"))

    # A corrupt pointer reads as "nothing yet", never raises.
    (tmp_path / POINTER_NAME).write_text("{ not json")
    assert read_latest_pointer(tmp_path) is None
    (tmp_path / POINTER_NAME).write_text(json.dumps({"no_file_key": 1}))
    assert read_latest_pointer(tmp_path) is None


def test_retention_keeps_last_k_plus_milestones(tmp_path):
    for step in range(1000, 9001, 1000):
        _write(tmp_path, step, keep_last=2, keep_every=4000)
    kept = [s for s, _ in list_snapshots(tmp_path)]
    assert kept == [4000, 8000, 8000 + 1000]
    assert read_latest_pointer(tmp_path)["step"] == 9000


def test_retention_ignores_unrelated_files(tmp_path):
    (tmp_path / "trigs_step_000100_pct001.pt").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("keep me")
    for step in (1, 2, 3, 4):
        _write(tmp_path, step, keep_last=1)
    assert [s for s, _ in list_snapshots(tmp_path)] == [4]
    assert (tmp_path / "trigs_step_000100_pct001.pt").exists()
    assert (tmp_path / "notes.txt").exists()


def test_apply_default_snapshot_dir(tmp_path):
    off = types.SimpleNamespace(viewer_snapshot_every=0, viewer_snapshot_dir=None)
    assert apply_default_snapshot_dir(off, tmp_path) is None
    assert off.viewer_snapshot_dir is None

    on = types.SimpleNamespace(viewer_snapshot_every=500, viewer_snapshot_dir=None)
    assert apply_default_snapshot_dir(on, tmp_path) == tmp_path / "viewer_snapshots"
    assert on.viewer_snapshot_dir == str(tmp_path / "viewer_snapshots")

    explicit = types.SimpleNamespace(viewer_snapshot_every=500, viewer_snapshot_dir="/x/y")
    assert str(apply_default_snapshot_dir(explicit, tmp_path)) == "/x/y"


def test_legacy_safety_checkpoint_pointer(tmp_path):
    assert read_latest_pointer(tmp_path, allow_legacy=True) is None
    ckpt = tmp_path / "trigs_step_002500_pct025.pt"
    ckpt.write_bytes(b"x")
    (tmp_path / LEGACY_POINTER_NAME).write_text(ckpt.name + "\n")

    assert read_latest_pointer(tmp_path) is None  # opt-in only
    ptr = read_latest_pointer(tmp_path, allow_legacy=True)
    assert ptr["file"] == ckpt.name and ptr["step"] == 2500

    assert list_snapshots(tmp_path) == []
    assert [s for s, _ in list_snapshots(tmp_path, include_legacy=True)] == [2500]

    # A pointer to a missing or oddly named file reads as "nothing yet".
    (tmp_path / LEGACY_POINTER_NAME).write_text("trigs_step_000001_pct001.pt\n")
    assert read_latest_pointer(tmp_path, allow_legacy=True) is None
    (tmp_path / LEGACY_POINTER_NAME).write_text("other.pt\n")
    assert read_latest_pointer(tmp_path, allow_legacy=True) is None

    # A real viewer pointer wins over the legacy one.
    _write(tmp_path, 7)
    assert read_latest_pointer(tmp_path, allow_legacy=True)["step"] == 7
