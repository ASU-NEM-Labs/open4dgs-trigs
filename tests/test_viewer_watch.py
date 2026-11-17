"""Viewer --watch mode: load, live swap, pinning, pruning, row-aligned NPZ.

viser / nerfview / gsplat are replaced by fakes (and the device is CPU), so
this exercises the viewer's own loading, locking and swap logic anywhere.
"""

import contextlib
import sys
import types

import numpy as np
import pytest
import torch

from trigs.core.viewer_snapshot import write_viewer_snapshot

N, K, N_FRAMES = 24, 15, 20


def _ensure_importable(monkeypatch):
    """The real packages are used when installed; otherwise empty stand-ins
    let ``trigs.core.viewer`` import."""
    for name in ("viser", "nerfview", "gsplat", "gsplat.rendering"):
        try:
            __import__(name)
        except ImportError:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    if not hasattr(sys.modules["gsplat.rendering"], "rasterization"):
        sys.modules["gsplat.rendering"].rasterization = lambda *a, **k: None
    monkeypatch.delitem(sys.modules, "trigs.core.viewer", raising=False)


class _Handle:
    def __init__(self, value=None, options=None, content=""):
        self.value, self.options, self.content = value, options, content
        self.update_cbs, self.click_cbs = [], []

    def on_update(self, fn):
        self.update_cbs.append(fn)
        return fn

    def on_click(self, fn):
        self.click_cbs.append(fn)
        return fn


class _Gui:
    def add_folder(self, *a, **k):
        return contextlib.nullcontext()

    def __getattr__(self, name):
        if not name.startswith("add_"):
            raise AttributeError(name)

        def make(label=None, *a, initial_value=None, options=None, **k):
            content = label if name == "add_markdown" else ""
            return _Handle(initial_value, options, content)
        return make


class _FakeServer:
    def __init__(self, *a, **k):
        self.gui = _Gui()


class _FakeNerfViewer:
    def __init__(self, *a, **k):
        self.rerenders = 0

    def rerender(self, _):
        self.rerenders += 1


@pytest.fixture
def viewer_mod(monkeypatch):
    _ensure_importable(monkeypatch)
    import importlib
    mod = importlib.import_module("trigs.core.viewer")
    monkeypatch.setattr(mod.viser, "ViserServer", _FakeServer, raising=False)
    monkeypatch.setattr(mod.nerfview, "Viewer", _FakeNerfViewer, raising=False)
    return mod


def _params(seed: int = 0, n: int = N):
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g) * 0.1  # noqa: E731
    p = {
        "means": r(n, 3), "quats": torch.nn.functional.normalize(r(n, 4) + 1, dim=-1),
        "scales": r(n, 3) - 3.0, "opacities": r(n) + 1.0,
        "sh0": r(n, 1, 3), "shN": r(n, K, 3),
        "times": torch.rand(n, 1, generator=g), "log_durations": torch.full((n, 1), -1.5),
        "anchors": r(n, 3), "is_static": (torch.arange(n) % 4 == 0).float().reshape(n, 1),
        "dynamic_protect": torch.zeros(n, 1),
    }
    for k in ("zeta_base_r", "zeta_base_t"):
        p[k] = r(n, 3)
    for i in range(3):
        p[f"zeta_c{i}_r"], p[f"zeta_c{i}_t"] = r(n, 3), r(n, 3)
    return p


def _snap(d, step, seed=0, *, active=1, extra=None, dtype="float32", n=N, config=None):
    cfg = types.SimpleNamespace(**(config or {"relocate": False}))
    return write_viewer_snapshot(
        d, params=_params(seed, n), step=step, n_steps=1000, fps=24.0,
        n_frames=N_FRAMES, sh_degree=3, sh_degree_active=active, config=cfg,
        extra=extra or {"mode": "chunked", "round": 1, "n_rounds": 2,
                        "window_idx": 2, "n_windows": 4, "trained_window": [0.2, 0.5]},
        keep_last=10, dtype=dtype,
    )[0]


def _cfg(viewer_mod, d, **kw):
    return viewer_mod.ConfigTriGS(
        watch_dir=str(d), device="cpu", poll_s=0.01, precompute_visibility=True, **kw,
    )


def test_first_load_resolves_meta_from_snapshot(viewer_mod, tmp_path):
    _snap(tmp_path, 100, active=2)
    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, tmp_path))
    assert v._loaded_file == "snap_s0000100.pt"
    assert v.cfg.fps == 24.0 and v.cfg.total_frames == N_FRAMES
    assert v.splats.sh_degree_render == 2  # active degree, not the final 3
    assert v.splats.N == N and v.splats.shN.shape == (N, K, 3)
    assert "step 100 / 1,000" in v.snap_status_md.content
    assert "window 2/4" in v.snap_status_md.content


def test_explicit_cli_values_beat_the_snapshot_and_survive_reloads(viewer_mod, tmp_path):
    _snap(tmp_path, 100)
    v = viewer_mod.ViewerTriGS(
        _cfg(viewer_mod, tmp_path, fps=60.0, total_frames=10, sh_degree=0)
    )
    assert (v.cfg.fps, v.cfg.total_frames, v.splats.sh_degree_render) == (60.0, 10, 0)
    _snap(tmp_path, 200, active=3)
    v._poll_once()
    assert v._loaded_file == "snap_s0000200.pt"
    assert (v.cfg.fps, v.cfg.total_frames, v.splats.sh_degree_render) == (60.0, 10, 0)


def test_new_snapshot_is_swapped_in_and_rerenders(viewer_mod, tmp_path):
    _snap(tmp_path, 100, seed=0)
    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, tmp_path))
    v.viewer = _FakeNerfViewer()
    before = v.splats.means.clone()

    _snap(tmp_path, 200, seed=1)
    v._poll_once()
    assert v._loaded_file == "snap_s0000200.pt"
    assert not torch.equal(v.splats.means, before)
    assert torch.equal(v.splats.means, _params(1)["means"])
    assert v.viewer.rerenders == 1
    assert v.snap_dd.value == "snap_s0000200.pt"
    assert v.snap_dd.options[0] == "snap_s0000200.pt"

    v._poll_once()  # nothing new: no churn
    assert v.viewer.rerenders == 1


def test_half_precision_shN_is_widened_on_load(viewer_mod, tmp_path):
    _snap(tmp_path, 100, dtype="float16")
    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, tmp_path))
    assert v.splats.shN.dtype == torch.float32


def test_auto_reload_off_waits_for_the_button(viewer_mod, tmp_path):
    _snap(tmp_path, 100)
    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, tmp_path, auto_reload=False))
    v.viewer = _FakeNerfViewer()
    _snap(tmp_path, 300, seed=3)

    v._poll_once()
    assert v._loaded_file == "snap_s0000100.pt"
    assert "newer available" in v.snap_status_md.content

    v.reload_btn.click_cbs[0](None)
    v._poll_once()
    assert v._loaded_file == "snap_s0000300.pt"


def test_picking_an_older_snapshot_pins_it(viewer_mod, tmp_path):
    _snap(tmp_path, 100, seed=0)
    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, tmp_path))
    v.viewer = _FakeNerfViewer()
    _snap(tmp_path, 200, seed=1)
    v._poll_once()
    assert v._loaded_file == "snap_s0000200.pt"

    v.snap_dd.value = "snap_s0000100.pt"
    v.snap_dd.update_cbs[0](None)
    assert v.auto_reload_cb.value is False
    v._poll_once()
    assert v._loaded_file == "snap_s0000100.pt"
    assert torch.equal(v.splats.means, _params(0)["means"])

    _snap(tmp_path, 400, seed=4)
    v._poll_once()  # pinned: a newer file must not steal the view
    assert v._loaded_file == "snap_s0000100.pt"


def test_pruned_or_corrupt_snapshot_keeps_the_current_one(viewer_mod, tmp_path):
    _snap(tmp_path, 100)
    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, tmp_path))
    kept = v.splats

    v._load_and_swap("snap_s0000999.pt")
    assert v.splats is kept and "pruned" in v._status_note

    (tmp_path / "snap_s0000500.pt").write_bytes(b"not a torch file")
    v._load_and_swap("snap_s0000500.pt")
    assert v.splats is kept and "could not read" in v._status_note
    assert v._loaded_file == "snap_s0000100.pt"


def test_safety_checkpoint_dir_can_be_followed(viewer_mod, tmp_path):
    """--watch also works on a plain --config.checkpoint-dir."""
    src = _snap(tmp_path / "tmp", 100)
    ckpt = tmp_path / "trigs_step_000100_pct010.pt"
    src.rename(ckpt)
    (tmp_path / "latest_checkpoint.txt").write_text(ckpt.name + "\n")
    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, tmp_path))
    assert v._loaded_file == ckpt.name and v.splats.N == N


def test_jump_to_trained_window_sets_the_matching_frame(viewer_mod, tmp_path):
    _snap(tmp_path, 100)
    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, tmp_path))
    v.viewer = _FakeNerfViewer()
    v.jump_btn.click_cbs[0](None)
    # window [0.2, 0.5] s -> 0.35 s of a (N_FRAMES-1)/24 s sequence
    expected = round(0.35 / ((N_FRAMES - 1) / 24.0) * (N_FRAMES - 1))
    assert v.current_frame == expected == v.frame_slider.value


def _write_npz(path, n, *, static_from=None):
    static = (np.arange(n) % 4 == 0) if static_from is None else static_from
    np.savez(path, source_frames=np.arange(n) % 5, is_static=static)


def test_npz_mismatch_is_fatal_for_a_checkpoint_but_degrades_when_watching(
    viewer_mod, tmp_path,
):
    snaps = tmp_path / "snaps"
    _snap(snaps, 100)
    npz = tmp_path / "keyframes.npz"
    _write_npz(npz, N + 7)  # row count does not match

    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, snaps, npz_path=str(npz)))
    assert v.splats.source_frames is None
    assert "row mismatch" in v.splats.npz_note
    assert "source-cohort diagnostics off" in v.snap_status_md.content

    # The same mismatch on a plain checkpoint keeps its old hard error.
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"params": _params(), "fps": 24.0, "n_frames": N_FRAMES}, ckpt)
    cfg = viewer_mod.ConfigTriGS(ckpt=str(ckpt), device="cpu", npz_path=str(npz))
    with pytest.raises(ValueError, match="row mismatch"):
        viewer_mod.ViewerTriGS(cfg)


def test_npz_cohorts_load_and_flag_staleness_after_relocation(viewer_mod, tmp_path):
    npz = tmp_path / "keyframes.npz"
    _write_npz(npz, N)
    d = tmp_path / "snaps"
    _snap(d, 100, config={"relocate": True, "relocate_start_iter": 5000})
    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, d, npz_path=str(npz)))
    assert v.splats.source_frames is not None and v.splats.npz_note == ""

    _snap(d, 6000, config={"relocate": True, "relocate_start_iter": 5000})
    v._poll_once()
    assert v.splats.source_frames is not None
    assert "stale" in v.splats.npz_note and "stale" in v.snap_status_md.content


def test_plain_checkpoint_path_still_works_without_watch(viewer_mod, tmp_path):
    ckpt = tmp_path / "trigs_ckpt.pt"
    torch.save({"params": _params(), "selfcap_eval_meta": {"fps": 12.0, "n_frames": 7}}, ckpt)
    v = viewer_mod.ViewerTriGS(
        viewer_mod.ConfigTriGS(ckpt=str(ckpt), device="cpu")
    )
    assert (v.cfg.fps, v.cfg.total_frames, v.splats.sh_degree_render) == (12.0, 7, 3)
    assert not hasattr(v, "snap_status_md")


def test_render_runs_under_the_lock_and_sees_the_swapped_model(
    viewer_mod, tmp_path, monkeypatch,
):
    seen = []

    def fake_raster(*, means, **kw):
        seen.append(means.detach().clone())
        h, w = kw["height"], kw["width"]
        return torch.zeros(1, h, w, 4), None, None

    monkeypatch.setattr(viewer_mod, "rasterization", fake_raster)
    _snap(tmp_path, 100, seed=0)
    v = viewer_mod.ViewerTriGS(_cfg(viewer_mod, tmp_path))
    v.viewer = _FakeNerfViewer()

    cam = types.SimpleNamespace(
        c2w=np.eye(4, dtype=np.float32),
        get_K=lambda wh: np.array(
            [[50.0, 0, wh[0] / 2], [0, 50.0, wh[1] / 2], [0, 0, 1]], np.float32
        ),
    )
    img = v._render(cam, (16, 12))
    assert img.shape == (12, 16, 3)
    n_before = seen[-1].shape[0]
    assert n_before > 0

    _snap(tmp_path, 200, seed=1)
    v._poll_once()
    v._render(cam, (16, 12))
    assert len(seen) == 2
    unchanged = seen[0].shape == seen[1].shape and torch.equal(seen[0], seen[1])
    assert not unchanged
