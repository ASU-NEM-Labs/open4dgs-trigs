"""COLMAP evaluator: run config and per-camera values come from the right place."""

import dataclasses
import types

import torch

from trigs.cli.eval_colmap import (
    _blur_errors,
    _blur_regions,
    _blur_summary,
    _box_px,
    _config_from_snapshot,
    _gauss,
    _load_run,
)
from trigs.core.trainer import TrigsTrainingConfig
from trigs.core.viewer_snapshot import VIEW_KEYS, write_viewer_snapshot


def _params(n: int = 4):
    p = {k: torch.zeros(n, 3) for k in VIEW_KEYS}
    p["opacities"] = torch.zeros(n)
    return p


def _snap(run, step, params, config=None):
    cfg = config or types.SimpleNamespace(fps=60.0, lambda_reg=0.001)
    write_viewer_snapshot(
        run / "viewer_snapshots", params=params, step=step, n_steps=100,
        fps=60.0, n_frames=10, sh_degree=3, sh_degree_active=2, config=cfg,
    )


def test_config_from_snapshot_keeps_run_values_and_ignores_unknown_keys():
    saved = {**vars(TrigsTrainingConfig()), "lambda_reg": 0.001, "batch_cameras": 20,
             "field_from_the_future": 1}
    cfg = _config_from_snapshot(saved)
    assert cfg.lambda_reg == 0.001 and cfg.batch_cameras == 20
    assert not hasattr(cfg, "field_from_the_future")
    assert _config_from_snapshot({}) == TrigsTrainingConfig()
    assert dataclasses.is_dataclass(cfg)


def test_snapshot_camera_params_win(tmp_path):
    params = _params()
    params["camera_time_offsets"] = torch.tensor([0.004, -0.004])
    _snap(tmp_path, 10, params)
    loaded, payload, source = _load_run(tmp_path, 10)
    assert source == "snapshot"
    assert torch.equal(loaded["camera_time_offsets"], params["camera_time_offsets"])
    assert payload["sh_degree_active"] == 2


def test_old_snapshot_takes_camera_params_from_final_checkpoint(tmp_path):
    _snap(tmp_path, 10, _params())
    final = {"params": {**_params(), "camera_time_offsets": torch.tensor([0.001, 0.002])}}
    torch.save(final, tmp_path / "trigs_ckpt.pt")
    loaded, _, source = _load_run(tmp_path, 10)
    assert source.startswith("final checkpoint")
    assert torch.equal(loaded["camera_time_offsets"], torch.tensor([0.001, 0.002]))

    (tmp_path / "trigs_ckpt.pt").unlink()
    loaded, _, source = _load_run(tmp_path, 10)
    assert "camera_time_offsets" not in loaded and "zero offsets" in source


def test_final_checkpoint_uses_latest_snapshot_config(tmp_path):
    _snap(tmp_path, 50, _params())
    torch.save({"params": _params(2)}, tmp_path / "trigs_ckpt.pt")
    loaded, payload, source = _load_run(tmp_path, -1)
    assert source == "final checkpoint" and payload["step"] == 100
    assert loaded["means"].shape[0] == 2 and payload["config"]["lambda_reg"] == 0.001


def _texture(h=96, w=96, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.rand(3, h // 4, w // 4, generator=g)
    return torch.nn.functional.interpolate(img[None], size=(h, w), mode="nearest")[0]


def _measure(pred, photo):
    mask = torch.zeros(photo.shape[1:], dtype=torch.bool)
    mask[8:-8, 8:-8] = True
    per_view = [{**_blur_errors(pred, photo, {"static": mask}), "moving": [torch.zeros(1), 0]}]
    return _blur_summary(per_view)["static"]


def test_blur_recovers_known_sigma():
    photo = _texture()
    for sigma in (0.0, 1.0, 2.5):
        r = _measure(_gauss(photo, sigma), photo)
        assert abs(r["sigma_px"] - sigma) < 0.2, (sigma, r)
        assert r["mean_shift_px"] == 0.0


def test_shift_is_not_reported_as_blur_once_shift_is_allowed():
    photo = _texture(seed=1)
    pred = torch.roll(_gauss(photo, 1.0), shifts=(0, 2), dims=(1, 2))
    r = _measure(pred, photo)
    assert r["sigma_px"] > 1.5  # a 2 px shift looks like extra blur...
    assert abs(r["sigma_px_after_shift"] - 1.0) < 0.2  # ...until the shift is allowed
    assert abs(r["mean_shift_px"] - 2.0) < 1e-6


def test_blur_regions_split_moving_from_static():
    photo = _texture(h=128, w=128, seed=2)
    before, after = photo.clone(), photo.clone()
    after[:, 10:30, 10:30] = 1.0 - after[:, 10:30, 10:30]
    masks = _blur_regions(photo, before, after)
    assert masks["moving"][20, 20] and not masks["static"][20, 20]
    assert not masks["moving"][100, 100]
    assert masks["static"].any() and not (masks["static"] & masks["moving"]).any()


def test_box_px():
    assert _box_px("0.25,0.1,0.75,0.9", 1080, 1920) == (270, 192, 810, 1728)


def test_main_does_not_shadow_module_imports():
    """A function-level `import json` in one branch of main() made the final json.dumps crash after a full eval."""
    from trigs.cli import eval_colmap

    assert "json" not in eval_colmap.main.__code__.co_varnames
