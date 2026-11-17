from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from trigs.init.from_colmap import build_canonical_from_npz
from trigs.math.twist_init import screw_deform_means

FPS = 60.0
CTRL = ("zeta_c0_t", "zeta_c1_t", "zeta_c2_t", "zeta_c0_r", "zeta_c1_r", "zeta_c2_r")
V, A, J = np.array([0.02, 0.0, 0.0]), np.array([0.0, -0.002, 0.0]), np.array([0.0, 0.0004, 0.0])  # per frame^k


def _track(x0, t):
    """Sheet position at frame offset t: jerk the one-step velocity/acceleration can't see."""
    return x0 + V * t + 0.5 * A * t * t + J * t ** 3 / 6.0


def _npz(path: Path, with_chains: bool = True) -> dict:
    rng = np.random.default_rng(0)
    offs = np.array([-12, -9, -6, -3, 3, 6, 9, 12], np.int32)
    pos, vel, acc, sf, chain = [], [], [], [], []
    for f in range(1, 38, 3):
        t = f - 19.0
        x0 = np.c_[rng.random((300, 2)) * 0.3, np.zeros(300)]
        pos.append(_track(x0, t)); sf.append(np.full(300, f))
        vel.append(np.tile(V + A * t + 0.5 * J * t * t, (300, 1))); acc.append(np.tile(A + J * t, (300, 1)))
        c = np.stack([_track(x0, t + o) for o in offs], axis=1)
        c[100:200, [0, 1, 6, 7]] = np.nan                    # rows 100-199 reach +-6 frames
        c[200:, 5:] = np.nan                                  # rows 200-299 reach 12 back, 3 forward
        chain.append(c)
    n = 300 * 13
    data = dict(
        positions=np.concatenate(pos).astype(np.float32), velocities=np.concatenate(vel).astype(np.float32),
        accelerations=np.concatenate(acc).astype(np.float32), has_acceleration=np.ones(n, bool),
        colors=np.full((n, 3), 0.5, np.float32), times=np.zeros((n, 1), np.float32),
        is_static=np.zeros(n, bool), dynamic_protect=np.zeros(n, bool), dynamic_allowed=np.ones(n, bool),
        stage_keep=np.ones(n, bool), source_frames=np.concatenate(sf).reshape(-1, 1).astype(np.int32),
        source_frame_start=np.asarray(1, np.int32), source_frame_end=np.asarray(38, np.int32),
        keyframe_step=np.asarray(3, np.int32), velocity_units=np.asarray("world_per_video_frame"),
    )
    if with_chains:
        data["chain_pos"] = np.concatenate(chain).astype(np.float32)
        data["chain_offsets_frames"] = offs
    np.savez_compressed(path, **data)
    return data


def _build(path, chain_fit):
    params, _, _ = build_canonical_from_npz(
        path, fps=FPS, keyframe_stride=3, device="cpu", compute_device="cpu",
        source_frame_start=1, source_frame_end=38, init_sigma_t=4 / FPS, init_sigma_t_mode="uniform",
        init_accel_mode="fit", init_accel_scale=1.0, init_accel_fit_iters=200,
        init_chain_fit=chain_fit, init_chain_min_span_frames=6,
    )
    return {k: v.detach() for k, v in params.items()}


def _miss(P, data, offset):
    """Distance from each row's starting curve at (its keyframe + offset) to the true track."""
    kf = torch.from_numpy((data["source_frames"].reshape(-1, 1) - 1) / FPS).float()
    x = screw_deform_means(P["means"], P["anchors"], P["times"], P["log_durations"], P["zeta_base_t"],
                           P["zeta_base_r"], {k: P[k] for k in CTRL}, kf + offset / FPS)
    rng = np.random.default_rng(0)
    truth = []
    for f in range(1, 38, 3):
        x0 = np.c_[rng.random((300, 2)) * 0.3, np.zeros(300)]
        truth.append(_track(x0, f - 19.0 + offset))
    return (x - torch.from_numpy(np.concatenate(truth)).float()).norm(dim=-1)


def test_chain_fit_widens_lifetimes_and_follows_the_track_further(tmp_path):
    data = _npz(tmp_path / "kf.npz")
    local = _build(tmp_path / "kf.npz", chain_fit=False)
    chained = _build(tmp_path / "kf.npz", chain_fit=True)
    k = torch.from_numpy(np.arange(len(data["positions"])) % 300)
    far, mid, back = k < 100, (k >= 100) & (k < 200), k >= 200
    sig = chained["log_durations"].exp().reshape(-1) * FPS
    assert torch.allclose(sig[far], torch.full_like(sig[far], 6.0))
    assert torch.allclose(sig[mid | back], torch.full_like(sig[mid | back], 4.0))   # reach 6 -> sigma 3 < start 4
    for o in (9, 12, -12):
        before, after = _miss(local, data, o)[far].median(), _miss(chained, data, o)[far].median()
        assert after < 0.5 * before, (o, float(before), float(after))
    torch.testing.assert_close(chained["means"][far | mid], local["means"][far | mid])
    # Rows chaining only backward: centre moved 6 frames back onto the chained point,
    # anchor moved with it, and the curve follows the track over the back half. (Keyframes 1
    # and 4 can't move 6 frames back without leaving the clip, so their centre is clamped.)
    kf = torch.from_numpy((data["source_frames"].reshape(-1) - 1) / FPS).float()
    clamped = back & (kf < 6 / FPS)
    assert torch.all(chained["times"].reshape(-1)[clamped] >= 0)
    back = back & ~clamped
    torch.testing.assert_close(chained["times"].reshape(-1)[back], kf[back] - 6 / FPS)
    torch.testing.assert_close(chained["means"][back], torch.from_numpy(data["chain_pos"][:, 2]).float()[back])
    torch.testing.assert_close(chained["anchors"][back] - chained["means"][back], local["anchors"][back] - local["means"][back])
    for o in (-12, -9):
        before, after = _miss(local, data, o)[back].median(), _miss(chained, data, o)[back].median()
        assert after < 0.5 * before, (o, float(before), float(after))


def test_chain_fit_without_chain_fields_is_an_error(tmp_path):
    _npz(tmp_path / "kf.npz", with_chains=False)
    with pytest.raises(ValueError, match="chain-targets-npz"):
        _build(tmp_path / "kf.npz", chain_fit=True)
