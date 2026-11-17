from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from trigs.init import from_colmap
from trigs.init.from_colmap import build_canonical_from_npz
from trigs.math.twist_init import fit_local_twist, local_knn_indices


def test_per_keyframe_fits_match_per_time_mask_reference(tmp_path: Path) -> None:
    """Grouping rows by keyframe once gives exactly the per-time-mask results,
    including when rows of different keyframes are interleaved in the NPZ."""
    rng = np.random.default_rng(0)
    frames = rng.permutation(np.repeat([1, 3, 5, 7], 40))  # interleaved keyframes
    n = len(frames)
    is_static = np.zeros(n, dtype=bool)
    is_static[:6] = True
    positions = rng.normal(size=(n, 3)).astype(np.float32)
    velocities = (rng.normal(size=(n, 3)) * 0.01).astype(np.float32)
    path = tmp_path / "kf.npz"
    np.savez_compressed(
        path, positions=positions, colors=np.full((n, 3), 0.5, np.float32),
        velocities=velocities, times=np.zeros((n, 1), np.float32),
        is_static=is_static, dynamic_protect=np.zeros(n, bool),
        dynamic_allowed=~is_static, stage_keep=np.ones(n, bool),
        source_frames=frames.reshape(-1, 1).astype(np.int32),
        source_frame_start=np.asarray(1, np.int32), source_frame_end=np.asarray(8, np.int32),
        keyframe_step=np.asarray(2, np.int32), velocity_units=np.asarray("world_per_video_frame"),
    )
    fps = 30.0
    params, _, _ = build_canonical_from_npz(
        path, fps=fps, keyframe_stride=2, device="cpu", compute_device="cpu",
        source_frame_start=1, source_frame_end=8,
    )

    # Reference: the original per-time boolean-mask loop.
    pos = torch.from_numpy(positions)
    nu = torch.from_numpy(velocities) * fps
    dyn = ~torch.from_numpy(is_static)
    t = torch.from_numpy(frames).float()
    anchors = pos.clone(); omega = torch.zeros_like(pos); scales = torch.zeros(n, 3)
    for t_val in torch.unique(t[dyn]).sort().values:
        idx = ((t == t_val) & dyn).nonzero().squeeze(-1)
        kf_pos, kf_vel = pos[idx], nu[idx]
        knn_d = from_colmap._knn_distance(kf_pos, k=4)
        scales[idx] = torch.log(torch.clamp(knn_d * 0.3, min=1e-6)).unsqueeze(-1).expand(-1, 3)
        ext = (kf_pos.max(0).values - kf_pos.min(0).values).max().clamp(min=1e-6)
        anchors[idx] = from_colmap._radius_ball_centroid(kf_pos, 0.1 * ext.item())
        nu_k, om_k, anchors[idx] = fit_local_twist(kf_pos, kf_vel, anchors[idx], local_knn_indices(kf_pos, 16), reg=1e-4)
        omega[idx] = om_k; nu[idx] = nu_k

    d = dyn
    torch.testing.assert_close(params["anchors"].detach()[d], anchors[d])
    torch.testing.assert_close(params["scales"].detach()[d], scales[d])
    torch.testing.assert_close(params["zeta_base_t"].detach()[d], nu[d])
    torch.testing.assert_close(params["zeta_base_r"].detach()[d], omega[d])
    assert params["means"].device.type == "cpu"


def _durations_npz(path: Path, durations_norm: np.ndarray) -> None:
    n = len(durations_norm)
    rng = np.random.default_rng(1)
    np.savez_compressed(
        path, positions=rng.normal(size=(n, 3)).astype(np.float32),
        colors=np.full((n, 3), 0.5, np.float32), velocities=np.zeros((n, 3), np.float32),
        times=np.zeros((n, 1), np.float32), durations=durations_norm.reshape(-1, 1).astype(np.float32),
        is_static=np.zeros(n, bool), dynamic_protect=np.zeros(n, bool),
        dynamic_allowed=np.ones(n, bool), stage_keep=np.ones(n, bool),
        source_frames=np.full((n, 1), 6, np.int32),
        source_frame_start=np.asarray(1, np.int32), source_frame_end=np.asarray(301, np.int32),
        frame_start=np.asarray(0), frame_end=np.asarray(300),
        time_units=np.asarray("normalized_0_1_over_source_frame_range"),
        keyframe_step=np.asarray(5, np.int32), velocity_units=np.asarray("world_per_video_frame"),
    )


def test_npz_lifetimes_are_converted_and_clamped(tmp_path: Path) -> None:
    fps, stride = 30.0, 5
    frames = np.array([0.5, 1.0, 2.5, 4.0, 9.0])          # durations in frames
    path = tmp_path / "d.npz"
    _durations_npz(path, frames / 300.0)                    # normalized over 300 frames
    kw = dict(fps=fps, keyframe_stride=stride, device="cpu", source_frame_start=1, source_frame_end=301)

    uniform, _, _ = build_canonical_from_npz(path, **kw)
    torch.testing.assert_close(uniform["log_durations"].exp().reshape(-1), torch.full((5,), stride / fps))

    npz_mode, _, _ = build_canonical_from_npz(path, init_sigma_t_mode="npz", init_sigma_t_floor_frac=0.4, **kw)
    want = np.clip(frames / fps, 0.4 * stride / fps, stride / fps)
    torch.testing.assert_close(
        npz_mode["log_durations"].exp().reshape(-1), torch.from_numpy(want).float(), rtol=1e-5, atol=1e-6,
    )


def test_npz_acceleration_fit_starts_rows_on_the_parabola(tmp_path: Path) -> None:
    """End to end: --centred-velocity NPZ -> rows start on x + v dt + 1/2 s a dt^2."""
    from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig

    rng = np.random.default_rng(0)
    n, fps = 120, 30.0
    positions = (rng.normal(size=(n, 3)) * 0.2).astype(np.float32)
    v_pf = np.tile(np.array([[0.02, -0.01, 0.005]], np.float32), (n, 1))
    a_pf = np.tile(np.array([[-0.004, 0.002, 0.0]], np.float32), (n, 1))
    path = tmp_path / "accel.npz"
    np.savez(
        path,
        positions=positions, colors=np.full((n, 3), 0.5, np.float32), velocities=v_pf,
        accelerations=a_pf, has_acceleration=np.ones(n, bool),
        times=np.full((n, 1), 10 / 20, np.float32), durations=np.full((n, 1), 0.25, np.float32),
        source_frames=np.full((n, 1), 11, np.int32), is_static=np.zeros(n, bool),
        frame_start=np.array(0, np.int32), frame_end=np.array(20, np.int32), keyframe_step=np.array(5, np.int32),
        velocity_units=np.array("world_per_video_frame"),
        time_units=np.array("normalized_0_1_over_source_frame_range"),
    )
    scale = 0.5
    params, _, _ = build_canonical_from_npz(
        path, fps=fps, keyframe_stride=5, device="cpu", init_sigma_t_mode="uniform",
        init_accel_mode="fit", init_accel_scale=scale, init_accel_fit_iters=600,
    )
    P = {k: v.detach() for k, v in params.items()}
    cfg = TrigsTrainingConfig(fps=fps, temporal_active_cull=False, verbose=False)
    v_s, a_s = torch.from_numpy(v_pf) * fps, torch.from_numpy(a_pf) * fps * fps * scale
    for dt in (-4 / fps, -2 / fps, 2 / fps, 4 / fps):
        x, _, _ = TrigsTrainer._deform(None, P, P["times"] + dt, cfg)
        target = torch.from_numpy(positions) + v_s * dt + 0.5 * a_s * dt * dt
        assert (x - target).norm(dim=-1).max() < 2e-3
    off, _, _ = build_canonical_from_npz(path, fps=fps, keyframe_stride=5, device="cpu", init_sigma_t_mode="uniform")
    assert all(float(off[k].abs().max()) == 0.0 for k in ("zeta_c0_t", "zeta_c1_t", "zeta_c2_t"))


def test_speed_aware_lifetimes(tmp_path: Path) -> None:
    """init_sigma_t_mode='speed': sigma = K * local spacing / speed, clamped to
    [floor_frac x init_sigma_t, init_sigma_t_max]. Still rows hit the cap, fast
    rows the floor, and a single noisy velocity inside a still patch does not
    shorten that row (neighbourhood-median speed)."""
    spacing = 0.01
    g = np.stack(np.meshgrid(np.arange(8), np.arange(8), np.arange(2)), -1).reshape(-1, 3) * spacing
    still = g.astype(np.float32)
    slow = (g + [1.0, 0, 0]).astype(np.float32)          # well separated patches
    fast = (g + [2.0, 0, 0]).astype(np.float32)
    pos = np.concatenate([still, slow, fast])
    n = len(pos); n0 = len(g)
    vel = np.zeros((n, 3), np.float32)                   # world per video frame
    vel[n0:2 * n0, 0] = 0.001                            # 0.1 spacing / frame
    vel[2 * n0:, 0] = 0.05                               # 5 spacings / frame
    vel[3, 0] = 0.05                                     # one noisy track inside the still patch
    path = tmp_path / "speed.npz"
    np.savez_compressed(
        path, positions=pos, colors=np.full((n, 3), 0.5, np.float32), velocities=vel,
        times=np.zeros((n, 1), np.float32), is_static=np.zeros(n, bool),
        dynamic_protect=np.zeros(n, bool), dynamic_allowed=np.ones(n, bool), stage_keep=np.ones(n, bool),
        source_frames=np.full((n, 1), 1, np.int32), source_frame_start=np.asarray(1, np.int32),
        source_frame_end=np.asarray(8, np.int32), keyframe_step=np.asarray(2, np.int32),
        velocity_units=np.asarray("world_per_video_frame"),
    )
    fps = 30.0
    params, _, _ = build_canonical_from_npz(
        path, fps=fps, keyframe_stride=2, device="cpu", compute_device="cpu",
        source_frame_start=1, source_frame_end=8, init_sigma_t=0.1667, init_sigma_t_mode="speed",
        init_sigma_t_speed_spacings=1.0, init_sigma_t_max=0.5,
    )
    sig = params["log_durations"].detach().exp().reshape(-1).numpy()
    floor = 0.4 * 0.1667
    np.testing.assert_allclose(sig[:n0], 0.5, rtol=1e-5)            # still -> cap (incl. the noisy row 3)
    np.testing.assert_allclose(sig[2 * n0:], floor, rtol=1e-5)      # 5 spacings/frame -> floor
    knn = from_colmap._knn_distance(torch.from_numpy(slow), k=4).numpy()
    expect = np.clip(knn / (0.001 * fps), floor, 0.5)               # 1 spacing / (0.03 world/s)
    np.testing.assert_allclose(sig[n0:2 * n0], expect, rtol=1e-4)
    assert (expect > floor).all() and (expect < 0.5).all()          # the slow patch is in between


def test_speed_aware_lifetimes_bound_acceleration(tmp_path: Path) -> None:
    """Under init_accel_mode='fit' the speed rule bounds the whole starting curve:
    |v| s + 1/2 |a| s^2 = K d, so a still-but-accelerating patch is not given a
    long life over which the accel term would carry it many spacings."""
    spacing = 0.01
    g = (np.stack(np.meshgrid(np.arange(8), np.arange(8), np.arange(2)), -1).reshape(-1, 3) * spacing)
    pos = np.concatenate([g, g + [1.0, 0, 0]]).astype(np.float32)
    n = len(pos); n0 = len(g)
    vel = np.zeros((n, 3), np.float32)
    vel[:, 0] = 0.001                                    # 0.1 spacing / frame everywhere
    acc = np.zeros((n, 3), np.float32)
    acc[n0:, 1] = 0.002                                  # second patch accelerates (world / frame^2)
    path = tmp_path / "speed_acc.npz"
    np.savez_compressed(
        path, positions=pos, colors=np.full((n, 3), 0.5, np.float32), velocities=vel,
        accelerations=acc, has_acceleration=np.ones(n, bool),
        times=np.zeros((n, 1), np.float32), is_static=np.zeros(n, bool),
        dynamic_protect=np.zeros(n, bool), dynamic_allowed=np.ones(n, bool), stage_keep=np.ones(n, bool),
        source_frames=np.full((n, 1), 1, np.int32), source_frame_start=np.asarray(1, np.int32),
        source_frame_end=np.asarray(8, np.int32), keyframe_step=np.asarray(2, np.int32),
        velocity_units=np.asarray("world_per_video_frame"),
    )
    fps, k_sp, a_scale = 30.0, 2.0, 0.5
    params, _, _ = build_canonical_from_npz(
        path, fps=fps, keyframe_stride=2, device="cpu", compute_device="cpu",
        source_frame_start=1, source_frame_end=8, init_sigma_t=0.1667, init_sigma_t_mode="speed",
        init_sigma_t_speed_spacings=k_sp, init_sigma_t_max=1.0,
        init_accel_mode="fit", init_accel_scale=a_scale, init_accel_fit_iters=5,
    )
    sig = params["log_durations"].detach().exp().reshape(-1).numpy()
    knn = from_colmap._knn_distance(torch.from_numpy(pos[:n0]), k=4).numpy()
    v, a = 0.001 * fps, 0.002 * fps * fps * a_scale      # world/s, smoothed+scaled world/s^2
    np.testing.assert_allclose(sig[:n0], k_sp * knn / v, rtol=1e-4)            # velocity only
    kd = k_sp * knn
    expect = (-v + np.sqrt(v * v + 2 * a * kd)) / a                          # v s + a s^2 / 2 = K d
    np.testing.assert_allclose(sig[n0:], expect, rtol=1e-3)
    np.testing.assert_allclose(v * sig[n0:] + 0.5 * a * sig[n0:] ** 2, kd, rtol=1e-3)
    assert (sig[n0:] < 0.5 * sig[:n0]).all()
