"""Smoke tests for the TRiGS trainer.

Covers:
- the end-to-end deformation (Eqs. 4–9) producing finite tensors,
- identity-motion invariance at initialization (ζ ≡ 0 ⇒ µ(t) = µ),
- one training-style backward pass producing gradients on every trainable
  parameter family,
- L_motion vanishing when all Bézier control points coincide,
- motion-guided relocation (now a gsplat Strategy) clones active primitives
  over inactive slots.
"""

import dataclasses
import itertools
import math
import random
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from trigs.core.losses import (
    bezier_motion_loss,
    rigid_coherence_loss,
    scale_size_loss,
)
from trigs.core.trainer import (
    TrigsTrainer,
    TrigsTrainingConfig,
    _bounded_diagnostic_sample,
    _motion_residual_magnitude_loss,
    opacity_diag_summary,
)
from trigs.strategy.motion_guided_relocation import (
    MotionGuidedRelocationConfig,
    _motion_guided_relocation,
)


def _make_params(N: int = 64, device: str = "cpu") -> nn.ParameterDict:
    """Build a minimal ParameterDict matching ``TrigsTrainer`` expectations."""
    means = torch.randn(N, 3, device=device) * 0.5
    quats = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.full((N, 3), -3.0, device=device)
    opacities = torch.randn(N, device=device)
    sh0 = torch.randn(N, 3, device=device) * 0.1
    times = torch.rand(N, 1, device=device)
    log_durations = torch.full((N, 1), math.log(0.1), device=device)

    def _z3():
        return torch.zeros(N, 3, device=device)
    return nn.ParameterDict({
        "means": nn.Parameter(means, requires_grad=False),
        "quats": nn.Parameter(quats, requires_grad=False),
        "scales": nn.Parameter(scales, requires_grad=False),
        "sh0": nn.Parameter(sh0, requires_grad=False),
        "opacities": nn.Parameter(opacities, requires_grad=True),
        "times": nn.Parameter(times, requires_grad=True),
        "log_durations": nn.Parameter(log_durations, requires_grad=True),
        "anchors": nn.Parameter(means.detach().clone(), requires_grad=True),
        "zeta_base_t": nn.Parameter(_z3(), requires_grad=True),
        "zeta_base_r": nn.Parameter(_z3(), requires_grad=True),
        "zeta_c0_t":   nn.Parameter(_z3(), requires_grad=True),
        "zeta_c0_r":   nn.Parameter(_z3(), requires_grad=True),
        "zeta_c1_t":   nn.Parameter(_z3(), requires_grad=True),
        "zeta_c1_r":   nn.Parameter(_z3(), requires_grad=True),
        "zeta_c2_t":   nn.Parameter(_z3(), requires_grad=True),
        "zeta_c2_r":   nn.Parameter(_z3(), requires_grad=True),
    })


def test_bounded_diagnostic_sample_is_deterministic_and_includes_endpoints():
    values = torch.arange(20, dtype=torch.float32)
    sample = _bounded_diagnostic_sample(values, max_samples=5)

    assert torch.equal(sample, torch.tensor([0.0, 4.0, 9.0, 14.0, 19.0]))
    assert torch.equal(
        _bounded_diagnostic_sample(values, max_samples=100),
        values,
    )


def test_motion_residual_loss_is_zero_for_all_static_empty_selection():
    controls = torch.zeros(5, 6)
    no_dynamic_rows = torch.zeros(5, dtype=torch.bool)

    loss = _motion_residual_magnitude_loss(
        controls,
        controls,
        controls,
        no_dynamic_rows,
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_chunked_result_device_is_opt_in_and_preserves_default(
    monkeypatch: pytest.MonkeyPatch,
):
    """Existing chunked callers still receive trainer-device parameters."""
    from trigs.core.temporal_chunks import ChunkResidency

    requested_devices: list[str] = []
    original_materialize = ChunkResidency.materialize

    def _capture_materialize(self, idx, device):
        requested_devices.append(str(device))
        # Keep this CPU-only unit test independent of CUDA availability.
        return original_materialize(self, idx, "cpu")

    monkeypatch.setattr(ChunkResidency, "materialize", _capture_materialize)

    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        n_steps=0,
        temporal_chunk_len_s=1.0,
        temporal_chunk_rounds=1,
        temporal_chunk_entry_diagnostics=False,
        relocate=False,
        verbose=False,
    )
    trainer = TrigsTrainer(device="cpu")
    trainer.device = torch.device("cuda:7")

    common = {
        "sh_degree": 0,
        "scene_extent": 1.0,
        "cameras": [],
        "n_frames": 1,
        "gt_images_per_frame": {},
        "gt_alphas_per_frame": {},
        "config": config,
    }
    trainer.train_chunked(params=_make_params(N=4), **common)
    assert requested_devices[-1] == "cuda:7"

    trainer.train_chunked(
        params=_make_params(N=4),
        result_device="cpu",
        **common,
    )
    assert requested_devices[-1] == "cpu"


def _capture_chunk_configs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rounds: int,
    n_steps: int = 90_000,
    n_frames: int = 300,
    fps: float = 60.0,
    chunk_len_s: float = 2.0,
) -> list[TrigsTrainingConfig]:
    """Run ``train_chunked`` with training stubbed out and return the config
    each chunk call would have trained with.
    """
    from trigs.core.trainer import TrigsTrainingResult

    configs: list[TrigsTrainingConfig] = []

    def _fake_train_from_params(self, *, params, config, **kwargs):
        configs.append(config)
        return TrigsTrainingResult(
            params=params, temporal_params={}, motion_params={}, metrics={},
        )

    monkeypatch.setattr(
        TrigsTrainer, "train_from_params", _fake_train_from_params,
    )

    # Spread mu_t over the whole sequence so every window owns rows; windows
    # with no resident rows are skipped and would not exercise the schedule.
    params = _make_params(N=256)
    t_max = (n_frames - 1) / fps
    with torch.no_grad():
        params["times"].copy_(
            torch.linspace(0.0, t_max, 256).reshape(-1, 1)
        )

    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        n_steps=n_steps,
        fps=fps,
        temporal_chunk_len_s=chunk_len_s,
        temporal_chunk_overlap_s=0.5,
        temporal_chunk_rounds=rounds,
        temporal_chunk_supervision_pad_s=0.0,
        temporal_chunk_entry_diagnostics=False,
        relocate=False,
        verbose=False,
    )
    TrigsTrainer(device="cpu").train_chunked(
        params=params,
        sh_degree=0,
        scene_extent=1.0,
        cameras=[],
        n_frames=n_frames,
        gt_images_per_frame={},
        gt_alphas_per_frame={},
        config=config,
    )
    return configs


def _run_chunked_with_viewer_snapshots(
    monkeypatch: pytest.MonkeyPatch, out_dir, *, every: int, n_steps: int = 9_000,
):
    """Chunked run with training stubbed to nudge ``means`` once per call.
    Returns (per-call chunk configs, result)."""
    from trigs.core.trainer import TrigsTrainingResult

    configs: list[TrigsTrainingConfig] = []

    def _fake_train_from_params(self, *, params, config, **kwargs):
        configs.append(config)
        with torch.no_grad():
            params["means"].add_(1.0)
        return TrigsTrainingResult(
            params=params, temporal_params={}, motion_params={}, metrics={},
        )

    monkeypatch.setattr(TrigsTrainer, "train_from_params", _fake_train_from_params)

    n_frames, fps = 300, 60.0
    params = _make_params(N=256)
    with torch.no_grad():
        params["times"].copy_(
            torch.linspace(0.0, (n_frames - 1) / fps, 256).reshape(-1, 1)
        )
    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        n_steps=n_steps, fps=fps,
        temporal_chunk_len_s=2.0, temporal_chunk_overlap_s=0.5,
        temporal_chunk_rounds=2, temporal_chunk_entry_diagnostics=False,
        relocate=False, verbose=False,
        viewer_snapshot_every=every, viewer_snapshot_dir=str(out_dir),
        viewer_snapshot_keep_last=1000,
    )
    result = TrigsTrainer(device="cpu").train_chunked(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[], n_frames=n_frames,
        gt_images_per_frame={}, gt_alphas_per_frame={}, config=config,
        result_device="cpu",
    )
    return configs, result


def test_chunked_viewer_snapshots_are_whole_model_at_global_step_cadence(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
):
    from trigs.core.viewer_snapshot import (
        list_snapshots, load_snapshot, read_latest_pointer, should_snapshot,
    )

    configs, result = _run_chunked_with_viewer_snapshots(
        monkeypatch, tmp_path, every=2_000,
    )

    # Chunk calls must never write: they only hold part of the model.
    assert configs and all(c.viewer_snapshot_every == 0 for c in configs)

    # Replay the cadence over the cumulative global step at each call's end.
    expected, last, total = [], 0, 0
    for c in configs:
        total += int(c.n_steps)
        if should_snapshot(total, 2_000, last):
            expected.append(total)
            last = total
    assert total == 9_000
    if expected[-1] != total:
        expected.append(total)  # the final snapshot
    snaps = list_snapshots(tmp_path)
    assert [s for s, _ in snaps] == expected
    assert len(expected) >= 3
    assert read_latest_pointer(tmp_path)["step"] == total

    first = load_snapshot(snaps[0][1])
    final = load_snapshot(snaps[-1][1])
    assert first["extra"]["mode"] == "chunked"
    assert first["extra"]["n_rounds"] == 2 and "trained_window" in first["extra"]
    assert final["step"] == 9_000 and final["progress"] == 1.0

    # Every snapshot is the full row count, and the last one IS the result.
    for _, path in snaps:
        assert load_snapshot(path)["params"]["means"].shape[0] == 256
    for key, value in final["params"].items():
        assert torch.equal(value, result.params[key].detach().cpu()), key
    # ...while earlier snapshots show earlier training progress.
    assert not torch.equal(first["params"]["means"], final["params"]["means"])


def test_chunked_viewer_snapshots_off_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
):
    _run_chunked_with_viewer_snapshots(monkeypatch, tmp_path / "snaps", every=0)
    assert not (tmp_path / "snaps").exists()


def test_viewer_snapshots_without_a_directory_are_off():
    cfg = TrigsTrainingConfig(viewer_snapshot_every=100, viewer_snapshot_dir=None)
    assert TrigsTrainer._viewer_snapshot_dir(cfg) is None
    assert TrigsTrainer._viewer_snapshot_dir(TrigsTrainingConfig(viewer_snapshot_every=0, viewer_snapshot_dir="x")) is None
    assert TrigsTrainer._viewer_snapshot_dir(TrigsTrainingConfig(viewer_snapshot_dir="x")) == Path("x")


def test_viewer_snapshot_write_failure_does_not_raise(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    cfg = TrigsTrainingConfig(verbose=False)
    ok = TrigsTrainer(device="cpu")._write_viewer_snapshot(
        out_dir=blocker / "sub", params=_make_params(N=4), step=1, config=cfg,
        n_frames=1, sh_degree=0, sh_degree_active=0, extra={},
    )
    assert ok is False


def _capture_chunk_lr_bands(
    monkeypatch: pytest.MonkeyPatch, **kwargs,
) -> list[tuple[float, float]]:
    """Per chunk call, the (start, end) LR-schedule progress band that call
    would traverse.
    """
    from trigs.core.trainer import _lr_schedule_progress

    return [
        (_lr_schedule_progress(c, 0), _lr_schedule_progress(c, c.n_steps))
        for c in _capture_chunk_configs(monkeypatch, **kwargs)
    ]


def test_chunked_lr_band_is_shared_by_every_window_in_a_round(
    monkeypatch: pytest.MonkeyPatch,
):
    """All windows of a round traverse the same slice of the LR schedule.

    Regression test for the sequence-long quality ramp: keying
    ``lr_schedule_offset_steps`` off the running global step made a window's
    LR depend on its position in the time-ordered chunk loop, so the final
    (authoritative) write for early frames ran at ~10x the LR of the write
    for late frames.
    """
    rounds = 3
    bands = _capture_chunk_lr_bands(monkeypatch, rounds=rounds)

    n_windows = len(bands) // rounds
    assert n_windows > 1, "test needs >1 window per round to be meaningful"
    assert len(bands) == rounds * n_windows

    for round_i in range(rounds):
        in_round = bands[round_i * n_windows:(round_i + 1) * n_windows]
        first = in_round[0]
        for band in in_round[1:]:
            assert band == pytest.approx(first, abs=1e-6), (
                f"round {round_i}: windows disagree on LR band "
                f"{first} vs {band}"
            )


def test_chunked_lr_bands_tile_the_schedule_across_rounds(
    monkeypatch: pytest.MonkeyPatch,
):
    """Rounds still anneal: their bands are contiguous and reach the floor."""
    rounds = 3
    bands = _capture_chunk_lr_bands(monkeypatch, rounds=rounds)
    n_windows = len(bands) // rounds
    per_round = [bands[i * n_windows] for i in range(rounds)]

    assert per_round[0][0] == pytest.approx(0.0, abs=1e-6)
    assert per_round[-1][1] == pytest.approx(1.0, abs=1e-3)
    for (_, prev_end), (next_start, _) in itertools.pairwise(per_round):
        assert next_start == pytest.approx(prev_end, abs=1e-3)
    for start, end in per_round:
        assert end > start


def test_chunked_single_window_lr_schedule_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    """One window per round degenerates to the original global schedule."""
    rounds = 2
    bands = _capture_chunk_lr_bands(
        monkeypatch, rounds=rounds, chunk_len_s=1000.0,
    )

    assert len(bands) == rounds
    assert bands[0][0] == pytest.approx(0.0, abs=1e-6)
    assert bands[0][1] == pytest.approx(0.5, abs=1e-3)
    assert bands[1][1] == pytest.approx(1.0, abs=1e-3)


def test_lr_schedule_progress_scale_maps_local_steps_onto_a_band():
    """The scale stretches a short call across a longer schedule slice."""
    from trigs.core.trainer import _lr_schedule_progress

    config = TrigsTrainingConfig(
        n_steps=1000,
        lr_schedule_total_steps=100_001,
        lr_schedule_offset_steps=50_000,
        lr_schedule_step_scale=25.0,
    )

    assert _lr_schedule_progress(config, 0) == pytest.approx(0.5)
    assert _lr_schedule_progress(config, 1000) == pytest.approx(0.75)
    # Clamped, never extrapolated past the floor.
    assert _lr_schedule_progress(config, 10_000) == pytest.approx(1.0)

    # A non-positive scale falls back to 1:1 rather than freezing the LR.
    unscaled = dataclasses.replace(config, lr_schedule_step_scale=0.0)
    assert _lr_schedule_progress(unscaled, 1000) == pytest.approx(0.51)


def test_lr_schedule_progress_is_call_local_without_shared_schedule():
    """Default (unchunked) configs keep the per-call schedule."""
    from trigs.core.trainer import _lr_schedule_progress

    config = TrigsTrainingConfig(n_steps=10_001)
    assert _lr_schedule_progress(config, 0) == pytest.approx(0.0)
    assert _lr_schedule_progress(config, 5_000) == pytest.approx(0.5)
    assert _lr_schedule_progress(config, 10_000) == pytest.approx(1.0)


def test_sh_warmup_runs_once_on_cumulative_steps(
    monkeypatch: pytest.MonkeyPatch,
):
    """Only the first chunk call warms SH up; later calls stay at full degree.

    Regression test: the degree was keyed on the per-call step, so every
    chunk call dropped back to degree 0 for 1000 steps.
    """
    from trigs.core.trainer import _sh_degree_for_step

    configs = _capture_chunk_configs(monkeypatch, rounds=2)
    assert len(configs) > 2

    first = configs[0]
    assert _sh_degree_for_step(first, 0, 3) == 0
    assert _sh_degree_for_step(first, 2_999, 3) == 2
    assert _sh_degree_for_step(first, 3_000, 3) == 3

    cumulative = 0
    for config in configs:
        assert config.sh_warmup_step_offset == cumulative
        cumulative += config.n_steps
    for config in configs[1:]:
        assert _sh_degree_for_step(config, 0, 3) == 3


def test_deform_identity_at_init():
    """With ζ_base = ζ^(k) = 0 the deformation is the identity at every t."""
    torch.manual_seed(0)
    svc = TrigsTrainer(device="cpu")
    params = _make_params(N=32)
    config = TrigsTrainingConfig(verbose=False)

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        means_t, quats_t, gamma = svc._deform(params, t, config)
        assert torch.allclose(means_t, params["means"], atol=1e-5), f"t={t}"
        # q_rel = identity ⇒ quats_t = ±quats (hemisphere sign); accept both.
        canon = torch.nn.functional.normalize(params["quats"], dim=-1)
        q_new = torch.nn.functional.normalize(quats_t, dim=-1)
        agree = torch.allclose(q_new, canon, atol=1e-5) or torch.allclose(
            q_new, -canon, atol=1e-5,
        )
        assert agree, f"quat mismatch at t={t}"
        assert torch.isfinite(gamma).all()


def test_deform_nonzero_motion_is_finite():
    """Non-zero ζ produces finite means/quats and non-trivial motion."""
    torch.manual_seed(0)
    svc = TrigsTrainer(device="cpu")
    params = _make_params(N=32)
    config = TrigsTrainingConfig(verbose=False)
    with torch.no_grad():
        for name in ("zeta_base_t", "zeta_base_r", "zeta_c1_t", "zeta_c1_r"):
            params[name].copy_(torch.randn_like(params[name]) * 0.2)

    means_t, quats_t, gamma = svc._deform(params, t=0.3, config=config)
    assert torch.isfinite(means_t).all() and torch.isfinite(quats_t).all()
    assert not torch.allclose(means_t, params["means"], atol=1e-4)


@pytest.mark.parametrize("detach", [True, False])
def test_bezier_window_detach_blocks_temporal_retiming(detach: bool):
    """σ_t reaches means_t only through τ, so detaching the window must zero
    its gradient there; μ_t keeps its Δt path either way.
    """
    torch.manual_seed(0)
    svc = TrigsTrainer(device="cpu")
    params = _make_params(N=32)
    config = TrigsTrainingConfig(verbose=False, bezier_window_detach=detach)
    with torch.no_grad():
        for name in ("zeta_base_t", "zeta_base_r"):
            params[name].copy_(torch.randn_like(params[name]) * 0.2)
        for name in ("zeta_c0_t", "zeta_c1_t", "zeta_c2_t", "zeta_c0_r", "zeta_c2_r"):
            params[name].copy_(torch.randn_like(params[name]) * 0.3)
        params["times"].fill_(0.5)

    means_t, quats_t, _ = svc._deform(params, t=0.6, config=config)
    (means_t.sum() + quats_t.sum()).backward()

    sigma_grad = params["log_durations"].grad
    sigma_moves = sigma_grad is not None and bool((sigma_grad != 0).any())
    assert sigma_moves is (not detach)
    assert params["times"].grad is not None
    assert (params["times"].grad != 0).any()


def test_backward_populates_gradients():
    """One backward pass on a fabricated render-free loss populates grads on
    every TRiGS-trainable parameter family."""
    torch.manual_seed(0)
    svc = TrigsTrainer(device="cpu")
    params = _make_params(N=48)
    config = TrigsTrainingConfig(verbose=False)

    with torch.no_grad():
        for name in ("zeta_base_t", "zeta_base_r",
                     "zeta_c0_t", "zeta_c0_r",
                     "zeta_c1_t", "zeta_c1_r",
                     "zeta_c2_t", "zeta_c2_r"):
            params[name].copy_(torch.randn_like(params[name]) * 0.1)

    means_t, quats_t, gamma = svc._deform(params, t=0.4, config=config)

    # Surrogate loss: match deformed means against a displaced target plus
    # a Bézier-acceleration term on the reassembled 6-vector.
    target = params["means"] + 0.01
    loss = (means_t - target).pow(2).mean() + (1.0 - gamma).mean()
    c0_full = torch.cat([params["zeta_c0_t"], params["zeta_c0_r"]], dim=-1)
    c1_full = torch.cat([params["zeta_c1_t"], params["zeta_c1_r"]], dim=-1)
    c2_full = torch.cat([params["zeta_c2_t"], params["zeta_c2_r"]], dim=-1)
    loss = loss + bezier_motion_loss(c0_full, c1_full, c2_full)
    loss.backward()

    for name in ("anchors",
                 "zeta_base_t", "zeta_base_r",
                 "zeta_c0_t", "zeta_c1_t", "zeta_c2_t",
                 "zeta_c0_r", "zeta_c1_r", "zeta_c2_r",
                 "times", "log_durations"):
        grad = params[name].grad
        assert grad is not None, f"{name} has no gradient"
        assert torch.isfinite(grad).all(), f"{name} grad has NaN/Inf"
        assert grad.abs().sum() > 0, f"{name} grad is identically zero"


def test_motion_loss_vanishes_on_constant_bezier():
    """L_motion = ‖c0 − 2 c1 + c2‖² = 0 when c0 = c1 = c2."""
    c = torch.randn(10, 6)
    loss = bezier_motion_loss(c, c.clone(), c.clone())
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_rigid_loss_vanishes_when_zeta_uniform():
    """L_rigid = 0 when all neighbors share the same ζ^base."""
    N, K = 16, 4
    zeta = torch.randn(1, 6).expand(N, 6).contiguous()
    colors = torch.randn(N, 3)
    # Any kNN indexing will do; equal ζ ⇒ diffs are zero.
    nn_idx = torch.arange(N).unsqueeze(1).repeat(1, K)
    loss = rigid_coherence_loss(zeta, colors, nn_idx, lambda_c=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("detach", [True, False])
def test_rigid_color_weight_gradient_reaches_sh0_only_when_attached(detach: bool):
    """Official TRiGS keeps K_ij differentiable; the flag restores detach."""
    torch.manual_seed(0)
    N, K = 16, 3
    zeta = torch.randn(N, 6, requires_grad=True)
    colors = (torch.randn(N, 3) * 0.05).requires_grad_(True)
    nn_idx = torch.randint(0, N, (N, K))
    rigid_coherence_loss(
        zeta, colors, nn_idx, lambda_c=50.0, detach_color_weight=detach,
    ).backward()
    assert zeta.grad is not None and (zeta.grad != 0).any()
    color_moves = colors.grad is not None and bool((colors.grad != 0).any())
    assert color_moves is (not detach)


def test_scale_size_loss_matches_official_per_row_gradient():
    """d/ds [λ·mean(exp(s))] = λ·exp(s)/(3·N_total), whatever subset is resident."""
    torch.manual_seed(0)
    log_scales = (torch.randn(10, 3) * 0.5).requires_grad_(True)
    rows = torch.zeros(10, dtype=torch.bool)
    rows[:6] = True
    scale_size_loss(log_scales, n_total_rows=40, rows=rows).backward()

    expected = log_scales.detach().exp() / (3 * 40)
    expected[~rows] = 0.0
    assert torch.allclose(log_scales.grad, expected, atol=1e-7)

    full = scale_size_loss(log_scales.detach(), n_total_rows=10)
    assert full.item() == pytest.approx(log_scales.detach().exp().mean().item())


def test_scale_size_loss_normalizes_by_full_model_in_chunks():
    """A chunk-resident subset is normalized by the master row count."""
    params = _make_params(N=8)
    assert TrigsTrainer._total_model_rows(params) == 8
    params["chunk_total_rows"] = nn.Parameter(
        torch.tensor(500.0), requires_grad=False,
    )
    assert TrigsTrainer._total_model_rows(params) == 500


def test_rigid_knn_space_selects_anchor_neighbors():
    """The default graph is built on anchors (official), not canonical means."""
    torch.manual_seed(0)
    svc = TrigsTrainer(device="cpu")
    params = _make_params(N=32)
    with torch.no_grad():
        params["anchors"].copy_(params["means"][torch.randperm(32)])

    anchors_cfg = TrigsTrainingConfig(verbose=False)
    means_cfg = TrigsTrainingConfig(verbose=False, rigid_knn_space="means")
    assert anchors_cfg.rigid_knn_space == "anchors"

    knn_default = svc._compute_knn(
        TrigsTrainer._rigid_knn_points(params, anchors_cfg).detach(), 3,
    )
    knn_means = svc._compute_knn(
        TrigsTrainer._rigid_knn_points(params, means_cfg).detach(), 3,
    )
    assert torch.equal(knn_default, svc._compute_knn(params["anchors"].detach(), 3))
    assert not torch.equal(knn_default, knn_means)

    with pytest.raises(ValueError, match="rigid_knn_space"):
        TrigsTrainer._rigid_knn_points(
            params, TrigsTrainingConfig(verbose=False, rigid_knn_space="bogus"),
        )


def test_angular_motion_lrs_decay_like_official():
    """Official TRiGS schedules the angular base and Bézier LRs too."""
    from trigs.core.trainer import _lr_decay_ratios

    config = TrigsTrainingConfig(verbose=False)
    ratios = _lr_decay_ratios(config)
    assert ratios["zeta_base_r"] == pytest.approx(1e-2)
    assert ratios["zeta_bezier_r"] == pytest.approx(1.0 / 16.0)
    assert ratios["zeta_base_t"] == pytest.approx(1e-2)
    assert "means" in ratios

    frozen = TrigsTrainingConfig(verbose=False, freeze_geometry=True)
    assert "means" not in _lr_decay_ratios(frozen)


@pytest.mark.parametrize("batch", [1, 4])
def test_batch_optimizer_rule_matches_official(batch: int):
    """lr * sqrt(B), eps / sqrt(B), betas ** B, temporal params included."""
    svc = TrigsTrainer(device="cpu")
    params = _make_params(N=8)
    config = TrigsTrainingConfig(verbose=False, batch_cameras=batch, batch_lr_scaling=True)
    opts = svc._build_optimizers(params, config, scene_extent=2.0)

    group = opts["scales"].param_groups[0]
    assert group["lr"] == pytest.approx(config.lr_scales * batch ** 0.5)
    assert group["eps"] == pytest.approx(1e-15 / batch ** 0.5)
    assert group["betas"] == pytest.approx((0.9 ** batch, 0.999 ** batch))

    means = opts["means"].param_groups[0]
    assert means["lr"] == pytest.approx(config.lr_means * 2.0 * batch ** 0.5)

    times = opts["times"].param_groups[0]
    assert config.temporal_optimizer_beta1 == 0.9  # official default
    assert times["betas"] == pytest.approx((0.9 ** batch, 0.999 ** batch))
    no_momentum = dataclasses.replace(config, temporal_optimizer_beta1=0.0)
    times0 = svc._build_optimizers(params, no_momentum, 2.0)["times"].param_groups[0]
    assert times0["betas"][0] == 0.0

    unscaled = TrigsTrainingConfig(
        verbose=False, batch_cameras=batch, batch_optimizer_scaling=False,
    )
    plain = svc._build_optimizers(params, unscaled, scene_extent=2.0)
    assert plain["scales"].param_groups[0]["lr"] == pytest.approx(config.lr_scales)


@pytest.mark.parametrize("batch", [1, 20])
def test_default_batch_rule_keeps_lrs_as_effective_rates(batch: int):
    """Default: lr_* are the effective rates at any batch; eps and betas keep the batch rule.
    Same optimizer as the tuned runs (lr_* / sqrt(B) with the full rule)."""
    svc = TrigsTrainer(device="cpu")
    params = _make_params(N=8)
    config = TrigsTrainingConfig(verbose=False, batch_cameras=batch)
    assert config.batch_lr_scaling is False
    g = svc._build_optimizers(params, config, scene_extent=2.0)["scales"].param_groups[0]
    assert g["lr"] == pytest.approx(config.lr_scales)
    assert g["eps"] == pytest.approx(1e-15 / batch ** 0.5)
    assert g["betas"] == pytest.approx((0.9 ** batch, 0.999 ** batch))
    tuned = TrigsTrainingConfig(verbose=False, batch_cameras=batch, batch_lr_scaling=True,
                                lr_scales=config.lr_scales / batch ** 0.5)
    t = svc._build_optimizers(params, tuned, scene_extent=2.0)["scales"].param_groups[0]
    assert t["lr"] == pytest.approx(g["lr"]) and t["eps"] == g["eps"] and t["betas"] == g["betas"]


def test_training_camera_extent_is_official_radius_without_holdouts():
    """Max distance of training camera centers from their mean."""
    from trigs.core.trainer import training_camera_extent

    centers = torch.tensor([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, -3.0, 0.0],
        [50.0, 0.0, 0.0],  # held out: must not inflate the extent
    ])
    c2w = torch.eye(4).repeat(5, 1, 1)
    c2w[:, :3, 3] = centers
    viewmats = torch.linalg.inv(c2w)

    assert training_camera_extent(viewmats, exclude=(4,)) == pytest.approx(3.0)
    assert training_camera_extent(viewmats) > 30.0
    with pytest.raises(ValueError):
        training_camera_extent(viewmats[:1], exclude=(0,))


def test_config_fps_target_count_optional():
    """FPS knobs default to None/1.0 — inert unless set."""
    cfg = TrigsTrainingConfig()
    assert cfg.target_count is None
    assert cfg.fps_time_weight == 1.0


def test_config_new_defaults_match_paper_spirit():
    """Defaults critical for long-sequence performance."""
    cfg = TrigsTrainingConfig()
    # Geometry trains alongside motion — TriGS is a 3DGS extension.
    assert cfg.freeze_geometry is False
    assert cfg.init_sigma_t == pytest.approx(0.0)
    assert cfg.fps == pytest.approx(30.0)
    assert cfg.nn_velocity_init is True


def test_freeze_static_means_zeros_only_static_gradients():
    params = nn.ParameterDict({
        "means": nn.Parameter(torch.zeros(3, 3)),
        "is_static": nn.Parameter(
            torch.tensor([1.0, 0.0, 1.0]), requires_grad=False,
        ),
    })
    params["means"].grad = torch.ones_like(params["means"])
    cfg = TrigsTrainingConfig(freeze_static_means=True)

    TrigsTrainer._zero_static_mean_grads(params, cfg)

    assert torch.equal(
        params["means"].grad,
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
    )


def test_scale_anisotropy_caps_longest_axis_relative_to_median():
    params = nn.ParameterDict({
        "scales": nn.Parameter(torch.log(torch.tensor([
            [1e-4, 1e-2, 8e-2],
            [1e-2, 2e-2, 3e-2],
        ]))),
    })
    cfg = TrigsTrainingConfig(
        max_scale_exp=0.08,
        max_scale_anisotropy=4.0,
    )

    TrigsTrainer._post_step_safety_clamp(
        params=params, config=cfg, n_frames=2, scene_extent=1.0,
    )

    scales = torch.exp(params["scales"])
    median = scales.median(dim=-1).values
    assert torch.all(scales.max(dim=-1).values <= median * 4.0 + 1e-7)
    # The thin surface-normal axis is retained; the needle axis is shortened.
    assert scales[0, 0].item() == pytest.approx(1e-4)
    assert scales[0, 2].item() == pytest.approx(4e-2)


def test_nn_velocity_init_produces_trajectory():
    """At t_k + Δt the rough trajectory µ + ν·Δt reaches (mean of) the k-NN
    matched primitives in the next keyframe."""
    torch.manual_seed(0)
    svc = TrigsTrainer(device="cpu")
    N_per_kf = 20
    d = torch.tensor([0.3, -0.1, 0.2])  # rigid displacement
    X = torch.randn(N_per_kf, 3) * 100.0
    means = torch.cat([X, X + d], dim=0)
    times = torch.cat([
        torch.zeros(N_per_kf, 1),
        torch.full((N_per_kf, 1), 0.25),
    ], dim=0)

    params = nn.ParameterDict({
        "means": nn.Parameter(means, requires_grad=False),
        "times": nn.Parameter(times, requires_grad=False),
        "zeta_base_t": nn.Parameter(torch.zeros(2 * N_per_kf, 3)),
    })
    n = svc._nn_velocity_init(params, verbose=False, k=1)
    assert n == 2 * N_per_kf

    nu = params["zeta_base_t"].data
    expected_fwd = d / 0.25
    assert torch.allclose(nu[:N_per_kf], expected_fwd.expand_as(nu[:N_per_kf]), atol=1e-4)
    assert torch.allclose(nu[N_per_kf:], expected_fwd.expand_as(nu[N_per_kf:]), atol=1e-4)


def test_knn_time_aware_excludes_cross_time_neighbors():
    """With large `time_weight`, primitives at the same 3D position but
    different times must NOT be neighbors of each other."""
    svc = TrigsTrainer(device="cpu")
    pos = torch.zeros(8, 3)
    times = torch.tensor(
        [[0.0], [0.0], [0.0], [0.0], [0.8], [0.8], [0.8], [0.8]],
    )

    idx_aware = svc._compute_knn(pos, k=3, times=times, time_weight=100.0)
    for i in range(4):
        assert all(int(n) in {0, 1, 2, 3} for n in idx_aware[i].tolist())
    for i in range(4, 8):
        assert all(int(n) in {4, 5, 6, 7} for n in idx_aware[i].tolist())


def test_sigma_floor_one_sided():
    """L_sigma_floor is quadratic below target and zero above."""
    target = 0.04
    log_sigma = torch.tensor([[-5.0], [-3.0], [-2.0], [0.0]], requires_grad=True)
    sigma_t = torch.exp(log_sigma)
    loss = torch.clamp(target - sigma_t, min=0.0).pow(2).mean()
    loss.backward()

    per_primitive = torch.clamp(target - sigma_t, min=0.0).pow(2).detach()
    assert per_primitive[0].item() > 0
    assert per_primitive[1].item() == pytest.approx(0.0, abs=1e-10)
    assert per_primitive[2].item() == pytest.approx(0.0, abs=1e-10)
    assert per_primitive[3].item() == pytest.approx(0.0, abs=1e-10)

    grads = log_sigma.grad.flatten()
    assert grads[0].item() < 0
    assert grads[1].item() == pytest.approx(0.0, abs=1e-10)
    assert grads[2].item() == pytest.approx(0.0, abs=1e-10)
    assert grads[3].item() == pytest.approx(0.0, abs=1e-10)


def test_radius_ball_centroid_uniform_grid():
    """On a regular grid, interior-point centroid = self."""
    svc = TrigsTrainer(device="cpu")
    x, y = torch.meshgrid(torch.arange(5.0), torch.arange(5.0), indexing="ij")
    z = torch.zeros_like(x)
    means = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    centroids = svc._radius_ball_centroid(means, radius=1.5)
    assert centroids.shape == means.shape
    center_idx = 12
    assert torch.allclose(centroids[center_idx], means[center_idx], atol=1e-5)


def test_radius_ball_centroid_isolated_points():
    """Isolated points fall back to self."""
    svc = TrigsTrainer(device="cpu")
    means = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    centroids = svc._radius_ball_centroid(means, radius=1.0)
    assert torch.allclose(centroids, means)


def test_omega_init_pure_rotation():
    """v = ω × r recovered perfectly by ω = (r × v)/‖r‖²."""
    N = 32
    torch.manual_seed(42)
    omega_gt = torch.tensor([0.0, 0.0, 1.0])
    anchor = torch.zeros(N, 3)
    means = torch.randn(N, 3) * 2.0
    means[:, 2] = 0.0

    r = means - anchor
    v = torch.cross(omega_gt.expand_as(r), r, dim=-1)

    r_sq = (r * r).sum(dim=-1, keepdim=True).clamp(min=1e-8)
    omega_est = torch.cross(r, v, dim=-1) / r_sq
    nu_residual = v - torch.cross(omega_est, r, dim=-1)

    assert torch.allclose(omega_est, omega_gt.expand_as(omega_est), atol=1e-5)
    assert nu_residual.abs().max() < 1e-5


# =========================================================================
# Motion-guided relocation (now a gsplat Strategy)
# =========================================================================

def test_motion_guided_relocation_swaps_inactive_slots():
    """Relocation overwrites low-opacity rows with (scaled) clones of actives."""
    torch.manual_seed(0)
    params = _make_params(N=24)
    cfg = MotionGuidedRelocationConfig(
        relocate_tau_alpha=0.5, relocate_mcmc_split=False,
    )

    with torch.no_grad():
        params["opacities"].data[:6] = -10.0
        params["opacities"].data[6:] = 3.0
        params["zeta_base_t"].data[6:] = torch.randn(18, 3)

    optimizers: dict = {}
    n_before = params["means"].shape[0]
    means_before = params["means"].data.clone()

    zero_cue = torch.zeros(n_before)
    n_relocated = _motion_guided_relocation(
        params, optimizers, cfg, zero_cue, zero_cue.clone(),
    )

    assert n_relocated == 6
    assert params["means"].shape[0] == n_before
    for i in range(6):
        match = any(
            torch.allclose(params["means"].data[i], means_before[j])
            for j in range(6, n_before)
        )
        assert match, f"relocated slot {i} did not match any active canonical mean"

    expected = torch.full((6, 3), -3.0 + math.log(cfg.relocate_scale_factor))
    assert torch.allclose(params["scales"].data[:6], expected, atol=1e-5)


@pytest.mark.parametrize("per_visit", [True, False])
def test_relocation_time_cue_per_visit_vs_official_total(per_visit: bool):
    """Same total |dL/dmu|: per-visit favours the briefly active row, the
    official total treats both donors alike."""
    torch.manual_seed(0)
    n = 402
    params = _make_params(N=n)
    with torch.no_grad():
        params["opacities"].data[:] = -10.0
        params["opacities"].data[-2:] = 3.0
        params["zeta_base_t"].data[:] = 0.0
        params["means"].data[-2] = torch.tensor([10.0, 0.0, 0.0])
        params["means"].data[-1] = torch.tensor([-10.0, 0.0, 0.0])
    grad_t = torch.zeros(n)
    grad_t[-2:] = 1.0
    visits = torch.ones(n)
    visits[-2] = 1.0
    visits[-1] = 100.0
    cfg = MotionGuidedRelocationConfig(
        relocate_tau_alpha=0.5, relocate_mcmc_split=False,
        relocate_source_duration_divisor=1.0, relocate_source_time_jitter_frac=0.0,
        relocate_time_cue_per_visit=per_visit,
    )
    _motion_guided_relocation(
        params, {}, cfg, torch.zeros(n), grad_t, visit_accum=visits,
    )
    from_short = int((params["means"].data[:-2, 0] > 0).sum())
    share = from_short / (n - 2)
    # Cues: opacity 1/2 each, time per-visit ~0.99/0.01 or official 1/2 each.
    if per_visit:
        assert share > 0.65
    else:
        assert 0.4 < share < 0.6


def _single_donor_params(N: int = 8) -> nn.ParameterDict:
    """Row N-1 is the only active row, so every clone comes from it."""
    torch.manual_seed(0)
    params = _make_params(N=N)
    with torch.no_grad():
        params["opacities"].data[:] = -10.0
        params["opacities"].data[-1] = math.log(0.6 / 0.4)
        params["times"].data[-1] = 0.5
        params["log_durations"].data[-1] = math.log(0.2)
        params["zeta_base_t"].data[-1] = torch.tensor([0.3, -0.2, 0.1])
        params["zeta_base_r"].data[-1] = torch.tensor([0.0, 0.8, 0.2])
        params["zeta_c0_t"].data[-1] = torch.tensor([0.2, 0.0, 0.0])
        params["zeta_c2_t"].data[-1] = torch.tensor([-0.2, 0.1, 0.0])
        params["zeta_c1_r"].data[-1] = torch.tensor([0.1, 0.0, -0.3])
        params["anchors"].data[-1] = params["means"].data[-1] + 0.1
    return params


def test_relocation_shrinks_donor_window_once_and_clones_copy_it():
    """Official: donor sigma / 1.5 once per event, mu_t within +/-0.25 sigma."""
    params = _single_donor_params()
    old_mu = float(params["times"].data[-1])
    old_log_sigma = float(params["log_durations"].data[-1])
    cfg = MotionGuidedRelocationConfig(
        relocate_tau_alpha=0.5, relocate_mcmc_split=False,
    )
    n = _motion_guided_relocation(
        params, {}, cfg, torch.zeros(8), torch.zeros(8),
    )
    assert n == 7

    log_sigma = params["log_durations"].data.reshape(-1)
    assert log_sigma[-1].item() == pytest.approx(old_log_sigma - math.log(1.5))
    assert torch.allclose(log_sigma[:-1], log_sigma[-1].expand(7))

    mu = params["times"].data.reshape(-1)
    assert abs(mu[-1].item() - old_mu) <= 0.25 * math.exp(old_log_sigma) + 1e-6
    assert torch.allclose(mu[:-1], mu[-1].expand(7))


def test_relocation_reanchor_keeps_donor_where_it_is_at_new_center():
    """The donor's canonical state moves to its old deformed state at mu'."""
    params = _single_donor_params()
    old_rows = {k: v.data.clone() for k, v in params.items()}
    tcfg = TrigsTrainingConfig(verbose=False)

    def deform_fn(rows, t):
        means_t, quats_t, _ = TrigsTrainer._deform(None, rows, t, tcfg)
        return means_t, quats_t

    cfg = MotionGuidedRelocationConfig(
        relocate_tau_alpha=0.5, relocate_mcmc_split=False,
        relocate_source_time_jitter_frac=0.25, relocate_time_max=10.0,
    )
    torch.manual_seed(3)
    _motion_guided_relocation(
        params, {}, cfg, torch.zeros(8), torch.zeros(8), deform_fn=deform_fn,
    )
    new_mu = params["times"].data[-1:].clone()
    assert not torch.allclose(new_mu, old_rows["times"][-1:])

    donor_old = {k: v[-1:] for k, v in old_rows.items()}
    expect_means, expect_quats = deform_fn(donor_old, new_mu)
    assert torch.allclose(params["means"].data[-1:], expect_means, atol=1e-5)
    q_new = params["quats"].data[-1:]
    assert (
        torch.allclose(q_new, expect_quats, atol=1e-5)
        or torch.allclose(q_new, -expect_quats, atol=1e-5)
    )

    # Constant residual (zero curvature) equal to the old curve at mu'.
    for suffix in ("_t", "_r"):
        c0 = params[f"zeta_c0{suffix}"].data[-1]
        assert torch.allclose(params[f"zeta_c1{suffix}"].data[-1], c0)
        assert torch.allclose(params[f"zeta_c2{suffix}"].data[-1], c0)

    # The re-anchored donor renders exactly where the old one was at mu'.
    donor_new = {k: v.data[-1:] for k, v in params.items()}
    at_center, _ = deform_fn(donor_new, new_mu)
    assert torch.allclose(at_center, expect_means, atol=1e-5)

    # Clones copy the re-anchored donor.
    assert torch.allclose(
        params["means"].data[:-1], params["means"].data[-1].expand(7, 3),
    )


def test_relocation_mcmc_split_matches_closed_form():
    """Donor + k clones share 1 - (1 - a)^(1/(k+1)) opacity (official)."""
    params = _single_donor_params()
    # tau_alpha is also the split's opacity floor, so keep it below the
    # expected per-copy opacity.
    cfg = MotionGuidedRelocationConfig(
        relocate_tau_alpha=0.01, relocate_source_duration_divisor=1.0,
        relocate_source_time_jitter_frac=0.0,
    )
    assert cfg.relocate_mcmc_split is True
    _motion_guided_relocation(params, {}, cfg, torch.zeros(8), torch.zeros(8))

    alpha = torch.sigmoid(params["opacities"].data).reshape(-1)
    expected = 1.0 - (1.0 - 0.6) ** (1.0 / 8.0)
    assert torch.allclose(alpha, torch.full_like(alpha, expected), atol=1e-5)
    scales = params["scales"].data
    assert torch.allclose(scales, scales[-1].expand_as(scales))
    assert (scales[-1] < -3.0).all()  # shrunk from the -3.0 init


def test_relocation_split_torch_port_matches_gsplat_kernel():
    """CPU torch port == gsplat's CUDA relocation kernel."""
    if not torch.cuda.is_available():
        pytest.skip("gsplat relocation kernel needs CUDA")
    from gsplat.relocation import compute_relocation

    from trigs.strategy.motion_guided_relocation import (
        _compute_relocation_torch,
        _relocation_binoms,
    )

    torch.manual_seed(0)
    op = torch.rand(256) * 0.9 + 0.05
    sc = torch.rand(256, 3) + 0.01
    ratios = torch.randint(1, 20, (256,), dtype=torch.int32)
    cpu_op, cpu_sc = _compute_relocation_torch(
        op, sc, ratios, _relocation_binoms("cpu"),
    )
    gpu_op, gpu_sc = compute_relocation(
        opacities=op.cuda(), scales=sc.cuda(), ratios=ratios.cuda().clone(),
        binoms=_relocation_binoms("cuda"),
    )
    assert torch.allclose(cpu_op, gpu_op.cpu(), atol=1e-5)
    assert torch.allclose(cpu_sc, gpu_sc.cpu(), rtol=1e-3, atol=1e-6)


def test_accumulate_2d_counts_only_projected_views():
    """Official 2-D cue: NDC-scaled grads, averaged over radii > 0 views."""
    from trigs.strategy.motion_guided_relocation import MotionGuidedRelocationStrategy

    strat = MotionGuidedRelocationStrategy(MotionGuidedRelocationConfig())
    state = strat.initialize_state(n_primitives=6, device="cpu")
    means2d = torch.zeros(1, 3, 2, requires_grad=True)
    means2d.grad = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
    info = {
        "means2d": means2d,
        "radii": torch.tensor([[[3, 3], [2, 2], [0, 0]]]),
        "active_idx": torch.tensor([1, 3, 5]),
        "width": 200, "height": 100,
    }
    strat.accumulate_2d(state, info, batch_size=2)
    assert state["grad_2d_accum"].tolist() == pytest.approx([0, 200, 0, 100, 0, 0])
    assert state["visible_accum"].tolist() == [0, 1, 0, 1, 0, 0]

    params = _make_params(N=6)
    strat.accumulate_time(state, params, torch.tensor([1, 3, 5]))
    assert state["visit_accum"].tolist() == [0, 1, 0, 1, 0, 1]


def test_relocation_stops_at_global_progress():
    from trigs.core.trainer import _relocation_past_stop

    config = TrigsTrainingConfig(verbose=False, n_steps=30_001)
    assert not _relocation_past_stop(config, 24_999)
    assert _relocation_past_stop(config, 25_000)
    always = dataclasses.replace(config, relocate_stop_frac=1.0)
    assert not _relocation_past_stop(always, 30_000)


def test_camera_batch_epoch_covers_every_pair_with_distinct_groups():
    import random

    from trigs.core.trainer import _build_camera_batch_epoch

    random.seed(0)
    frames, views = list(range(5)), list(range(7))
    epoch = _build_camera_batch_epoch(frames, views, batch=3)
    assert len(epoch) == 5 * 3  # ceil(7 / 3) rounds over 5 frames
    seen = set()
    for fi, group in epoch:
        assert len(group) == 3 and len(set(group)) == 3
        seen.update((fi, v) for v in group)
    assert seen == {(f, v) for f in frames for v in views}


def test_camera_batch_epoch_b1_reproduces_single_pair_schedule():
    """B=1 consumes the RNG exactly like the historical pair schedule."""
    import random

    from trigs.core.trainer import _build_camera_batch_epoch

    frames, views = list(range(6)), [0, 2, 3, 5]

    random.seed(7)
    frame_order = list(frames)
    random.shuffle(frame_order)
    by_frame = {}
    for fi in frame_order:
        v = list(views)
        random.shuffle(v)
        by_frame[fi] = v
    legacy = [(fi, by_frame[fi][r]) for r in range(len(views)) for fi in frame_order]

    random.seed(7)
    batched = _build_camera_batch_epoch(frames, views, batch=1)
    assert [(fi, g[0]) for fi, g in batched] == legacy


def test_official_camera_sampling_matches_loader_semantics():
    """Official TRiGS: a uniform random frame, B cameras with replacement."""
    import random

    from trigs.core.trainer import _draw_official_camera_batch

    random.seed(0)
    frames, views = list(range(5)), [0, 2, 3, 5]
    draws = [_draw_official_camera_batch(frames, views, batch=4) for _ in range(4000)]
    assert all(f in frames and len(g) == 4 and set(g) <= set(views) for f, g in draws)
    counts = [sum(f == k for f, _ in draws) for k in frames]
    assert min(counts) > 650 and max(counts) < 950                  # ~800 each
    assert any(len(set(g)) < 4 for _, g in draws)                  # duplicates allowed


@pytest.mark.parametrize("clip_edges", [True, False])
def test_relocation_speed_cue_edge_clipping_switch(clip_edges: bool):
    """Same speed and lifetime: clipping halves the cue of a row at the clip
    start; the official cue treats both donors alike."""
    torch.manual_seed(0)
    n = 402
    params = _make_params(N=n)
    with torch.no_grad():
        params["opacities"].data[:] = -10.0
        params["opacities"].data[-2:] = 3.0
        params["zeta_base_t"].data[:] = 0.0
        params["zeta_base_t"].data[-2:] = torch.tensor([1.0, 0.0, 0.0])
        params["log_durations"].data[-2:] = math.log(0.2)
        params["times"].data[-2] = 0.0                                # at the clip start
        params["times"].data[-1] = 2.0                                # interior
        params["means"].data[-2] = torch.tensor([10.0, 0.0, 0.0])
        params["means"].data[-1] = torch.tensor([-10.0, 0.0, 0.0])
    cfg = MotionGuidedRelocationConfig(
        relocate_tau_alpha=0.5, relocate_mcmc_split=False,
        relocate_source_duration_divisor=1.0, relocate_source_time_jitter_frac=0.0,
        relocate_speed_cue_clip_edges=clip_edges, relocate_time_max=4.0,
    )
    _motion_guided_relocation(params, {}, cfg, torch.zeros(n), torch.zeros(n))
    share_edge = int((params["means"].data[:-2, 0] > 0).sum()) / (n - 2)
    # Cues: opacity 1/2 each; speed 1/3 vs 2/3 when clipped, 1/2 each when not.
    if clip_edges:
        assert share_edge < 0.47
    else:
        assert 0.42 < share_edge < 0.58


@pytest.mark.parametrize("query_time", [10.0, torch.tensor(10.0)])
def test_render_rejects_empty_temporal_selection_before_gsplat(monkeypatch, query_time):
    """Out-of-support timestamps must not reach gsplat's N=0 division."""
    params = _make_params(N=4)
    with torch.no_grad():
        params["times"].fill_(0.25)
        params["log_durations"].fill_(math.log(0.01))

    def unexpected_rasterization(**kwargs):
        pytest.fail("An empty temporal selection reached native rasterization")

    monkeypatch.setattr("trigs.core.trainer.rasterization", unexpected_rasterization)
    trainer = TrigsTrainer(device="cpu")
    with pytest.raises(RuntimeError, match="No temporally active Gaussians") as exc:
        trainer._render(
            params, cameras=None, cam_idx=2, t=query_time, sh_degree=0,
            config=TrigsTrainingConfig(temporal_hierarchy_enabled=False), step=0,
        )
    message = str(exc.value)
    assert "camera=2, step=0, t=10.000000000 s" in message
    assert "primitive_center_time_range=[0.250000, 0.250000] s" in message
    assert "frame_timestamps.json/sync.json" in message


def _fake_render(self, params, cameras, cam_idx, t, sh_degree, config, step=None):
    """Differentiable stand-in for gsplat: real deform, toy image."""
    n = int(params["means"].shape[0])
    means_t, _, gamma = self._deform(params, t, config)
    means2d = means_t[None, :, :2]
    means2d.retain_grad()
    opa = torch.sigmoid(params["opacities"]).reshape(-1, 1)
    color = (opa * gamma * (params["sh0"] + params["scales"].exp())).mean(dim=0)
    color = color + 0.01 * means2d.mean() + 0.1 * float(cam_idx)
    image = color.reshape(1, 1, 1, 3).expand(1, 4, 4, 3)
    alphas = torch.full((1, 4, 4, 1), 0.5)
    info = {
        "means2d": means2d,
        "radii": torch.ones(1, n, 2, dtype=torch.int32),
        "width": 4, "height": 4,
        "active_idx": torch.arange(n),
        "n_active": n, "n_total": n,
    }
    return image, alphas, gamma.detach(), info


@pytest.mark.parametrize("batch,sampling", [(1, "epoch"), (3, "epoch"), (3, "official")])
def test_train_loop_runs_with_camera_batches(
    monkeypatch: pytest.MonkeyPatch, batch: int, sampling: str,
):
    """End-to-end CPU smoke test of the batched step and relocation."""
    calls: list[tuple[int, int | None]] = []

    def _counting_render(self, params, cameras, cam_idx, t, *args, **kwargs):
        calls.append((int(cam_idx), kwargs.get("step")))
        return _fake_render(self, params, cameras, cam_idx, t, *args, **kwargs)

    monkeypatch.setattr(TrigsTrainer, "_render", _counting_render)
    torch.manual_seed(0)
    params = _make_params(N=24)
    with torch.no_grad():
        params["opacities"].data[:4] = -10.0
        params["zeta_base_t"].data[4:] = torch.randn(20, 3) * 0.1
    for p in params.values():
        p.requires_grad_(True)

    n_frames, n_cams = 3, 4
    gt = {
        f: [torch.full((4, 4, 3), 60 * c, dtype=torch.uint8) for c in range(n_cams)]
        for f in range(n_frames)
    }
    masks = {
        f: [torch.full((4, 4), 255, dtype=torch.uint8) for _ in range(n_cams)]
        for f in range(n_frames)
    }
    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        verbose=False, n_steps=8, fps=2.0, batch_cameras=batch,
        relocate_start_iter=2, relocate_every=3, relocate_tau_alpha=0.01,
        knn_recompute_every=2, camera_sampling=sampling,
    )
    result = TrigsTrainer(device="cpu").train_from_params(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[None] * n_cams,
        n_frames=n_frames, gt_images_per_frame=gt, gt_alphas_per_frame=masks,
        config=config,
    )
    out = result.params
    assert int(out["means"].shape[0]) == 24
    for name in ("means", "scales", "times", "log_durations", "zeta_base_t"):
        assert torch.isfinite(out[name]).all(), name
    # The dead rows were recycled.
    assert (torch.sigmoid(out["opacities"][:4]) > 0.01).all()
    # Every step rendered B cameras (distinct unless sampled officially,
    # which draws with replacement).
    assert len(calls) == 8 * batch
    for step in range(8):
        cams = [c for c, s in calls if s == step]
        assert len(cams) == batch
        if sampling == "epoch":
            assert len(set(cams)) == batch


def test_train_loop_writes_viewer_snapshots(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
):
    """Unchunked runs snapshot at the end of the step body (so ``step`` updates
    are in the file) every N steps, plus once at the very end."""
    from trigs.core.viewer_snapshot import list_snapshots, load_snapshot

    monkeypatch.setattr(TrigsTrainer, "_render", _fake_render)
    torch.manual_seed(0)
    params = _make_params(N=24)
    with torch.no_grad():
        params["opacities"].data[:4] = -10.0
    for p in params.values():
        p.requires_grad_(True)
    gt = {
        f: [torch.full((4, 4, 3), 60, dtype=torch.uint8)] for f in range(3)
    }
    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        verbose=False, n_steps=10, fps=2.0,
        relocate_start_iter=2, relocate_every=3, relocate_tau_alpha=0.01,
        viewer_snapshot_every=4, viewer_snapshot_dir=str(tmp_path),
        viewer_snapshot_keep_last=10,
    )
    result = TrigsTrainer(device="cpu").train_from_params(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[None],
        n_frames=3, gt_images_per_frame=gt, gt_alphas_per_frame=None,
        config=config,
    )
    snaps = list_snapshots(tmp_path)
    assert [s for s, _ in snaps] == [4, 8, 10]

    first, last = (load_snapshot(snaps[0][1]), load_snapshot(snaps[-1][1]))
    assert first["extra"]["mode"] == "unchunked" and first["extra"]["n_rows"] == 24
    assert last["step"] == 10 and last["progress"] == 1.0
    assert last["n_frames"] == 3 and last["fps"] == 2.0
    # The final snapshot is the model the run returns; the first is earlier.
    for key, value in last["params"].items():
        assert torch.equal(value, result.params[key].detach().cpu()), key
    assert not torch.equal(first["params"]["means"], last["params"]["means"]) or \
        not torch.equal(first["params"]["opacities"], last["params"]["opacities"])


def _relocate_once(with_lineage: bool) -> nn.ParameterDict:
    torch.manual_seed(0)
    params = _make_params(N=24)
    with torch.no_grad():
        params["opacities"].data[:6] = -10.0
        params["opacities"].data[6:] = 3.0
        params["zeta_base_t"].data[6:] = torch.randn(18, 3)
    if with_lineage:
        TrigsTrainer._add_relocation_lineage(params, donor_count=True)
    _motion_guided_relocation(
        params, {}, MotionGuidedRelocationConfig(relocate_tau_alpha=0.5),
        torch.zeros(24), torch.zeros(24),
    )
    return params


def test_relocation_lineage_counts_clones_and_donors():
    """Clones get generation +1 and donor count 0; each clone adds one to its
    donor's count; nothing else changes."""
    params = _relocate_once(with_lineage=True)
    gen = params["relocation_generation"].data.reshape(-1)
    donors = params["relocation_donor_count"].data.reshape(-1)
    assert torch.equal(gen[:6], torch.ones(6)) and torch.equal(gen[6:], torch.zeros(18))
    assert torch.equal(donors[:6], torch.zeros(6))
    assert float(donors.sum()) == 6.0
    # Every clone is an exact copy of a counted donor's colour.
    sh0 = params["sh0"].data
    for i in range(6):
        assert any(torch.equal(sh0[i], sh0[j]) for j in range(6, 24) if donors[j] > 0), i

    plain = _relocate_once(with_lineage=False)
    for key in plain.keys():
        assert torch.equal(plain[key].data, params[key].data), key


def _lineage_run(monkeypatch, tmp_path, record: bool):
    monkeypatch.setattr(TrigsTrainer, "_render", _fake_render)
    torch.manual_seed(0)
    random.seed(0)
    params = _make_params(N=24)
    with torch.no_grad():
        params["opacities"].data[:4] = -10.0
    for p in params.values():
        p.requires_grad_(True)
    gt = {f: [torch.full((4, 4, 3), 60, dtype=torch.uint8)] for f in range(3)}
    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        verbose=False, n_steps=10, fps=2.0,
        relocate_start_iter=2, relocate_every=3, relocate_tau_alpha=0.01,
        checkpoint_dir=str(tmp_path), checkpoint_progress_interval=0.1,
        checkpoint_until_step=5, checkpoint_include_optimizers=True,
        opacity_diag_every=3, record_relocation_lineage=record,
    )
    return TrigsTrainer(device="cpu").train_from_params(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[None],
        n_frames=3, gt_images_per_frame=gt, gt_alphas_per_frame=None,
        config=config,
    )


def test_relocation_lineage_is_logging_only(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """The lineage flag leaves training bit-identical, the counters reach the
    full checkpoints, and checkpoint_until_step stops writing after its step."""
    plain = _lineage_run(monkeypatch, tmp_path / "plain", record=False)
    traced = _lineage_run(monkeypatch, tmp_path / "traced", record=True)
    assert "relocation_donor_count" not in plain.params
    for key, value in plain.params.items():
        assert torch.equal(value, traced.params[key]), key

    gen = traced.params["relocation_generation"].reshape(-1)
    donors = traced.params["relocation_donor_count"].reshape(-1)
    assert float(gen[:4].min()) >= 1.0  # the dead rows were refilled
    assert float(donors.sum()) >= 4.0

    ckpts = sorted((tmp_path / "traced").glob("trigs_step_*.pt"))
    assert [int(p.name.split("_")[2]) for p in ckpts] == [1, 2, 3, 4, 5]
    ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    for key in ("relocation_generation", "relocation_donor_count"):
        assert key in ck["params"], key
    assert "optimizer_states" in ck and "reloc_state" in ck
    assert "visit_accum" in ck["reloc_state"]


def test_train_loop_all_static_without_size_penalty(monkeypatch: pytest.MonkeyPatch):
    """Every regularizer can be a constant; the step must not try to
    backpropagate it separately."""
    monkeypatch.setattr(TrigsTrainer, "_render", _fake_render)
    torch.manual_seed(0)
    params = _make_params(N=8)
    params["is_static"] = nn.Parameter(torch.ones(8, 1), requires_grad=False)
    for name, p in params.items():
        if name != "is_static":
            p.requires_grad_(True)
    gt = {0: [torch.zeros(4, 4, 3, dtype=torch.uint8)]}
    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        verbose=False, n_steps=2, fps=1.0, lambda_scale_reg=0.0, relocate=False,
    )
    TrigsTrainer(device="cpu").train_from_params(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[None],
        n_frames=1, gt_images_per_frame=gt, gt_alphas_per_frame=None,
        config=config,
    )


def test_temporal_rows_changed_catches_momentum_moves():
    """With beta1 > 0 a zero-gradient row still moves; it must be re-filed."""
    params = _make_params(N=6)
    params["is_static"] = nn.Parameter(torch.zeros(6, 1), requires_grad=False)
    params["is_static"].data[5] = 1.0
    opt = torch.optim.Adam([params["times"]], lr=0.1, betas=(0.9, 0.999))
    params["times"].grad = torch.ones_like(params["times"])
    opt.step()

    before = TrigsTrainer._temporal_snapshot(params)
    params["times"].grad = torch.zeros_like(params["times"])
    opt.step()  # momentum only
    changed = TrigsTrainer._temporal_rows_changed(params, before)
    assert changed.tolist() == [0, 1, 2, 3, 4]


def _fake_render_culled(self, params, cameras, cam_idx, t, sh_degree, config, step=None):
    """Toy render through the real active-set query (hierarchy included)."""
    active_idx = self._active_idx_for_time(params, t, config, step=step)
    rows = self._slice_rowwise_params(params, active_idx)
    means_t, _, gamma = self._deform(rows, t, config)
    means2d = means_t[None, :, :2]
    means2d.retain_grad()
    opa = torch.sigmoid(rows["opacities"]).reshape(-1, 1)
    color = (opa * gamma * rows["sh0"]).sum(dim=0) + 0.01 * means2d.mean()
    image = color.reshape(1, 1, 1, 3).expand(1, 4, 4, 3)
    n = int(params["means"].shape[0])
    gamma_full = params["means"].new_zeros((n, 1))
    gamma_full[active_idx] = gamma.detach()
    info = {
        "means2d": means2d,
        "radii": torch.ones(1, int(active_idx.numel()), 2, dtype=torch.int32),
        "width": 4, "height": 4,
        "active_idx": active_idx,
        "n_active": int(active_idx.numel()), "n_total": n,
    }
    return image, torch.full((1, 4, 4, 1), 0.5), gamma_full, info


@pytest.mark.parametrize("tracking", [True, False])
def test_hierarchy_stays_exact_with_temporal_momentum(
    monkeypatch: pytest.MonkeyPatch, tracking: bool,
):
    """Culled training with beta1 = 0.9: the hierarchy must match the direct
    query at every step. Disabling the change tracking must break it, which
    shows the test can see staleness.
    """
    monkeypatch.setattr(TrigsTrainer, "_render", _fake_render_culled)
    mismatches: list[str] = []
    original = TrigsTrainer._validate_temporal_hierarchy

    def _recording(self, **kwargs):
        result = original(self, **kwargs)
        if result is not None:
            mismatches.append(result)
        return result

    monkeypatch.setattr(TrigsTrainer, "_validate_temporal_hierarchy", _recording)
    if not tracking:
        monkeypatch.setattr(
            TrigsTrainer, "_temporal_rows_changed",
            staticmethod(lambda params, before: torch.zeros(0, dtype=torch.long)),
        )

    # Dense enough that rows leave the active set while still carrying
    # momentum and later become active again: the gradient-only tracking
    # this replaced went stale on 8 of these 600 steps.
    torch.manual_seed(0)
    n_frames, fps, n_rows = 40, 10.0, 512
    params = _make_params(N=n_rows)
    with torch.no_grad():
        params["times"].copy_(torch.rand(n_rows, 1) * (n_frames - 1) / fps)
        params["log_durations"].copy_(torch.log(torch.rand(n_rows, 1) * 0.2 + 0.03))
        params["sh0"].copy_(torch.randn(n_rows, 3))
    for p in params.values():
        p.requires_grad_(True)
    gt = {
        f: [torch.full((4, 4, 3), 50 + 4 * f, dtype=torch.uint8)]
        for f in range(n_frames)
    }
    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        verbose=False, n_steps=600, fps=fps, relocate=False,
        temporal_hierarchy_enabled=True, temporal_hierarchy_validate_every=1,
        temporal_hierarchy_rebuild_every=100_000,
        temporal_hierarchy_overflow_rebuild_fraction=1.0,
        temporal_optimizer_beta1=0.9, lr_times=0.02, lr_durations=0.02,
        max_sigma_t=-1.0,
    )
    TrigsTrainer(device="cpu").train_from_params(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[None],
        n_frames=n_frames, gt_images_per_frame=gt, gt_alphas_per_frame=None,
        config=config,
    )
    if tracking:
        assert mismatches == []
    else:
        assert mismatches, "untracked momentum moves should leave the hierarchy stale"


def test_psnr_does_not_rescale_overbright_float_renders():
    """A float render with one pixel above 2.0 is still in [0, 1] units.

    Regression: the scale was guessed from the max, so one over-bright pixel
    divided the whole render by 255 and PSNR collapsed to ~7 dB.
    """
    from trigs.core.trainer import compute_psnr_metrics

    torch.manual_seed(0)
    gt_u8 = (torch.rand(16, 16, 3) * 255).to(torch.uint8)
    gt = gt_u8.float() / 255.0
    pred = gt.clone()
    pred[0, 0] = 2.5  # clamped to 1.0 by the metric, not a scale change
    good = compute_psnr_metrics(pred, gt)["psnr_full"]
    assert good > 25.0
    # uint8 GT is still rescaled.
    assert compute_psnr_metrics(pred, gt_u8)["psnr_full"] == pytest.approx(good)


def test_relocation_strategy_initialize_state():
    """Strategy.initialize_state returns grad accumulator tensors of correct shape."""
    from trigs.strategy.motion_guided_relocation import MotionGuidedRelocationStrategy

    strat = MotionGuidedRelocationStrategy(MotionGuidedRelocationConfig())
    state = strat.initialize_state(n_primitives=17, device="cpu")
    assert state["grad_2d_accum"].shape == (17,)
    assert state["grad_t_accum"].shape == (17,)
    assert (state["grad_2d_accum"] == 0).all()
    assert (state["grad_t_accum"] == 0).all()


def test_relocation_strategy_check_sanity_rejects_missing_keys():
    """check_sanity raises when required TriGS params are missing."""
    from trigs.strategy.motion_guided_relocation import MotionGuidedRelocationStrategy

    strat = MotionGuidedRelocationStrategy(MotionGuidedRelocationConfig())
    bad_params = nn.ParameterDict({"means": nn.Parameter(torch.zeros(4, 3))})
    with pytest.raises(ValueError, match="missing required params"):
        strat.check_sanity(bad_params, {})


def test_relocation_strategy_step_post_backward_cadence():
    """step_post_backward only fires when (step >= start) and (step % every == 0)."""
    from trigs.strategy.motion_guided_relocation import MotionGuidedRelocationStrategy

    cfg = MotionGuidedRelocationConfig(
        relocate_start_iter=10, relocate_every=5, relocate_tau_alpha=0.5,
    )
    strat = MotionGuidedRelocationStrategy(cfg)
    params = _make_params(N=12)
    state = strat.initialize_state(n_primitives=12, device="cpu")

    with torch.no_grad():
        params["opacities"].data[:4] = -10.0
        params["opacities"].data[4:] = 3.0

    # Below start iter — no-op.
    assert strat.step_post_backward(
        params=params, optimizers={}, state=state, step=5,
    ) == 0
    # At start iter but off-cadence — no-op.
    assert strat.step_post_backward(
        params=params, optimizers={}, state=state, step=11,
    ) == 0
    # On-cadence — fires.
    n = strat.step_post_backward(
        params=params, optimizers={}, state=state, step=10,
    )
    assert n == 4


def _fake_render_active(self, params, cameras, cam_idx, t, sh_degree, config, step=None):
    """Toy image from exactly the rows the trainer's temporal culling selects."""
    idx = self._active_idx_for_time(params, t, config, step=step)
    rows = self._slice_rowwise_params(params, idx)
    _, _, gamma = self._deform(rows, t, config)
    opa = torch.sigmoid(rows["opacities"]).reshape(-1, 1)
    color = (opa * gamma.reshape(-1, 1) * rows["sh0"]).sum(dim=0) + 0.1 * float(cam_idx)
    image = color.clamp(0, 1).reshape(1, 1, 1, 3).expand(1, 4, 4, 3)
    alphas = torch.full((1, 4, 4, 1), 0.5)
    return image, alphas, gamma.detach(), {}


def _eval_fixture(n_cams: int = 3, n_frames: int = 61, fps: float = 20.0):
    torch.manual_seed(0)
    params = _make_params(N=300)
    t_max = (n_frames - 1) / fps
    with torch.no_grad():
        params["times"].copy_(torch.rand(300, 1) * t_max)
        params["log_durations"].fill_(math.log(0.15))
        params["sh0"].copy_(torch.rand(300, 3) * 0.05)
    params["camera_time_offsets"] = nn.Parameter(
        torch.tensor([0.02, -0.03, 0.01]), requires_grad=False,
    )
    gt = {
        f: [torch.full((4, 4, 3), (7 * f + 40 * c) % 256, dtype=torch.uint8) for c in range(n_cams)]
        for f in range(n_frames)
    }
    masks = {
        f: [torch.cat([torch.full((2, 4), 255), torch.zeros(2, 4)]).to(torch.uint8) for _ in range(n_cams)]
        for f in range(n_frames)
    }
    return params, gt, masks, t_max


def test_chunked_eval_matches_full_model_eval(monkeypatch: pytest.MonkeyPatch):
    """Rendering each pair from its owning window's rows must give exactly the
    metrics of rendering the whole model under the same temporal culling."""
    from trigs.core.temporal_chunks import ChunkResidency
    from trigs.core.trainer import _make_psnr_eval_pairs

    monkeypatch.setattr(TrigsTrainer, "_render", _fake_render_active)
    params, gt, masks, t_max = _eval_fixture()
    config = TrigsTrainingConfig(
        fps=20.0, verbose=False, temporal_active_cull=True,
        temporal_active_threshold=0.05, psnr_eval_num_samples=12,
        holdout_views=(2,),
    )
    trainer = TrigsTrainer(device="cpu")
    common = dict(
        cameras=[None] * 3, sh_degree=0, config=config,
        gt_images_per_frame=gt, gt_alphas_per_frame=masks,
        gt_times_per_frame=None, step=0,
    )
    chunked = trainer._chunked_psnr_eval(
        residency=ChunkResidency.from_params(params),
        ownership_intervals=[(0.0, 1.0), (1.0, 2.2), (2.2, t_max + 0.5 / 20.0)],
        n_frames=61, cull_threshold=trainer._effective_temporal_cull_threshold(config),
        **common,
    )
    want_eval = trainer._evaluate_psnr_subset(
        params=params,
        eval_pairs=_make_psnr_eval_pairs(61, 3, 12, config.psnr_eval_seed, views=[0, 1]),
        **common,
    )
    want_holdout = trainer._evaluate_psnr_subset(
        params=params,
        eval_pairs=_make_psnr_eval_pairs(61, 3, 12, config.psnr_eval_seed, views=[2]),
        **common,
    )
    want_holdout = {k.replace("eval_", "holdout_"): v for k, v in want_holdout.items()}

    assert set(chunked) == {"eval", "holdout"}
    for got, want in ((chunked["eval"], want_eval), (chunked["holdout"], want_holdout)):
        assert set(got) == set(want)
        assert "psnr/eval_masked" in want or "psnr/holdout_masked" in want
        for key, value in want.items():
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-9), key


def test_chunked_training_logs_eval_after_each_round(monkeypatch: pytest.MonkeyPatch):
    from trigs.core.trainer import TrigsTrainingResult

    def _fake_train_from_params(self, *, params, config, **kwargs):
        assert config.psnr_eval_every == 0  # chunk calls never eval themselves
        return TrigsTrainingResult(params=params, temporal_params={}, motion_params={}, metrics={})

    monkeypatch.setattr(TrigsTrainer, "train_from_params", _fake_train_from_params)
    monkeypatch.setattr(TrigsTrainer, "_render", _fake_render_active)

    class _Writer:
        def __init__(self):
            self.scalars = []

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, float(value), int(step)))

    params, gt, masks, _ = _eval_fixture()
    writer = _Writer()
    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        n_steps=1000, fps=20.0, verbose=False, temporal_chunk_len_s=1.2,
        temporal_chunk_overlap_s=0.2, temporal_chunk_rounds=2,
        temporal_chunk_entry_diagnostics=False, relocate=False,
        psnr_eval_every=500, psnr_eval_num_samples=12,
    )
    result = TrigsTrainer(device="cpu").train_chunked(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[None] * 3,
        n_frames=61, gt_images_per_frame=gt, gt_alphas_per_frame=masks,
        config=config, writer=writer,
    )
    steps = sorted({s for tag, _, s in writer.scalars if tag == "psnr/eval_full"})
    assert steps == [500, 1000]  # end of each round (every 500 steps)
    assert [e["round"] for e in result.metrics["psnr_eval"]] == [0, 1]
    assert {tag for tag, _, _ in writer.scalars} >= {"psnr/eval_full", "psnr/eval_masked"}


def _capture_visits(monkeypatch: pytest.MonkeyPatch, **overrides):
    """Run train_chunked with training stubbed; return (config, window lo) per call."""
    from trigs.core.trainer import TrigsTrainingResult

    calls: list[tuple[TrigsTrainingConfig, float]] = []

    def _fake_train_from_params(self, *, params, config, **kwargs):
        calls.append((config, float(params["chunk_supervision_lo"])))
        return TrigsTrainingResult(params=params, temporal_params={}, motion_params={}, metrics={})

    monkeypatch.setattr(TrigsTrainer, "train_from_params", _fake_train_from_params)
    params = _make_params(N=256)
    with torch.no_grad():
        params["times"].copy_(torch.linspace(0.0, 299 / 60.0, 256).reshape(-1, 1))
    kwargs = dict(
        n_steps=30_000, fps=60.0, temporal_chunk_len_s=2.0, temporal_chunk_overlap_s=0.5,
        temporal_chunk_entry_diagnostics=False, verbose=False, psnr_eval_every=0,
        temporal_chunk_visit_steps=500,
    )
    kwargs.update(overrides)
    TrigsTrainer(device="cpu").train_chunked(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[], n_frames=300,
        gt_images_per_frame={}, gt_alphas_per_frame={}, config=TrigsTrainingConfig(frame_sampling_fast_visits=0.0, **kwargs),
    )
    return calls


def test_interleaved_visits_cycle_every_window_in_shuffled_order(monkeypatch: pytest.MonkeyPatch):
    from trigs.core.temporal_chunks import chunk_windows

    n_windows = len(chunk_windows(299 / 60.0, 2.0, 0.5))
    calls = _capture_visits(monkeypatch)
    rounds = round(30_000 / (500 * n_windows))
    assert len(calls) == rounds * n_windows
    assert sum(c.n_steps for c, _ in calls) == 30_000
    orders = []
    for r in range(rounds):
        block = calls[r * n_windows:(r + 1) * n_windows]
        orders.append([lo for _, lo in block])
        assert len(set(orders[-1])) == n_windows          # every window once per cycle
        assert len({c.lr_schedule_offset_steps for c, _ in block}) == 1  # one LR band per cycle
        assert all(abs(c.n_steps - 500) <= 150 for c, _ in block)
    assert len({tuple(o) for o in orders}) > 1             # order is reshuffled


def test_interleaved_visits_relocate_on_the_whole_run_schedule(monkeypatch: pytest.MonkeyPatch):
    calls = _capture_visits(monkeypatch, relocate_every=100, relocate_start_iter=500)
    offset = 0
    for config, _ in calls:
        assert config.relocate                          # including the final cycle
        assert config.relocate_settle_steps == 0
        assert config.relocate_start_iter == max(100, 500 - offset)
        assert config.lambda_temporal_chunk_outside_mass == 0.0
        offset += config.n_steps


def test_interleaved_short_visits_still_recycle_dead_rows(monkeypatch: pytest.MonkeyPatch):
    """End to end on CPU: relocation fires inside visits shorter than the old
    per-call start/settle windows."""
    monkeypatch.setattr(TrigsTrainer, "_render", _fake_render)
    torch.manual_seed(0)
    params = _make_params(N=24)
    with torch.no_grad():
        params["times"].copy_(torch.rand(24, 1) * 2.5)
        params["opacities"].data[:4] = -10.0
        params["zeta_base_t"].data[4:] = torch.randn(20, 3) * 0.1
    for p in params.values():
        p.requires_grad_(True)
    n_frames, n_cams = 6, 2
    gt = {f: [torch.full((4, 4, 3), 60 * c, dtype=torch.uint8) for c in range(n_cams)] for f in range(n_frames)}
    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        verbose=False, n_steps=48, fps=2.0, temporal_chunk_len_s=1.5,
        temporal_chunk_overlap_s=0.5, temporal_chunk_visit_steps=6,
        temporal_chunk_entry_diagnostics=False, psnr_eval_every=0,
        relocate_start_iter=2, relocate_every=3, relocate_tau_alpha=0.01,
        relocate_stop_frac=1.0, knn_recompute_every=2,
    )
    result = TrigsTrainer(device="cpu").train_chunked(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[None] * n_cams,
        n_frames=n_frames, gt_images_per_frame=gt, gt_alphas_per_frame=None,
        config=config, result_device="cpu",
    )
    out = result.params
    assert int(out["means"].shape[0]) == 24
    assert (torch.sigmoid(out["opacities"][:4]) > 0.01).all()
    for name in ("means", "times", "log_durations", "zeta_base_t"):
        assert torch.isfinite(out[name]).all(), name


def test_optimizer_state_export_load_round_trip():
    from trigs.core.optimizer_state import export_param_state, load_param_state

    torch.manual_seed(0)
    params = _make_params(N=10)
    for p in params.values():
        p.requires_grad_(True)
    trainer = TrigsTrainer(device="cpu")
    config = TrigsTrainingConfig(verbose=False)
    opts = trainer._build_optimizers(params, config, 1.0)
    for _ in range(3):
        for p in params.values():
            p.grad = torch.randn_like(p)
        for o in opts.values():
            o.step()
    saved = export_param_state(opts, params)
    assert {"means", "opacities", "zeta_c1_t", "log_durations"} <= set(saved)

    fresh = trainer._build_optimizers(params, config, 1.0)
    load_param_state(fresh, params, saved, step=1234)
    st = fresh["zeta_bezier_t"].state[params["zeta_c1_t"]]
    assert float(st["step"]) == 1234.0
    torch.testing.assert_close(st["exp_avg"], saved["zeta_c1_t"]["exp_avg"])
    torch.testing.assert_close(st["exp_avg_sq"], saved["zeta_c1_t"]["exp_avg_sq"])


def test_interleaved_visits_carry_adam_moments_with_rows(monkeypatch: pytest.MonkeyPatch):
    """Each visit starts from the moments last written for its trainable rows
    (zero for frozen context rows) at the global step it starts at."""
    from trigs.core.trainer import TrigsTrainingResult

    master_m: dict[int, float] = {}
    seen: list[str] = []
    visit = {"n": 0, "offset": 0}

    def _fake_train_from_params(self, *, params, config, initial_optimizer_state=None,
                                initial_optimizer_step=0, return_optimizer_state=False, **kwargs):
        assert return_optimizer_state
        assert initial_optimizer_step == visit["offset"]
        rows = params["row_id"].detach().reshape(-1).long()
        train = params["chunk_train_mask"].detach().reshape(-1) > 0.5
        if initial_optimizer_state:
            got = initial_optimizer_state["opacities"]["exp_avg"]
            for i, r in enumerate(rows.tolist()):
                want = master_m.get(r, 0.0) if bool(train[i]) else 0.0
                assert float(got[i]) == want, (visit["n"], r)
            seen.append("checked")
        visit["n"] += 1
        visit["offset"] += int(config.n_steps)
        value = torch.full((rows.numel(),), float(visit["n"]))
        for i, r in enumerate(rows.tolist()):
            if bool(train[i]):
                master_m[r] = float(visit["n"])
        result = TrigsTrainingResult(params=params, temporal_params={}, motion_params={}, metrics={})
        result.optimizer_state = {
            "opacities": {"exp_avg": value, "exp_avg_sq": value.clone()},
            "camera_time_offsets": {"exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
        }
        return result

    monkeypatch.setattr(TrigsTrainer, "train_from_params", _fake_train_from_params)
    params = _make_params(N=120)
    with torch.no_grad():
        params["times"].copy_(torch.linspace(0.0, 299 / 60.0, 120).reshape(-1, 1))
        params["log_durations"].fill_(math.log(0.4))  # rows span window boundaries
    params["row_id"] = nn.Parameter(torch.arange(120.0).reshape(-1, 1), requires_grad=False)
    config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
        n_steps=6_000, fps=60.0, temporal_chunk_len_s=2.0, temporal_chunk_overlap_s=0.5,
        temporal_chunk_entry_diagnostics=False, verbose=False, psnr_eval_every=0,
        temporal_chunk_visit_steps=500, temporal_active_threshold=0.01,
        learn_camera_time_offsets=True,
    )
    TrigsTrainer(device="cpu").train_chunked(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[None, None], n_frames=300,
        gt_images_per_frame={}, gt_alphas_per_frame={}, config=config,
    )
    assert len(seen) == visit["n"] - 1  # every visit after the first got carried state


def test_opacity_diag_is_off_by_default():
    assert TrigsTrainingConfig().opacity_diag_every == 0


def test_opacity_diag_summary_bands_and_signs():
    # rows: short (0-3 fr), mid (3-6 fr), long (>200 fr), long but untouched, still-class
    out = opacity_diag_summary(
        sigma_frames=torch.tensor([1.0, 4.0, 300.0, 300.0, 5.0]),
        is_static=torch.tensor([False, False, False, False, True]),
        opacity=torch.tensor([0.9, 0.8, 0.1, 0.2, 0.95]),
        photo=torch.tensor([-1.0, 2.0, -3.0, 0.0, -1.0]),
        reg=torch.tensor([0.1, 0.1, 4.0, 0.1, 0.0]),
        delta=torch.tensor([0.5, -0.5, -0.2, 0.0, 0.0]),
        photo_neg=torch.tensor([3.0, 0.0, 1.0, 0.0, 2.0]),
        photo_nz=torch.tensor([4.0, 2.0, 4.0, 0.0, 2.0]),
    )
    # the untouched long row is excluded; empty bands are omitted
    assert set(out) == {"sigma_0_3", "sigma_3_6", "sigma_200_inf", "still_class"}
    long = out["sigma_200_inf"]
    assert long["count"] == 1.0
    assert long["photo_wants_more"] == 1.0        # photo sum < 0
    assert long["net_drains"] == 1.0              # -3 + 4 > 0: the regulariser wins
    assert long["steps_say_less"] == 1.0          # only 1 of 4 touched steps said "more"
    assert long["fell"] == 1.0 and long["rose"] == 0.0
    assert long["reg_over_photo_p50"] == pytest.approx(4.0 / 3.0)
    assert out["sigma_0_3"]["steps_say_less"] == 0.0   # 3 of 4 steps said "more"
    assert out["sigma_3_6"]["photo_wants_more"] == 0.0
    assert out["still_class"]["opacity_p50"] == pytest.approx(0.95)


def test_opacity_diag_runs_in_train_loop_without_changing_training(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    """The diagnostics only observe: the same seed gives identical params with them on or off."""
    monkeypatch.setattr(TrigsTrainer, "_render", _fake_render)
    n_frames, n_cams = 3, 4
    gt = {
        f: [torch.full((4, 4, 3), 60 * c, dtype=torch.uint8) for c in range(n_cams)]
        for f in range(n_frames)
    }
    masks = {
        f: [torch.full((4, 4), 255, dtype=torch.uint8) for _ in range(n_cams)]
        for f in range(n_frames)
    }

    def run(diag_every: int):
        torch.manual_seed(0)
        random.seed(0)
        params = _make_params(N=24)
        for p in params.values():
            p.requires_grad_(True)
        config = TrigsTrainingConfig(frame_sampling_fast_visits=0.0, 
            verbose=False, n_steps=6, fps=2.0, batch_cameras=2,
            relocate_start_iter=2, relocate_every=3, relocate_tau_alpha=0.01,
            knn_recompute_every=2, opacity_diag_every=diag_every,
        )
        return TrigsTrainer(device="cpu").train_from_params(
            params=params, sh_degree=0, scene_extent=1.0, cameras=[None] * n_cams,
            n_frames=n_frames, gt_images_per_frame=gt, gt_alphas_per_frame=masks,
            config=config,
        ).params

    off = run(0)
    on = run(3)
    assert "[opacity_diag] step 3" in capsys.readouterr().out
    for name in ("opacities", "means", "scales", "log_durations"):
        assert torch.equal(off[name], on[name]), name


def test_short_lived_rows_per_frame_counts_windows_and_skips_long_and_static():
    from trigs.core.trainer import short_lived_rows_per_frame

    fps = 60.0
    frame_mu = torch.tensor([10.0, 10.0, 20.0, 15.0], dtype=torch.float64)
    sigma_frames = torch.tensor([1.0, 1.0, 1.0, 5.0], dtype=torch.float64)   # last row is long-lived
    params = {
        "times": (frame_mu / fps).float().reshape(-1, 1),
        "log_durations": (sigma_frames / fps).log().float().reshape(-1, 1),
        "is_static": torch.tensor([0.0, 1.0, 0.0, 0.0]).reshape(-1, 1),       # second row is static
    }
    frames = list(range(0, 25))
    counts = short_lived_rows_per_frame(params, frames, fps, 1.5)
    expected = [1.0 if 8 <= f <= 12 or 18 <= f <= 22 else 0.0 for f in frames]   # mu +/- 2 sigma
    assert counts == expected
    assert short_lived_rows_per_frame(params, frames, fps, 0.5) == [0.0] * len(frames)


def test_weighted_frame_draw_follows_weights_and_keeps_the_uniform_stream_when_off():
    from trigs.core.trainer import _draw_official_camera_batch as draw

    frames, views = list(range(10)), [0, 1, 2]
    random.seed(7)
    plain = [draw(frames, views, 2) for _ in range(50)]
    random.seed(7)
    no_weights = [draw(frames, views, 2, frame_weights=None, weighted_frac=0.5) for _ in range(50)]
    random.seed(7)
    frac_zero = [draw(frames, views, 2, frame_weights=[1.0] * 10, weighted_frac=0.0) for _ in range(50)]
    assert plain == no_weights == frac_zero

    weights = [0.0] * 9 + [1.0]
    random.seed(11)
    all_weighted = {draw(frames, views, 1, frame_weights=weights, weighted_frac=1.0)[0] for _ in range(200)}
    assert all_weighted == {9}
    random.seed(11)
    half = [draw(frames, views, 1, frame_weights=weights, weighted_frac=0.5)[0] for _ in range(4000)]
    share = sum(f == 9 for f in half) / len(half)
    assert abs(share - (0.5 + 0.5 / 10)) < 0.03                                  # weighted half + uniform tenth
    assert len(set(half)) == 10                                                  # every frame still visited


def test_weighted_frame_sampling_is_rejected_where_it_does_not_apply():
    svc = TrigsTrainer(device="cpu")
    cfg = TrigsTrainingConfig(verbose=False, frame_sampling_fast_frac=0.5, temporal_chunk_len_s=1.0)
    with pytest.raises(ValueError, match="frame_sampling_fast_frac"):
        svc.train_chunked(
            params=_make_params(N=8), sh_degree=0, scene_extent=1.0, cameras=None, n_frames=4,
            gt_images_per_frame={}, gt_alphas_per_frame={}, config=cfg,
        )


def test_fast_frame_topup_adds_steps_only_for_fast_frames_and_caps_them():
    from trigs.core.trainer import fast_frame_topup

    counts = [0, 0, 100, 400, 1000, 0]                 # rows; full top-up at 200
    visits, extra = fast_frame_topup(counts, n_steps=600, target_visits=400, full_rows=200)
    base = 600 / 6
    assert visits[0] == visits[1] == visits[5] == base                    # slow frames: normal training only
    assert visits[2] == pytest.approx(base + 0.5 * (400 - base))           # partial below full_rows
    assert visits[3] == visits[4] == 400                                   # capped at the target
    assert extra == round(sum(v - base for v in visits))
    assert fast_frame_topup(counts, n_steps=6000, target_visits=400, full_rows=200)[1] == 0   # already above target


def test_fast_frame_topup_with_multiplicity_tops_up_against_each_frames_own_normal_visits():
    from trigs.core.trainer import fast_frame_topup

    # Frames 1-2 sit in a window overlap (trained twice): normal visits 600 * m / sum(m).
    visits, extra = fast_frame_topup([0, 400, 0, 400], 600, 300.0, 200, multiplicity=[1, 2, 2, 1])
    assert visits == pytest.approx([100.0, 300.0, 200.0, 300.0])
    assert extra == 300
    assert fast_frame_topup([0, 400], 600, 300.0, 200) == fast_frame_topup([0, 400], 600, 300.0, 200, [1, 1])


def test_alive_rows_per_frame_bins_live_dynamic_rows_by_mu():
    from trigs.core.trainer import alive_rows_per_frame

    fps = 10.0
    params = {
        "times": torch.tensor([0.0, 0.1, 0.1, 0.5, 0.2]).reshape(-1, 1),
        "opacities": torch.tensor([5.0, 5.0, -20.0, 5.0, 5.0]).reshape(-1, 1),   # row 2 dead
        "is_static": torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]).reshape(-1, 1),    # row 4 static
    }
    assert alive_rows_per_frame(params, 6, fps, 0.005).tolist() == [1, 1, 0, 0, 0, 1]


def test_capacity_log_reports_shift_toward_fast_frames():
    class W:
        def __init__(self):
            self.rows = {}

        def add_scalar(self, k, v, s):
            self.rows[k] = v

    fps, n = 10.0, 4
    mask = torch.tensor([False, False, True, True])
    start = torch.tensor([10.0, 10.0, 10.0, 10.0])
    params = {"times": torch.tensor([0.0] * 5 + [0.1] * 10 + [0.2] * 10 + [0.3] * 15).reshape(-1, 1),
              "opacities": torch.full((40, 1), 5.0)}
    w, metrics = W(), {}
    TrigsTrainer._log_capacity(params, TrigsTrainingConfig(verbose=False, fps=fps), n, mask, start, w, 7, metrics)
    row = metrics["capacity"][0]
    assert row["fast_share_start"] == pytest.approx(0.5) and row["fast_share"] == pytest.approx(25 / 40)
    assert row["slow_ratio_min"] == pytest.approx(0.5) and w.rows["capacity/fast_frames_row_share"] == row["fast_share"]


def test_fast_visits_is_on_by_default():
    assert TrigsTrainingConfig().frame_sampling_fast_visits == 387.0
    assert TrigsTrainingConfig().frame_sampling_fast_frac == 0.0


def _run_chunked_capture_calls(monkeypatch, *, fast_visits, n_steps=30_000, verbose=False, visit_steps=500):
    """Chunked run (training stubbed) over 300 frames: long-lived rows everywhere plus short-lived
    rows at frames 100-110. Returns (per-call records, result metrics)."""
    from trigs.core.trainer import TrigsTrainingResult

    calls = []

    def _fake(self, *, params, config, frame_indices=None, frame_weights=None, progress_bar=None, **kw):
        calls.append({"steps": int(config.n_steps), "frames": list(frame_indices),
                      "weights": None if frame_weights is None else list(frame_weights),
                      "inner_fast_visits": config.frame_sampling_fast_visits, "verbose": config.verbose})
        if progress_bar is not None:
            progress_bar.update(int(config.n_steps))
        return TrigsTrainingResult(params=params, temporal_params={}, motion_params={}, metrics={})

    monkeypatch.setattr(TrigsTrainer, "train_from_params", _fake)
    n_frames, fps = 300, 60.0
    slow = _make_params(N=256)
    with torch.no_grad():
        slow["times"].copy_(torch.linspace(0.0, (n_frames - 1) / fps, 256).reshape(-1, 1))
        slow["log_durations"].fill_(math.log(10 / fps))
        slow["opacities"].fill_(5.0)
    fast = _make_params(N=330)
    with torch.no_grad():
        fast["times"].copy_((torch.arange(100, 111).repeat_interleave(30).double() / fps).float().reshape(-1, 1))
        fast["log_durations"].fill_(math.log(1 / fps))
        fast["opacities"].fill_(5.0)
    params = nn.ParameterDict({k: nn.Parameter(torch.cat([slow[k].detach(), fast[k].detach()]),
                                               requires_grad=slow[k].requires_grad) for k in slow})
    config = TrigsTrainingConfig(
        n_steps=n_steps, fps=fps, frame_sampling_fast_visits=fast_visits,
        temporal_chunk_len_s=1.5, temporal_chunk_overlap_s=0.5, temporal_chunk_visit_steps=visit_steps,
        temporal_chunk_entry_diagnostics=False, relocate=False, verbose=verbose,
        psnr_eval_every=0, viewer_snapshot_every=0,
    )
    result = TrigsTrainer(device="cpu").train_chunked(
        params=params, sh_degree=0, scene_extent=1.0, cameras=[], n_frames=n_frames,
        gt_images_per_frame={}, gt_alphas_per_frame={}, config=config, result_device="cpu",
    )
    return calls, result.metrics


def _expected_visits(calls, n_frames=300):
    v = [0.0] * n_frames
    for c in calls:
        w = c["weights"] or [1.0] * len(c["frames"])
        tot = sum(w)
        for f, wf in zip(c["frames"], w):
            v[f] += c["steps"] * wf / tot
    return v


def test_chunked_fast_topup_adds_steps_and_leaves_other_frames_their_visits(monkeypatch):
    from trigs.core.trainer import fast_frame_topup

    off, _ = _run_chunked_capture_calls(monkeypatch, fast_visits=0.0)
    on, metrics = _run_chunked_capture_calls(monkeypatch, fast_visits=400.0)
    assert all(c["inner_fast_visits"] == 0.0 for c in on)               # calls never top up again
    assert sum(c["steps"] for c in off) == 30_000
    # Fast frames: short-lived rows at 100-110 with +/-2 sigma windows -> 98..112 (30 rows/frame,
    # under the 200-row full top-up, so partial).
    counts = [30.0 * sum(1 for m in range(100, 111) if abs(f - m) <= 2) for f in range(300)]
    v_off, v_on = _expected_visits(off), _expected_visits(on)
    # Normal visits as the chunk plan gives them (overlap frames are trained by two windows).
    visits, extra = fast_frame_topup(counts, 30_000, 400.0, 200, multiplicity=[v / 100.0 for v in v_off])
    assert extra > 0 and sum(c["steps"] for c in on) == 30_000 + extra
    # Every frame gets exactly its planned visits (within integer rounding of per-call steps): other
    # frames keep what they get without the top-up, fast frames gain.
    for f in range(300):
        assert v_on[f] == pytest.approx(visits[f], rel=0.005), f
    for f in list(range(0, 90)) + list(range(120, 300)):
        assert v_on[f] == pytest.approx(v_off[f], rel=0.005), f
    for f in range(100, 111):
        assert v_on[f] > 1.5 * v_off[f], f
    assert any(c["weights"] is not None for c in on) and any(c["weights"] is None for c in on)
    assert metrics["capacity"] and "fast_share" in metrics["capacity"][-1]


def test_chunked_progress_is_one_bar_with_persistent_chunk_lines(monkeypatch, capsys):
    calls, _ = _run_chunked_capture_calls(monkeypatch, fast_visits=400.0, verbose=True, n_steps=6_000)
    out = capsys.readouterr()
    text = out.out + out.err
    assert text.count("Chunk cycle") == len(calls)                       # one line per call, kept in the log
    assert "of " in text and "global 0 -> " in text
    assert "Fast frames:" in text and "Capacity:" in text
    assert calls[0]["verbose"] and not any(c["verbose"] for c in calls[1:])


def test_window_leak_counts_rows_reaching_past_their_owner_window():
    from trigs.core.temporal_chunks import window_leak_stats

    fps, t_max = 10.0, 1.5
    windows = [(0.0, 1.0), (0.5, 1.5)]                     # ownership boundary at 0.75
    params = {
        "times": torch.tensor([0.2, 0.6, 1.2, 0.6]).reshape(-1, 1),
        "log_durations": torch.tensor([0.01, 0.3, 0.01, 0.3]).log().reshape(-1, 1),
        "opacities": torch.tensor([5.0, 5.0, 5.0, -20.0]).reshape(-1, 1),   # last row dead
    }
    s = window_leak_stats(params, windows, t_max, fps, cull_threshold=0.01)
    assert s["rows"] == 3
    assert s["leak_frac"] == pytest.approx(1 / 3)          # only the long row owned by window 0
    # support 0.6 +/- 0.91 s is clipped to the clip [0, 1.5]: 0.5 s = 5 frames past window 0's end
    assert s["overshoot_p50_frames"] == pytest.approx(5.0)
    assert 0.0 < s["mass_outside"] < 1 / 3


def test_interleaved_visits_drop_outside_mass_unless_asked(monkeypatch):
    seen = []

    from trigs.core.trainer import TrigsTrainingResult

    def _fake(self, *, params, config, **kw):
        seen.append(config.lambda_temporal_chunk_outside_mass)
        return TrigsTrainingResult(params=params, temporal_params={}, motion_params={}, metrics={})

    monkeypatch.setattr(TrigsTrainer, "train_from_params", _fake)
    for flag in (False, True):
        params = _make_params(N=64)
        with torch.no_grad():
            params["times"].copy_(torch.linspace(0.0, 2.0, 64).reshape(-1, 1))
        cfg = TrigsTrainingConfig(n_steps=2_000, fps=30.0, frame_sampling_fast_visits=0.0, relocate=False,
                                  temporal_chunk_len_s=1.0, temporal_chunk_visit_steps=500,
                                  temporal_chunk_visit_outside_mass=flag, verbose=False,
                                  psnr_eval_every=0, viewer_snapshot_every=0)
        TrigsTrainer(device="cpu").train_chunked(
            params=params, sh_degree=0, scene_extent=1.0, cameras=[], n_frames=61,
            gt_images_per_frame={}, gt_alphas_per_frame={}, config=cfg, result_device="cpu")
    half = len(seen) // 2
    assert set(seen[:half]) == {0.0} and set(seen[half:]) == {TrigsTrainingConfig().lambda_temporal_chunk_outside_mass}
