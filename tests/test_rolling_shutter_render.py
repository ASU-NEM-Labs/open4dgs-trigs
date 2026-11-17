"""Rolling-shutter aware rendering in the trainer.

Checks the property that matters: a primitive's capture time must depend on the
image COLUMN it lands in, with the sign taken from its camera, and the whole
thing must be an exact no-op when the feature is off.
"""
from __future__ import annotations

import math

import pytest
import torch

from trigs.core.camera import CameraConfig
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig


def _cameras(n: int = 2, width: int = 1080, height: int = 1920, rs_signs=None):
    viewmats = torch.eye(4).unsqueeze(0).repeat(n, 1, 1)
    viewmats[:, 2, 3] = 4.0  # push the scene 4 m in front of the camera
    Ks = torch.zeros(n, 3, 3)
    Ks[:, 0, 0] = 1300.0
    Ks[:, 1, 1] = 1300.0
    Ks[:, 0, 2] = width / 2.0
    Ks[:, 1, 2] = height / 2.0
    Ks[:, 2, 2] = 1.0
    return CameraConfig.from_matrices(
        viewmats=viewmats, Ks=Ks,
        widths=[width] * n, heights=[height] * n,
        rs_signs=rs_signs,
    )


def _means_spanning_columns(width: int = 1080) -> torch.Tensor:
    """Points that project to the left edge, centre, and right edge."""
    z = 4.0
    f, cx = 1300.0, width / 2.0
    us = torch.tensor([0.0, cx, float(width)])
    xs = (us - cx) * z / f
    return torch.stack([xs, torch.zeros(3), torch.zeros(3)], dim=1)


def test_disabled_by_default_is_a_no_op():
    cfg = TrigsTrainingConfig()
    assert not cfg.rs_enabled(), "rolling shutter must default to OFF"
    trainer = TrigsTrainer.__new__(TrigsTrainer)
    cams = _cameras(rs_signs=[1.0, -1.0])
    out = trainer._rolling_shutter_times(
        _means_spanning_columns(), cams, 0, 1.0, cfg
    )
    assert out is None


def test_no_signs_means_no_correction():
    cfg = TrigsTrainingConfig(rs_readout_ms=13.0)
    assert cfg.rs_enabled()
    trainer = TrigsTrainer.__new__(TrigsTrainer)
    cams = _cameras(rs_signs=None)          # global-shutter rig
    assert trainer._rolling_shutter_times(
        _means_spanning_columns(), cams, 0, 1.0, cfg
    ) is None


def test_time_varies_across_columns_by_the_readout():
    """Left edge and right edge must differ by exactly rho, centre by zero."""
    rho_ms = 13.0
    cfg = TrigsTrainingConfig(rs_readout_ms=rho_ms)
    trainer = TrigsTrainer.__new__(TrigsTrainer)
    cams = _cameras(rs_signs=[1.0, -1.0])
    t0 = 2.5

    t_rs = trainer._rolling_shutter_times(
        _means_spanning_columns(), cams, 0, t0, cfg
    )
    assert t_rs.shape == (3, 1)
    left, centre, right = (float(v) for v in t_rs.reshape(-1))

    assert centre == pytest.approx(t0, abs=1e-6), "centre column is the reference"
    assert right - left == pytest.approx(rho_ms / 1000.0, abs=1e-6)
    assert left < t0 < right, "'l' camera: time must increase with +u"


def test_sign_flips_between_l_and_r_cameras():
    """The whole point: two cameras with opposite extract rotation shear the
    same moving subject in OPPOSITE directions. A per-camera scalar offset
    cannot express that, which is why this lives in the renderer."""
    cfg = TrigsTrainingConfig(rs_readout_ms=13.0)
    trainer = TrigsTrainer.__new__(TrigsTrainer)
    cams = _cameras(rs_signs=[1.0, -1.0])
    means = _means_spanning_columns()

    t_l = trainer._rolling_shutter_times(means, cams, 0, 2.5, cfg).reshape(-1)
    t_r = trainer._rolling_shutter_times(means, cams, 1, 2.5, cfg).reshape(-1)

    assert torch.allclose(t_l - 2.5, -(t_r - 2.5), atol=1e-6)
    assert (t_l[2] - t_l[0]) > 0 and (t_r[2] - t_r[0]) < 0


def test_zero_sign_camera_is_skipped():
    cfg = TrigsTrainingConfig(rs_readout_ms=13.0)
    trainer = TrigsTrainer.__new__(TrigsTrainer)
    cams = _cameras(n=2, rs_signs=[0.0, 1.0])
    assert trainer._rolling_shutter_times(
        _means_spanning_columns(), cams, 0, 1.0, cfg
    ) is None
    assert trainer._rolling_shutter_times(
        _means_spanning_columns(), cams, 1, 1.0, cfg
    ) is not None


def test_behind_camera_points_get_no_shift():
    """A primitive at or behind the near plane has no meaningful column; it must
    fall back to the nominal time rather than divide by ~0."""
    cfg = TrigsTrainingConfig(rs_readout_ms=13.0)
    trainer = TrigsTrainer.__new__(TrigsTrainer)
    cams = _cameras(rs_signs=[1.0, -1.0])
    # viewmats translate +4 in z, so world z <= -4 lands at or behind the
    # camera origin.
    means = torch.tensor([[0.0, 0.0, -10.0], [0.0, 0.0, -4.0], [1.0, 0.0, -5.0]])
    t_rs = trainer._rolling_shutter_times(means, cams, 0, 3.0, cfg)
    assert torch.allclose(t_rs.reshape(-1), torch.full((3,), 3.0), atol=1e-6)
    assert torch.isfinite(t_rs).all()


def test_shift_is_bounded_by_half_the_readout():
    """Primitives projecting far outside the frame are pinned to the nearest
    edge, so the shift never exceeds +/- rho/2 and can never reach outside a
    temporal support (sigma_t is hundreds of ms; rho/2 is ~6 ms)."""
    rho_ms = 13.0
    cfg = TrigsTrainingConfig(rs_readout_ms=rho_ms)
    trainer = TrigsTrainer.__new__(TrigsTrainer)
    cams = _cameras(rs_signs=[1.0, -1.0])
    far = torch.tensor([[-500.0, 0.0, 4.0], [500.0, 0.0, 4.0], [0.0, 0.0, 4.0]])
    t_rs = trainer._rolling_shutter_times(far, cams, 0, 1.0, cfg).reshape(-1)
    # float32 subtraction near 1.0 loses ~1e-7, so compare at that scale.
    assert torch.all((t_rs - 1.0).abs() <= rho_ms / 1000.0 / 2.0 + 1e-6)
    assert float((t_rs[1] - t_rs[0]).abs()) == pytest.approx(
        rho_ms / 1000.0, abs=1e-6
    ), "opposite off-frame edges must still span the full readout"


def test_learned_time_offset_keeps_its_gradient():
    """The correction is added on top of a possibly-learned camera time offset;
    that offset must still receive gradient through the RS path."""
    cfg = TrigsTrainingConfig(rs_readout_ms=13.0)
    trainer = TrigsTrainer.__new__(TrigsTrainer)
    cams = _cameras(rs_signs=[1.0, -1.0])
    offset = torch.zeros((), requires_grad=True)
    t = 2.0 + offset
    t_rs = trainer._rolling_shutter_times(_means_spanning_columns(), cams, 0, t, cfg)
    t_rs.sum().backward()
    assert offset.grad is not None
    assert float(offset.grad) == pytest.approx(3.0, abs=1e-6)


def test_deform_accepts_per_row_times():
    """_deform must broadcast an (N, 1) time, since that is what the RS path
    hands it. A scalar and a constant per-row tensor must agree exactly."""
    from trigs.core.trainer import TrigsTrainer as T
    n = 64
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.randn(n, 3) * 0.1),
        "quats": torch.nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1)
        ),
        "anchors": torch.nn.Parameter(torch.randn(n, 3) * 0.1),
        "times": torch.nn.Parameter(torch.rand(n, 1)),
        "log_durations": torch.nn.Parameter(torch.full((n, 1), math.log(0.1))),
        "zeta_base_t": torch.nn.Parameter(torch.randn(n, 3) * 0.05),
        "zeta_base_r": torch.nn.Parameter(torch.randn(n, 3) * 0.05),
        **{
            f"zeta_c{i}_{k}": torch.nn.Parameter(torch.randn(n, 3) * 0.01)
            for i in range(3) for k in ("t", "r")
        },
    })
    cfg = TrigsTrainingConfig()
    trainer = T.__new__(T)
    m_scalar, q_scalar, g_scalar = trainer._deform(params, 0.5, cfg)
    per_row = torch.full((n, 1), 0.5)
    m_row, q_row, g_row = trainer._deform(params, per_row, cfg)
    assert torch.allclose(m_scalar, m_row, atol=1e-6)
    assert torch.allclose(q_scalar, q_row, atol=1e-6)
    assert torch.allclose(g_scalar, g_row, atol=1e-6)
