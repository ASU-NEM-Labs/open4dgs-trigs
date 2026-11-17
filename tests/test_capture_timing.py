"""The shared capture-time convention (trigs.capture_timing)."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from trigs import capture_timing as ct

K_RAW = np.array([[1302.7, 0.0, 960.0], [0.0, 1305.1, 540.0], [0.0, 0.0, 1.0]])
DIST = np.array([0.0149, -0.0151, 0.00099, -0.00244, 0.0])


def _delivered(factor: int):
    """What io/dataset.py does: scale K by 1/factor, undistort with alpha 0, crop to the ROI."""
    K = K_RAW.copy(); K[:2] /= factor
    W, H = 1920 // factor, 1080 // factor
    Ku, roi = cv2.getOptimalNewCameraMatrix(K, DIST[:4], (W, H), 0)
    Kd = Ku.copy(); Kd[0, 2] -= roi[0]; Kd[1, 2] -= roi[1]
    return K, Kd, (W, H), roi


@pytest.mark.parametrize("factor", [1, 2])
def test_delivered_to_raw_inverts_undistort_and_crop(factor):
    K, Kd, (W, H), roi = _delivered(factor)
    rng = np.random.default_rng(0)
    raw = np.c_[rng.uniform(5, W - 5, 400), rng.uniform(5, H - 5, 400)]
    und = cv2.undistortPoints(raw.reshape(-1, 1, 2), K, DIST[:4], P=Kd).reshape(-1, 2)
    u, v = ct.delivered_to_raw(und[:, 0], und[:, 1], Kd, K, DIST)
    assert np.max(np.abs(np.c_[u, v] - raw)) < 1e-3


def test_fraction_sign_and_origin():
    H = 1080.0
    rows = np.array([0.0, 540.0, 1080.0, -50.0, 2000.0])
    f = ct.readout_fraction(rows, H)
    assert np.allclose(f, [-0.5, 0.0, 0.5, -0.5, 0.5])
    t = ct.capture_offset_s(0.002, 0.015, f)
    # midpoint row carries exactly the constant; top/bottom rows are -/+ readout/2 around it
    assert np.allclose(t, [0.002 - 0.0075, 0.002, 0.002 + 0.0075, 0.002 - 0.0075, 0.002 + 0.0075])


def test_numpy_and_torch_agree():
    K, Kd, _, _ = _delivered(1)
    u = np.linspace(0, 1900, 50); v = np.linspace(0, 1070, 50)
    un, vn = ct.delivered_to_raw(u, v, Kd, K, DIST)
    ut, vt = ct.delivered_to_raw(torch.tensor(u), torch.tensor(v), torch.tensor(Kd), torch.tensor(K), DIST.tolist())
    assert np.allclose(un, ut.numpy()) and np.allclose(vn, vt.numpy())
    assert np.allclose(ct.readout_fraction(vn, 1080.0), ct.readout_fraction(vt, 1080.0).numpy())


def _write(tmp_path, **over):
    d = {"convention_id": "midpoint_v1", "readout_ms": 15.0, "readout_axis": "row", "fps": 60.0,
         "gauge_camera": "0021",
         "cameras": {"0003": {"c_ms": 0.2, "rs": True, "status": "measured"},
                     "0011": {"c_ms": -6.16, "rs": False, "status": "unresolved"}}}
    d.update(over)
    p = tmp_path / "timing.json"; p.write_text(json.dumps(d))
    return p


def test_timing_candidate_loads_and_looks_up(tmp_path):
    tc = ct.load_timing_candidate(_write(tmp_path))
    assert tc.readout_s == pytest.approx(0.015) and tc.axis == "row"
    assert tc.lookup("3").c_s == pytest.approx(0.0002) and tc.lookup("03").rs
    assert not tc.lookup("11").rs and tc.lookup("11").status == "unresolved"
    assert tc.lookup("99") is None


def test_timing_candidate_rejects_bad_files(tmp_path):
    with pytest.raises(ValueError):
        ct.load_timing_candidate(_write(tmp_path, convention_id="sync_json"))
    with pytest.raises(ValueError):
        ct.load_timing_candidate(_write(tmp_path, readout_axis="diagonal"))
    with pytest.raises(ValueError):
        ct.load_timing_candidate(_write(tmp_path, cameras={"0003": {"c_ms": None, "rs": False, "status": "unresolved"}}))
