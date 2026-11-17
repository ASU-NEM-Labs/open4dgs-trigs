"""Capture-time convention shared by prep, the sync tools, the trainer and eval.

    t_capture(camera k, sensor coordinate s) = t_nominal + c_k + readout_s * (s / S - 0.5)

  t_nominal  frame time on the common clock (frame index / fps)
  c_k        the camera's MIDPOINT constant: capture time of the middle sensor line (s = S/2) minus nominal,
             PLUS sign (actual = nominal + c), seconds
  readout_s  time from the first to the last sensor line; > 0 means later lines are captured later
  s, S       position along the readout axis in RAW sensor pixels and its extent:
               axis "row"    -> raw image row v, S = raw height (unrotated landscape capture)
               axis "column" -> delivered image column u, S = delivered width, with a per-camera sign
                                (legacy path for extracts rotated +-90 deg; unchanged behaviour)
The trainer works on undistorted, ROI-cropped, factor-resized images; `delivered_to_raw` maps such a pixel back
to the raw (distorted) image grid with the camera's OPENCV model so the readout coordinate is the true sensor
line. All functions accept numpy arrays or torch tensors (no other dependencies), so prep and the trainer use
the same arithmetic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

CONVENTION = ("t = t_nominal + c_k + readout_s*(s/S - 0.5); c_k = midpoint constant (PLUS); "
              "readout_s > 0: later sensor lines captured later")


def _is_torch(x) -> bool:
    return type(x).__module__.startswith("torch")


def _clip(x, lo, hi):
    return x.clamp(lo, hi) if _is_torch(x) else x.clip(lo, hi)


def distort_opencv(xn, yn, dist):
    """OPENCV distortion of normalised coordinates. dist = (k1, k2, p1, p2[, k3]) as in cv2."""
    k1, k2, p1, p2 = (float(d) for d in list(dist)[:4])
    k3 = float(list(dist)[4]) if len(list(dist)) > 4 else 0.0
    r2 = xn * xn + yn * yn
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return xd, yd


def delivered_to_raw(u, v, K_delivered, K_raw, dist):
    """Pixel in the trainer's delivered image (undistorted with K_delivered, which already includes the ROI
    crop shift and the 1/factor scale) -> pixel in the raw distorted image at the same scale (K_raw = original
    K scaled by 1/factor). Exact inverse of the dataset's undistort+crop for an OPENCV camera."""
    fx, fy, cx, cy = (float(K_delivered[0][0]), float(K_delivered[1][1]), float(K_delivered[0][2]), float(K_delivered[1][2]))
    xn = (u - cx) / fx
    yn = (v - cy) / fy
    xd, yd = distort_opencv(xn, yn, dist)
    return float(K_raw[0][0]) * xd + float(K_raw[0][2]), float(K_raw[1][1]) * yd + float(K_raw[1][2])


def readout_fraction(coord, extent):
    """(s / S - 0.5) clamped to [-0.5, 0.5]: outside the sensor there are no lines, so off-image points take the
    nearest edge's time (same rule as the existing column path)."""
    return _clip(coord / max(float(extent), 1.0) - 0.5, -0.5, 0.5)


def capture_offset_s(c_s, readout_s, fraction):
    """Capture time minus nominal, seconds."""
    return c_s + readout_s * fraction


@dataclass
class CameraTiming:
    c_s: float                 # midpoint constant (or the fallback sync value when rs is False)
    rs: bool                   # apply the readout term for this camera
    status: str                # measured | weak | converted | unresolved
    note: str = ""


@dataclass
class TimingCandidate:
    readout_ms: float
    axis: str
    fps: float
    gauge_camera: str
    cameras: Dict[str, CameraTiming] = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    path: Optional[str] = None

    @property
    def readout_s(self) -> float:
        return float(self.readout_ms) / 1000.0

    def lookup(self, folder: str) -> Optional[CameraTiming]:
        """Match a camera folder ("3", "03", "0003", "Camera 3") to an entry, like sync.json keys."""
        cands = [str(folder)]
        if str(folder).isdigit():
            n = int(folder)
            cands += [f"{n:04d}", f"{n:02d}", str(n), f"Camera {n}"]
        for k in cands:
            if k in self.cameras:
                return self.cameras[k]
        return None

    def as_sync_offsets(self) -> Dict[str, float]:
        """sync.json-shaped dict {key: seconds} of the per-camera constants (midpoint c_k, or fallback)."""
        return {k: float(v.c_s) for k, v in self.cameras.items()}


def load_timing_candidate(path) -> TimingCandidate:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("convention_id") != "midpoint_v1":
        raise ValueError(f"{path}: convention_id must be 'midpoint_v1' (got {raw.get('convention_id')!r})")
    axis = str(raw["readout_axis"]).lower()
    if axis not in ("row", "column"):
        raise ValueError(f"{path}: readout_axis must be 'row' or 'column'")
    cams = {}
    for k, v in raw["cameras"].items():
        c_ms = v.get("c_ms")
        if c_ms is None:
            raise ValueError(f"{path}: camera {k} has no c_ms (unresolved cameras must carry their fallback value)")
        cams[str(k)] = CameraTiming(c_s=float(c_ms) / 1000.0, rs=bool(v["rs"]), status=str(v["status"]), note=str(v.get("note", "")))
    return TimingCandidate(readout_ms=float(raw["readout_ms"]), axis=axis, fps=float(raw["fps"]),
                           gauge_camera=str(raw.get("gauge_camera", "")), cameras=cams, source=raw.get("source", {}),
                           path=str(path))
