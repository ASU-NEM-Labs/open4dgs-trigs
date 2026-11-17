"""SelfCap (4DV) dataset loader.

SelfCap — a.k.a. the LongVolcap dataset (Xu et al., SIGGRAPH Asia 2024) —
ships multi-view videos plus EasyVolcap-style calibration
(``optimized/extri.yml`` + ``optimized/intri.yml``), an optional per-camera
timing correction (``optimized/sync.json``), and per-frame sparse point
clouds (``pcds/FFFFFF.ply``).

Dataset page: https://zju3dv.github.io/longvolcap/

This module expects the scene to have been preprocessed by
``scripts/preprocess_selfcap.py``, which extracts a frame range from the
mp4s into jpgs (0-indexed) and writes a ``preprocess_meta.json`` recording
the absolute-frame origin of the extraction.

The parser exposes the same attribute surface as
:class:`trigs.io.dataset.FreeTimeParser` (``camtoworlds``, ``Ks_dict``,
``params_dict``, ``imsize_dict``, ``mask_dict``, ``points``, ``points_rgb``,
``points_err``, ``camera_names``, ``campaths``, …) so downstream code that
consumes a FreeTimeParser also works here. Distortion coefficients are loaded
into ``params_dict`` as OpenCV vectors (typically
``[k1, k2, p1, p2, k3]`` for SelfCap ``intri.yml``), so the inherited
``_setup_undistortion`` logic runs unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .dataset import FreeTimeDataset, _detect_image_format
from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EasyVolcap YAML calibration


def _read_opencv_yaml(path: Path) -> cv2.FileStorage:
    """Open an EasyVolcap YAML file.

    Files in the SelfCap release lack the ``%YAML:1.0`` directive line that
    ``cv2.FileStorage`` requires in FILE mode, so when direct open fails we
    prepend the directive in memory and reopen via ``FILE_STORAGE_MEMORY``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot open YAML: {path}")
    try:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            return fs
    except (cv2.error, SystemError):
        pass

    text = path.read_text()
    if not text.startswith("%YAML"):
        text = "%YAML:1.0\n---\n" + text
    fs = cv2.FileStorage(
        text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY,
    )
    if not fs.isOpened():
        raise ValueError(f"Failed to parse YAML (even with directive): {path}")
    return fs


def read_easyvolcap_cameras(
    optimized_dir: Path,
) -> Tuple[List[str], np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray],
           Dict[int, Tuple[int, int]]]:
    """Parse EasyVolcap extri.yml + intri.yml.

    Returns:
        camera_names: list of N camera-name strings (e.g. ``"0000"``)
        camtoworlds:  [N, 4, 4] float32 camera-to-world matrices
        Ks_dict:      {camera_id: 3x3 K}
        params_dict:  {camera_id: (N,) float32 OpenCV distortion vector}
        imsize_dict:  {camera_id: (W, H)}

    ``camera_id`` is the 1-based positional index (matches the existing
    FreeTimeParser convention that image tensors report ``camera_id - 1``).
    """
    extri = _read_opencv_yaml(optimized_dir / "extri.yml")
    intri = _read_opencv_yaml(optimized_dir / "intri.yml")

    names_node = extri.getNode("names")
    if names_node.empty():
        raise ValueError(f"'names' missing in {optimized_dir / 'extri.yml'}")
    camera_names = [names_node.at(i).string() for i in range(names_node.size())]

    N = len(camera_names)
    camtoworlds = np.zeros((N, 4, 4), dtype=np.float32)
    Ks_dict: Dict[int, np.ndarray] = {}
    params_dict: Dict[int, np.ndarray] = {}
    imsize_dict: Dict[int, Tuple[int, int]] = {}

    bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)

    for i, name in enumerate(camera_names):
        rot_node = extri.getNode(f"Rot_{name}")
        t_node = extri.getNode(f"T_{name}")
        if rot_node.empty() or t_node.empty():
            raise ValueError(f"Extrinsic missing for camera {name}")
        R = rot_node.mat().astype(np.float32)      # (3, 3) world->cam
        t = t_node.mat().astype(np.float32).reshape(3, 1)
        w2c = np.concatenate([np.concatenate([R, t], axis=1), bottom], axis=0)
        camtoworlds[i] = np.linalg.inv(w2c)

        K_node = intri.getNode(f"K_{name}")
        D_node = intri.getNode(f"D_{name}")
        H_node = intri.getNode(f"H_{name}")
        W_node = intri.getNode(f"W_{name}")
        if K_node.empty() or H_node.empty() or W_node.empty():
            raise ValueError(f"Intrinsic missing for camera {name}")

        K = K_node.mat().astype(np.float32).copy()
        H = int(H_node.real())
        W = int(W_node.real())

        # EasyVolcap stores OpenCV distortion as [k1, k2, p1, p2, k3].
        # Keep the full vector so OpenCV undistortion can use k3 when present.
        if D_node.empty():
            D = np.empty(0, dtype=np.float32)
        else:
            D = D_node.mat().astype(np.float32).reshape(-1)

        cam_id = i + 1  # 1-based to match FreeTimeParser
        Ks_dict[cam_id] = K
        params_dict[cam_id] = D
        imsize_dict[cam_id] = (W, H)

    return camera_names, camtoworlds, Ks_dict, params_dict, imsize_dict


# ---------------------------------------------------------------------------
# Point clouds (sparse xyz + rgb PLYs from RealityCapture / COLMAP)


def _load_xyzrgb_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal reader for SelfCap pcds: xyz float + rgb uchar, no normals.

    Returns ``(positions[N, 3] float32, colors[N, 3] uint8)``.
    """
    with open(path, "rb") as f:
        # Parse ASCII header
        header_lines: List[bytes] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {path}")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break

        format_line = next(
            (ln for ln in header_lines if ln.startswith(b"format")), b"",
        ).decode().strip()
        if "binary_little_endian" not in format_line:
            # Fall back to a slower but more tolerant path via cv2? Easier:
            # demand binary LE, which is what SelfCap ships.
            raise NotImplementedError(
                f"Only binary_little_endian PLY supported, got: {format_line}"
            )

        n = 0
        dtype_fields: List[Tuple[str, str]] = []
        for ln in header_lines:
            s = ln.decode().strip()
            if s.startswith("element vertex"):
                n = int(s.split()[-1])
            elif s.startswith("property"):
                parts = s.split()
                ty, name = parts[1], parts[-1]
                np_ty = {
                    "float": "<f4", "float32": "<f4",
                    "double": "<f8", "float64": "<f8",
                    "uchar": "u1", "uint8": "u1",
                    "char": "i1", "int8": "i1",
                    "ushort": "<u2", "uint16": "<u2",
                    "short": "<i2", "int16": "<i2",
                    "uint": "<u4", "uint32": "<u4",
                    "int": "<i4", "int32": "<i4",
                }[ty]
                dtype_fields.append((name, np_ty))

        dt = np.dtype(dtype_fields)
        raw = np.frombuffer(f.read(n * dt.itemsize), dtype=dt, count=n)

    xyz = np.stack(
        [raw["x"].astype(np.float32),
         raw["y"].astype(np.float32),
         raw["z"].astype(np.float32)], axis=1,
    )
    rgb = np.stack(
        [raw["red"].astype(np.uint8),
         raw["green"].astype(np.uint8),
         raw["blue"].astype(np.uint8)], axis=1,
    ) if "red" in raw.dtype.names else np.full((n, 3), 200, dtype=np.uint8)
    return xyz, rgb


def load_pcd_frame(
    pcd_dir: Path, absolute_frame: int, frame_digits: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``pcds/{absolute_frame:0{frame_digits}d}.ply`` as (xyz, rgb)."""
    p = pcd_dir / f"{absolute_frame:0{frame_digits}d}.ply"
    if not p.exists():
        raise FileNotFoundError(f"Missing PLY: {p}")
    return _load_xyzrgb_ply(p)


def _probe_camera_image_size(
    camera_dir: Path,
    extension: str,
    frame_digits: int,
    frame_start_offset: int,
) -> Optional[Tuple[int, int]]:
    """Return ``(W, H)`` for one readable image in ``camera_dir``."""
    expected = camera_dir / (
        f"{frame_start_offset:0{frame_digits}d}{extension}"
    )
    if expected.exists():
        candidates = [expected]
    else:
        candidates = sorted(camera_dir.glob(f"*{extension}"))

    for path in candidates:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            h, w = image.shape[:2]
            return int(w), int(h)
    return None


def _probe_camera_image(
    camera_dir: Path,
    extension: str,
    frame_digits: int,
    frame_start_offset: int,
) -> Optional[np.ndarray]:
    """Return one readable image array from ``camera_dir``."""
    expected = camera_dir / (
        f"{frame_start_offset:0{frame_digits}d}{extension}"
    )
    if expected.exists():
        candidates = [expected]
    else:
        candidates = sorted(camera_dir.glob(f"*{extension}"))

    for path in candidates:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            return image
    return None


# ---------------------------------------------------------------------------
# Parser


class SelfCapParser:
    """FreeTimeParser-compatible parser for preprocessed SelfCap scenes.

    The parser expects the directory layout produced by
    ``scripts/preprocess_selfcap.py``:

    .. code::

        <data_dir>/
        ├── images/0000/000000.jpg … NNNNNN.jpg   (0-indexed, ffmpeg-extracted)
        ├── optimized/extri.yml, intri.yml, sync.json
        ├── pcds/XXXXXX.ply                        (absolute-frame indexed)
        └── preprocess_meta.json

    Time semantics: the dataset emits ``time`` normalized to [0, 1] over the
    preprocessed frame window *and* ``time_sec`` in seconds with per-camera
    sync correction applied (``time_sec = frame_offset / fps − sync[cam]``).
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        factor: int = 1,
        normalize: bool = False,
        pcd_source: str = "pcds",
        apply_sync: bool = True,
        strict_validation: bool = True,
        max_size_mismatch_px: int = 1,
        min_undistort_valid_ratio: float = 0.5,
    ):
        """
        Args:
            data_dir: Preprocessed SelfCap directory (must contain
                ``preprocess_meta.json`` + ``optimized/``).
            factor: Additional runtime image downsampling factor applied by the
                dataset loader. The preprocessing factor from
                ``preprocess_meta.json["factor"]`` is combined with this,
                unless preprocessing already baked pinhole calibration
                (``images_are_undistorted=true``), in which case only runtime
                factor is applied.
            normalize: Align world coordinates so cameras sit around origin
                with principal axes aligned.
            pcd_source: Subdirectory under ``data_dir`` to pull the
                initial-frame point cloud from (``"pcds"`` for sparse,
                ``"dense_pcds"`` for RealityCapture-dense).
            apply_sync: Apply ``sync.json`` per-camera offsets when the
                dataset computes ``time_sec``.
            strict_validation: If True, fail fast on inconsistent image-size /
                calibration / undistortion state instead of silently continuing.
            max_size_mismatch_px: Allowed absolute per-axis mismatch (in pixels)
                between expected post-factor size and decoded image size after
                runtime resize.
            min_undistort_valid_ratio: Minimum in-bounds fraction of undistort
                map coordinates for distorted cameras.
        """
        data_dir = Path(data_dir)
        self.data_dir = data_dir
        runtime_factor = int(factor)
        if runtime_factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        self.factor = runtime_factor
        self.normalize = normalize
        self.apply_sync = apply_sync
        self.strict_validation = bool(strict_validation)
        self.max_size_mismatch_px = max(int(max_size_mismatch_px), 0)
        self.min_undistort_valid_ratio = float(min_undistort_valid_ratio)
        if not (0.0 < self.min_undistort_valid_ratio <= 1.0):
            raise ValueError(
                "min_undistort_valid_ratio must be in (0, 1], "
                f"got {self.min_undistort_valid_ratio}",
            )

        meta_path = data_dir / "preprocess_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing preprocess_meta.json in {data_dir}. "
                f"Run scripts/preprocess_selfcap.py first."
            )
        meta = json.loads(meta_path.read_text())
        self.meta = meta
        self.start_frame: int = int(meta["start_frame"])
        self.end_frame: int = int(meta["end_frame"])
        self.total_frames: int = int(meta["n_frames"])
        self.fps: float = float(meta["fps"])
        self.absolute_frame_origin: int = int(meta.get("absolute_frame_origin", self.start_frame))
        self.images_are_undistorted = bool(meta.get("images_are_undistorted", False))
        preprocess_factor = int(meta.get("factor", 1))
        if preprocess_factor < 1:
            raise ValueError(
                "preprocess_meta.json factor must be >= 1, "
                f"got {preprocess_factor}",
            )
        self.preprocess_factor = preprocess_factor
        # In pinhole-rectified preprocess mode, intrinsics are already baked
        # to extracted image resolution (and crop), so only runtime factor
        # remains to be applied here.
        self.total_factor = self.factor if self.images_are_undistorted else (self.preprocess_factor * self.factor)

        # --- calibration -------------------------------------------------
        optimized_dir = data_dir / "optimized"
        camera_names, camtoworlds, Ks_dict, params_dict, imsize_dict = \
            read_easyvolcap_cameras(optimized_dir)

        # Scale calibration to the final image resolution seen by the dataset:
        # preprocess downsample × runtime downsample.
        for cam_id in Ks_dict:
            K = Ks_dict[cam_id].copy()
            K[:2, :] /= self.total_factor
            Ks_dict[cam_id] = K
            W, H = imsize_dict[cam_id]
            imsize_dict[cam_id] = (
                max(W // self.total_factor, 1),
                max(H // self.total_factor, 1),
            )
            if self.images_are_undistorted:
                params_dict[cam_id] = np.empty(0, dtype=np.float32)

        # --- sync.json (optional) ---------------------------------------
        self.sync_offsets: Dict[str, float] = {}
        sync_path = optimized_dir / "sync.json"
        if sync_path.exists() and apply_sync:
            self.sync_offsets = {k: float(v) for k, v in json.loads(sync_path.read_text()).items()}

        # --- image folder layout (preprocessing output) ------------------
        images_dir = data_dir / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing images/ in {data_dir}")
        folder_names = sorted(
            name for name in os.listdir(images_dir)
            if (images_dir / name).is_dir()
        )
        folder_set = set(folder_names)
        missing = [name for name in camera_names if name not in folder_set]
        if missing:
            raise FileNotFoundError(
                "[SelfCapParser] missing image folders for YAML cameras: "
                f"{missing}"
            )
        extra = sorted(folder_set - set(camera_names))
        if extra:
            log.warning(
                "[SelfCapParser] ignoring extra image folders not in YAML: "
                f"{extra}"
            )

        # Critical: keep image paths in YAML camera order to guarantee
        # camtoworlds/K/params alignment.
        self.campaths = [str(images_dir / name) for name in camera_names]

        camera_ids = [i + 1 for i in range(len(camera_names))]

        # --- initial points from first-frame PLY ------------------------
        pcd_dir = data_dir / pcd_source
        if pcd_dir.is_dir():
            try:
                positions, colors = load_pcd_frame(
                    pcd_dir, self.absolute_frame_origin, frame_digits=6,
                )
            except FileNotFoundError:
                # Fallback: pick the first available PLY in the dir.
                plys = sorted(pcd_dir.glob("*.ply"))
                if not plys:
                    raise
                positions, colors = _load_xyzrgb_ply(plys[0])
                log.warning(
                    f"[SelfCapParser] no PLY for absolute frame "
                    f"{self.absolute_frame_origin}; using {plys[0].name}"
                )
        else:
            log.warning(f"[SelfCapParser] {pcd_dir} not found; points empty")
            positions = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)

        points = positions.astype(np.float32)
        points_rgb = colors.astype(np.uint8)
        points_err = np.zeros(points.shape[0], dtype=np.float32)

        # --- normalize world coords (optional) --------------------------
        if normalize and points.shape[0] > 0:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)
            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)
            transform = T2 @ T1
        else:
            transform = np.eye(4)

        # --- image format inference -------------------------------------
        # Preprocessing writes 6-digit 0-indexed jpgs but fall back to auto.
        if self.campaths and os.path.isdir(self.campaths[0]):
            try:
                fmt = _detect_image_format(self.campaths[0])
                self.image_format = fmt["extension"]
                self.frame_digits = fmt["frame_digits"]
                self.frame_start_offset = fmt["frame_start"]
            except ValueError:
                self.image_format = meta.get("extension", ".jpg")
                self.frame_digits = int(meta.get("frame_digits", 6))
                self.frame_start_offset = int(meta.get("frame_start_offset", 0))
        else:
            self.image_format = meta.get("extension", ".jpg")
            self.frame_digits = int(meta.get("frame_digits", 6))
            self.frame_start_offset = int(meta.get("frame_start_offset", 0))

        # Reconcile expected final size with decoded images (including runtime
        # resize behavior) per camera.
        for cam_idx, cam_name in enumerate(camera_names):
            cam_id = cam_idx + 1
            expected_w, expected_h = imsize_dict[cam_id]
            actual_wh = _probe_camera_image_size(
                Path(self.campaths[cam_idx]),
                self.image_format,
                self.frame_digits,
                self.frame_start_offset,
            )
            if actual_wh is None:
                msg = (
                    "[SelfCapParser] could not probe image size for camera "
                    f"{cam_name}; expected {expected_w}x{expected_h}."
                )
                if self.strict_validation:
                    raise FileNotFoundError(msg)
                log.warning(msg)
                continue

            src_w, src_h = actual_wh
            final_w_raw = src_w // self.factor
            final_h_raw = src_h // self.factor
            if final_w_raw < 1 or final_h_raw < 1:
                raise ValueError(
                    "[SelfCapParser] runtime factor too large for camera "
                    f"{cam_name}: decoded={src_w}x{src_h}, factor={self.factor}",
                )
            final_w = final_w_raw
            final_h = final_h_raw
            if (final_w, final_h) == (expected_w, expected_h):
                continue

            dw = abs(final_w - expected_w)
            dh = abs(final_h - expected_h)
            if (dw > self.max_size_mismatch_px) or (dh > self.max_size_mismatch_px):
                msg = (
                    "[SelfCapParser] large size mismatch for camera "
                    f"{cam_name}: expected {expected_w}x{expected_h} from "
                    f"(intri/meta factors), got {final_w}x{final_h} from decoded "
                    f"{src_w}x{src_h} and runtime factor={self.factor}. "
                    "Check preprocess_meta.json factor and extracted image sizes."
                )
                if self.strict_validation:
                    raise ValueError(msg)
                log.warning(msg)

            sx = final_w / max(expected_w, 1)
            sy = final_h / max(expected_h, 1)
            K = Ks_dict[cam_id].copy()
            K[0, :] *= sx
            K[1, :] *= sy
            Ks_dict[cam_id] = K
            imsize_dict[cam_id] = (final_w, final_h)
            log.warning(
                "[SelfCapParser] camera %s size mismatch: expected %dx%d, "
                "decoded %dx%d (runtime factor=%d -> %dx%d); "
                "rescaled K by (sx=%.6f, sy=%.6f).",
                cam_name,
                expected_w,
                expected_h,
                src_w,
                src_h,
                self.factor,
                final_w,
                final_h,
                sx,
                sy,
            )

        # --- store everything -------------------------------------------
        self.camera_names = camera_names
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict: Dict[int, Optional[np.ndarray]] = {cid: None for cid in camera_ids}
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.transform = transform
        self.max_camera_id = max(camera_ids) if camera_ids else 0
        self.colmap_format = "flat"
        self.camtype = "perspective"

        # Undistortion maps (re-use the FreeTimeParser helper by delegating).
        from .dataset import FreeTimeParser  # local import avoids cycle at module load
        self.mapx_dict: Dict[int, np.ndarray] = {}
        self.mapy_dict: Dict[int, np.ndarray] = {}
        self.roi_undist_dict: Dict[int, list] = {}
        pre_undist_imsize = dict(self.imsize_dict)
        FreeTimeParser._setup_undistortion(self)  # type: ignore[arg-type]
        self._validate_undistortion_setup(pre_undist_imsize)

        # Scene scale. (diameter of the camera trajectory, matches FreeTimeGS).
        cam_locs = camtoworlds[:, :3, 3]
        center = cam_locs.mean(axis=0)
        dists = np.linalg.norm(cam_locs - center, axis=1)
        self.scene_scale = float(np.max(dists)) * 2.0
        self.extent = self.scene_scale

        log.info(
            f"[SelfCapParser] {len(camera_names)} cameras, "
            f"frames=[{self.start_frame},{self.end_frame}) @ {self.fps} fps, "
            f"points={points.shape[0]}, scene_scale={self.scene_scale:.3f}, "
            f"factors(preprocess={self.preprocess_factor}, "
            f"runtime={self.factor}, total={self.total_factor}), "
            f"images_are_undistorted={self.images_are_undistorted}"
        )

    def _validate_undistortion_setup(
        self,
        pre_undist_imsize: Dict[int, Tuple[int, int]],
    ) -> None:
        """Fail fast when calibration/image/undistort state is inconsistent."""
        for cam_idx, cam_id in enumerate(self.camera_ids):
            params = self.params_dict.get(cam_id, np.empty(0, dtype=np.float32))
            if len(params) == 0:
                continue

            if cam_id not in self.mapx_dict or cam_id not in self.mapy_dict:
                raise ValueError(
                    f"[SelfCapParser] missing undistort map for camera_id={cam_id}",
                )
            if cam_id not in self.roi_undist_dict:
                raise ValueError(
                    f"[SelfCapParser] missing undistort ROI for camera_id={cam_id}",
                )

            mapx = self.mapx_dict[cam_id]
            mapy = self.mapy_dict[cam_id]
            if mapx.shape != mapy.shape:
                raise ValueError(
                    f"[SelfCapParser] map shape mismatch for camera_id={cam_id}: "
                    f"mapx={mapx.shape}, mapy={mapy.shape}",
                )
            map_h, map_w = mapx.shape
            expected_w, expected_h = pre_undist_imsize[cam_id]
            if (map_w, map_h) != (expected_w, expected_h):
                raise ValueError(
                    f"[SelfCapParser] undistort map size mismatch for camera_id={cam_id}: "
                    f"map={map_w}x{map_h}, expected={expected_w}x{expected_h}",
                )

            x, y, w_roi, h_roi = [int(v) for v in self.roi_undist_dict[cam_id]]
            if w_roi < 1 or h_roi < 1:
                raise ValueError(
                    f"[SelfCapParser] invalid ROI for camera_id={cam_id}: "
                    f"{[x, y, w_roi, h_roi]}",
                )
            if x < 0 or y < 0 or (x + w_roi) > map_w or (y + h_roi) > map_h:
                raise ValueError(
                    f"[SelfCapParser] ROI out of bounds for camera_id={cam_id}: "
                    f"roi={[x, y, w_roi, h_roi]}, map={map_w}x{map_h}",
                )

            valid = (
                (mapx >= 0.0)
                & (mapx <= float(map_w - 1))
                & (mapy >= 0.0)
                & (mapy <= float(map_h - 1))
            )
            valid_ratio = float(valid.mean())
            if valid_ratio < self.min_undistort_valid_ratio:
                raise ValueError(
                    f"[SelfCapParser] undistort map appears invalid for camera_id={cam_id}: "
                    f"valid_ratio={valid_ratio:.4f} < {self.min_undistort_valid_ratio:.4f}",
                )

            K = self.Ks_dict[cam_id]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            if fx <= 0.0 or fy <= 0.0:
                raise ValueError(
                    f"[SelfCapParser] non-positive focal length after undistortion "
                    f"for camera_id={cam_id}: fx={fx}, fy={fy}",
                )
            if not (0.0 <= cx < float(w_roi) and 0.0 <= cy < float(h_roi)):
                raise ValueError(
                    f"[SelfCapParser] principal point out of ROI after undistortion "
                    f"for camera_id={cam_id}: cx={cx}, cy={cy}, roi={w_roi}x{h_roi}",
                )

            probe = _probe_camera_image(
                Path(self.campaths[cam_idx]),
                self.image_format,
                self.frame_digits,
                self.frame_start_offset,
            )
            if probe is None:
                if self.strict_validation:
                    raise FileNotFoundError(
                        f"[SelfCapParser] cannot load probe image for camera_id={cam_id}",
                    )
                continue
            if probe.ndim == 2:
                probe = cv2.cvtColor(probe, cv2.COLOR_GRAY2BGR)
            if self.factor > 1:
                probe = cv2.resize(
                    probe,
                    (probe.shape[1] // self.factor, probe.shape[0] // self.factor),
                    interpolation=cv2.INTER_LINEAR,
                )
            if (probe.shape[1], probe.shape[0]) != (map_w, map_h):
                raise ValueError(
                    f"[SelfCapParser] probe image size mismatch for camera_id={cam_id}: "
                    f"probe={probe.shape[1]}x{probe.shape[0]}, map={map_w}x{map_h}",
                )
            remapped = cv2.remap(probe, mapx, mapy, cv2.INTER_LINEAR)
            cropped = remapped[y:y + h_roi, x:x + w_roi]
            if (cropped.shape[1], cropped.shape[0]) != (w_roi, h_roi):
                raise ValueError(
                    f"[SelfCapParser] remap/crop output mismatch for camera_id={cam_id}: "
                    f"got={cropped.shape[1]}x{cropped.shape[0]}, expected={w_roi}x{h_roi}",
                )

    # Convenience -------------------------------------------------------
    def absolute_frame(self, frame_offset: int) -> int:
        return self.absolute_frame_origin + frame_offset

    def pcd_path_for(self, frame_offset: int, subdir: str = "pcds") -> Path:
        f = self.absolute_frame(frame_offset)
        return self.data_dir / subdir / f"{f:0{self.frame_digits}d}.ply"


# ---------------------------------------------------------------------------
# Dataset


class SelfCapDataset(FreeTimeDataset):
    """FreeTimeDataset with 0-indexed frame lookup and sync.json correction.

    SelfCap preprocessing extracts mp4s into ``CAM/000000.jpg …`` (0-based
    offset into the preprocessing window), whereas FreeTimeDataset assumes
    filenames are the absolute frame index. We override the path resolver
    to use ``frame_offset`` directly. Each sample additionally exposes
    ``time_sec`` — the sync-corrected time in seconds.
    """

    def _get_image_path(self, camera_path: str, frame_idx: int) -> str:  # type: ignore[override]
        # In FreeTimeDataset.__getitem__, frame_idx is start_frame + offset.
        # SelfCap jpgs are 0-indexed by offset.
        frame_offset = frame_idx - self.parser.start_frame
        name = f"{frame_offset + self.parser.frame_start_offset:0{self.parser.frame_digits}d}{self.parser.image_format}"
        return os.path.join(camera_path, name)

    def __getitem__(self, item: int) -> Optional[Dict[str, Any]]:  # type: ignore[override]
        data = super().__getitem__(item)
        if data is None:
            return None

        cam_idx = int(data["camera_idx"])
        frame_offset = int(data["frame_offset"])
        cam_name = self.parser.camera_names[cam_idx]
        offsets = getattr(self.parser, "sync_offsets", {}) or {}
        sync_off = float(offsets.get(cam_name, 0.0))

        fps = float(getattr(self.parser, "fps", 60.0))
        data["time_sec"] = float(frame_offset) / max(fps, 1e-6) - sync_off
        return data
