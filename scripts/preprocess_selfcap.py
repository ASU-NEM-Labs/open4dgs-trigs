"""Preprocess a SelfCap (4DV) scene for TriGS training.

Extracts a window from each per-camera mp4 as jpgs, copies the matching
subset of point clouds and the EasyVolcap calibration, and writes a
``preprocess_meta.json`` so :class:`trigs.io.selfcap.SelfCapParser` knows
the absolute-frame origin of the extraction.

Frame-index conventions
-----------------------
SelfCap's mp4 files and pcd filenames use *different* frame origins:

- ``videos/CAM.mp4`` — frames are 0-indexed *within the mp4*. Use
  ``--start-frame`` / ``--end-frame`` in this space.
- ``pcds/FFFFFF.ply`` — filenames carry absolute source-footage indices
  (e.g. corgi's mp4 covers absolute frames 5200–8699, so the pcds span
  005200.ply–008699.ply).

This script auto-detects the pcd origin (minimum filename in ``pcds/``)
and copies the matching slice of pcds into the output directory,
preserving original filenames. The resulting ``preprocess_meta.json``
records ``absolute_frame_origin = pcd_origin + start_frame`` so the
loader can map extracted-jpg frame 0 back to its pcd.

Example:
    uv run python scripts/preprocess_selfcap.py \
        --scene-dir /data/shared/datasets/4DV/corgi-release \
        --output-dir /data/shared/aly/data/corgi_first1k \
        --start-frame 0 --end-frame 1000 \
        --factor 2 --jpeg-quality 2 --workers 24
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _extract_one(
    video_path: Path,
    out_dir: Path,
    start_frame: int,
    end_frame: int,
    factor: int,
    jpeg_quality: int,
) -> tuple[str, int]:
    """ffmpeg-extract ``[start_frame, end_frame)`` (mp4-relative) to 0-indexed jpgs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    vf_parts = [
        f"select='between(n\\,{start_frame}\\,{end_frame - 1})'",
        "setpts=N/FR/TB",
    ]
    if factor > 1:
        vf_parts.append(f"scale=iw/{factor}:ih/{factor}")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", vf,
        "-fps_mode", "passthrough",
        "-q:v", str(jpeg_quality),
        "-start_number", "0",
        str(out_dir / "%06d.jpg"),
    ]
    subprocess.check_call(cmd)
    n = len(list(out_dir.glob("*.jpg")))
    return video_path.stem, n


def _detect_pcd_origin(pcd_dir: Path) -> int | None:
    """Return the smallest integer filename stem in ``pcd_dir``, or None."""
    if not pcd_dir.is_dir():
        return None
    stems = []
    for p in pcd_dir.iterdir():
        if p.suffix != ".ply":
            continue
        try:
            stems.append(int(p.stem))
        except ValueError:
            continue
    return min(stems) if stems else None


def _copy_pcd_range(
    src_dir: Path,
    dst_dir: Path,
    absolute_start: int,
    absolute_end_excl: int,
    stride: int = 1,
    frame_digits: int = 6,
) -> int:
    """Copy ``{absolute_start .. absolute_end_excl} step stride`` PLYs from
    ``src_dir`` into ``dst_dir``. Files missing from src are silently skipped
    (dense_pcds are every-1000 so many strides won't have a match)."""
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in range(absolute_start, absolute_end_excl, stride):
        src = src_dir / f"{f:0{frame_digits}d}.ply"
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)
            copied += 1
    return copied


def _copy_any_pcds_in_range(
    src_dir: Path,
    dst_dir: Path,
    absolute_start: int,
    absolute_end_excl: int,
) -> int:
    """Copy every .ply in ``src_dir`` whose numeric stem is in
    ``[absolute_start, absolute_end_excl)``. Used for dense_pcds where the
    sampling stride isn't known a priori."""
    if not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in sorted(src_dir.glob("*.ply")):
        try:
            f = int(p.stem)
        except ValueError:
            continue
        if absolute_start <= f < absolute_end_excl:
            shutil.copy2(p, dst_dir / p.name)
            copied += 1
    return copied


def _read_opencv_yaml(path: Path) -> cv2.FileStorage:
    """Open OpenCV YAML files that may miss the header directive."""
    try:
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            return fs
    except (cv2.error, SystemError):
        # Some OpenCV builds throw immediately on header-less YAML.
        pass
    text = path.read_text()
    if not text.startswith("%YAML"):
        text = "%YAML:1.0\n---\n" + text
    fs = cv2.FileStorage(text, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    if not fs.isOpened():
        raise ValueError(f"Failed to parse YAML: {path}")
    return fs


def _read_easyvolcap_intri(intri_path: Path) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[int, int]]]:
    """Read EasyVolcap intrinsics/distortion by camera name."""
    fs = _read_opencv_yaml(intri_path)
    names_node = fs.getNode("names")
    if names_node.empty():
        raise ValueError(f"'names' missing in {intri_path}")
    names = [names_node.at(i).string() for i in range(names_node.size())]

    K_by_name: dict[str, np.ndarray] = {}
    D_by_name: dict[str, np.ndarray] = {}
    size_by_name: dict[str, tuple[int, int]] = {}
    for name in names:
        K_node = fs.getNode(f"K_{name}")
        D_node = fs.getNode(f"D_{name}")
        H_node = fs.getNode(f"H_{name}")
        W_node = fs.getNode(f"W_{name}")
        if K_node.empty() or H_node.empty() or W_node.empty():
            raise ValueError(f"Incomplete intrinsics for camera {name}")
        K = K_node.mat().astype(np.float64)
        D = np.empty(0, dtype=np.float64) if D_node.empty() else D_node.mat().astype(np.float64).reshape(-1)
        H = int(H_node.real())
        W = int(W_node.real())
        K_by_name[name] = K
        D_by_name[name] = D
        size_by_name[name] = (W, H)
    return names, K_by_name, D_by_name, size_by_name


def _write_easyvolcap_intri_pinhole(
    intri_path: Path,
    names: list[str],
    K_by_name: dict[str, np.ndarray],
    size_by_name: dict[str, tuple[int, int]],
) -> None:
    """Write pinhole intrinsics (zero distortion) at per-camera final sizes."""
    fs = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_WRITE)
    fs.write("names", names)
    D0 = np.zeros((5, 1), dtype=np.float64)
    ccm = np.eye(3, dtype=np.float64)
    for name in names:
        width, height = size_by_name[name]
        fs.write(f"K_{name}", K_by_name[name].astype(np.float64))
        fs.write(f"D_{name}", D0)
        fs.write(f"H_{name}", float(height))
        fs.write(f"W_{name}", float(width))
        fs.write(f"ccm_{name}", ccm)
    fs.release()


def _rectify_images_and_calibration(
    *,
    out_dir: Path,
    preprocess_factor: int,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Undistort/crop extracted images and rewrite intri.yml as pinhole.

    Returns:
        local_roi_stats: {camera_name: local ROI area}
        final_size_by_camera: {camera_name: {"width": W, "height": H}}
    """
    optimized_dir = out_dir / "optimized"
    images_dir = out_dir / "images"
    intri_path = optimized_dir / "intri.yml"
    names, K_raw, D_raw, size_raw = _read_easyvolcap_intri(intri_path)

    maps_by_name: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    K_u_full_by_name: dict[str, np.ndarray] = {}
    roi_by_name: dict[str, tuple[int, int, int, int]] = {}

    for name in names:
        cam_dir = images_dir / name
        if not cam_dir.is_dir():
            raise FileNotFoundError(f"Missing extracted image folder: {cam_dir}")
        first = cam_dir / "000000.jpg"
        if not first.exists():
            imgs = sorted(cam_dir.glob("*.jpg"))
            if not imgs:
                raise FileNotFoundError(f"No extracted jpgs in {cam_dir}")
            first = imgs[0]
        probe = cv2.imread(str(first), cv2.IMREAD_COLOR)
        if probe is None:
            raise ValueError(f"Could not read extracted image: {first}")
        h_img, w_img = probe.shape[:2]

        K = K_raw[name].copy()
        w_raw, h_raw = size_raw[name]
        K[:2, :] /= max(preprocess_factor, 1)
        # Align intrinsics to actual extracted size (ffmpeg rounding-safe).
        w_expect = max(w_raw // max(preprocess_factor, 1), 1)
        h_expect = max(h_raw // max(preprocess_factor, 1), 1)
        K[0, :] *= (w_img / max(w_expect, 1))
        K[1, :] *= (h_img / max(h_expect, 1))

        D = D_raw[name]
        has_distortion = D.size > 0 and np.any(np.abs(D) > 0)
        if has_distortion:
            K_u, roi = cv2.getOptimalNewCameraMatrix(
                K.astype(np.float32), D.astype(np.float32), (w_img, h_img), 0,
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K.astype(np.float32),
                D.astype(np.float32),
                None,
                K_u.astype(np.float32),
                (w_img, h_img),
                cv2.CV_32FC1,
            )
            maps_by_name[name] = (mapx, mapy)
            K_u_full_by_name[name] = K_u.astype(np.float64)
            x, y, w, h = [int(v) for v in roi]
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid undistortion ROI for camera {name}: {(x, y, w, h)}")
            roi_by_name[name] = (x, y, w, h)
        else:
            maps_by_name[name] = None
            K_u_full_by_name[name] = K.astype(np.float64)
            roi_by_name[name] = (0, 0, w_img, h_img)

    # Rewrite extracted images as undistorted, per-camera ROI-cropped pinhole views.
    for name in names:
        x, y, w, h = roi_by_name[name]
        cam_dir = images_dir / name
        image_paths = sorted(cam_dir.glob("*.jpg"))
        for p in image_paths:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not read extracted image: {p}")
            maps = maps_by_name[name]
            if maps is not None:
                mapx, mapy = maps
                img_u = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
            else:
                img_u = img
            crop = img_u[y:y + h, x:x + w]
            if crop.shape[0] != h or crop.shape[1] != w:
                raise ValueError(
                    f"Local ROI crop mismatch for {name}/{p.name}: "
                    f"got {crop.shape[1]}x{crop.shape[0]}, expected {w}x{h}",
                )
            ok = cv2.imwrite(str(p), crop)
            if not ok:
                raise RuntimeError(f"Failed writing rectified frame: {p}")

    # Convert intrinsics to cropped pinhole space.
    K_final_by_name: dict[str, np.ndarray] = {}
    final_size_by_name: dict[str, tuple[int, int]] = {}
    for name in names:
        x, y, w, h = roi_by_name[name]
        Kf = K_u_full_by_name[name].copy()
        Kf[0, 2] -= x
        Kf[1, 2] -= y
        K_final_by_name[name] = Kf
        final_size_by_name[name] = (int(w), int(h))

    _write_easyvolcap_intri_pinhole(
        intri_path=optimized_dir / "intri.yml",
        names=names,
        K_by_name=K_final_by_name,
        size_by_name=final_size_by_name,
    )
    local_roi_stats = {name: int(roi_by_name[name][2] * roi_by_name[name][3]) for name in names}
    final_size = {
        name: {"width": int(final_size_by_name[name][0]), "height": int(final_size_by_name[name][1])}
        for name in names
    }
    return local_roi_stats, final_size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", type=Path, required=True,
                    help="Path to <scene>-release directory")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, required=True,
                    help="First mp4-relative frame (inclusive, 0-indexed)")
    ap.add_argument("--end-frame", type=int, required=True,
                    help="Last mp4-relative frame (exclusive)")
    ap.add_argument("--factor", type=int, default=1,
                    help="Spatial downsample factor (1=full, 2=half)")
    ap.add_argument("--jpeg-quality", type=int, default=2,
                    help="ffmpeg -q:v (1=best, 31=worst; default 2)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=0,
                    help="Parallel ffmpeg jobs (0 = one per camera)")
    ap.add_argument("--pcd-origin", type=int, default=None,
                    help="Absolute footage frame of mp4-frame-0. "
                         "Auto-detected from min pcds/*.ply filename if omitted.")
    ap.add_argument("--skip-pcds", action="store_true",
                    help="Don't copy per-frame pcds (useful if the caller "
                         "will feed canonical points from elsewhere).")
    ap.add_argument("--skip-dense-pcds", action="store_true",
                    help="Don't copy dense_pcds*.")
    ap.add_argument(
        "--no-rectify-pinhole",
        action="store_true",
        help="Keep extracted images distorted and keep original intrinsics in optimized/intri.yml.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scene_dir = args.scene_dir
    out_dir = args.output_dir
    videos_dir = scene_dir / "videos"
    optimized_src = scene_dir / "optimized"

    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Missing videos/ in {scene_dir}")
    if not optimized_src.is_dir():
        raise FileNotFoundError(f"Missing optimized/ in {scene_dir}")

    mp4s = sorted(videos_dir.glob("*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No .mp4 files in {videos_dir}")

    # -- resolve pcd origin -------------------------------------------------
    pcd_origin = args.pcd_origin
    if pcd_origin is None:
        pcd_origin = _detect_pcd_origin(scene_dir / "pcds")
        if pcd_origin is None:
            pcd_origin = 0
            log.warning(
                "Could not detect pcd origin from pcds/; defaulting to 0. "
                "Pass --pcd-origin explicitly if this is wrong."
            )
        else:
            log.info(f"Auto-detected pcd_origin = {pcd_origin}")

    absolute_frame_origin = pcd_origin + args.start_frame
    absolute_frame_end_excl = pcd_origin + args.end_frame

    out_dir.mkdir(parents=True, exist_ok=True)
    images_out = out_dir / "images"
    images_out.mkdir(exist_ok=True)

    # -- copy calibration (tiny) -------------------------------------------
    optimized_dst = out_dir / "optimized"
    optimized_dst.mkdir(exist_ok=True)
    for name in ("extri.yml", "intri.yml", "sync.json"):
        src = optimized_src / name
        if src.exists():
            shutil.copy2(src, optimized_dst / name)
            log.info(f"  copied optimized/{name}")

    # -- copy in-range pcds -------------------------------------------------
    pcd_counts: dict[str, int] = {}
    if not args.skip_pcds:
        n = _copy_pcd_range(
            scene_dir / "pcds", out_dir / "pcds",
            absolute_frame_origin, absolute_frame_end_excl, stride=1,
        )
        pcd_counts["pcds"] = n
        log.info(f"  copied pcds: {n} files")
        # pcds_j10 = every 10th frame; copy any that fall in range.
        n10 = _copy_any_pcds_in_range(
            scene_dir / "pcds_j10", out_dir / "pcds_j10",
            absolute_frame_origin, absolute_frame_end_excl,
        )
        if n10 > 0:
            pcd_counts["pcds_j10"] = n10
            log.info(f"  copied pcds_j10: {n10} files")

    if not args.skip_dense_pcds:
        for name in ("dense_pcds", "dense_pcds_bbox", "dense_pcds_rc_bbox_180k"):
            nd = _copy_any_pcds_in_range(
                scene_dir / name, out_dir / name,
                absolute_frame_origin, absolute_frame_end_excl,
            )
            if nd > 0:
                pcd_counts[name] = nd
                log.info(f"  copied {name}: {nd} files")

    # -- parallel ffmpeg extraction -----------------------------------------
    n_workers = args.workers or len(mp4s)
    log.info(
        f"Extracting mp4 frames [{args.start_frame}, {args.end_frame}) from "
        f"{len(mp4s)} cameras via {n_workers} workers (factor={args.factor})"
    )

    per_cam_counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _extract_one,
                mp4, images_out / mp4.stem,
                args.start_frame, args.end_frame,
                args.factor, args.jpeg_quality,
            ): mp4 for mp4 in mp4s
        }
        for fut in as_completed(futures):
            mp4 = futures[fut]
            try:
                cam_name, n = fut.result()
                per_cam_counts[cam_name] = n
                log.info(f"  {cam_name}: {n} frames")
            except Exception as e:
                log.error(f"  {mp4.stem}: FAILED ({e})")
                raise

    # -- optional one-time undistort + pinhole calibration bake -------------
    local_roi_stats: dict[str, int] | None = None
    final_image_size_by_camera: dict[str, dict[str, int]] | None = None
    images_are_undistorted = not args.no_rectify_pinhole
    if images_are_undistorted:
        log.info("Rectifying extracted images and baking pinhole intrinsics...")
        local_roi_stats, final_image_size_by_camera = _rectify_images_and_calibration(
            out_dir=out_dir,
            preprocess_factor=args.factor,
        )
        sizes = list(final_image_size_by_camera.values())
        min_w = min(s["width"] for s in sizes)
        max_w = max(s["width"] for s in sizes)
        min_h = min(s["height"] for s in sizes)
        max_h = max(s["height"] for s in sizes)
        log.info(f"  Rectified pinhole sizes by camera: width {min_w}-{max_w}, height {min_h}-{max_h}")

    # -- meta ---------------------------------------------------------------
    meta = {
        "scene_dir": str(scene_dir.resolve()),
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "n_frames": args.end_frame - args.start_frame,
        "fps": args.fps,
        "factor": args.factor,
        "jpeg_quality": args.jpeg_quality,
        "extension": ".jpg",
        "frame_digits": 6,
        "frame_start_offset": 0,
        "pcd_origin": pcd_origin,
        "absolute_frame_origin": absolute_frame_origin,
        "per_camera_frame_counts": per_cam_counts,
        "pcd_counts": pcd_counts,
        "images_are_undistorted": images_are_undistorted,
        "calibration_baked_to_images": images_are_undistorted,
        "shared_roi_xywh": None,
        "final_image_size": None,
        "final_image_size_by_camera": final_image_size_by_camera,
        "local_roi_area_by_camera": local_roi_stats,
    }
    (out_dir / "preprocess_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(f"Wrote {out_dir / 'preprocess_meta.json'}")
    log.info("Done.")


if __name__ == "__main__":
    main()
