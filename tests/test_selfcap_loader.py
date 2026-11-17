"""Tests for the SelfCap dataset loader."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from trigs.io.selfcap import (
    SelfCapParser,
    SelfCapDataset,
    _load_xyzrgb_ply,
    read_easyvolcap_cameras,
)


# ---------------------------------------------------------------------------
# Fixture construction


def _write_extri_intri(
    optimized_dir: Path,
    *,
    H: int = 1080,
    W: int = 1920,
    d_by_name: dict[str, np.ndarray] | None = None,
) -> None:
    """Write a minimal 2-camera EasyVolcap YAML pair."""
    optimized_dir.mkdir(parents=True, exist_ok=True)

    # Two cameras: one at origin looking +Z, one translated in +X.
    R = np.eye(3, dtype=np.float64)
    t0 = np.array([[0.0], [0.0], [1.0]])
    t1 = np.array([[1.0], [0.0], [1.0]])
    fx = 0.8 * float(W)
    fy = 0.8 * float(H)
    cx = float(W) * 0.5
    cy = float(H) * 0.5
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([[0.10], [-0.15], [0.001], [-0.001], [0.02]], dtype=np.float64)

    import cv2
    extri = cv2.FileStorage(str(optimized_dir / "extri.yml"), cv2.FILE_STORAGE_WRITE)
    extri.write("names", ["0000", "0001"])
    for name, t in [("0000", t0), ("0001", t1)]:
        extri.write(f"Rot_{name}", R)
        extri.write(f"T_{name}", t)
    extri.release()

    intri = cv2.FileStorage(str(optimized_dir / "intri.yml"), cv2.FILE_STORAGE_WRITE)
    intri.write("names", ["0000", "0001"])
    for name in ["0000", "0001"]:
        D_cur = d_by_name[name] if d_by_name is not None and name in d_by_name else D
        intri.write(f"K_{name}", K)
        intri.write(f"D_{name}", D_cur)
        intri.write(f"H_{name}", float(H))
        intri.write(f"W_{name}", float(W))
    intri.release()


def _write_xyzrgb_ply(path: Path, positions: np.ndarray, colors: np.ndarray) -> None:
    """Write a minimal binary-LE PLY with x/y/z float + r/g/b uchar."""
    n = positions.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode()
    with open(path, "wb") as f:
        f.write(header)
        for p, c in zip(positions, colors):
            f.write(struct.pack("<fff", *p.astype(np.float32)))
            f.write(struct.pack("<BBB", *c.astype(np.uint8)))


def _write_tiny_jpg(path: Path, h: int = 64, w: int = 64) -> None:
    import cv2
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Unit tests


def test_read_easyvolcap_cameras_roundtrip(tmp_path: Path):
    _write_extri_intri(tmp_path)
    names, c2w, Ks, params, imsize = read_easyvolcap_cameras(tmp_path)

    assert names == ["0000", "0001"]
    assert c2w.shape == (2, 4, 4)

    # Rotation block orthogonal.
    for i in range(2):
        R = c2w[i, :3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)

    # w2c = [R|t], c2w = inv(w2c) → c2w translation should be -R.T @ t.
    # For identity R, t0=(0,0,1), c2w translation should be (0,0,-1).
    np.testing.assert_allclose(c2w[0, :3, 3], [0, 0, -1], atol=1e-5)
    np.testing.assert_allclose(c2w[1, :3, 3], [-1, 0, -1], atol=1e-5)

    # Intrinsics + distortion shape.
    assert Ks[1].shape == (3, 3)
    assert params[1].shape == (5,)  # [k1, k2, p1, p2, k3]
    np.testing.assert_allclose(params[1], [0.10, -0.15, 0.001, -0.001, 0.02], atol=1e-7)
    assert imsize[1] == (1920, 1080)


def test_ply_roundtrip(tmp_path: Path):
    pts = np.random.default_rng(0).standard_normal((50, 3)).astype(np.float32)
    cols = (np.random.default_rng(1).integers(0, 255, size=(50, 3))).astype(np.uint8)
    path = tmp_path / "005200.ply"
    _write_xyzrgb_ply(path, pts, cols)

    xyz, rgb = _load_xyzrgb_ply(path)
    np.testing.assert_allclose(xyz, pts, atol=1e-6)
    np.testing.assert_array_equal(rgb, cols)


def _build_tiny_scene(
    root: Path,
    *,
    start_frame: int = 5200,
    n_frames: int = 3,
    preprocess_factor: int = 1,
    image_hw: tuple[int, int] = (64, 64),
    calib_hw: tuple[int, int] | None = None,
    d_by_name: dict[str, np.ndarray] | None = None,
    images_are_undistorted: bool = False,
) -> Path:
    """Assemble a tiny preprocessed-SelfCap scene for end-to-end parser tests."""
    if calib_hw is None:
        calib_h = int(image_hw[0]) * int(preprocess_factor)
        calib_w = int(image_hw[1]) * int(preprocess_factor)
    else:
        calib_h = int(calib_hw[0])
        calib_w = int(calib_hw[1])
    _write_extri_intri(
        root / "optimized",
        H=calib_h,
        W=calib_w,
        d_by_name=d_by_name,
    )

    # sync.json with non-zero offsets.
    (root / "optimized" / "sync.json").write_text(json.dumps({
        "0000": 0.0, "0001": 0.01,
    }))

    # Images dir (one jpg per camera per frame).
    for cam in ("0000", "0001"):
        cam_dir = root / "images" / cam
        cam_dir.mkdir(parents=True)
        for fi in range(n_frames):
            _write_tiny_jpg(
                cam_dir / f"{fi:06d}.jpg",
                h=int(image_hw[0]),
                w=int(image_hw[1]),
            )

    # Pcds dir with one PLY at the start frame.
    pcds = root / "pcds"
    pcds.mkdir()
    pts = np.random.default_rng(0).standard_normal((20, 3)).astype(np.float32)
    cols = np.full((20, 3), 100, dtype=np.uint8)
    _write_xyzrgb_ply(pcds / f"{start_frame:06d}.ply", pts, cols)

    # preprocess_meta.json.
    meta = {
        "scene_dir": str(root),
        "start_frame": start_frame,
        "end_frame": start_frame + n_frames,
        "n_frames": n_frames,
        "fps": 60.0,
        "factor": int(preprocess_factor),
        "jpeg_quality": 2,
        "extension": ".jpg",
        "frame_digits": 6,
        "frame_start_offset": 0,
        "absolute_frame_origin": start_frame,
        "images_are_undistorted": bool(images_are_undistorted),
    }
    (root / "preprocess_meta.json").write_text(json.dumps(meta))
    return root


def test_selfcap_parser_and_dataset(tmp_path: Path):
    scene = _build_tiny_scene(tmp_path)
    parser = SelfCapParser(scene, factor=1)

    assert parser.camera_names == ["0000", "0001"]
    assert parser.camtoworlds.shape == (2, 4, 4)
    assert parser.points.shape == (20, 3)
    assert parser.absolute_frame_origin == 5200
    assert parser.total_frames == 3
    assert parser.fps == 60.0
    assert parser.sync_offsets["0001"] == pytest.approx(0.01)

    ds = SelfCapDataset(parser, split="train", test_set=[])
    # 2 cameras × 3 frames
    assert len(ds) == 6

    # time_sec: cam 0001, frame 2 → 2/60 − 0.01
    # Locate that pair.
    for i in range(len(ds)):
        s = ds[i]
        assert s is not None
        if ds.parser.camera_names[int(s["camera_idx"])] == "0001" and int(s["frame_offset"]) == 2:
            assert s["time_sec"] == pytest.approx(2.0 / 60.0 - 0.01, abs=1e-6)
            break
    else:
        pytest.fail("Could not find cam 0001 frame 2 sample")


def test_selfcap_parser_combines_preprocess_and_runtime_factor(tmp_path: Path):
    # intri.yml fixture is 1920x1080. Simulate preprocess_factor=2 extracted
    # jpgs (960x540), then runtime factor=2 in parser/dataset -> 480x270 maps.
    scene = _build_tiny_scene(
        tmp_path,
        preprocess_factor=2,
        image_hw=(540, 960),
    )
    parser = SelfCapParser(scene, factor=2)

    assert parser.preprocess_factor == 2
    assert parser.factor == 2
    assert parser.total_factor == 4
    # map shape is [H, W] at the final resolution used for undistortion.
    assert parser.mapx_dict[1].shape == (270, 480)


def test_selfcap_parser_validates_yaml_camera_folders(tmp_path: Path):
    scene = _build_tiny_scene(tmp_path)
    # Add an unrelated extra folder; parser should ignore it rather than
    # reordering camera paths by lexicographic folder listing.
    (scene / "images" / "extra_camera").mkdir(parents=True, exist_ok=True)

    parser = SelfCapParser(scene, factor=1)
    assert parser.camera_names == ["0000", "0001"]
    assert len(parser.campaths) == 2
    assert parser.campaths[0].endswith("/images/0000")
    assert parser.campaths[1].endswith("/images/0001")


def test_selfcap_parser_raises_on_missing_yaml_camera_folder(tmp_path: Path):
    scene = _build_tiny_scene(tmp_path)
    # Remove one YAML-listed camera folder.
    for path in (scene / "images" / "0001").glob("*"):
        path.unlink()
    (scene / "images" / "0001").rmdir()

    with pytest.raises(FileNotFoundError):
        _ = SelfCapParser(scene, factor=1)


def test_selfcap_parser_fails_fast_on_large_factor_mismatch(tmp_path: Path):
    # intri expects 1920x1080 with preprocess_factor=1. Inject half-res images:
    # strict mode should raise instead of silently proceeding.
    scene = _build_tiny_scene(
        tmp_path,
        preprocess_factor=1,
        image_hw=(540, 960),
        calib_hw=(1080, 1920),
    )
    with pytest.raises(ValueError):
        _ = SelfCapParser(scene, factor=1)


def test_selfcap_parser_can_relax_large_factor_mismatch(tmp_path: Path):
    scene = _build_tiny_scene(
        tmp_path,
        preprocess_factor=1,
        image_hw=(540, 960),
        calib_hw=(1080, 1920),
    )
    parser = SelfCapParser(
        scene,
        factor=1,
        strict_validation=False,
    )
    # In relaxed mode parser rescales K/imsize to decoded image size.
    assert parser.mapx_dict[1].shape == (540, 960)


def test_selfcap_parser_baked_pinhole_skips_preprocess_rescale(tmp_path: Path):
    # Simulate a dataset preprocessed to pinhole where intrinsics already match
    # extracted images at preprocess factor 2.
    scene = _build_tiny_scene(
        tmp_path,
        preprocess_factor=2,
        image_hw=(540, 960),
        calib_hw=(540, 960),
        images_are_undistorted=True,
    )
    parser = SelfCapParser(scene, factor=1)

    assert parser.images_are_undistorted is True
    assert parser.total_factor == 1
    assert len(parser.params_dict[1]) == 0
    assert parser.imsize_dict[1] == (960, 540)
    assert parser.imsize_dict[2] == (960, 540)
    # No undistortion maps are needed for already-rectified pinhole inputs.
    assert parser.mapx_dict == {}
    assert parser.mapy_dict == {}


def test_selfcap_per_camera_roi_and_intrinsics_consistent(tmp_path: Path):
    import cv2

    # Force two different local OpenCV ROIs and ensure parser preserves each
    # camera's own ROI + image size.
    scene = _build_tiny_scene(
        tmp_path,
        preprocess_factor=1,
        image_hw=(360, 640),
        d_by_name={
            "0000": np.array([[0.18], [-0.22], [0.001], [-0.001], [0.03]], dtype=np.float64),
            "0001": np.array([[0.04], [-0.05], [0.0], [0.0], [0.01]], dtype=np.float64),
        },
    )

    # Compute local camera ROIs from OpenCV reference path.
    _, _, Ks_raw, params_raw, imsize_raw = read_easyvolcap_cameras(scene / "optimized")
    local_rois = {}
    for cam_id in (1, 2):
        w, h = imsize_raw[cam_id]
        _K_u, roi = cv2.getOptimalNewCameraMatrix(
            Ks_raw[cam_id], params_raw[cam_id], (w, h), 0,
        )
        local_rois[cam_id] = tuple(int(v) for v in roi)

    parser = SelfCapParser(scene, factor=1)

    roi0 = tuple(int(v) for v in parser.roi_undist_dict[1])
    roi1 = tuple(int(v) for v in parser.roi_undist_dict[2])
    assert roi0 == local_rois[1]
    assert roi1 == local_rois[2]
    assert parser.imsize_dict[1] == (roi0[2], roi0[3])
    assert parser.imsize_dict[2] == (roi1[2], roi1[3])

    # K must be in each camera's own cropped frame.
    for cam_id in (1, 2):
        w_out, h_out = parser.imsize_dict[cam_id]
        K = parser.Ks_dict[cam_id]
        assert 0.0 <= float(K[0, 2]) < float(w_out)
        assert 0.0 <= float(K[1, 2]) < float(h_out)

    ds = SelfCapDataset(parser, split="train", test_set=[])
    s0 = ds[0]
    # camera-major ordering: first n_frames belong to first train camera
    s1 = ds[parser.total_frames]
    assert s0 is not None and s1 is not None
    h0, w0 = parser.imsize_dict[1][1], parser.imsize_dict[1][0]
    h1, w1 = parser.imsize_dict[2][1], parser.imsize_dict[2][0]
    assert tuple(s0["image"].shape[:2]) == (h0, w0)
    assert tuple(s1["image"].shape[:2]) == (h1, w1)


# ---------------------------------------------------------------------------
# Integration smoke against the shared dataset (skipped when not mounted).

_DATASET_ROOT = Path("/data/shared/datasets/4DV/corgi-release")


@pytest.mark.skipif(not _DATASET_ROOT.is_dir(), reason="4DV dataset not mounted")
def test_corgi_calibration_smoke():
    names, c2w, Ks, params, imsize = read_easyvolcap_cameras(_DATASET_ROOT / "optimized")
    assert len(names) == 24
    assert c2w.shape == (24, 4, 4)
    # Rotation block orthogonal for every camera.
    for i in range(24):
        R = c2w[i, :3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-4)
    # Intrinsics reasonable (2160p master).
    first_id = 1
    assert imsize[first_id] == (3840, 2160)


@pytest.mark.skipif(not _DATASET_ROOT.is_dir(), reason="4DV dataset not mounted")
def test_corgi_pcd_smoke():
    xyz, rgb = _load_xyzrgb_ply(_DATASET_ROOT / "pcds" / "005200.ply")
    assert xyz.shape[0] > 0
    assert xyz.shape[1] == 3
    assert rgb.shape == xyz.shape


# ---------------------------------------------------------------------------
# Multi-keyframe canonical builder


def test_build_canonical_from_pcd_sequence(tmp_path: Path):
    """3-keyframe synthetic sequence with constant +x velocity at 60 fps.

    Expectations:
        - ParameterDict contains the full trainer-compatible schema.
        - ν init recovers ~ (+1, 0, 0) world/s on the middle keyframe
          (centered FD; boundary keyframes use one-sided FD).
    """
    from trigs.init.from_selfcap import build_canonical_from_pcd_sequence

    pcds = tmp_path / "pcds"
    pcds.mkdir()

    rng = np.random.default_rng(0)
    base = rng.standard_normal((30, 3)).astype(np.float32)
    cols = np.full((30, 3), 128, dtype=np.uint8)

    # +x motion: 1 world unit per second at 60 fps => +1/60 per frame.
    fps = 60.0
    start = 5200
    n_frames = 3
    dx = 1.0 / fps
    for fi in range(n_frames):
        shifted = base.copy()
        shifted[:, 0] += fi * dx
        _write_xyzrgb_ply(pcds / f"{start + fi:06d}.ply", shifted, cols)

    params, sh_degree, scene_extent = build_canonical_from_pcd_sequence(
        data_dir=tmp_path,
        absolute_frame_origin=start,
        n_frames=n_frames,
        fps=fps,
        keyframe_stride=1,
        nn_velocity_k=1,
        device="cpu",
        verbose=False,
    )

    assert sh_degree == 0
    for key in ("means", "quats", "scales", "sh0", "opacities", "times",
                "log_durations", "anchors", "zeta_base_t", "zeta_base_r",
                "zeta_c0_t", "zeta_c0_r", "zeta_c1_t", "zeta_c1_r",
                "zeta_c2_t", "zeta_c2_r"):
        assert key in params, f"missing param {key}"

    # 3 keyframes × 30 points.
    assert params["means"].shape == (90, 3)

    # Identity quats.
    np.testing.assert_allclose(params["quats"].detach().cpu().numpy()[:, 0], 1.0)

    # Middle keyframe (frame offset 1) has a centered FD; ν should be ≈ (1, 0, 0)
    # before ω subtraction. After ω-reduction, the residual ν may be smaller,
    # but ν + ω × r should still approximate the observed velocity. Check that
    # the recovered velocity (ν + ω × r) ≈ (1, 0, 0) on the middle bucket.
    means = params["means"].detach().cpu().numpy()
    anchors = params["anchors"].detach().cpu().numpy()
    nu = params["zeta_base_t"].detach().cpu().numpy()
    omega = params["zeta_base_r"].detach().cpu().numpy()
    times = params["times"].detach().cpu().squeeze(-1).numpy()

    # Middle bucket (t ≈ 1/60).
    mid_mask = np.isclose(times, 1.0 / fps, atol=1e-6)
    assert mid_mask.sum() > 0
    r = means[mid_mask] - anchors[mid_mask]
    v_recovered = nu[mid_mask] + np.cross(omega[mid_mask], r)
    np.testing.assert_allclose(
        v_recovered.mean(axis=0), [1.0, 0.0, 0.0], atol=0.1,
    )
