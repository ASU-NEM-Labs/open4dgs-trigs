from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import numpy as np
from PIL import Image

from trigs.cli.train_from_colmap import TrigsColmapArgs, _LazyGTMasks, _LazyGTImages


def _loader(tmp_path: Path, *, source_start_frame: int, mask_frame_offset: int = 0):
    dataset = SimpleNamespace(pairs=[(0, 0), (0, 119)])
    return _LazyGTMasks(
        dataset=dataset,
        n_cameras=1,
        orig_to_pos={0: 0},
        masks_root=tmp_path,
        cam_folders_by_pos=["00"],
        source_start_frame=source_start_frame,
        mask_frame_offset=mask_frame_offset,
    )


def test_default_mask_offset_matches_source_image_numbering(tmp_path: Path) -> None:
    assert TrigsColmapArgs.__dataclass_fields__["mask_frame_offset"].default == 0

    masks = _loader(tmp_path, source_start_frame=1)
    assert masks._mask_path(0, 0) == tmp_path / "00" / "000001.png"
    assert masks._mask_path(119, 0) == tmp_path / "00" / "000120.png"


def test_explicit_offset_supports_legacy_zero_based_source_range(tmp_path: Path) -> None:
    masks = _loader(tmp_path, source_start_frame=0, mask_frame_offset=1)
    assert masks._mask_path(0, 0) == tmp_path / "00" / "000001.png"
    assert masks._mask_path(119, 0) == tmp_path / "00" / "000120.png"


def test_boundary_validation_reports_frame_mapping_before_training(
    tmp_path: Path,
) -> None:
    camera_dir = tmp_path / "00"
    camera_dir.mkdir()
    (camera_dir / "000001.png").touch()
    (camera_dir / "000120.png").touch()

    masks = _loader(tmp_path, source_start_frame=1, mask_frame_offset=1)
    with pytest.raises(FileNotFoundError, match=r"000121\.png"):
        masks.validate_boundaries(120)


def test_foreground_loader_uses_source_numbering_and_rgba_png(tmp_path: Path) -> None:
    camera_dir = tmp_path / "00"
    camera_dir.mkdir()
    expected = np.zeros((4, 5, 3), dtype=np.uint8)
    expected[..., 0] = 17
    alpha = np.arange(20, dtype=np.uint8).reshape(4, 5)
    rgba = np.dstack((expected, alpha))
    Image.fromarray(rgba, mode="RGBA").save(camera_dir / "000001.png")

    images = _LazyGTImages(
        images_root=tmp_path,
        cam_folders_by_pos=["00"],
        source_start_frame=1,
        target_hw_by_pos=[(4, 5)],
        image_pattern="{frame:06d}.png",
    )
    images.validate_boundaries(1)

    got = images[0][0].numpy()
    np.testing.assert_array_equal(got, expected)


def test_foreground_loader_validates_last_frame(tmp_path: Path) -> None:
    camera_dir = tmp_path / "00"
    camera_dir.mkdir()
    Image.new("RGB", (5, 4)).save(camera_dir / "000001.png")
    images = _LazyGTImages(
        images_root=tmp_path,
        cam_folders_by_pos=["00"],
        source_start_frame=1,
        target_hw_by_pos=[(4, 5)],
    )

    with pytest.raises(FileNotFoundError, match=r"000002\.png"):
        images.validate_boundaries(2)


def _distorted_camera_warp():
    """A warp built by _raw_warps_by_pos from maps/ROI exactly as
    FreeTimeParser._setup_undistortion builds them."""
    import cv2

    from trigs.cli.train_from_colmap import _raw_warps_by_pos

    W, H = 96, 64
    K = np.array([[80.0, 0.0, 47.0], [0.0, 80.0, 33.0], [0.0, 0.0, 1.0]])
    dist = np.array([0.08, -0.02, 0.001, 0.0])
    K_undist, roi = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0)
    mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_undist, (W, H), cv2.CV_32FC1)
    parser = SimpleNamespace(
        camera_ids=[7, 9], params_dict={7: dist, 9: np.zeros(0)},
        mapx_dict={7: mapx}, mapy_dict={7: mapy}, roi_undist_dict={7: list(roi)},
    )
    warps = _raw_warps_by_pos(parser, [1, 0])
    assert warps[0] is None  # undistorted camera: loaded as-is, like the dataset
    return warps[1], (mapx, mapy, tuple(int(v) for v in roi)), (W, H)


def test_raw_warp_matches_dataset_grid_sample() -> None:
    """Masks/foregrounds must land on exactly the pixels of the dataset's RGB."""
    import torch
    import torch.nn.functional as F

    from trigs.cli.train_from_colmap import _warp_raw_like_dataset

    warp, (mapx, mapy, (x, y, w, h)), (W, H) = _distorted_camera_warp()
    raw = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)

    # FreeTimeDataset RGB path: grid_sample(align_corners=True) on the parser maps, .byte(), ROI crop.
    grid = torch.stack((
        2.0 * torch.from_numpy(mapx) / (W - 1) - 1.0,
        2.0 * torch.from_numpy(mapy) / (H - 1) - 1.0,
    ), dim=-1)
    img = torch.from_numpy(raw).permute(2, 0, 1).float().unsqueeze(0)
    ds = F.grid_sample(img, grid.unsqueeze(0), mode="bilinear", padding_mode="zeros",
                       align_corners=True)[0].byte().permute(1, 2, 0).numpy()[y:y + h, x:x + w]

    np.testing.assert_array_equal(_warp_raw_like_dataset(raw, warp), ds)
    np.testing.assert_array_equal(_warp_raw_like_dataset(raw[..., 0], warp), ds[..., 0])


def test_mask_and_foreground_loaders_undistort_raw_frames(tmp_path: Path) -> None:
    from trigs.cli.train_from_colmap import _warp_raw_like_dataset

    warp, (_, _, (_, _, w, h)), (W, H) = _distorted_camera_warp()
    raw_mask = np.zeros((H, W), dtype=np.uint8)
    raw_mask[6:18, 70:88] = 255  # off-centre, where distortion moves pixels most
    raw_fg = np.dstack([raw_mask] * 3)
    for root, arr in (("masks", raw_mask), ("fg", raw_fg)):
        (tmp_path / root / "00").mkdir(parents=True)
        Image.fromarray(arr).save(tmp_path / root / "00" / "000001.png")

    class _Dataset:
        pairs = [(0, 0)]

        def __getitem__(self, _):
            return {"image": np.zeros((h, w, 3), dtype=np.uint8)}

    masks = _LazyGTMasks(
        dataset=_Dataset(), n_cameras=1, orig_to_pos={0: 0},
        masks_root=tmp_path / "masks", cam_folders_by_pos=["00"],
        source_start_frame=1, raw_warps_by_pos=[warp],
    )
    fgs = _LazyGTImages(
        images_root=tmp_path / "fg", cam_folders_by_pos=["00"], source_start_frame=1,
        target_hw_by_pos=[(h, w)], image_pattern="{frame:06d}.png", raw_warps_by_pos=[warp],
    )
    want = _warp_raw_like_dataset(raw_mask, warp)
    np.testing.assert_array_equal(masks[0][0].numpy()[..., 0], want)
    np.testing.assert_array_equal(fgs[0][0].numpy()[..., 0], want)

    # The old resize-only path puts the region somewhere else.
    resized = np.asarray(Image.fromarray(raw_mask).resize((w, h), Image.Resampling.BILINEAR))

    def centroid(m):
        ys, xs = np.nonzero(m > 127)
        return np.array([xs.mean(), ys.mean()])

    assert np.linalg.norm(centroid(want) - centroid(resized)) > 1.0


def test_raw_warp_reuses_parser_grid() -> None:
    """On a training box the parser's grid already lives on the GPU; reuse it."""
    import torch

    from trigs.cli.train_from_colmap import _raw_warps_by_pos

    grid = torch.zeros(4, 6, 2)
    parser = SimpleNamespace(
        camera_ids=[7], params_dict={7: np.ones(4)}, grid_dict={7: grid},
        mapx_dict={7: np.zeros((4, 6), np.float32)}, mapy_dict={7: np.zeros((4, 6), np.float32)},
        roi_undist_dict={7: [1, 0, 5, 4]},
    )
    (got, roi), = _raw_warps_by_pos(parser, [0])
    assert got is grid and roi == (1, 0, 5, 4)
