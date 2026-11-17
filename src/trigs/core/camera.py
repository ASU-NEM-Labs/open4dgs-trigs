"""CameraConfig: Camera configuration for rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from gsply import GSTensor


@dataclass
class CameraConfig:
    """Camera configuration for rendering.

    Contains view matrices, intrinsics, and image dimensions.
    Width/height may differ per camera.
    """

    viewmats: Tensor  # (N, 4, 4) view matrices
    Ks: Tensor  # (N, 3, 3) intrinsic matrices
    widths: Tensor  # (N,) per-camera image widths
    heights: Tensor  # (N,) per-camera image heights

    # (N,) per-camera rolling-shutter readout direction, +1 / -1 / 0.
    # The rig records landscape and the extract rotates +/-90 per camera, so
    # readout runs along image COLUMNS with a per-camera sign. 0 disables the
    # correction for that camera. None means the rig is global-shutter (or the
    # map was never supplied) and no correction is applied anywhere.
    rs_signs: Tensor | None = None

    # Optional raw-sensor mapping for row-axis rolling shutter (see trigs.capture_timing): the camera's original
    # intrinsics scaled by 1/factor (N, 3, 3), OPENCV distortion (N, 5: k1, k2, p1, p2, k3) and raw image size at
    # that scale (N, 2: height, width). None = use delivered image coordinates (legacy behaviour).
    rs_sensor_K: Tensor | None = None
    rs_sensor_dist: Tensor | None = None
    rs_sensor_hw: Tensor | None = None

    def __post_init__(self):
        """Validate tensor shapes."""
        n = self.viewmats.shape[0]
        assert self.viewmats.shape == (n, 4, 4), f"viewmats shape {self.viewmats.shape}"
        assert self.Ks.shape == (n, 3, 3), f"Ks shape {self.Ks.shape}"
        assert self.widths.shape == (n,), f"widths shape {self.widths.shape}"
        assert self.heights.shape == (n,), f"heights shape {self.heights.shape}"
        if self.rs_signs is not None:
            assert self.rs_signs.shape == (n,), f"rs_signs shape {self.rs_signs.shape}"
            self.rs_signs = self.rs_signs.to(
                device=self.viewmats.device, dtype=self.viewmats.dtype
            )
        if self.widths.dtype not in (torch.int32, torch.int64):
            self.widths = self.widths.to(torch.int64)
        if self.heights.dtype not in (torch.int32, torch.int64):
            self.heights = self.heights.to(torch.int64)
        if torch.any(self.widths <= 0) or torch.any(self.heights <= 0):
            raise ValueError("Camera widths/heights must be positive.")

    def __len__(self) -> int:
        """Return number of cameras."""
        return self.viewmats.shape[0]

    @property
    def device(self) -> torch.device:
        """Return device of tensors."""
        return self.viewmats.device

    def to(self, device: str | torch.device) -> "CameraConfig":
        """Move all tensors to device."""
        return CameraConfig(
            viewmats=self.viewmats.to(device),
            Ks=self.Ks.to(device),
            widths=self.widths.to(device),
            heights=self.heights.to(device),
            rs_signs=None if self.rs_signs is None else self.rs_signs.to(device),
            rs_sensor_K=None if self.rs_sensor_K is None else self.rs_sensor_K.to(device),
            rs_sensor_dist=None if self.rs_sensor_dist is None else self.rs_sensor_dist.to(device),
            rs_sensor_hw=None if self.rs_sensor_hw is None else self.rs_sensor_hw.to(device),
        )

    def width_for(self, cam_idx: int) -> int:
        return int(self.widths[cam_idx].item())

    def height_for(self, cam_idx: int) -> int:
        return int(self.heights[cam_idx].item())

    def size_for(self, cam_idx: int) -> tuple[int, int]:
        return self.width_for(cam_idx), self.height_for(cam_idx)

    @property
    def width(self) -> int:
        """Legacy single-width accessor; only valid when all widths match."""
        if not torch.all(self.widths == self.widths[0]):
            raise ValueError("CameraConfig has per-camera widths; use width_for(cam_idx).")
        return int(self.widths[0].item())

    @property
    def height(self) -> int:
        """Legacy single-height accessor; only valid when all heights match."""
        if not torch.all(self.heights == self.heights[0]):
            raise ValueError("CameraConfig has per-camera heights; use height_for(cam_idx).")
        return int(self.heights[0].item())

    @classmethod
    def from_matrices(
        cls,
        viewmats: Tensor,
        Ks: Tensor,
        width: int | None = None,
        height: int | None = None,
        widths: Sequence[int] | Tensor | None = None,
        heights: Sequence[int] | Tensor | None = None,
        rs_signs: Sequence[float] | Tensor | None = None,
        rs_sensor_K: Tensor | None = None,
        rs_sensor_dist: Tensor | None = None,
        rs_sensor_hw: Tensor | None = None,
    ) -> "CameraConfig":
        """Create CameraConfig from pre-computed matrices.

        Args:
            viewmats: (N, 4, 4) view matrices
            Ks: (N, 3, 3) intrinsic matrices
            width: Shared image width (legacy path)
            height: Shared image height (legacy path)
            widths: Per-camera image widths
            heights: Per-camera image heights
            rs_signs: Per-camera rolling-shutter readout direction (+1/-1/0)
            rs_sensor_K, rs_sensor_dist, rs_sensor_hw: optional raw-sensor mapping (row-axis rolling shutter)

        Returns:
            CameraConfig instance
        """
        n = int(viewmats.shape[0])
        if widths is None:
            if width is None:
                raise ValueError("Either `widths` or `width` must be provided.")
            widths_t = torch.full((n,), int(width), dtype=torch.int64, device=viewmats.device)
        elif torch.is_tensor(widths):
            widths_t = widths.to(device=viewmats.device, dtype=torch.int64)
        else:
            widths_t = torch.tensor(list(widths), dtype=torch.int64, device=viewmats.device)

        if heights is None:
            if height is None:
                raise ValueError("Either `heights` or `height` must be provided.")
            heights_t = torch.full((n,), int(height), dtype=torch.int64, device=viewmats.device)
        elif torch.is_tensor(heights):
            heights_t = heights.to(device=viewmats.device, dtype=torch.int64)
        else:
            heights_t = torch.tensor(list(heights), dtype=torch.int64, device=viewmats.device)

        if int(widths_t.shape[0]) != n or int(heights_t.shape[0]) != n:
            raise ValueError(
                f"Per-camera sizes must have length {n}; got widths={tuple(widths_t.shape)}, "
                f"heights={tuple(heights_t.shape)}",
            )

        rs_t = None
        if rs_signs is not None:
            rs_t = (
                rs_signs.to(device=viewmats.device, dtype=viewmats.dtype)
                if torch.is_tensor(rs_signs)
                else torch.tensor(
                    list(rs_signs), dtype=viewmats.dtype, device=viewmats.device
                )
            )
            if int(rs_t.shape[0]) != n:
                raise ValueError(
                    f"rs_signs must have length {n}; got {tuple(rs_t.shape)}"
                )

        return cls(
            viewmats=viewmats,
            Ks=Ks,
            widths=widths_t,
            heights=heights_t,
            rs_signs=rs_t,
            rs_sensor_K=rs_sensor_K,
            rs_sensor_dist=rs_sensor_dist,
            rs_sensor_hw=rs_sensor_hw,
        )

    @classmethod
    def from_sphere(
        cls,
        gs: "GSTensor",
        longitude_steps: int = 15,
        latitude_steps: int = 10,
        longitude_min: float = 0.0,
        longitude_max: float = 360.0,
        latitude_min: float = -20.0,
        latitude_max: float = 60.0,
        fov: float = 60.0,
        width: int = 512,
        height: int = 512,
        margin: float = 1.2,
    ) -> "CameraConfig":
        """Generate cameras on a sphere around the scene.

        Cameras are distributed on a lat/lon grid looking at the scene center.

        Args:
            gs: GSTensor to compute bounding sphere from
            longitude_steps: Number of cameras around Y axis
            latitude_steps: Number of latitude levels
            longitude_min: Minimum longitude in degrees (0 = +Z direction)
            longitude_max: Maximum longitude in degrees (360 = full circle)
            latitude_min: Minimum latitude in degrees (-90 to 90)
            latitude_max: Maximum latitude in degrees (-90 to 90)
            fov: Field of view in degrees
            width: Image width
            height: Image height
            margin: Distance margin multiplier (1.2 = 20% extra distance)

        Returns:
            CameraConfig with generated cameras
        """
        viewmats, Ks = generate_sphere_cameras(
            means=gs.means,
            scales=gs.scales,
            longitude_steps=longitude_steps,
            latitude_steps=latitude_steps,
            longitude_min=longitude_min,
            longitude_max=longitude_max,
            latitude_min=latitude_min,
            latitude_max=latitude_max,
            fov=fov,
            width=width,
            height=height,
            margin=margin,
        )

        n = int(viewmats.shape[0])
        widths = torch.full((n,), int(width), dtype=torch.int64, device=viewmats.device)
        heights = torch.full((n,), int(height), dtype=torch.int64, device=viewmats.device)
        return cls(viewmats=viewmats, Ks=Ks, widths=widths, heights=heights)


def _batched_viewmats(
    positions: Tensor, target: Tensor, pole_threshold: float = 0.99
) -> Tensor:
    """Compute view matrices for multiple camera positions.

    Args:
        positions: (N, 3) camera positions
        target: (3,) look-at target point
        pole_threshold: threshold for switching up vector near poles

    Returns:
        (N, 4, 4) view matrices
    """
    n = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    # Look direction: target - position, then normalize
    lookdir = target.unsqueeze(0) - positions
    vec2 = lookdir / torch.linalg.norm(lookdir, dim=1, keepdim=True)

    # Up vector: use Y for most, Z near poles
    up = torch.zeros(n, 3, device=device, dtype=dtype)
    up[:, 1] = 1.0

    # Near poles, use Z-up to avoid gimbal lock
    dir_y = -vec2[:, 1]
    near_pole = torch.abs(dir_y) > pole_threshold
    up[near_pole, 1] = 0.0
    up[near_pole, 2] = 1.0

    # Build orthonormal basis
    vec0 = torch.linalg.cross(up, vec2)
    vec0 = vec0 / torch.linalg.norm(vec0, dim=1, keepdim=True)
    vec1 = torch.linalg.cross(vec2, vec0)

    # Build rotation matrices (N, 3, 3)
    R = torch.stack([vec0, vec1, vec2], dim=1)

    # Translation: t = -R @ position
    t = -torch.bmm(R, positions.unsqueeze(2)).squeeze(2)

    # Build 4x4 view matrices
    viewmats = torch.zeros(n, 4, 4, device=device, dtype=dtype)
    viewmats[:, :3, :3] = R
    viewmats[:, :3, 3] = t
    viewmats[:, 3, 3] = 1.0

    return viewmats


def generate_sphere_cameras(
    means: Tensor,
    scales: Tensor,
    longitude_steps: int = 12,
    latitude_steps: int = 5,
    longitude_min: float = 0.0,
    longitude_max: float = 360.0,
    latitude_min: float = 0.0,
    latitude_max: float = 80.0,
    fov: float = 60.0,
    width: int = 800,
    height: int = 800,
    margin: float = 1.2,
) -> Tuple[Tensor, Tensor]:
    """Generate cameras on a sphere using latitude/longitude grid.

    Assumes the scene is already normalized:
    - min_y = 0 (bottom at ground level)
    - centroid_xz = (0, 0) (centered horizontally)

    Coordinate system (Y-up):
    - Latitude 90° = directly above (north pole)
    - Latitude 0° = equator (horizontal)
    - Latitude -90° = directly below (south pole)
    - Longitude 0° = +Z direction
    - Longitude 90° = +X direction

    Args:
        means: (N, 3) Gaussian positions (already normalized)
        scales: (N, 3) Gaussian scales (log-space)
        longitude_steps: Number of cameras around Y axis
        latitude_steps: Number of cameras from latitude_min to latitude_max
        longitude_min: Minimum longitude in degrees (0 = +Z direction)
        longitude_max: Maximum longitude in degrees (360 = full circle)
        latitude_min: Minimum latitude in degrees (0 = equator, -90 = bottom)
        latitude_max: Maximum latitude in degrees (90 = top)
        fov: Field of view in degrees
        width: Image width
        height: Image height
        margin: Margin multiplier for camera distance (default 1.2 = 20% margin)

    Returns:
        viewmats: (N, 4, 4) view matrices
        Ks: (N, 3, 3) intrinsic matrices
    """
    device = means.device
    dtype = means.dtype

    # Compute scene bounds
    min_bounds = means.min(dim=0).values
    max_bounds = means.max(dim=0).values
    scene_height = (max_bounds[1] - min_bounds[1]).item()

    # Sphere center at middle of scene
    sphere_center = torch.tensor(
        [0.0, scene_height / 2, 0.0], device=device, dtype=dtype
    )

    # Account for Gaussian size when computing bounding radius
    gaussian_radii = torch.exp(scales).max(dim=1).values
    distances = torch.linalg.norm(means - sphere_center, dim=1)
    bounding_radius = (distances + gaussian_radii).max().item()

    # Camera distance to fit scene in FOV with margin
    fov_rad = math.radians(fov)
    radius = margin * bounding_radius / math.tan(fov_rad / 2)

    # Generate latitude/longitude grid
    lat_min_rad = math.radians(latitude_min)
    lat_max_rad = math.radians(latitude_max)
    lon_min_rad = math.radians(longitude_min)
    lon_max_rad = math.radians(longitude_max)

    positions = []

    for lat_idx in range(latitude_steps):
        # Latitude from min to max
        if latitude_steps > 1:
            t = lat_idx / (latitude_steps - 1)
        else:
            t = 0.5
        lat = lat_min_rad + t * (lat_max_rad - lat_min_rad)

        for lon_idx in range(longitude_steps):
            # Longitude from min to max (exclusive of max if full 360°)
            if longitude_steps > 1:
                # If full 360°, don't include the endpoint (it wraps)
                if abs(longitude_max - longitude_min - 360.0) < 1e-6:
                    s = lon_idx / longitude_steps
                else:
                    s = lon_idx / (longitude_steps - 1)
            else:
                s = 0.5
            lon = lon_min_rad + s * (lon_max_rad - lon_min_rad)

            # Convert spherical to Cartesian (Y-up)
            x = math.cos(lat) * math.sin(lon)
            y = math.sin(lat)
            z = math.cos(lat) * math.cos(lon)

            pos = sphere_center + torch.tensor(
                [x * radius, y * radius, z * radius],
                device=device, dtype=dtype
            )
            positions.append(pos)

    positions_t = torch.stack(positions)
    n_views = positions_t.shape[0]

    # Compute view matrices
    viewmats = _batched_viewmats(positions_t, sphere_center)

    # Intrinsic matrix
    focal = 0.5 * width / math.tan(0.5 * fov_rad)
    K = torch.tensor([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=dtype)
    Ks = K.unsqueeze(0).expand(n_views, -1, -1)

    return viewmats, Ks
