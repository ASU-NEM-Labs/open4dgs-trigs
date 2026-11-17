"""Camera configuration for CLI commands."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.camera import CameraConfig
    from gsply import GSTensor


@dataclass
class CameraCLIConfig:
    """Camera parameters for CLI (reused across all commands).

    This is a CLI-friendly wrapper around CameraConfig that provides
    sensible defaults and a build() method to create the actual camera.
    """

    longitude_steps: int = 12
    """Number of cameras around Y axis."""

    latitude_steps: int = 10
    """Number of latitude levels."""

    longitude_min: float = 0.0
    """Minimum longitude in degrees (0 = +Z direction)."""

    longitude_max: float = 360.0
    """Maximum longitude in degrees."""

    latitude_min: float = -60.0
    """Minimum latitude in degrees."""

    latitude_max: float = 60.0
    """Maximum latitude in degrees."""

    fov: float = 60.0
    """Field of view in degrees."""

    width: int = 512
    """Image width."""

    height: int = 512
    """Image height."""

    margin: float = 1.2
    """Distance margin multiplier (1.2 = 20% extra distance from scene)."""

    def build(self, gs: "GSTensor") -> "CameraConfig":
        """Convert CLI params to actual CameraConfig.

        Args:
            gs: GSTensor to compute camera sphere radius from

        Returns:
            CameraConfig with viewmats and intrinsics
        """
        from ..core.camera import CameraConfig

        return CameraConfig.from_sphere(gs, **asdict(self))

    @property
    def n_cameras(self) -> int:
        """Total number of cameras."""
        return self.longitude_steps * self.latitude_steps
