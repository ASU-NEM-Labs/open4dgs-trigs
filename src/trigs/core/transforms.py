"""Scene transform configuration for CLI commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gsply import GSTensor


@dataclass
class TransformConfig:
    """Scene transform (consistent across all frames).

    Provides a CLI-friendly interface for specifying rotation and translation
    as comma-separated strings.
    """

    rotation: str = "0,0,0"
    """Scene rotation as 'rx,ry,rz' Euler angles in degrees."""

    translation: str = "0,0,0"
    """Scene translation as 'tx,ty,tz'."""

    def parse(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Parse rotation and translation strings.

        Returns:
            Tuple of (rotation, translation) as float tuples
        """
        rx, ry, rz = map(float, self.rotation.split(","))
        tx, ty, tz = map(float, self.translation.split(","))
        return (rx, ry, rz), (tx, ty, tz)

    def apply(self, gs: "GSTensor") -> "GSTensor":
        """Apply transform to GSTensor if non-zero.

        Args:
            gs: GSTensor to transform

        Returns:
            Transformed GSTensor (or original if no transform needed)
        """
        rotation, translation = self.parse()
        if rotation != (0, 0, 0) or translation != (0, 0, 0):
            from ._scene_transform import transform
            return transform(gs, rotation=rotation, translation=translation)
        return gs

    @property
    def is_identity(self) -> bool:
        """Check if this is an identity transform (no-op)."""
        rotation, translation = self.parse()
        return rotation == (0, 0, 0) and translation == (0, 0, 0)
