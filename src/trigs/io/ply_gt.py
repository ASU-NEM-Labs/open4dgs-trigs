"""On-demand ground-truth rendering for the PLY input path.

``TrigsTrainer._render_gt`` rasterizes every frame from every camera up front
and holds the result as uint8 in CPU RAM.  That cache is
``n_frames * n_cameras * H * W * 4`` bytes, so for a 781-frame sequence it is
about five times larger than anything the trainer reads at once: chunked
training only ever touches the ~150 frames resident in the current window.
Resolution therefore ends up capped by sequence length rather than by
available memory, which starves the supervision of exactly the detail the
source PLYs contain.

This module renders a frame's views the first time they are asked for and
keeps a bounded LRU of recent frames.  The trainer only ever reaches GT as
``gt[frame_idx][cam_idx]``, so :class:`PlyGtStore` needs no cooperation from
it -- the views below satisfy that access pattern and nothing else.  Other
input paths keep passing plain dicts and are unaffected.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, List, Sequence, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..core.camera import CameraConfig
    from ..core.transforms import TransformConfig

_FrameEntry = Tuple[List[Tensor], List[Tensor]]


class _PlyGtView:
    """Read-only ``gt[frame_idx][cam_idx]`` facade over one slot of the store.

    Images and alphas come from a single rasterization, so both views share a
    store rather than each triggering its own render.
    """

    __slots__ = ("_store", "_slot")

    def __init__(self, store: "PlyGtStore", slot: int) -> None:
        self._store = store
        self._slot = int(slot)

    def __getitem__(self, frame_idx: int) -> List[Tensor]:
        return self._store.frame(int(frame_idx))[self._slot]


class PlyGtStore:
    """Render dense-PLY ground truth on demand, caching recent frames.

    Args:
        ply_files: Per-frame dense PLYs, indexed by frame number.
        cameras: Camera rig shared by training and GT.
        transform_config: Scene transform applied to every loaded PLY.
        device: Device used for rasterization; results are moved to CPU.
        capacity_frames: Frames retained. Must cover the frame span a chunk
            supervises, or every step evicts the frame the next step needs.
        verbose: Log the cache footprint and, on close, the hit rate.
    """

    def __init__(
        self,
        ply_files: Sequence[Path],
        cameras: "CameraConfig",
        transform_config: "TransformConfig",
        *,
        device: str,
        capacity_frames: int,
        verbose: bool = True,
    ) -> None:
        if int(capacity_frames) < 1:
            raise ValueError("capacity_frames must be >= 1")
        self._ply_files = list(ply_files)
        self._cameras = cameras
        self._transform = transform_config
        self._device = str(device)
        self._capacity = int(capacity_frames)
        self._verbose = bool(verbose)
        self._lru: "OrderedDict[int, _FrameEntry]" = OrderedDict()
        self._renderer = None
        self._hits = 0
        self._misses = 0

    @property
    def images(self) -> _PlyGtView:
        """View satisfying ``gt_images_per_frame[frame][cam]``."""
        return _PlyGtView(self, 0)

    @property
    def alphas(self) -> _PlyGtView:
        """View satisfying ``gt_alphas_per_frame[frame][cam]``."""
        return _PlyGtView(self, 1)

    def resident_bytes(self) -> int:
        """Upper bound on retained bytes: 3 uint8 for RGB + 1 for alpha."""
        n_views = int(self._cameras.viewmats.shape[0])
        per_view = sum(
            int(self._cameras.width_for(v)) * int(self._cameras.height_for(v))
            for v in range(n_views)
        )
        return self._capacity * per_view * 4

    @torch.no_grad()
    def _render_frame(self, frame_idx: int) -> _FrameEntry:
        from ..core.renderer import Renderer
        from .ply import load_ply

        if self._renderer is None:
            self._renderer = Renderer(device=self._device)

        gs = load_ply(self._ply_files[frame_idx], device=self._device)
        gs = self._transform.apply(gs)
        result = self._renderer.render(
            gs, self._cameras, return_alphas=True, show_progress=False,
        )
        # uint8 on CPU, matching _render_gt so the trainer's .to(device) and
        # .div(255) downstream stay valid.
        images = [
            img.clamp(0, 1).mul(255).to(torch.uint8).cpu() for img in result.images
        ]
        alphas = [
            a.clamp(0, 1).mul(255).to(torch.uint8).cpu() for a in result.alphas
        ]
        del gs, result
        return images, alphas

    def frame(self, frame_idx: int) -> _FrameEntry:
        """Return ``(images, alphas)`` for one frame, rendering on a miss."""
        entry = self._lru.get(frame_idx)
        if entry is not None:
            self._lru.move_to_end(frame_idx)
            self._hits += 1
            return entry

        self._misses += 1
        entry = self._render_frame(frame_idx)
        self._lru[frame_idx] = entry
        while len(self._lru) > self._capacity:
            self._lru.popitem(last=False)
        return entry

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / total) if total else 0.0,
            "resident_frames": len(self._lru),
            "capacity_frames": self._capacity,
        }
