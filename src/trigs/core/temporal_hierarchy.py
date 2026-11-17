from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


def _row_count(params) -> int:
    return int(params["means"].shape[0])


def _static_mask(params) -> Tensor:
    n = _row_count(params)
    device = params["means"].device
    if "is_static" not in params:
        return torch.zeros(n, device=device, dtype=torch.bool)
    return params["is_static"].detach().reshape(-1) > 0.5


def _radius_scale(opacity_threshold: float) -> float:
    threshold = min(max(float(opacity_threshold), 1e-8), 1.0 - 1e-8)
    return math.sqrt(-2.0 * math.log(threshold))


def _influence_radius(
    log_durations: Tensor,
    opacity_threshold: float,
    extra_margin_s: float,
) -> Tensor:
    sigma = log_durations.detach().exp().reshape(-1).clamp_min(1e-6)
    return sigma * _radius_scale(opacity_threshold) + float(extra_margin_s)


def _indexed_influence_radius(
    log_durations: Tensor,
    idx: Tensor,
    opacity_threshold: float,
    extra_margin_s: float,
) -> Tensor:
    log_sigma = log_durations.detach().reshape(-1)[idx]
    sigma = log_sigma.exp().clamp_min(1e-6)
    return sigma * _radius_scale(opacity_threshold) + float(extra_margin_s)


@torch.no_grad()
def direct_temporal_active_idx(
    params,
    t: float | Tensor,
    *,
    opacity_threshold: float = 0.05,
    extra_margin_s: float = 0.0,
    include_static: bool = True,
    base_mask: Optional[Tensor] = None,
) -> Tensor:
    """Exact O(N) reference query."""
    device = params["means"].device
    t_tensor = torch.as_tensor(
        t,
        device=device,
        dtype=params["times"].dtype,
    ).reshape(())

    mu = params["times"].detach().reshape(-1)
    radius = _influence_radius(
        params["log_durations"],
        opacity_threshold,
        extra_margin_s,
    )
    active = torch.abs(t_tensor - mu) <= radius

    if include_static:
        active |= _static_mask(params)

    if base_mask is not None:
        active &= base_mask.detach().reshape(-1).to(
            device=device,
            dtype=torch.bool,
        )

    return torch.nonzero(active, as_tuple=False).reshape(-1)


@dataclass
class TemporalGaussianHierarchy:
    """
    Exact dynamic hierarchy with localized staleness handling.

    Key properties, in decreasing order of importance for making the index
    actually pay off on learned temporal supports:

    1. Sequence-bounds clipping. Influence intervals are intersected with
       [t_min - pad, t_max + pad] before root sizing and segment assignment.
       Queries only occur inside the padded bounds, so an interval's clipped
       form contains every queryable t its raw form contains. This stops a
       handful of wide-sigma rows from inflating the dyadic root far beyond
       the captured sequence (previously 80 s for an 11.65 s clip).

    2. Coverage promotion. A dynamic row whose clipped interval covers at
       least ``global_coverage_fraction`` of the sequence is placed in the
       global bucket. Such rows would be retrieved at nearly every timestamp
       anyway; keeping them out of the tree preserves fine segments for
       compactly supported rows and keeps the root small.

    3. Bounded overflow instead of a global drift envelope. Per-row endpoint
       drift is compared against a fixed slack (a fraction of the finest
       segment). Rows exceeding the slack join an overflow set that is always
       included in the candidates. The query window is the constant
       [t - slack, t + slack], never the worst-case drift, so one runaway row
       costs one extra candidate rather than a full-tree scan. A rebuild is
       requested when the overflow set exceeds a fraction of the dynamic rows.

    4. Optional half-shifted offset grids per level, in the spirit of
       LongVolCap's per-level corrected offsets. Each level carries a second
       grid offset by half a segment; an interval that straddles a boundary
       in one grid frequently fits the other, avoiding promotion to a
       coarser level.

    Exactness argument for the query: for a non-overflow row, both cached
    endpoints are within ``drift_slack`` of the current endpoints. If its
    current interval contains t, the cached interval must intersect
    [t - slack, t + slack], so its cached segment is visited. Overflow rows
    are unconditionally candidates. Static and coverage-promoted rows are
    global. The exact current-parameter test then removes false positives.
    As before, this requires every temporally-updatable row to be passed to
    update_changed_rows() after each optimizer step (use beta1=0 for the
    temporal optimizer so zero-gradient rows cannot move).
    """

    levels: int = 9
    root_segment_s: float = 10.0
    opacity_threshold: float = 0.05
    extra_margin_s: float = 0.0
    include_static: bool = True

    # Sequence bounds for interval clipping. None disables clipping and
    # reproduces the old unbounded-root behavior.
    t_min: Optional[float] = None
    t_max: Optional[float] = None
    # Queries slightly outside [t_min, t_max] (e.g. learned camera time
    # offsets) remain exact within this pad.
    bounds_pad_s: float = 0.05

    # Dynamic rows whose clipped interval covers at least this fraction of
    # the padded sequence go to the global bucket. <= 0 or >= 1 disables.
    global_coverage_fraction: float = 0.5

    # Drift slack as a fraction of the finest segment. Rows drifting past
    # this join the overflow set instead of widening the query window.
    drift_slack_finest: float = 0.5

    # Second half-shifted grid per level to reduce boundary straddling.
    offset_grids: bool = True

    def __post_init__(self) -> None:
        self.global_idx: Optional[Tensor] = None
        # level -> grid -> rows / lookup
        self.level_rows: list[list[Tensor]] = []
        self.level_lookup: list[list[Dict[int, Tuple[int, int]]]] = []

        self.allowed_mask: Optional[Tensor] = None
        self.build_lo: Optional[Tensor] = None
        self.build_hi: Optional[Tensor] = None
        self.overflow_mask: Optional[Tensor] = None

        self.n_rows: int = 0
        self.n_dynamic: int = 0
        self.device: Optional[torch.device] = None
        self.build_step: int = -1
        self.rebuild_count: int = 0
        self.valid: bool = False

        self.origin_s: float = 0.0
        self.effective_root_segment_s: float = float(self.root_segment_s)
        self.max_endpoint_drift_s: float = 0.0
        self.overflow_count: int = 0

        self.last_candidate_count: int = 0
        self.last_active_count: int = 0
        self.last_segments_visited: int = 0

    def invalidate(self) -> None:
        self.valid = False
        self.build_step = -1

    @property
    def finest_segment_s(self) -> float:
        return float(self.effective_root_segment_s) / float(
            2 ** max(int(self.levels) - 1, 0)
        )

    @property
    def drift_slack_s(self) -> float:
        return max(float(self.drift_slack_finest), 0.0) * self.finest_segment_s

    @property
    def dirty_count(self) -> int:
        # Old TensorBoard plumbing maps "dirty" onto the overflow set.
        return int(self.overflow_count)

    @property
    def dirty_fraction(self) -> float:
        return float(self.overflow_count) / float(max(self.n_dynamic, 1))

    @staticmethod
    def _next_dyadic_multiple(value: float, base: float) -> float:
        value = max(float(value), 1e-9)
        base = max(float(base), 1e-9)
        if value <= base:
            return base
        exponent = math.ceil(math.log2(value / base))
        return base * float(2 ** exponent)

    def _clip_bounds(self) -> Optional[Tuple[float, float]]:
        if self.t_min is None or self.t_max is None:
            return None
        pad = max(float(self.bounds_pad_s), 0.0)
        lo = float(self.t_min) - pad
        hi = float(self.t_max) + pad
        if hi <= lo:
            hi = lo + 1e-6
        return lo, hi

    @torch.no_grad()
    def rebuild(
        self,
        params,
        *,
        base_mask: Optional[Tensor] = None,
        step: int = 0,
    ) -> None:
        """Vectorized full rebuild. Loops over levels, never over rows."""
        device = params["means"].device
        n = _row_count(params)

        if int(self.levels) <= 0:
            raise ValueError("levels must be > 0")
        if float(self.root_segment_s) <= 0:
            raise ValueError("root_segment_s must be > 0")

        self.device = device
        self.n_rows = n

        mu = params["times"].detach().reshape(-1)
        radius = _influence_radius(
            params["log_durations"],
            self.opacity_threshold,
            self.extra_margin_s,
        )
        influence_lo = mu - radius
        influence_hi = mu + radius

        # Clip to the queryable time range. Exact for any query t inside the
        # padded bounds: t in raw interval and t in bounds implies t in the
        # clipped interval.
        clip = self._clip_bounds()
        if clip is not None:
            clip_lo, clip_hi = clip
            influence_lo = influence_lo.clamp(min=clip_lo, max=clip_hi)
            influence_hi = influence_hi.clamp(min=clip_lo, max=clip_hi)

        static = (
            _static_mask(params)
            if self.include_static
            else torch.zeros(n, device=device, dtype=torch.bool)
        )

        if base_mask is None:
            allowed = torch.ones(n, device=device, dtype=torch.bool)
        else:
            allowed = base_mask.detach().reshape(-1).to(
                device=device,
                dtype=torch.bool,
            )
            if int(allowed.numel()) != n:
                raise ValueError(
                    f"base_mask length {allowed.numel()} does not match rows {n}"
                )

        dynamic = allowed & (~static)
        self.n_dynamic = int(dynamic.sum().item())

        # Promote near-persistent dynamic rows to the global bucket before
        # sizing the root, so they cannot distort it and do not populate the
        # coarse levels.
        coverage_global = torch.zeros(n, device=device, dtype=torch.bool)
        frac = float(self.global_coverage_fraction)
        if clip is not None and 0.0 < frac < 1.0:
            span_total = clip[1] - clip[0]
            coverage = (influence_hi - influence_lo) / max(span_total, 1e-9)
            coverage_global = dynamic & (coverage >= frac)

        tree_rows_mask = dynamic & (~coverage_global)
        tree_rows = torch.nonzero(tree_rows_mask, as_tuple=False).reshape(-1)

        # Build one dyadic root containing every tree-bound interval.
        if int(tree_rows.numel()) > 0:
            tree_lo = influence_lo[tree_rows]
            tree_hi = influence_hi[tree_rows]
            lo_min = float(tree_lo.min().item())
            hi_max = float(tree_hi.max().item())
            span = max(hi_max - lo_min, 1e-6)

            effective_root = self._next_dyadic_multiple(
                span * (1.0 + 1e-6),
                float(self.root_segment_s),
            )
            slack = max(effective_root - span, 0.0)
            origin = lo_min - 0.5 * slack
        else:
            effective_root = float(self.root_segment_s)
            origin = 0.0 if clip is None else clip[0]

        self.origin_s = float(origin)
        self.effective_root_segment_s = float(effective_root)

        n_grids = 2 if self.offset_grids else 1

        assigned = torch.zeros(n, device=device, dtype=torch.bool)
        level_rows: list[list[Tensor]] = [
            [
                torch.empty(0, device=device, dtype=torch.long)
                for _ in range(n_grids)
            ]
            for _ in range(int(self.levels))
        ]
        level_lookup: list[list[Dict[int, Tuple[int, int]]]] = [
            [{} for _ in range(n_grids)] for _ in range(int(self.levels))
        ]

        # Fine -> coarse: place each interval in the shortest segment (over
        # both grids) that fully contains it.
        for level in reversed(range(int(self.levels))):
            segment_s = self.effective_root_segment_s / float(2**level)

            for grid in range(n_grids):
                offset = 0.5 * segment_s * grid

                segment_lo = torch.floor(
                    (influence_lo - self.origin_s - offset) / segment_s
                ).to(torch.int64)
                segment_hi = torch.floor(
                    (influence_hi - self.origin_s - offset) / segment_s
                ).to(torch.int64)

                take = tree_rows_mask & (~assigned) & (segment_lo == segment_hi)
                rows = torch.nonzero(take, as_tuple=False).reshape(-1)
                if int(rows.numel()) == 0:
                    continue

                row_segments = segment_lo[rows]
                order = torch.argsort(row_segments)
                rows_sorted = rows[order]
                segments_sorted = row_segments[order]

                unique_segments, counts = torch.unique_consecutive(
                    segments_sorted,
                    return_counts=True,
                )

                lookup: Dict[int, Tuple[int, int]] = {}
                start = 0
                for segment_id, count in zip(
                    unique_segments.cpu().tolist(),
                    counts.cpu().tolist(),
                ):
                    end = start + int(count)
                    lookup[int(segment_id)] = (start, end)
                    start = end

                level_rows[level][grid] = rows_sorted
                level_lookup[level][grid] = lookup
                assigned |= take

        # Global bucket: static rows, coverage-promoted rows, and numerical
        # stragglers.
        global_mask = allowed & (
            static | coverage_global | (tree_rows_mask & (~assigned))
        )

        self.global_idx = torch.nonzero(
            global_mask,
            as_tuple=False,
        ).reshape(-1)
        self.level_rows = level_rows
        self.level_lookup = level_lookup
        self.allowed_mask = allowed

        # Full cached (clipped) endpoints permit exact per-row drift checks.
        self.build_lo = influence_lo.detach().clone()
        self.build_hi = influence_hi.detach().clone()
        self.overflow_mask = torch.zeros(n, device=device, dtype=torch.bool)
        self.overflow_count = 0
        self.max_endpoint_drift_s = 0.0

        self.build_step = int(step)
        self.rebuild_count += 1
        self.valid = True
        self.last_candidate_count = 0
        self.last_active_count = 0
        self.last_segments_visited = 0

    @torch.no_grad()
    def update_changed_rows(self, params, row_idx: Tensor) -> float:
        """
        Compare changed rows' current (clipped) endpoints against the rebuild
        snapshot. Rows drifting past the slack join the overflow set. Call
        after optimizer.step() and post-step clamps; repeated rows are fine.
        """
        if not self.valid:
            return self.max_endpoint_drift_s
        if (
            self.build_lo is None
            or self.build_hi is None
            or self.overflow_mask is None
        ):
            raise RuntimeError("Hierarchy build endpoints are unavailable.")

        idx = row_idx.detach().reshape(-1).to(
            device=self.device,
            dtype=torch.long,
        )
        if int(idx.numel()) == 0:
            return self.max_endpoint_drift_s

        if int(idx.min().item()) < 0 or int(idx.max().item()) >= self.n_rows:
            raise IndexError("update_changed_rows received an out-of-range row")

        if self.allowed_mask is not None:
            idx = idx[self.allowed_mask[idx]]

        if self.include_static and "is_static" in params:
            static = params["is_static"].detach().reshape(-1) > 0.5
            idx = idx[~static[idx]]

        # Rows already in overflow are permanent candidates; skip re-checking.
        idx = idx[~self.overflow_mask[idx]]

        if int(idx.numel()) == 0:
            return self.max_endpoint_drift_s

        mu = params["times"].detach().reshape(-1)[idx]
        radius = _indexed_influence_radius(
            params["log_durations"],
            idx,
            self.opacity_threshold,
            self.extra_margin_s,
        )
        current_lo = mu - radius
        current_hi = mu + radius

        clip = self._clip_bounds()
        if clip is not None:
            current_lo = current_lo.clamp(min=clip[0], max=clip[1])
            current_hi = current_hi.clamp(min=clip[0], max=clip[1])

        drift = torch.maximum(
            (current_lo - self.build_lo[idx]).abs(),
            (current_hi - self.build_hi[idx]).abs(),
        )

        self.max_endpoint_drift_s = max(
            self.max_endpoint_drift_s,
            float(drift.max().item()),
        )

        over = drift > self.drift_slack_s
        if bool(over.any()):
            self.overflow_mask[idx[over]] = True
            self.overflow_count = int(self.overflow_mask.sum().item())

        return self.max_endpoint_drift_s

    # Backward-compatible alias for the first dynamic implementation.
    @torch.no_grad()
    def mark_dirty(self, row_idx: Tensor, params=None) -> int:
        if params is None:
            raise ValueError("mark_dirty now requires params")
        self.update_changed_rows(params, row_idx)
        return int(row_idx.numel())

    def needs_rebuild(
        self,
        *,
        n_rows: int,
        step: int,
        rebuild_every: int,
        overflow_fraction: float = 0.05,
        max_drift_finest_segments: float = 0.0,
        **_ignored,
    ) -> bool:
        if not self.valid:
            return True
        if int(n_rows) != int(self.n_rows):
            return True

        if int(rebuild_every) > 0:
            if int(step) - int(self.build_step) >= int(rebuild_every):
                return True

        if float(overflow_fraction) > 0 and self.n_dynamic > 0:
            if (
                self.overflow_count
                >= float(overflow_fraction) * float(self.n_dynamic)
            ):
                return True

        # Legacy outlier-drift trigger; usually unnecessary with overflow.
        if float(max_drift_finest_segments) > 0:
            drift_limit = (
                float(max_drift_finest_segments) * self.finest_segment_s
            )
            if self.max_endpoint_drift_s >= drift_limit:
                return True

        return False

    @torch.no_grad()
    def query_candidates(self, t: float | Tensor) -> Tensor:
        """
        Return every cached row that could currently be active.

        Non-overflow rows have endpoints within drift_slack of their cached
        values, so any such row whose current interval contains t has a
        cached interval intersecting [t - slack, t + slack]. Overflow rows
        are included unconditionally.
        """
        if not self.valid or self.device is None:
            raise RuntimeError("Temporal hierarchy queried before rebuild().")

        t_float = (
            float(t.detach().reshape(()).item())
            if torch.is_tensor(t)
            else float(t)
        )
        slack = self.drift_slack_s
        query_lo = t_float - slack
        query_hi = t_float + slack

        parts: list[Tensor] = []
        segments_visited = 0

        if self.global_idx is not None and int(self.global_idx.numel()) > 0:
            parts.append(self.global_idx)

        n_grids = 2 if self.offset_grids else 1

        for level in range(int(self.levels)):
            segment_s = self.effective_root_segment_s / float(2**level)

            for grid in range(n_grids):
                offset = 0.5 * segment_s * grid
                first_segment = math.floor(
                    (query_lo - self.origin_s - offset) / segment_s
                )
                last_segment = math.floor(
                    (query_hi - self.origin_s - offset) / segment_s
                )

                lookup = self.level_lookup[level][grid]
                if not lookup:
                    continue
                rows = self.level_rows[level][grid]
                for segment_id in range(first_segment, last_segment + 1):
                    span = lookup.get(segment_id)
                    if span is None:
                        continue
                    start, end = span
                    parts.append(rows[start:end])
                    segments_visited += 1

        has_overflow = (
            self.overflow_mask is not None and self.overflow_count > 0
        )
        if has_overflow:
            parts.append(
                torch.nonzero(self.overflow_mask, as_tuple=False).reshape(-1)
            )

        if not parts:
            candidates = torch.empty(
                0,
                device=self.device,
                dtype=torch.long,
            )
        elif len(parts) == 1:
            candidates = parts[0]
        else:
            candidates = torch.cat(parts, dim=0)
            # Tree/global membership is disjoint, but overflow rows still sit
            # in their stale segments, so duplicates are possible.
            if has_overflow:
                candidates = torch.unique(candidates)

        self.last_candidate_count = int(candidates.numel())
        self.last_segments_visited = int(segments_visited)
        return candidates

    @torch.no_grad()
    def query(self, params, t: float | Tensor) -> Tensor:
        """Exact current-parameter query over conservative candidates."""
        candidates = self.query_candidates(t)
        if int(candidates.numel()) == 0:
            self.last_active_count = 0
            return candidates

        device = params["means"].device
        t_tensor = torch.as_tensor(
            t,
            device=device,
            dtype=params["times"].dtype,
        ).reshape(())

        mu = params["times"].detach().reshape(-1)[candidates]
        radius = _indexed_influence_radius(
            params["log_durations"],
            candidates,
            self.opacity_threshold,
            self.extra_margin_s,
        )
        active = torch.abs(t_tensor - mu) <= radius

        if self.include_static and "is_static" in params:
            static = (
                params["is_static"].detach().reshape(-1)[candidates] > 0.5
            )
            active |= static

        if self.allowed_mask is not None:
            active &= self.allowed_mask[candidates]

        result = candidates[active]
        self.last_active_count = int(result.numel())
        return result

    def distribution_stats(self, params=None) -> dict:
        global_count = (
            int(self.global_idx.numel())
            if self.global_idx is not None
            else 0
        )
        level_counts = [
            int(sum(int(g.numel()) for g in grids))
            for grids in self.level_rows
        ]
        out = {
            "n_rows": int(self.n_rows),
            "n_dynamic": int(self.n_dynamic),
            "global": global_count,
            "levels": level_counts,
            "build_step": int(self.build_step),
            "rebuild_count": int(self.rebuild_count),
            "origin_s": float(self.origin_s),
            "configured_root_s": float(self.root_segment_s),
            "effective_root_s": float(self.effective_root_segment_s),
            "finest_segment_s": float(self.finest_segment_s),
            "drift_slack_s": float(self.drift_slack_s),
            "max_endpoint_drift_s": float(self.max_endpoint_drift_s),
            "overflow": int(self.overflow_count),
            "overflow_fraction": float(self.dirty_fraction),
            "last_candidates": int(self.last_candidate_count),
            "last_active": int(self.last_active_count),
            "last_segments_visited": int(self.last_segments_visited),
            # Old dirty-overflow naming.
            "dirty": int(self.overflow_count),
            "dirty_fraction": float(self.dirty_fraction),
        }

        if (
            params is not None
            and self.global_idx is not None
            and "is_static" in params
        ):
            static = params["is_static"].detach().reshape(-1) > 0.5
            global_static = int(static[self.global_idx].sum().item())
            out["global_static"] = global_static
            out["global_dynamic"] = global_count - global_static

        return out