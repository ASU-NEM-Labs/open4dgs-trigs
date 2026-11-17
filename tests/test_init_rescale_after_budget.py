from __future__ import annotations

import math

import torch
from scipy.spatial import cKDTree

from trigs.cli.train_from_colmap import (
    _apply_split_budget_to_prebuilt_params,
    _rescale_dynamic_init_after_budget,
)


def _params(n_static: int = 50, n_per_kf: int = 2000, kf_times=(0.0, 0.05)) -> dict:
    gen = torch.Generator().manual_seed(0)
    means, times, static = [], [], []
    for t in kf_times:
        means.append(torch.rand(n_per_kf, 3, generator=gen))
        times.append(torch.full((n_per_kf, 1), t))
        static.append(torch.zeros(n_per_kf, 1))
    means.append(torch.rand(n_static, 3, generator=gen) + 5.0)
    times.append(torch.zeros(n_static, 1))
    static.append(torch.ones(n_static, 1))
    means = torch.cat(means)
    n = means.shape[0]
    return {
        "means": torch.nn.Parameter(means),
        "times": torch.nn.Parameter(torch.cat(times)),
        "scales": torch.nn.Parameter(torch.full((n, 3), math.log(1e-3))),
        "is_static": torch.cat(static),
    }


def _expected_log_scale(points: torch.Tensor, k: int = 4, factor: float = 0.3) -> torch.Tensor:
    p = points.double().numpy()
    d = cKDTree(p).query(p, k=k + 1)[0][:, 1:].mean(axis=1)
    return torch.from_numpy(d * factor).log().float()


def test_rescale_sizes_each_keyframe_on_the_kept_rows_only() -> None:
    params = _params()
    params, _ = _apply_split_budget_to_prebuilt_params(params, target_count=450, static_target_count=50)
    static = params["is_static"].reshape(-1) > 0.5
    static_before = params["scales"].data[static].clone()

    n = _rescale_dynamic_init_after_budget(params)

    assert n == int((~static).sum())
    assert torch.equal(params["scales"].data[static], static_before)
    times = params["times"].data.reshape(-1)
    for t in times[~static].unique():
        rows = (~static) & (times == t)
        expected = _expected_log_scale(params["means"].data[rows])
        got = params["scales"].data[rows]
        assert torch.allclose(got, expected.unsqueeze(-1).expand_as(got), atol=1e-5)


def test_rescaled_splats_match_their_thinned_spacing() -> None:
    dense = _params()
    full_scale = {}
    times = dense["times"].data.reshape(-1)
    for t in (0.0, 0.05):
        rows = (dense["is_static"].reshape(-1) <= 0.5) & (times == t)
        full_scale[t] = _expected_log_scale(dense["means"].data[rows]).exp().median()
    thinned, _ = _apply_split_budget_to_prebuilt_params(dense, target_count=250, static_target_count=50)
    _rescale_dynamic_init_after_budget(thinned)
    dyn = thinned["is_static"].reshape(-1) <= 0.5
    t_all = thinned["times"].data.reshape(-1)
    for t in (0.0, 0.05):
        rows = dyn & (t_all == t)
        thinned_scale = thinned["scales"].data[rows, 0].exp().median()
        # 2000 -> ~100 rows of a unit cube: spacing grows by about (20)^(1/3) ~ 2.7x.
        assert thinned_scale > 2.0 * full_scale[t]
