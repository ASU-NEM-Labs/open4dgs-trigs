"""Regression coverage for the dense per-frame PLY input path."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
from trigs.cli.train_from_plys import (
    TrigsPipelineArgs,
    _apply_compact_profile,
    _apply_vanilla_profile,
    _configure_ply_scale_guard,
    _initialize_ply_opacity,
)
from trigs.init.from_plys import (
    allocate_keyframe_quotas,
    count_keyframe_gaussians,
    discover_ply_files,
    read_ply_vertex_count,
    select_keyframe_gaussians,
)
from trigs.io import ply as ply_io


class _FakeGSTensor:
    frames: dict[str, "_FakeGSTensor"] = {}

    def __init__(self, means: torch.Tensor) -> None:
        n = means.shape[0]
        self.means = means
        self.quats = torch.zeros(n, 4)
        self.quats[:, 0] = 1.0
        self.scales = torch.full((n, 3), -2.0)
        self.opacities = torch.zeros(n)
        self.sh0 = torch.zeros(n, 3)
        self.shN = None

    def __len__(self) -> int:
        return self.means.shape[0]

    def to(self, device: torch.device) -> "_FakeGSTensor":
        for name in ("means", "quats", "scales", "opacities", "sh0"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @classmethod
    def load(
        cls,
        path: str,
        device: str | None = None,
    ) -> "_FakeGSTensor":
        src = cls.frames[path]
        result = cls(src.means.clone())
        return result.to(device) if device is not None else result


class _Translate:
    def apply(self, gs: _FakeGSTensor) -> _FakeGSTensor:
        gs.means = gs.means + torch.tensor([10.0, 0.0, 0.0])
        return gs


def test_original_3dgs_normals_layout_uses_compatibility_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    property_names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        *(f"f_rest_{i}" for i in range(45)),
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        "element vertex 1",
        *(f"property float {name}" for name in property_names),
        "end_header",
        "",
    ]).encode("ascii")
    values = np.arange(len(property_names), dtype="<f4")
    path = tmp_path / "original_3dgs.ply"
    path.write_bytes(header + values.tobytes())

    captured: dict[str, object] = {}

    class _RejectingGSTensor:
        @classmethod
        def load(cls, path: str, device: str | None = None):
            raise ValueError("strict gsply layout rejection")

        @classmethod
        def from_arrays(cls, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(ply_io, "GSTensor", _RejectingGSTensor)
    result = ply_io.load_ply(path, device="cpu")

    assert torch.equal(result.means, torch.tensor([[0.0, 1.0, 2.0]]))
    assert torch.equal(result.sh0, torch.tensor([[6.0, 7.0, 8.0]]))
    assert result.shN.shape == (1, 15, 3)
    assert torch.equal(result.shN[0, 0], torch.tensor([9.0, 24.0, 39.0]))
    assert result.opacities.item() == pytest.approx(54.0)
    assert torch.equal(result.scales, torch.tensor([[55.0, 56.0, 57.0]]))
    assert torch.equal(result.quats, torch.tensor([[58.0, 59.0, 60.0, 61.0]]))
    assert captured["device"] == "cpu"


def test_discover_ply_files_is_sorted_and_validates_directory(tmp_path: Path) -> None:
    (tmp_path / "frame_000002.ply").touch()
    (tmp_path / "ignore.txt").touch()
    (tmp_path / "frame_000001.ply").touch()

    assert [p.name for p in discover_ply_files(tmp_path)] == [
        "frame_000001.ply",
        "frame_000002.ply",
    ]
    with pytest.raises(NotADirectoryError):
        discover_ply_files(tmp_path / "missing")


def test_ply_header_counts_drive_keyframe_population(tmp_path: Path) -> None:
    paths = []
    for frame, count in enumerate((80, 90, 100, 110, 120)):
        path = tmp_path / f"frame_{frame:06d}.ply"
        path.write_bytes(
            (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {count}\n"
                "property float x\n"
                "end_header\n"
            ).encode("ascii"),
        )
        paths.append(path)

    assert read_ply_vertex_count(paths[0]) == 80
    assert count_keyframe_gaussians(paths, keyframe_stride=2) == 300


def test_ply_target_helpers_are_exact_and_reproducible() -> None:
    assert allocate_keyframe_quotas(11, 3) == [4, 4, 3]
    with pytest.raises(ValueError):
        allocate_keyframe_quotas(2, 3)

    means = torch.stack([
        torch.linspace(-2.0, 2.0, 20),
        torch.sin(torch.linspace(0.0, 3.0, 20)),
        torch.cos(torch.linspace(0.0, 3.0, 20)),
    ], dim=-1)
    keep_a = select_keyframe_gaussians(means, 7, seed=123)
    keep_b = select_keyframe_gaussians(means, 7, seed=123)

    assert keep_a.shape == (7,)
    assert torch.equal(keep_a, keep_b)
    assert int(torch.unique(keep_a).numel()) == 7
    assert bool((keep_a[1:] > keep_a[:-1]).all())

    # A small disconnected component must not be discarded merely because it
    # contributes little opacity/area compared with the main body.
    main_component = torch.randn(1_000, 3) * 0.05
    small_component = torch.tensor([[5.0, 0.0, 0.0], [5.1, 0.0, 0.0]])
    disconnected = torch.cat([main_component, small_component])
    keep_disconnected = select_keyframe_gaussians(disconnected, 32, seed=456)
    assert bool((keep_disconnected >= 1_000).any())


def test_ply_compact_profile_is_ply_scoped_and_respects_overrides(tmp_path: Path) -> None:
    plain = TrigsPipelineArgs(
        input_dir=tmp_path,
        output_dir=tmp_path / "out-plain",
    )
    _apply_compact_profile(plain, keyframe_row_count=20_815_495)
    assert plain.config.masked_rgb_loss is False
    assert plain.config.max_sigma_t == TrigsTrainingConfig().max_sigma_t        # profile off: defaults untouched
    assert plain.config.opacity_logit_upper_clamp == TrigsTrainingConfig().opacity_logit_upper_clamp
    assert plain.camera.n_cameras == 120

    args = TrigsPipelineArgs(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        compact_profile=True,
    )
    _apply_compact_profile(args, keyframe_row_count=20_815_495)

    assert args.config.target_count == 8_326_198
    assert args.config.init_sigma_t == pytest.approx(0.1)
    assert args.config.max_sigma_t == pytest.approx(0.2)
    assert args.config.unfreeze_temporal_step == 5_000
    assert args.config.omega_init is True
    assert args.config.masked_rgb_loss is True
    assert args.config.opacity_logit_upper_clamp == pytest.approx(4.0)
    assert args.config.opacity_logit_upper_clamp_until_step == args.config.n_steps
    assert args.initial_opacity == pytest.approx(0.0)
    assert args.camera.n_cameras == 120
    assert TrigsTrainingConfig().target_count is None
    assert TrigsTrainingConfig().init_sigma_t == pytest.approx(0.0)
    assert TrigsTrainingConfig().max_sigma_t == pytest.approx(-1.0)
    assert TrigsTrainingConfig().unfreeze_temporal_step == 0
    assert TrigsTrainingConfig().masked_rgb_loss is False
    assert TrigsTrainingConfig().opacity_logit_upper_clamp is None
    assert TrigsTrainingConfig().opacity_logit_upper_clamp_until_step == 25_000

    overridden = TrigsPipelineArgs(
        input_dir=tmp_path,
        output_dir=tmp_path / "out-override",
        compact_profile=True,
        config=replace(
            TrigsTrainingConfig(),
            target_count=750_000,
            init_sigma_t=1.5,
            max_sigma_t=2.5,
            unfreeze_temporal_step=2_000,
            opacity_logit_upper_clamp=3.0,
            opacity_logit_upper_clamp_until_step=12_345,
        ),
        initial_opacity=0.2,
    )
    _apply_compact_profile(overridden)
    assert overridden.config.target_count == 750_000
    assert overridden.config.init_sigma_t == pytest.approx(1.5)
    assert overridden.config.max_sigma_t == pytest.approx(2.5)
    assert overridden.config.unfreeze_temporal_step == 2_000
    assert overridden.config.opacity_logit_upper_clamp == pytest.approx(3.0)
    assert overridden.config.opacity_logit_upper_clamp_until_step == 12_345
    assert overridden.initial_opacity == pytest.approx(0.2)


def test_ply_vanilla_profile_preserves_ply_safeguards(tmp_path: Path) -> None:
    args = TrigsPipelineArgs(
        input_dir=tmp_path,
        output_dir=tmp_path / "out-vanilla",
        vanilla_profile=True,
        config=replace(
            TrigsTrainingConfig(),
            unfreeze_temporal_step=3_000,
            omega_init=True,
            freeze_geometry=True,
            temporal_active_cull=True,
            temporal_gamma_smooth_cull=True,
            temporal_hierarchy_enabled=True,
            temporal_chunk_len_s=1.0,
            clamp_log_durations=True,
            clamp_log_scales=True,
            max_sigma_t=-1.0,
            max_scale_exp=0.25,
            masked_rgb_loss=True,
            lambda_sigma_floor=0.1,
            sigma_floor_target=0.1,
            lambda_motion_residual=1e-3,
            temporal_optimizer_beta1=0.0,
            relocate_dynamic_only=True,
            relocate_mcmc_split=True,
            relocate_time_jitter_sigma=0.25,
            relocate_recenter_residual=True,
            relocate_settle_steps=500,
        ),
    )
    _apply_vanilla_profile(args)

    config = args.config
    assert config.unfreeze_temporal_step == 3_000
    assert config.omega_init is False
    assert config.freeze_geometry is False
    assert config.temporal_active_cull is False
    assert config.temporal_gamma_smooth_cull is False
    assert config.temporal_hierarchy_enabled is False
    assert config.temporal_chunk_len_s == pytest.approx(0.0)
    assert config.clamp_log_durations is False
    assert config.clamp_log_scales is False
    assert config.max_sigma_t == pytest.approx(0.0)
    assert config.max_scale_exp == pytest.approx(0.0)
    assert config.masked_rgb_loss is True
    assert config.lambda_sigma_floor == pytest.approx(0.1)
    assert config.sigma_floor_target == pytest.approx(0.1)
    assert config.lambda_motion_residual == pytest.approx(0.0)
    assert config.temporal_optimizer_beta1 == pytest.approx(0.9)
    assert config.relocate_dynamic_only is False
    assert config.relocate_mcmc_split is False
    assert config.relocate_time_jitter_sigma == pytest.approx(0.0)
    assert config.relocate_recenter_residual is False
    assert config.relocate_settle_steps == 0


def test_ply_vanilla_profile_base_defaults_remain_upstream_compatible(
    tmp_path: Path,
) -> None:
    args = TrigsPipelineArgs(
        input_dir=tmp_path,
        output_dir=tmp_path / "out-vanilla-defaults",
        vanilla_profile=True,
    )
    _apply_vanilla_profile(args)

    assert args.config.unfreeze_temporal_step == 0
    assert args.config.masked_rgb_loss is False
    assert args.config.lambda_sigma_floor == pytest.approx(0.0)
    assert args.config.sigma_floor_target == pytest.approx(0.0)


def test_ply_vanilla_and_compact_profiles_are_mutually_exclusive(tmp_path: Path) -> None:
    args = TrigsPipelineArgs(
        input_dir=tmp_path,
        output_dir=tmp_path / "out-conflict",
        vanilla_profile=True,
        compact_profile=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _apply_vanilla_profile(args)


def test_ply_scale_guard_is_source_derived() -> None:
    scales = torch.tensor([
        [-4.0, -4.0, -4.0],
        [-3.0, -3.0, -3.0],
        [-2.0, -2.0, -2.0],
        [0.0, 0.0, 0.0],
    ])
    params = torch.nn.ParameterDict({
        "scales": torch.nn.Parameter(scales),
    })
    config = TrigsTrainingConfig()
    _configure_ply_scale_guard(params, config)

    assert 0.0 < config.max_scale_exp <= 1.05
    explicit = TrigsTrainingConfig(max_scale_exp=0.25)
    _configure_ply_scale_guard(params, explicit)
    assert explicit.max_scale_exp == pytest.approx(0.25)


def test_ply_opacity_initialization_resets_logits() -> None:
    params = torch.nn.ParameterDict({
        "opacities": torch.nn.Parameter(torch.tensor([-4.0, 0.0, 4.0])),
    })
    _initialize_ply_opacity(params, 0.1)
    assert torch.allclose(
        torch.sigmoid(params["opacities"]),
        torch.full((3,), 0.1),
    )


def test_ply_frames_build_current_canonical_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"frame_{i:06d}.ply" for i in range(3)]
    for path in paths:
        path.touch()

    # Keep points well separated so nearest-neighbor correspondence is
    # unambiguous after the one-unit per-frame translation.
    base = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    _FakeGSTensor.frames = {
        str(path): _FakeGSTensor(base + torch.tensor([float(i), 0.0, 0.0]))
        for i, path in enumerate(paths)
    }
    monkeypatch.setitem(
        sys.modules,
        "gsply",
        types.SimpleNamespace(GSTensor=_FakeGSTensor),
    )

    trainer = TrigsTrainer(device="cpu")
    config = TrigsTrainingConfig(
        keyframe_stride=2,
        fps=2.0,
        nn_velocity_init=True,
        nn_velocity_k=1,
        nn_anchor_init=False,
        omega_init=False,
        freeze_geometry=True,
        verbose=False,
    )
    params, sh_degree, scene_extent = trainer._build_canonical_set(
        paths,
        config,
        _Translate(),
        n_frames=len(paths),
    )

    assert set(params) == {
        "means",
        "quats",
        "scales",
        "sh0",
        "opacities",
        "times",
        "log_durations",
        "anchors",
        "zeta_base_t",
        "zeta_base_r",
        "zeta_c0_t",
        "zeta_c0_r",
        "zeta_c1_t",
        "zeta_c1_r",
        "zeta_c2_t",
        "zeta_c2_r",
    }
    assert params["means"].shape == (4, 3)
    assert torch.allclose(params["times"].flatten(), torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert torch.allclose(params["log_durations"].exp(), torch.full((4, 1), 1.0))
    assert torch.allclose(
        params["zeta_base_t"],
        torch.tensor([[2.0, 0.0, 0.0]]).expand(4, -1),
    )
    assert torch.allclose(params["anchors"], params["means"])
    assert params["means"].requires_grad is False
    assert sh_degree == 0
    assert scene_extent == pytest.approx(12.0)


def test_ply_target_count_is_applied_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = [tmp_path / f"frame_{i:06d}.ply" for i in range(3)]
    for path in paths:
        path.touch()

    base = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    _FakeGSTensor.frames = {
        str(path): _FakeGSTensor(base + float(i))
        for i, path in enumerate(paths)
    }
    monkeypatch.setitem(
        sys.modules,
        "gsply",
        types.SimpleNamespace(GSTensor=_FakeGSTensor),
    )

    config = TrigsTrainingConfig(
        keyframe_stride=2,
        target_count=6,
        nn_velocity_init=False,
        nn_anchor_init=False,
        omega_init=False,
        verbose=False,
    )
    params, _, _ = TrigsTrainer(device="cpu")._build_canonical_set(
        paths,
        config,
        _Translate(),
        n_frames=len(paths),
    )

    assert params["means"].shape[0] == 6
    unique_times, counts = torch.unique(
        params["times"].detach().flatten(),
        sorted=True,
        return_counts=True,
    )
    assert torch.equal(unique_times, torch.tensor([0.0, 2.0 / 30.0]))
    assert torch.equal(counts, torch.tensor([3, 3]))
