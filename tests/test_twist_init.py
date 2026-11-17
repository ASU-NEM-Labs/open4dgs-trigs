"""The local twist init must hand the model the velocities it was fit to."""

import torch

from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
from trigs.math.twist_init import fit_local_twist, local_knn_indices


def _model_velocity(means, anchors, nu, omega, eps=1e-3):
    """Instantaneous velocity of TrigsTrainer._deform at t = mu_t."""
    n = means.shape[0]
    params = {
        "means": means, "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1),
        "anchors": anchors, "times": torch.zeros(n, 1),
        "log_durations": torch.zeros(n, 1),
        "zeta_base_t": nu, "zeta_base_r": omega,
        **{f"zeta_c{k}_{s}": torch.zeros(n, 3) for k in "012" for s in "tr"},
    }
    cfg = TrigsTrainingConfig(temporal_active_cull=False, verbose=False)
    plus, _, _ = TrigsTrainer._deform(None, params, torch.tensor(eps), cfg)
    minus, _, _ = TrigsTrainer._deform(None, params, torch.tensor(-eps), cfg)
    return (plus - minus) / (2 * eps)


def _cloud(n=400):
    torch.manual_seed(0)
    return torch.randn(n, 3, dtype=torch.float64).float() * 0.3


def test_rigid_screw_motion_keeps_every_velocity():
    """Rotation about an off-centre axis plus translation: the old init kept
    only the axis-parallel translation and lost the rest."""
    x = _cloud()
    omega0 = torch.tensor([0.3, -1.2, 0.5])
    centre = torch.tensor([0.8, 0.1, -0.4])
    v = torch.tensor([0.9, 0.4, -0.6]) + torch.cross(omega0.expand_as(x), x - centre, dim=-1)
    nu, omega, anchors = fit_local_twist(x, v, x.mean(0).expand_as(x).clone(), local_knn_indices(x, 16))
    got = _model_velocity(x, anchors, nu, omega)
    assert torch.allclose(got, v, atol=2e-3), (got - v).norm(dim=-1).max()
    assert torch.allclose(omega, omega0.expand_as(omega), atol=5e-3)


def test_pure_translation_keeps_velocity():
    x = _cloud()
    v = torch.tensor([1.5, 0.0, -0.7]).expand_as(x).clone()
    nu, omega, anchors = fit_local_twist(x, v, x.clone(), local_knn_indices(x, 16))
    got = _model_velocity(x, anchors, nu, omega)
    assert torch.allclose(got, v, atol=2e-3)



from trigs.math.twist_init import (  # noqa: E402
    fit_bezier_to_acceleration,
    screw_deform_means,
    smooth_accelerations,
)

_CTRL = ("zeta_c0_t", "zeta_c1_t", "zeta_c2_t", "zeta_c0_r", "zeta_c1_r", "zeta_c2_r")


def test_screw_deform_means_matches_trainer_deform():
    torch.manual_seed(3)
    n = 64
    P = {
        "means": torch.randn(n, 3), "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "anchors": torch.randn(n, 3), "times": torch.rand(n, 1), "log_durations": torch.full((n, 1), -2.0),
        "zeta_base_t": torch.randn(n, 3), "zeta_base_r": torch.randn(n, 3),
        **{k: 0.3 * torch.randn(n, 3) for k in _CTRL},
    }
    cfg = TrigsTrainingConfig(temporal_active_cull=False, verbose=False)
    t = P["times"] + 0.07
    want, _, _ = TrigsTrainer._deform(None, P, t, cfg)
    got = screw_deform_means(P["means"], P["anchors"], P["times"], P["log_durations"],
                             P["zeta_base_t"], P["zeta_base_r"], {k: P[k] for k in _CTRL}, t)
    assert torch.allclose(got, want, atol=1e-5)


def _fit_and_check(means, anchors, nu, omega, v, a, sigma=0.1):
    n = means.shape[0]
    mu = torch.zeros(n, 1); ls = torch.full((n, 1), float(torch.log(torch.tensor(sigma))))
    ctrl = fit_bezier_to_acceleration(means, anchors, mu, ls, nu, omega, v, a, iters=600, lr=0.05)
    for dt in (-0.15, -0.05, 0.05, 0.15):
        t = torch.full((n, 1), dt)
        x = screw_deform_means(means, anchors, mu, ls, nu, omega, ctrl, t)
        target = means + v * dt + 0.5 * a * dt * dt
        assert (x - target).norm(dim=-1).max() < 2e-3, (dt, (x - target).norm(dim=-1).max())


def test_fit_reproduces_an_accelerating_translation():
    torch.manual_seed(0)
    x = torch.randn(32, 3) * 0.2
    v = torch.tensor([[0.8, -0.3, 0.1]]).expand(32, 3).clone()
    a = torch.tensor([[-6.0, 2.0, 0.5]]).expand(32, 3).clone()           # a reversal within the window
    _fit_and_check(x, x.clone(), v, torch.zeros(32, 3), v, a)


def test_fit_reproduces_acceleration_across_a_rotation_axis():
    """With rotation the model keeps only the axis-parallel translation, so
    the sideways acceleration has to come through the rotation controls."""
    torch.manual_seed(1)
    x = torch.randn(32, 3) * 0.2
    v = torch.tensor([0.9, 0.4, -0.6]) + torch.cross(torch.tensor([0.3, -1.2, 0.5]).expand(32, 3), x, dim=-1)
    nu, omega, anchors = fit_local_twist(x, v, x.mean(0).expand_as(x).clone(), local_knn_indices(x, 16))
    v_model = _model_velocity(x, anchors, nu, omega)
    a = torch.tensor([[3.0, -4.0, 1.0]]).expand(32, 3).clone()
    _fit_and_check(x, anchors, nu, omega, v_model, a)


def test_weighted_batch_fit_equals_separate_fits():
    """Two keyframes fitted in one batch with 1/len weights match fitting each alone."""
    torch.manual_seed(2)
    sets = []
    for n in (24, 40):
        x = torch.randn(n, 3) * 0.2
        v = torch.tensor([0.9, 0.4, -0.6]) + torch.cross(torch.tensor([0.3, -1.2, 0.5]).expand(n, 3), x, dim=-1)
        nu, omega, anchors = fit_local_twist(x, v, x.mean(0).expand_as(x).clone(), local_knn_indices(x, 8))
        a = torch.randn(1, 3).expand(n, 3).clone() * 3
        mu = torch.full((n, 1), 0.1 * len(sets)); ls = torch.full((n, 1), -2.3)
        sets.append((x, anchors, mu, ls, nu, omega, _model_velocity(x, anchors, nu, omega), a))
    alone = [fit_bezier_to_acceleration(*s, iters=50) for s in sets]
    batch = fit_bezier_to_acceleration(
        *[torch.cat([s[i] for s in sets]) for i in range(8)], iters=50,
        row_weight=torch.cat([torch.full((s[0].shape[0],), 1.0 / s[0].shape[0]) for s in sets]),
    )
    for k in _CTRL:
        assert torch.allclose(batch[k], torch.cat([r[k] for r in alone]), atol=1e-6), k


def test_target_fit_recovers_a_long_curved_trajectory():
    """Positions sampled from the model's own curve over a wide window are matched again
    from a zero start, using only the samples marked valid."""
    from trigs.math.twist_init import fit_bezier_to_targets
    torch.manual_seed(4)
    n = 48
    x = torch.randn(n, 3) * 0.2
    v = torch.tensor([0.9, 0.4, -0.6]) + torch.cross(torch.tensor([0.3, -1.2, 0.5]).expand(n, 3), x, dim=-1)
    nu, omega, anchors = fit_local_twist(x, v, x.mean(0).expand_as(x).clone(), local_knn_indices(x, 8))
    mu = torch.zeros(n, 1); ls = torch.full((n, 1), float(torch.log(torch.tensor(0.18))))
    true_ctrl = {k: 0.4 * torch.randn(n, 3) for k in _CTRL}
    dt = torch.linspace(-0.35, 0.35, 15)
    targets = torch.stack([screw_deform_means(x, anchors, mu, ls, nu, omega, true_ctrl, mu + d) for d in dt], dim=1)
    valid = torch.ones(n, 15, dtype=torch.bool); valid[:, :3] = False      # chain stopped early on one side
    targets[~valid] = float("nan")
    ctrl = fit_bezier_to_targets(x, anchors, mu, ls, nu, omega, dt, targets, valid, iters=1500, lr=0.05)
    for s in range(3, 15):
        got = screw_deform_means(x, anchors, mu, ls, nu, omega, ctrl, mu + dt[s])
        miss = (got - targets[:, s]).norm(dim=-1)
        assert miss.median() < 1e-3 and miss.max() < 1e-2, (s, float(miss.median()), float(miss.max()))


def test_smooth_accelerations_uses_only_measured_neighbours():
    acc = torch.tensor([[1.0, 0, 0], [3.0, 0, 0], [100.0, 0, 0], [2.0, 0, 0]])
    has = torch.tensor([True, True, False, True])
    nn = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
    out = smooth_accelerations(acc, has, nn)
    assert torch.allclose(out[:, 0], torch.tensor([2.0, 2.0, 2.0, 2.0]))     # the unmeasured 100 never counts
    none = smooth_accelerations(acc, torch.zeros(4, dtype=torch.bool), nn)
    assert torch.all(none == 0)


def test_bezier_rewindow_keeps_the_curve_timing():
    from trigs.math.se3 import bezier_rewindow
    torch.manual_seed(5)
    c0, c1, c2 = (torch.randn(32, 3) for _ in range(3))
    for r in (0.7, 1.0, 1.3):
        ratio = torch.full((32, 1), r)
        n0, n1, n2 = bezier_rewindow(c0, c1, c2, ratio)
        for tau_new in torch.linspace(0, 1, 9):
            tau_old = r * tau_new + (1 - r) / 2          # same absolute time, old window
            q = lambda t, a, b, c: (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t * t * c  # noqa: E731
            torch.testing.assert_close(q(tau_new, n0, n1, n2), q(tau_old, c0, c1, c2), atol=1e-5, rtol=1e-5)


def test_keep_bezier_timing_holds_positions_when_sigma_changes():
    torch.manual_seed(6)
    n = 64
    P = {
        "means": torch.randn(n, 3), "quats": torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1),
        "anchors": torch.randn(n, 3), "times": torch.zeros(n, 1), "log_durations": torch.full((n, 1), -2.5),
        "zeta_base_t": torch.randn(n, 3), "zeta_base_r": torch.randn(n, 3) * 0.5,
        **{k: 0.3 * torch.randn(n, 3) for k in _CTRL},
    }
    P = {k: torch.nn.Parameter(v) for k, v in P.items()}
    cfg = TrigsTrainingConfig(temporal_active_cull=False, verbose=False)
    sigma = float(torch.exp(torch.tensor(-2.5)))
    ts = [torch.full((n, 1), f * sigma) for f in (-1.5, -0.5, 0.5, 1.5)]   # inside old and new windows
    before = [TrigsTrainer._deform(None, P, t, cfg)[0].detach().clone() for t in ts]
    log_before = P["log_durations"].data.clone()
    P["log_durations"].data += torch.linspace(-0.2, 0.3, n).unsqueeze(-1)  # an optimiser step moved sigma
    stretched = [TrigsTrainer._deform(None, P, t, cfg)[0].detach() for t in ts]
    assert max(float((s - b).norm(dim=-1).max()) for s, b in zip(stretched, before)) > 1e-3
    assert TrigsTrainer._keep_bezier_timing(P, log_before) == n
    for t, b in zip(ts, before):
        torch.testing.assert_close(TrigsTrainer._deform(None, P, t, cfg)[0].detach(), b, atol=1e-4, rtol=1e-4)
