"""CPU checks for the diagnostic's interventions and cohort definition."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'floater_render_probe',
    Path(__file__).resolve().parents[1] / 'scripts/floater_render_probe.py',
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_interventions_do_not_change_originals():
    opacity = torch.tensor([0., 1.])
    shn = torch.ones(2, 15, 3)
    scales = torch.tensor([[.001, .01, .2], [.000001, .01, .02]]).log()
    originals = [x.clone() for x in (opacity, shn, scales)]
    op, _, _ = probe.variant_values(opacity, shn, scales, 'opacity_90pct')
    torch.testing.assert_close(op.sigmoid(), opacity.sigmoid() * .9)
    _, sh, _ = probe.variant_values(opacity, shn, scales, 'dc_only')
    assert not sh.any()
    _, _, sc = probe.variant_values(opacity, shn, scales, 'cap_longest_3x_middle')
    torch.testing.assert_close(sc.exp(), torch.tensor([
        [.001, .01, .03], [.000001, .01, .02],
    ]))
    hidden, _, _ = probe.variant_values(opacity, shn, scales, 'suppressed')
    assert (hidden.sigmoid() < 1e-12).all()
    for actual, original in zip((opacity, shn, scales), originals):
        torch.testing.assert_close(actual, original)


def test_ring_selection_excludes_static_and_surface_rows():
    params = {'means': torch.zeros(22, 3), 'is_static': torch.zeros(22, 1),
              'opacities': torch.zeros(22)}
    params['means'][:20, 0] = torch.linspace(-.005, .005, 20)
    params['means'][20:, 0] = .1
    params['is_static'][21] = 1

    class LinearMotion:
        def _deform(self, rows, time, config):
            xyz = rows['means'].clone()
            xyz[:, 1] += torch.where(xyz[:, 0] < .01, .01 * time, 0.)
            return xyz, None, torch.ones(len(xyz), 1)

    viewmats = torch.eye(4)[None]
    viewmats[0, 2, 3] = 3
    ring, surface = probe.select_ring(
        params, SimpleNamespace(viewmats=viewmats), LinearMotion(), None, 1.5, 1302,
    )
    assert ring.tolist() == [20]
    assert surface.tolist() == list(range(20))


def test_error_delta_sign_and_empty_affected_region():
    from trigs.core import losses
    gt = torch.zeros(16, 16, 3)
    original = torch.full_like(gt, .2)
    better = torch.full_like(gt, .1)
    affected = torch.ones(16, 16, dtype=torch.bool)
    with patch.object(losses, '_SSIM_BACKEND', 'custom'):
        result = probe.image_metrics(better, gt, original, affected)
    assert result['affected_l1_delta'] < 0
    assert result['affected_mse_delta'] < 0
    with patch.object(losses, '_SSIM_BACKEND', 'custom'):
        empty = probe.image_metrics(better, gt, original, ~affected)
    assert empty['affected_l1_delta'] is None
    assert empty['affected_mse_delta'] is None


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(fn) for fn in (
        test_interventions_do_not_change_originals,
        test_ring_selection_excludes_static_and_surface_rows,
        test_error_delta_sign_and_empty_affected_region,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
