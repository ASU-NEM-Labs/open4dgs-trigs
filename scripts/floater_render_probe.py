#!/usr/bin/env python3
"""Read-only render interventions for the unmasked example capture checkpoint.

Run with the training machine's trigs environment. Writes reports/images, never
checkpoints. Source frames are inclusive and camera numbers are zero-based in
the trainer's order. GT uses the trainer's actual undistort/crop/dataset path.
This is a fixed-checkpoint experiment, not a retraining ablation.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree


def select_ring(params, cameras, trainer, config, t, focal):
    """Handoff definition: visible, isolated, dynamic, near moving surface.

    Units assume the reconstruction uses metres, as in example capture. The
    pixel scale uses the raw capture focal length and median camera distance.
    Neither a geometric surface oracle nor an occlusion-aware visibility test.
    """
    with torch.no_grad():
        xyz, _, gamma = trainer._deform(params, t, config)
        alpha = params['opacities'].sigmoid().reshape(-1) * gamma.reshape(-1)
        ids = torch.where(alpha >= .05)[0]
        if len(ids) < 9:
            raise ValueError('Fewer than nine visible rows; cannot measure eighth neighbour')
        static = params['is_static'].reshape(-1) > .5
        centers = torch.linalg.inv(cameras.viewmats)[:, :3, 3]
        x = xyz[ids]
        ppu = torch.cat([
            focal / torch.cdist(part, centers).median(dim=1).values.clamp_min(.1)
            for part in x.split(50000)
        ]).cpu().numpy()
        x_cpu = x.cpu().numpy()
        distances, _ = cKDTree(x_cpu).query(x_cpu, k=9, workers=-1)
        isolated = distances[:, -1] * ppu > 15
        rows = {k: v[ids] for k, v in params.items()
                if v.ndim and v.shape[0] == len(xyz)}
        before = trainer._deform(rows, t - .25, config)[0]
        after = trainer._deform(rows, t + .25, config)[0]
        dynamic = ~static[ids].cpu().numpy()
        moving = ((after - before).norm(dim=-1).cpu().numpy() * ppu > 1)
        moving &= dynamic & ~isolated
        if not moving.any():
            raise ValueError('No moving surface rows at the requested time')
        distance, _ = cKDTree(x_cpu[moving]).query(x_cpu, workers=-1)
        selected = isolated & dynamic & (distance > .01) & (distance < .25)
        ring = ids[torch.as_tensor(selected, device=ids.device)]
        surface = ids[torch.as_tensor(moving, device=ids.device)]
    return ring, surface


def geometry_stats(params, ids):
    scales = params['scales'][ids].detach().exp().sort(dim=-1).values
    fields = {
        'smallest_axis_cm': scales[:, 0] * 100,
        'middle_axis_cm': scales[:, 1] * 100,
        'longest_axis_cm': scales[:, 2] * 100,
        'longest_over_middle': scales[:, 2] / scales[:, 1],
        'sigma_t_s': params['log_durations'][ids].detach().exp().reshape(-1),
        'base_opacity': params['opacities'][ids].detach().sigmoid().reshape(-1),
        'shN_norm': params['shN'][ids].detach().flatten(1).norm(dim=-1),
    }
    return {'count': len(ids), 'quantiles': [.1, .5, .9, .99], **{
        k: np.quantile(v.cpu().numpy(), [.1, .5, .9, .99]).tolist()
        for k, v in fields.items()
    }}


def variant_values(opacity, shn, scales, variant):
    """Return cohort-only replacements; originals are never changed."""
    op, sh, sc = opacity.clone(), shn.clone(), scales.clone()
    if variant == 'suppressed':
        op.fill_(-30)
    elif variant == 'opacity_90pct':
        op = torch.logit(op.sigmoid() * .9)
    elif variant == 'dc_only':
        sh.zero_()
    elif variant == 'cap_longest_3x_middle':
        middle = sc.median(dim=-1, keepdim=True).values
        sc = torch.minimum(sc, middle + math.log(3))
    elif variant != 'original':
        raise ValueError(variant)
    return op, sh, sc


def image_metrics(pred, gt, baseline, affected):
    from trigs.core.losses import reconstruction_loss
    residual = pred - gt
    mse = residual.square().mean().item()
    baseline_residual = baseline - gt
    return {
        'l1': residual.abs().mean().item(),
        'mse': mse,
        'psnr': -10 * math.log10(max(mse, 1e-20)),
        'photo_loss': reconstruction_loss(pred, gt, ssim_lambda=.2).item(),
        'affected_pixels': int(affected.sum()),
        'affected_l1_delta': (
            (residual.abs()[affected] - baseline_residual.abs()[affected]).mean().item()
            if affected.any() else None
        ),
        'affected_mse_delta': (
            (residual.square()[affected] - baseline_residual.square()[affected]).mean().item()
            if affected.any() else None
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--start-frame', type=int, default=1)
    ap.add_argument('--end-frame', type=int, default=200)
    ap.add_argument('--frames', default='31,91,151', help='Source image frame numbers')
    ap.add_argument('--cams', default='0,7,14,21,28,35,42', help='Comma list or all')
    ap.add_argument('--fps', type=float, default=60)
    ap.add_argument('--data-factor', type=int, default=1)
    ap.add_argument('--sync', choices=['none', 'file'], default='none',
                    help='r07 used no sync.json; use file only if the run did')
    ap.add_argument('--lambda-reg', type=float, default=.01)
    ap.add_argument('--lambda-scale-reg', type=float, default=.01)
    ap.add_argument('--no-images', action='store_true')
    args = ap.parse_args()
    if not torch.cuda.is_available():
        ap.error('CUDA required for the actual training rasterizer')
    from PIL import Image
    from trigs.core.camera import CameraConfig
    from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
    from trigs.io.dataset import FreeTimeDataset, FreeTimeParser
    from trigs.cli.train_from_colmap import (
        _DatasetGTImages, _build_gt_times_from_sync, _infer_camera_folders,
    )
    from trigs.io.colmap import read_cameras_binary

    device = 'cuda'
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    raw = ck.get('config') or {}
    valid = {f.name for f in dataclasses.fields(TrigsTrainingConfig)}
    cfg = TrigsTrainingConfig(**{k: v for k, v in raw.items() if k in valid})
    if cfg.masked_rgb_loss or cfg.rs_enabled():
        ap.error('This probe supports unmasked training without rolling shutter only')
    cfg.fps = args.fps
    cfg.ssim_lambda = .2
    cfg.lambda_render = 1.
    cfg.clamp_render_for_loss = True
    cfg.bezier_window_detach = True
    cfg.temporal_active_cull = False  # explicit match to results-07-b4
    cfg.temporal_hierarchy_enabled = False
    cfg.verbose = False
    params = torch.nn.ParameterDict({
        k: torch.nn.Parameter(v.to(device), requires_grad=False)
        for k, v in ck['params'].items()
    })
    del ck
    parser = FreeTimeParser(data_dir=str(args.data_dir), factor=args.data_factor,
                            start_frame=args.start_frame, end_frame=args.end_frame + 1)
    dataset = FreeTimeDataset(parser, split='train', test_set=[])
    camera_ids = dataset.camera_indices
    orig_to_pos = {int(orig): pos for pos, orig in enumerate(camera_ids)}
    first = {}
    for i, (orig, _) in enumerate(dataset.pairs):
        first.setdefault(orig_to_pos[int(orig)], i)
    samples = [dataset[first[pos]] for pos in range(len(camera_ids))]
    cameras = CameraConfig.from_matrices(
        viewmats=torch.linalg.inv(torch.from_numpy(parser.camtoworlds[camera_ids]).float().cuda()),
        Ks=torch.stack([s['K'] for s in samples]).cuda(),
        widths=[int(s['image'].shape[1]) for s in samples],
        heights=[int(s['image'].shape[0]) for s in samples],
    )
    del samples
    gt_images = _DatasetGTImages(dataset, len(camera_ids), orig_to_pos, cache_size=2)
    raw_cameras = read_cameras_binary(str(args.data_dir / 'sparse/0/cameras.bin'))
    focal = float(np.median([c.params[0] for c in raw_cameras.values()]))
    folders = _infer_camera_folders(parser, args.data_dir)
    times = None
    if args.sync == 'file':
        if not (args.data_dir / 'sync.json').exists():
            ap.error('--sync file requires data-dir/sync.json')
        times = _build_gt_times_from_sync(
            sync_file=args.data_dir / 'sync.json', parser=parser,
            data_dir=args.data_dir, training_cam_ids=camera_ids,
            n_frames=parser.total_frames, fps=args.fps,
        )
    cam_positions = list(range(len(camera_ids))) if args.cams == 'all' else [int(x) for x in args.cams.split(',')]
    frames = [int(x) for x in args.frames.split(',')]
    if any(x < 0 or x >= len(camera_ids) for x in cam_positions):
        ap.error('Camera index outside the training camera list')
    if any(x < args.start_frame or x > args.end_frame for x in frames):
        ap.error('Frame outside start/end range')
    if 'camera_time_offsets' in params and len(params['camera_time_offsets']) != len(camera_ids):
        ap.error('Checkpoint camera offset count does not match the dataset')
    trainer = TrigsTrainer(device=device)
    degree = math.isqrt(params['shN'].shape[1] + 1) - 1
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'checkpoint_has_config': bool(raw), 'cohorts': {}, 'views': []}
    report['assumptions'] = {
        'temporal_active_cull': False, 'masked_rgb_loss': False,
        'rolling_shutter': False, 'lambda_render': 1., 'ssim_lambda': .2,
        'clamp_render_for_loss': True, 'static_exclude_from_temporal_reg': True,
        'scale_reg_include_static': True, 'units': 'metres',
        'affected_pixel_threshold': 1 / 255,
        'scope': 'results-07-b4; no inference of missing checkpoint training settings',
    }
    variants = ('original', 'suppressed', 'opacity_90pct', 'dc_only', 'cap_longest_3x_middle')
    dynamic_count = int((params['is_static'].reshape(-1) <= .5).sum())
    model_count = len(params['means'])
    with torch.no_grad():
        for frame in frames:
            offset = frame - args.start_frame
            t_nominal = offset / args.fps
            ring, surface = select_ring(params, cameras, trainer, cfg, t_nominal, focal)
            if not len(ring):
                raise ValueError(f'No ring rows selected for frame {frame}')
            np.save(args.out / f'rows_f{frame:06d}.npy', ring.cpu().numpy())
            report['cohorts'][str(frame)] = {
                'floaters': geometry_stats(params, ring),
                'moving_surface': geometry_stats(params, surface),
            }
            originals = tuple(params[k][ring].clone() for k in ('opacities', 'shN', 'scales'))
            print(f'Frame {frame}: {len(ring):,} ring rows', flush=True)
            for cam in cam_positions:
                t = times[offset][cam] if times is not None else t_nominal
                if 'camera_time_offsets' in params:
                    scale = trainer._camera_time_offset_scale(params, len(camera_ids))
                    t += float(scale[cam] * params['camera_time_offsets'][cam])
                gt = gt_images[offset][cam].to(device).float() / 255
                predictions, regularizers = {}, {}
                try:
                    for mode in variants:
                        values = variant_values(*originals, mode)
                        for key, value in zip(('opacities', 'shN', 'scales'), values):
                            params[key][ring] = value
                        rgb, _, gamma, _ = trainer._render(params, cameras, cam, t, degree, cfg)
                        pred = rgb[0]
                        if 'cam_gain' in params:
                            pred = pred * params['cam_gain'][cam] + params['cam_bias'][cam]
                        predictions[mode] = pred.clamp(0, 1)
                        # All other regularizers cancel: no motion or sh0 changes.
                        regularizers[mode] = (
                            args.lambda_reg * (values[0].sigmoid().reshape(-1) * gamma[ring].reshape(-1)).sum().item() / dynamic_count
                            + args.lambda_scale_reg * values[2].exp().sum().item() / (3 * model_count)
                        )
                finally:
                    for key, value in zip(('opacities', 'shN', 'scales'), originals):
                        params[key][ring] = value
                baseline = predictions['original']
                diff = (baseline - predictions['suppressed']).abs().amax(-1)
                affected = diff > (1 / 255)
                scores = {mode: image_metrics(pred, gt, baseline, affected)
                          for mode, pred in predictions.items()}
                base_photo = scores['original']['photo_loss']
                for mode in variants:
                    scores[mode]['photo_loss_delta'] = scores[mode]['photo_loss'] - base_photo
                    scores[mode]['regularizer_delta'] = regularizers[mode] - regularizers['original']
                    scores[mode]['objective_delta'] = scores[mode]['photo_loss_delta'] + scores[mode]['regularizer_delta']
                row = {'frame': frame, 'cam': cam,
                       'folder': folders[int(camera_ids[cam])], 'time_s': t, 'variants': scores}
                report['views'].append(row)
                (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                print(f"  cam {cam:02d} suppression: photo delta={scores['suppressed']['photo_loss_delta']:+.7f}, "
                      f"objective delta={scores['suppressed']['objective_delta']:+.7f}, "
                      f"affected L1 delta={scores['suppressed']['affected_l1_delta']}", flush=True)
                if not args.no_images:
                    for name, im in {'gt': gt, **predictions,
                                     'suppression_difference_x8': (diff[..., None] * 8).expand(-1, -1, 3)}.items():
                        pixels = (im.clamp(0, 1) * 255).round().byte().cpu().numpy()
                        Image.fromarray(pixels).save(args.out / f'f{frame:06d}_c{cam:02d}_{name}.png')
    report['mean_deltas'] = {
        mode: {key: float(np.mean([v['variants'][mode][key] for v in report['views']]))
               for key in ('photo_loss_delta', 'regularizer_delta', 'objective_delta')}
        for mode in variants
    }
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['mean_deltas'], indent=2))
    print(f"Saved {args.out / 'report.json'}; positive delta means the intervention worsens the fit.")


if __name__ == '__main__':
    main()
