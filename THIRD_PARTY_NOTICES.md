# Third-party notices

## TriGSVanilla

This trainer is based on [OpsiClear-4DGS/TriGSVanilla](https://github.com/OpsiClear-4DGS/TriGSVanilla), an independent implementation of the TRiGS paper. This project modifies and extends that code. The PLY control path references upstream commit [bb51ea6482513bb3075a11e959a2beec46f1702b](https://github.com/OpsiClear-4DGS/TriGSVanilla/commit/bb51ea6482513bb3075a11e959a2beec46f1702b). The trainer is distributed under [AGPL-3.0](LICENSE); retain the upstream license and attribution when redistributing it.

## MultiNeRF

[`src/trigs/io/traj.py`](src/trigs/io/traj.py) contains modified camera trajectory code borrowed from [MultiNeRF camera_utils.py](https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/camera_utils.py). Copyright 2022 Google LLC. That code is licensed under Apache License 2.0; a copy is included in [LICENSE-MULTINERF](LICENSE-MULTINERF). The local file carries the original notice and identifies the adaptation.

## COLMAP

[`src/trigs/io/colmap.py`](src/trigs/io/colmap.py) includes the copyright notice, redistribution conditions, and disclaimer for ETH Zurich and UNC Chapel Hill in its source header. Preserve that notice; it also specifies how to reproduce it for binary distributions.

## Other dependencies

The trainer uses [gsplat](https://github.com/nerfstudio-project/gsplat), [PyTorch3D](https://github.com/facebookresearch/pytorch3d), and other independently licensed packages. Their respective licenses apply to those packages. See the [trainer README](README.md#citations-and-credits) for research citations.
