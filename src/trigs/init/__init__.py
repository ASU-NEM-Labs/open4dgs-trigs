"""Canonical-set initialization adapters for TriGS.

Two input modes are supported:

* :mod:`trigs.init.from_plys` — dense per-frame Gaussian PLYs (gsflow flow).
  The trainer's ``_build_canonical_set`` pulls keyframes directly from PLYs.

* :mod:`trigs.init.from_colmap` — multi-view videos with COLMAP sparse
  reconstruction + per-frame ROMA triangulated NPY files. Uses
  :mod:`trigs.init.roma_combine` to build an intermediate NPZ, then
  runs the same kNN anchor / ω init as the PLY path so both flows
  feed identical canonical params into the trainer.
"""
