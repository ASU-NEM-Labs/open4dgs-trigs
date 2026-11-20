# Open4DGS-Trigs installation

This guide sets up Open4DGS-Trigs in a Conda environment on Ubuntu. A
[Windows 11 recipe](#windows-11-with-nvidia-gpus) follows the Linux setup.
It covers the CUDA extension builds used by `gsplat`, PyTorch3D, and optional
fused SSIM. Preprocessing models such as RoMa and MEMFOF belong to the
preprocessing project and have separate setup instructions there.

## Linux with NVIDIA GPUs

### Supported setup recorded so far

The working setup reported for this repository is Ubuntu 24.04, Python 3.11,
PyTorch 2.10.0 with CUDA 12.8, and an RTX 3090 (compute capability 8.6). The
host driver was 595.84 and the Conda-provided compiler toolkit was CUDA 12.8.
The PyTorch, torchvision, gsplat, and PyTorch3D imports were smoke-tested on a
3090. Fused SSIM was separately built successfully on a 3090 environment.

These are machine-local installations. Repeat the setup and the smoke check on
each machine. The host CUDA driver and the CUDA toolkit used to compile
extensions are separate: `nvidia-smi` reports the driver, while `nvcc --version`
reports the compiler toolkit. The CUDA version printed by `nvidia-smi` is the
driver's supported CUDA level; it does not mean that version of `nvcc` is
installed. A host CUDA toolkit is not required if the Conda environment
supplies `nvcc`.

## 1. Install host tools

On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential git ninja-build curl ca-certificates \
  libgl1 libglib2.0-0 ffmpeg tmux
```

If the machine does not already have Conda, install
[Miniforge](https://github.com/conda-forge/miniforge) on x86_64 Ubuntu:

```bash
curl -L -o "$HOME/Miniforge3.sh" \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash "$HOME/Miniforge3.sh" -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Create the project environment with Python 3.11:

```bash
conda create -n trigs python=3.11 pip -y
conda activate trigs
```

Install `uv` for the current account if it is not installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

## 2. Install the CUDA compiler toolkit in the environment

For the validated 3090 setup, install CUDA 12.8 into the active environment:

```bash
conda install -y nvidia/label/cuda-12.8.0::cuda-toolkit
```

Set build variables in each shell before building CUDA extensions. Use the GPU
architecture for the machine that will run the code:

```bash
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6   # RTX 3090 / Ampere
```

For GeForce RTX 50-series Blackwell GPUs (including the RTX 5090), set
`TORCH_CUDA_ARCH_LIST=12.0`; do not use the Ada `8.9` target. Blackwell product
families do not all share one compute capability, so query the exact GPU and
use its target. NVIDIA's [compute-capability table](https://developer.nvidia.com/cuda/gpus)
lists current models. Rebuild native extensions on the target machine after
changing this value. The installed PyTorch wheel must also include a CUDA
runtime that supports the card. The setup below uses the PyTorch CUDA 12.8
wheel index.

Confirm the active host and tools before continuing:

```bash
hostname
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv
command -v nvcc
nvcc --version
python --version
```

If `nvcc` is missing, check that `trigs` is active and that `$CONDA_PREFIX/bin`
is on `PATH`.

## 3. Install TriGS dependencies

Run from the `open4dgs-trigs` checkout root (or `train/trigs/` in the working repository):

```bash
conda activate trigs
export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6   # use 12.0 for GeForce RTX 50-series

uv sync --locked --inexact --preview-features extra-build-dependencies
```

The project manifest and lockfile pin the core dependencies, including
PyTorch/torchvision from the CUDA 12.8 wheel index, gsplat, and PyTorch3D.
The direct package list also includes `gsply`, `torchmetrics`, `tqdm`,
`imageio`, `tyro`, `scipy`, `opencv-python`, `tensorboard`, `pycolmap`,
`viser`, `nerfview`, `matplotlib`, and `splines` (plus the build tools
`setuptools` and `wheel`). The `dev` dependency group adds `pytest`, `mypy`,
and `ruff`.
`--preview-features extra-build-dependencies` supplies build requirements that
gsplat and PyTorch3D do not declare themselves. `--inexact` lets uv install
locked packages into the Conda environment without removing the Conda CUDA
toolkit. `UV_PROJECT_ENVIRONMENT` makes uv use the active Conda environment
instead of creating `.venv`.

The lockfile is expected to be committed with this project. If you are using an
older checkout that has no `uv.lock`, run the same command once without
`--locked`, inspect the generated lockfile, and retain it for later installs.

## 4. Check the core GPU environment

```bash
python - <<'PY'
import sys
import torch
import torchvision
import gsplat
from pytorch3d.ops import knn_points

print("Python:", sys.executable)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
print("torchvision:", torchvision.__version__)
print("gsplat and PyTorch3D imports: OK")
PY
```

Check that CUDA is available, the expected GPU is listed, and the reported
compute capability matches the card. This import check does not prove every
training path or extension kernel works; use a small representative training
run before a long experiment.

## 5. Optional fused SSIM extension

Fused SSIM is not part of the locked TriGS project dependencies. Install it
after TriGS so its build can use the environment's installed PyTorch and CUDA
toolkit. The upstream project recommends disabling build isolation:

```bash
conda activate trigs
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
CUDA_ARCHITECTURES=86 uv pip install \
  --python "$CONDA_PREFIX/bin/python" \
  --no-build-isolation \
  "git+https://github.com/rahul-goel/fused-ssim.git"
```

For a GeForce RTX 50-series card, use `CUDA_ARCHITECTURES=120`. For other
Blackwell products, use the target matching that GPU's compute capability. If
the package has already been built for a different GPU architecture,
reinstall/rebuild it on the target machine with the correct architecture.
Upstream install and implementation details:
[fused-ssim](https://github.com/rahul-goel/fused-ssim).

Quick import check:

```bash
python -c 'from fused_ssim import fused_ssim; print("fused_ssim import: OK")'
```

## Windows 11 with NVIDIA GPUs

One reported installation and two end-to-end runs used Windows 11, Visual
Studio 2022 Community with C++ build tools, CUDA toolkit 12.9, Python 3.11.18,
PyTorch 2.10.0+cu128 and an RTX PRO 6000 Blackwell Workstation Edition
(compute capability 12.0). PyTorch3D 0.7.9 and gsplat 1.5.3 were built from
source. This is a field report for one configuration, not a validation of all
Windows systems. Install uv, the NVIDIA driver, CUDA toolkit with nvcc, Git,
and Visual Studio C++ build tools first.

Open an x64 Native Tools Command Prompt for Visual Studio 2022, go to the
Open4DGS-Trigs project directory, and confirm that cl, nvcc and uv are
available. The following uses the exact PyTorch versions in the reported run;
check the project lockfile when updating dependencies. TORCH_CUDA_ARCH_LIST
must match your GPU, and MAX_JOBS can be reduced for machines with less RAM.

```bat
cl
nvcc --version
uv --version
set TORCH_CUDA_ARCH_LIST=12.0
set MAX_JOBS=16
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set UV_CACHE_DIR=C:\uv-cache
set DISTUTILS_USE_SDK=1
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe ^
  --index-url https://download.pytorch.org/whl/cu128 ^
  torch==2.10.0+cu128 torchvision==0.25.0+cu128
uv pip install --python .venv\Scripts\python.exe setuptools wheel ninja numpy==2.4.4
uv sync --locked --python 3.11 --preview-features extra-build-dependencies ^
  --no-build-isolation-package pytorch3d --no-build-isolation-package gsplat
```

The non-isolated builds use the installed PyTorch rather than independently
resolving a potentially different version. `DISTUTILS_USE_SDK=1` lets the
PyTorch extension builder use the Visual Studio environment that the prompt has
already set up; without it, the PyTorch3D build stops with "the VC environment
is activated but DISTUTILS_USE_SDK is not set". The short cache path can also
avoid Windows path-length failures. The UTF-8 settings prevent a reported console
encoding error in help output. For a persistent setting, configure the
environment variables in Windows or set them in each terminal session.

Confirm imports and GPU access before a full run:

```bat
.venv\Scripts\python.exe -c "import torch, torchvision, gsplat; from pytorch3d.ops import knn_points; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

This verifies imports and CUDA availability, not all kernels or workflows. Run
a short representative training job before a long run. If a native extension
fails with a missing or unsupported GPU kernel, confirm the architecture and
rebuild it for the target GPU. A cache-lock timeout during an isolated build
was also observed in the field report; the recipe above avoids isolated builds
for these two packages.

## Sharing an environment with preparation tools

Follow the [two-repository setup](https://github.com/ASU-NEM-Labs/open4dgs-core/blob/main/SETUP.md). Install the trainer first, then the preparation backends into the same activated environment. Separate environments also work; transfer the prepared data directory between stages.

The trainer lockfile does not include preparation backends such as RoMa and
MEMFOF. A normal uv run synchronizes the environment and can remove packages
installed separately, including these backends and optional fused SSIM. After
installing everything needed, activate the environment and run the trainer's
console commands directly, or use uv run --no-sync from the project directory.
If a later sync removes a backend, reinstall it following the preparation setup.

## Troubleshooting

- **uv says `uv.lock` is missing with `--locked`:** this checkout may predate
  the committed lockfile. Run once without `--locked`, then keep/commit the
  generated lockfile.
- **`nvcc` is not found:** activate `trigs`, reinstall the Conda CUDA toolkit,
  and set `CUDA_HOME` and `PATH` as above. The NVIDIA driver alone does not
  install `nvcc`.
- **`no kernel image is available` or an extension reports an unsupported
  architecture:** rebuild the extension on this machine with
  `TORCH_CUDA_ARCH_LIST` / `CUDA_ARCHITECTURES` set for its GPU.
- **Imports come from the wrong Python:** check `which python` and
  `python -c 'import sys; print(sys.executable)'`; the path should be inside
  `$CONDA_PREFIX`.
- **A command fails and an SSH/tmux pane disappears:** avoid setting
  `set -euo pipefail` directly in an interactive shell. Put strict shell
  settings inside a script you intend to run. Use `Ctrl-b`, then `d` to detach
  from a tmux session without stopping it.

## Dependency boundaries

The TriGS `pyproject.toml` covers the training package. RoMa, MEMFOF, SAM2,
VGGT, EasyVolCap, and the dynamic-probe stack are not all part of that
environment. Install only the preprocessing backends and workflows required
for your data preparation, using the setup guide shipped with that project.
