#!/bin/bash
# Step 3 of 4 — see scripts/README.md and issue #44 for the full recipe.
#
# Purpose: build a BEoRN environment natively on a compute node, layered on
# top of whatever the site's Python module already provides (if any), so
# hard-to-build packages like mpi4py stay linked against the site's real MPI
# stack instead of a generic bundled one.
#
# IMPORTANT — decide ENVDIR's final location before running this: a venv's
# `bin/activate` script hardcodes its own absolute path at creation time and
# does NOT resolve it dynamically. If you move this environment's folder
# afterwards, activation will silently point at the old, now-missing path.
# Put ENVDIR somewhere permanent, OUTSIDE this git repository (it is build
# output, not source), before submitting.
#
# EDIT ME before submitting:
#   --account, --partition   your SLURM allocation / GPU partition
#   PY_MODULE                site Python module identified in step 1 (or "")
#   REPO                     path to your BEoRN checkout
#   ENVDIR                   final, permanent path for the new environment
#   CUDA extra (jax[cuda12]) match the CUDA toolkit version your site targets
#     if step 2 showed a different one resolving cleanly

#SBATCH --job-name=beorn-build-env
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --time=00:40:00
#SBATCH --output=/path/to/BEoRN/scripts/logs/build_env_%j.out

set -ex

PY_MODULE=""   # e.g. "GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb"
REPO=/path/to/BEoRN
ENVDIR=/path/outside/the/repo/beorn_gpu_env

[ -n "$PY_MODULE" ] && module load "$PY_MODULE"

echo "=== creating venv (inherits the module's numpy/scipy/mpi4py if present) ==="
python3 -m venv --system-site-packages "$ENVDIR"
source "$ENVDIR/bin/activate"
python -c "import platform, sys; print('venv python:', platform.machine(), sys.version)"

echo "=== installing missing core deps ==="
pip install --upgrade pip
pip install h5py astropy matplotlib tqdm pyyaml

echo "=== installing torch ==="
pip install torch

echo "=== installing jax + GPU plugin (adjust cuda12 -> your site's CUDA major version) ==="
pip install "jax[cuda12]"

echo "=== installing tools21cm ==="
pip install tools21cm
# Use the git master branch instead if you need an unreleased feature, e.g.:
#   pip install "git+https://github.com/sambit-giri/tools21cm@master"

echo "=== installing BEoRN itself, WITHOUT letting pip touch already-satisfied deps ==="
# --no-deps is deliberate: a plain 'pip install -e .' would try to satisfy
# every pin in pyproject.toml, which can silently upgrade a site-provided
# package (mpi4py in particular) to a generic PyPI build detached from the
# site's real MPI/interconnect. Install BEoRN's deps explicitly above, then
# add BEoRN itself with no further dependency resolution.
pip install --no-deps -e "$REPO"

echo "=== verification ==="
python - <<'PYEOF'
import platform
print("machine:", platform.machine())

import numpy, scipy, h5py, astropy, matplotlib, mpi4py, tools21cm
print("numpy", numpy.__version__, "scipy", scipy.__version__, "h5py", h5py.__version__,
      "astropy", astropy.__version__, "matplotlib", matplotlib.__version__,
      "mpi4py", mpi4py.__version__, "tools21cm", tools21cm.__version__)

import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch device:", torch.cuda.get_device_name(0))

import jax
print("jax", jax.__version__, "devices:", jax.devices())
# jax.devices() showing only CpuDevice means the CUDA plugin didn't load —
# re-check step 2's jaxlib/jax-cuda12-plugin/jax-cuda12-pjrt resolution.

import beorn
print("beorn", beorn.__file__)
PYEOF

echo "=== DONE — activate this environment in future jobs with: ==="
echo "    module load $PY_MODULE   # if PY_MODULE was set above"
echo "    source $ENVDIR/bin/activate"
