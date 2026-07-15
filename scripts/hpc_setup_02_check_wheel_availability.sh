#!/bin/bash
# Step 2 of 4 — see scripts/README.md and issue #44 for the full recipe.
#
# Purpose: before spending a full job building an environment, confirm that
# the GPU-capable packages BEoRN needs actually have prebuilt binary wheels
# for this compute node's architecture. `pip install --dry-run` resolves and
# downloads the wheel (so its size/existence is real) without installing —
# if a package instead falls back to a source build here, expect that step
# to be slow, to need extra system dev packages, or to fail outright.
#
# The important check is jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt
# specifically — `jax` itself is pure Python and will "succeed" even when
# there is no GPU wheel available, silently leaving you on CPU.
#
# EDIT ME before submitting:
#   --account, --partition   same as in hpc_setup_01_check_node_environment.sh
#   PY_MODULE                same module (if any) identified in step 1

#SBATCH --job-name=beorn-check-wheels
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --output=/path/to/BEoRN/scripts/logs/check_wheels_%j.out

set -x

PY_MODULE=""   # e.g. "GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb"
[ -n "$PY_MODULE" ] && module load "$PY_MODULE"

echo "=== torch ==="
pip3 install --dry-run --no-deps torch 2>&1 | tail -8

echo "=== jax GPU stack (jax itself is pure Python — check these three instead) ==="
pip3 install --dry-run --no-deps jaxlib 2>&1 | tail -8
pip3 install --dry-run --no-deps jax-cuda12-plugin 2>&1 | tail -8
pip3 install --dry-run --no-deps jax-cuda12-pjrt 2>&1 | tail -8

echo "=== BEoRN's remaining core dependencies ==="
pip3 install --dry-run --no-deps h5py astropy matplotlib tqdm pyyaml 2>&1 | tail -20

echo "=== DONE — anything that fell back to 'Building wheel ... from source' or"
echo "failed outright needs attention before running hpc_setup_03_build_beorn_env.sh ==="
