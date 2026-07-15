#!/bin/bash
# Step 1 of 4 — see scripts/README.md and issue #44 for the full recipe.
#
# Purpose: submit a short job to a COMPUTE node and report back its CPU
# architecture, GPU visibility/contention, and what the site's Python module
# (if any) already provides. Run this BEFORE building any environment — on
# many HPC clusters the login node (where you edit/submit from) is a
# different CPU architecture and/or has a different, often shared, GPU than
# the nodes your jobs actually run on. An environment built on the login node
# may simply not execute on a compute node of a different architecture.
#
# EDIT ME before submitting:
#   --account         your SLURM allocation
#   --partition       the GPU partition name on your cluster
#   PY_MODULE         a site-provided Python module for that partition, if one
#                     exists (`module avail` on a compute node to find it) —
#                     leave empty to skip and rely on your own environment
#   OUTPUT log path   where SBATCH should write stdout

#SBATCH --job-name=beorn-check-node-env
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/path/to/BEoRN/scripts/logs/check_node_env_%j.out

set -x

echo "=== compute-node architecture ==="
hostname
uname -m

echo "=== shared filesystem visibility (repo path reachable from this node?) ==="
ls -la /path/to/BEoRN | head -5

echo "=== GPU visibility and contention ==="
nvidia-smi
# A non-zero "Processes" section, or GPU-Util > 0 from a PID that isn't
# yours, means this GPU is shared — any timing measured on it is unreliable.

echo "=== site-provided Python module for this partition, if any ==="
PY_MODULE=""   # e.g. "GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb"
if [ -n "$PY_MODULE" ]; then
  module load "$PY_MODULE"
  module list
  python3 -c "import platform, sys; print(platform.machine(), sys.version)"

  echo "--- what the module bundle already provides ---"
  for pkg in numpy scipy mpi4py h5py pandas astropy matplotlib; do
    python3 -c "import $pkg; print('$pkg', $pkg.__version__)" 2>&1 | tail -1
  done
else
  echo "PY_MODULE not set — skipping module inspection."
  echo "Run 'module avail' on this node to see what's offered, if anything."
fi

echo "=== DONE — use these findings to fill in hpc_setup_02_check_wheel_availability.sh and hpc_setup_03_build_beorn_env.sh ==="
