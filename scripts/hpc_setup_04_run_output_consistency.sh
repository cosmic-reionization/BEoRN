#!/bin/bash
# Step 4 of 4 — see scripts/README.md and issue #44 for the full recipe.
#
# Purpose: validate that the environment built in step 3 actually reproduces
# BEoRN's expected physics — runs the MPI-parallel output-consistency check
# (tests/test_output_consistency.py) as a SLURM job, in the exact environment
# created by hpc_setup_03_build_beorn_env.sh. No GPU needed for this one.
#
# EDIT ME before submitting:
#   --account, --partition   your SLURM allocation / CPU partition
#   PY_MODULE, REPO, ENVDIR  must match what you used in hpc_setup_03_build_beorn_env.sh

#SBATCH --job-name=beorn-consistency-test
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=YOUR_CPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/BEoRN/scripts/logs/consistency_test_%j.out

set -ex

PY_MODULE=""   # must match hpc_setup_03_build_beorn_env.sh
REPO=/path/to/BEoRN
ENVDIR=/path/outside/the/repo/beorn_gpu_env

[ -n "$PY_MODULE" ] && module load "$PY_MODULE"
source "$ENVDIR/bin/activate"

# Relevant for the MPI-parallel parts of the test; harmless otherwise.
export OMPI_MCA_btl=self,tcp
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1

cd "$REPO"
python -m pytest -vv tests/test_output_consistency.py
EXIT_CODE=$?

echo "=================================================="
echo "Output consistency tests exit code: ${EXIT_CODE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=================================================="

exit ${EXIT_CODE}
