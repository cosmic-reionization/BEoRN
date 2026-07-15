#!/bin/bash
# HPC job example — see scripts/README.md and issue #44 for the full recipe.
#
# Purpose: the plainest possible batch job — run a Python script (no
# notebook, no interactivity) in the environment built by
# hpc_setup_03_build_beorn_env.sh. Shown here running
# examples/python_scripts/lpt_native_21cm.py; swap SCRIPT/SCRIPT_ARGS for
# your own script and arguments.
#
# For a headless notebook instead of a plain script, see
# hpc_example_job_notebook.sh. For a live, interactive session you can keep
# working in, see hpc_example_job_interactive.sh.
#
# EDIT ME before submitting:
#   --account, --partition   your SLURM allocation / GPU partition
#   PY_MODULE, REPO, ENVDIR  must match what you used in hpc_setup_03_build_beorn_env.sh
#   SCRIPT, SCRIPT_ARGS      the script (and arguments) to run

#SBATCH --job-name=beorn-run-script
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/path/to/BEoRN/scripts/logs/run_script_%j.out

set -ex

PY_MODULE=""   # must match hpc_setup_03_build_beorn_env.sh
REPO=/path/to/BEoRN
ENVDIR=/path/outside/the/repo/beorn_gpu_env
SCRIPT="$REPO/examples/python_scripts/lpt_native_21cm.py"
SCRIPT_ARGS="--param-file $REPO/examples/python_scripts/param.yaml"

[ -n "$PY_MODULE" ] && module load "$PY_MODULE"
source "$ENVDIR/bin/activate"

# XLA reserves ~75% of GPU memory by default — turn that off before jax is
# ever imported, especially if the GPU might be shared with anything else.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python "$SCRIPT" $SCRIPT_ARGS
EXIT_CODE=$?

echo "=================================================="
echo "Script exit code: ${EXIT_CODE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=================================================="

exit ${EXIT_CODE}
