#!/bin/bash
# HPC job example — see scripts/README.md and issue #44 for the full recipe.
#
# Purpose: submit ONE batch job that starts a Jupyter server on a compute
# node and keeps it running for the job's walltime, so you can `sbatch` this
# once and then keep working in a live notebook — no `srun --pty` interactive
# shell needed, and no reliance on a login node's (possibly shared, possibly
# wrong-architecture) GPU. The job itself just runs Jupyter in the
# foreground; the allocation stays alive for as long as Jupyter does.
#
# How to connect once it's running:
#   1. sbatch hpc_example_job_interactive.sh
#   2. Wait for the job to start, then read its log (the --output path
#      below) for a block like:
#          Compute node : gpu-042
#          Port         : 8888
#          (token URL printed a few lines below by Jupyter itself)
#   3. From your laptop, open an SSH tunnel THROUGH the login node to that
#      compute node and port — compute nodes are usually not reachable
#      directly from outside the cluster:
#          ssh -N -L 8888:gpu-042:8888 you@login.yourcluster.edu
#      (swap gpu-042/8888 for whatever the log printed)
#   4. Open the token URL from the log in your browser (it'll look like
#      http://127.0.0.1:8888/lab?token=...), or point VS Code's "Jupyter:
#      Connect to Existing Server" at that same URL — either way you're
#      talking to the compute node's Jupyter, running with the exact GPU
#      and environment the job was allocated.
#   5. scancel <job id> when you're done — otherwise the job (and the GPU
#      allocation) keeps running until --time expires.
#
# Binding Jupyter to 0.0.0.0 is required for the tunnel in step 3 to reach it
# (a compute node's 127.0.0.1 is not reachable from the login node), but
# Jupyter's own token auth still protects it, and compute-node networks are
# typically not internet-reachable in the first place.
#
# EDIT ME before submitting:
#   --account, --partition   your SLURM allocation / GPU partition
#   --time                   how long you want to keep working interactively,
#                             not just how long one computation takes
#   PY_MODULE, REPO, ENVDIR  must match what you used in hpc_setup_03_build_beorn_env.sh
#   PORT                     change only if this one happens to collide

#SBATCH --job-name=beorn-jupyter
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=/path/to/BEoRN/scripts/logs/jupyter_%j.out

set -e

PY_MODULE=""   # must match hpc_setup_03_build_beorn_env.sh
REPO=/path/to/BEoRN
ENVDIR=/path/outside/the/repo/beorn_gpu_env
PORT=8888

[ -n "$PY_MODULE" ] && module load "$PY_MODULE"
source "$ENVDIR/bin/activate"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
pip install --quiet jupyterlab

echo "=================================================="
echo "Compute node : $(hostname)"
echo "Port         : ${PORT}"
echo "Tunnel from your laptop with:"
echo "    ssh -N -L ${PORT}:$(hostname):${PORT} $(whoami)@YOUR_LOGIN_NODE"
echo "Then open the token URL Jupyter prints below in your browser."
echo "=================================================="

cd "$REPO/examples"
jupyter lab --no-browser --ip=0.0.0.0 --port="${PORT}"
# Runs in the foreground on purpose: the job (and your GPU allocation) stays
# alive for as long as this process does — end the session with `scancel`,
# not by expecting the script to exit on its own.
