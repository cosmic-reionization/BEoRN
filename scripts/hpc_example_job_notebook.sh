#!/bin/bash
# HPC job example — see scripts/README.md and issue #44 for the full recipe.
#
# Purpose: a minimal template for running something in the environment built
# by hpc_setup_03_build_beorn_env.sh, as a plain **batch** job (submit and
# walk away) — shown here executing a notebook headlessly via nbclient and
# overwriting it in place with fresh outputs (useful for validation/
# benchmarking notebooks such as examples/artificial_halo_gpu.ipynb).
#
# For a live, interactive notebook session instead of a headless run, see
# hpc_example_job_interactive.sh. For a plain Python script with no notebook
# involved, see hpc_example_job_run.sh.
#
# EDIT ME before submitting:
#   --account, --partition   your SLURM allocation / GPU partition
#   PY_MODULE, REPO, ENVDIR  must match what you used in hpc_setup_03_build_beorn_env.sh
#   NOTEBOOK                 the notebook to execute

#SBATCH --job-name=beorn-notebook-job
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/path/to/BEoRN/scripts/logs/notebook_job_%j.out

set -ex

PY_MODULE=""   # must match hpc_setup_03_build_beorn_env.sh
REPO=/path/to/BEoRN
ENVDIR=/path/outside/the/repo/beorn_gpu_env
NOTEBOOK="$REPO/examples/artificial_halo_gpu.ipynb"

[ -n "$PY_MODULE" ] && module load "$PY_MODULE"
source "$ENVDIR/bin/activate"

# XLA reserves ~75% of GPU memory by default — turn that off before jax is
# ever imported, especially if the GPU might be shared with anything else.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --- notebook example -------------------------------------------------------
pip install --quiet nbclient nbformat ipykernel
python -m ipykernel install --user --name python3 --display-name python3

python - <<PYEOF
import nbformat
from nbclient import NotebookClient

path = "$NOTEBOOK"
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=900, kernel_name='python3', allow_errors=True,
                        resources={'metadata': {'path': "$REPO/examples"}})
client.execute()

errs = [(i, o['ename'], o['evalue'][:150]) for i, c in enumerate(nb.cells)
        if c.cell_type == 'code' for o in c.get('outputs', [])
        if o.output_type == 'error']
print('code cells:', sum(1 for c in nb.cells if c.cell_type == 'code'), '| errors:', len(errs))
for e in errs[:6]:
    print(e)

nbformat.write(nb, path)   # overwrite in place with fresh outputs
PYEOF
# -----------------------------------------------------------------------------

echo "=== DONE ==="
