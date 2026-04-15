#!/bin/bash
#SBATCH --job-name=full_run
#SBATCH --output=/xdisk/timeifler/yhhuang/BEoRN-v2/log/full_run_%A.log
#SBATCH --error=/xdisk/timeifler/yhhuang/BEoRN-v2/log/full_run_%A.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=standard
#SBATCH --account=timeifler
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yhhuang@arizona.edu

# --------------------------------------------------
# Select which BEoRN run script to execute.
# Allowed values:
#   legacy -> full_run.py
#   fstar  -> full_run_fstar.py
RUN_MODE="fstar"
# --------------------------------------------------

# Load required modules for OpenMPI3 and conda environment
module load openmpi3
module load anaconda

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate v2

# Set environment variables for MPI and HDF5
export OMPI_MCA_btl=^openib  # Use TCP instead of InfiniBand if needed
export HDF5_USE_FILE_LOCKING=FALSE  # Important for parallel HDF5 I/O
export OMP_NUM_THREADS=1

# Verify MPI is available
which mpirun
mpirun --version

# Optional: check current account/user limits for Ocelote
# job-limits timeifler

# Change to the examples directory
cd /home/u15/yhhuang/cosmology/BEoRN-v2/examples

# Select the Python entrypoint based on RUN_MODE
case "$RUN_MODE" in
  legacy)
    RUN_SCRIPT="full_run.py"
    POSTPROCESS_SCRIPT=""
    ;;
  fstar)
    RUN_SCRIPT="full_run_fstar.py"
    POSTPROCESS_SCRIPT="postprocess_dtb_fstar.py"
    ;;
  *)
    echo "ERROR: Unknown RUN_MODE='$RUN_MODE'. Use 'legacy' or 'fstar'." >&2
    exit 2
    ;;
esac

# Keep the main run and postprocessing on the same parameter file.
# Override this at submission time if needed, e.g.
#   sbatch --export=BEORN_PARAMETER_FILE=parameters_fstar.yaml examples/run_beorn_hpc.slurm
export BEORN_PARAMETER_FILE="${BEORN_PARAMETER_FILE:-coarse_grid.yaml}"
export BEORN_SCRATCH_ROOT="${BEORN_SCRATCH_ROOT:-/xdisk/timeifler/yhhuang/BEoRN-v2/coarse_grid_test_main/}"
export BEORN_TREE_CACHE_FILE="${BEORN_TREE_CACHE_FILE:-/xdisk/timeifler/yhhuang/Thesan/postprocessing/trees/LHaloTree/tree_cache_v2.hdf5}"

# Memory note:
# Each MPI rank allocates 3D grids of size Ncell^3 (float64) for Grid_xHII, Grid_Temp, Grid_xal
# plus delta_b and intermediate arrays. For Ncell=512, one float64 grid is ~1.07 GiB.
# On Ocelote standard nodes memory is allocated at 6 GB per CPU. The
# cpus-per-task value is intentionally larger than OMP_NUM_THREADS so each MPI
# painting rank receives more memory without increasing the number of workers.
# With multiple MPI ranks on one node, memory adds up quickly. If you see OOM kills:
#   - reduce ntasks-per-node
#   - increase --cpus-per-task
#   - keep simulation.cores=1 in the YAML when running under MPI
#     (avoid MPI snapshot-parallelism *and* per-snapshot multiprocessing)
#   - consider reducing Ncell or store_grids in the YAML

# Total number of MPI processes = nodes * ntasks-per-node = 1 * 4 = 4
# Total allocated CPUs = tasks * cpus-per-task = 16, giving 96 GB on Ocelote
# standard nodes while still running only 4 MPI painting workers.
# The fallback keeps the command usable outside SLURM, but under SLURM
# `SLURM_NTASKS` should already be set to the correct total rank count.
mpirun -np ${SLURM_NTASKS:-1} python "$RUN_SCRIPT"
MAIN_EXIT_CODE=$?

POSTPROCESS_EXIT_CODE=0
if [ $MAIN_EXIT_CODE -eq 0 ] && [ -n "$POSTPROCESS_SCRIPT" ]; then
  echo "Main run completed successfully. Starting postprocessing with $POSTPROCESS_SCRIPT"
  python "$POSTPROCESS_SCRIPT"
  POSTPROCESS_EXIT_CODE=$?
fi

if [ $MAIN_EXIT_CODE -ne 0 ]; then
  EXIT_CODE=$MAIN_EXIT_CODE
else
  EXIT_CODE=$POSTPROCESS_EXIT_CODE
fi

# Print summary
echo "=================================================="
echo "BEoRN Full Run Completed ($RUN_MODE -> $RUN_SCRIPT)"
if [ -n "$POSTPROCESS_SCRIPT" ]; then
  echo "Postprocess script: $POSTPROCESS_SCRIPT"
  echo "Postprocess exit code: $POSTPROCESS_EXIT_CODE"
fi
echo "Main run exit code: $MAIN_EXIT_CODE"
echo "Exit code: $EXIT_CODE"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

exit $EXIT_CODE
