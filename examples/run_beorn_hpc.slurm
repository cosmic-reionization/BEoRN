#!/bin/bash
#SBATCH --job-name=full_run
#SBATCH --output=/xdisk/timeifler/yhhuang/BEoRN-v2/log/full_run_%A.log
#SBATCH --error=/xdisk/timeifler/yhhuang/BEoRN-v2/log/full_run_%A.err
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
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
    ;;
  fstar)
    RUN_SCRIPT="full_run_fstar.py"
    ;;
  *)
    echo "ERROR: Unknown RUN_MODE='$RUN_MODE'. Use 'legacy' or 'fstar'." >&2
    exit 2
    ;;
esac

# Memory note:
# Each MPI rank allocates 3D grids of size Ncell^3 (float64) for Grid_xHII, Grid_Temp, Grid_xal
# plus delta_b and intermediate arrays. For Ncell=512, one float64 grid is ~1.07 GiB.
# With multiple ranks per node, memory adds up quickly. If you see OOM kills:
#   - reduce ntasks-per-node
#   - increase --mem-per-cpu
#   - set simulation.cores=1 in the YAML when running under MPI (avoid MPI * multiprocessing)
#   - consider reducing Ncell or store_grids in the YAML

# Total number of MPI processes = nodes * ntasks-per-node = 4 * 4 = 16 (default is 32 if SLURM_NTASKS unset)
mpirun -np ${SLURM_NTASKS:-32} python "$RUN_SCRIPT"

# Capture exit status
EXIT_CODE=$?

# Print summary
echo "=================================================="
echo "BEoRN Full Run Completed ($RUN_MODE -> $RUN_SCRIPT)"
echo "Exit code: $EXIT_CODE"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

exit $EXIT_CODE
