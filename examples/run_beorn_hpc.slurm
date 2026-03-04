#!/bin/bash
#SBATCH --job-name=full_run
#SBATCH --output=/xdisk/timeifler/yhhuang/BEoRN-v2/log/full_run_%A.log
#SBATCH --error=/xdisk/timeifler/yhhuang/BEoRN-v2/log/full_run_%A.err
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=standard
#SBATCH --account=timeifler
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yhhuang@arizona.edu

# Load required modules for OpenMPI3 and conda environment
module load openmpi3
module load anaconda

# Activate the conda environment
source ~/.bashrc
conda activate v2

# Set environment variables for MPI and HDF5
export OMPI_MCA_btl=^openib  # Use TCP instead of InfiniBand if needed
export HDF5_USE_FILE_LOCKING=FALSE  # Important for parallel HDF5 I/O

# Verify MPI is available
which mpirun
mpirun --version

# Change to the examples directory
cd /Users/yhhuang/code/cosmology/BEoRN-v2/examples

# Run full_run.py with MPI
# Total number of MPI processes = nodes * ntasks-per-node = 4 * 8 = 32
mpirun -np 32 python full_run.py

# Capture exit status
EXIT_CODE=$?

# Print summary
echo "=================================================="
echo "BEoRN Full Run Completed"
echo "Exit code: $EXIT_CODE"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="

exit $EXIT_CODE
