#!/bin/bash
#SBATCH --job-name=post_dtb
#SBATCH --output=/xdisk/timeifler/yhhuang/BEoRN-v2/log/post_dtb_%A.log
#SBATCH --error=/xdisk/timeifler/yhhuang/BEoRN-v2/log/post_dtb_%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=standard
#SBATCH --account=timeifler
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yhhuang@arizona.edu

# This postprocessing job is intentionally single-node and single-task.
# It stays well within the UArizona HPC standard Ocelote job limits documented at:
# https://hpcdocs.hpc.arizona.edu/running_jobs/job_limits/

module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2

export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export BEORN_PARAMETER_FILE="${BEORN_PARAMETER_FILE:-coarse_grid.yaml}"
export BEORN_SCRATCH_ROOT="${BEORN_SCRATCH_ROOT:-/xdisk/timeifler/yhhuang/BEoRN-v2/coarse_grid_test_main/}"
export BEORN_TREE_CACHE_FILE="${BEORN_TREE_CACHE_FILE:-/xdisk/timeifler/yhhuang/Thesan/postprocessing/trees/LHaloTree/tree_cache_v2.hdf5}"

cd /home/u15/yhhuang/cosmology/BEoRN-v2/examples

echo "Job limits for account ${SLURM_ACCOUNT:-timeifler}:"
job-limits "${SLURM_ACCOUNT:-timeifler}" || true

echo "Running Grid_dTb postprocessing with BEORN_PARAMETER_FILE=${BEORN_PARAMETER_FILE}"
python postprocess_dtb_fstar.py
