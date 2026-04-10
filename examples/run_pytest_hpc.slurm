#!/bin/bash
#SBATCH --job-name=beorn-pytest
#SBATCH --output=/xdisk/timeifler/yhhuang/BEoRN-v2/log/pytest_%A.log
#SBATCH --error=/xdisk/timeifler/yhhuang/BEoRN-v2/log/pytest_%A.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=standard
#SBATCH --account=timeifler

module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2

export OMPI_MCA_btl=^openib
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export PYTHONPATH=/home/u15/yhhuang/cosmology/BEoRN-v2/src:${PYTHONPATH}

cd /home/u15/yhhuang/cosmology/BEoRN-v2

which mpirun
mpirun --version

mpirun -np ${SLURM_NTASKS:-1} python -m pytest -vv -s tests/test_solver.py tests/test_coordinator.py
