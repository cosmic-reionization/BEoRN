#!/bin/bash
#SBATCH --job-name=beorn-consistency-test
#SBATCH --output=/xdisk/timeifler/yhhuang/BEoRN-v2/log/consistency_test_%A.out
#SBATCH --error=/xdisk/timeifler/yhhuang/BEoRN-v2/log/consistency_test_%A.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=standard
#SBATCH --account=timeifler
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yhhuang@arizona.edu

module load openmpi3
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2

export OMPI_MCA_btl=self,tcp
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export PYTHONPATH=/home/u15/yhhuang/cosmology/BEoRN-v2/src:${PYTHONPATH}

cd /home/u15/yhhuang/cosmology/BEoRN-v2

python -m pytest -vv tests/test_output_consistency.py
EXIT_CODE=$?

echo "=================================================="
echo "Output consistency tests exit code: ${EXIT_CODE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "=================================================="

exit ${EXIT_CODE}
