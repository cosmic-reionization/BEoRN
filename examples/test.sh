#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/xdisk/timeifler/yhhuang/BEoRN-v2/log/test-%A.out
#SBATCH --error=/xdisk/timeifler/yhhuang/BEoRN-v2/log/test-%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=timeifler
#SBATCH --time=02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yhhuang@arizona.edu


source ~/.bashrc
conda activate v2

echo "Running test.py without MPI."
python test.py

echo "Test completed."
