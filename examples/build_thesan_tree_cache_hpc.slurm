#!/bin/bash
#SBATCH --job-name=thesan_tree_cache
#SBATCH --output=/xdisk/timeifler/yhhuang/BEoRN-v2/log/build_thesan_tree_cache_%A.log
#SBATCH --error=/xdisk/timeifler/yhhuang/BEoRN-v2/log/build_thesan_tree_cache_%A.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --partition=standard
#SBATCH --account=timeifler
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yhhuang@arizona.edu

set -euo pipefail

TREE_ROOT="/xdisk/timeifler/yhhuang/Thesan/postprocessing/trees/LHaloTree"
OUTPUT_FILE="${TREE_ROOT}/tree_cache.hdf5"

module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate v2

cd /home/u15/yhhuang/cosmology/BEoRN-v2/examples

echo "=================================================="
echo "Building THESAN tree cache"
echo "Tree root: ${TREE_ROOT}"
echo "Output: ${OUTPUT_FILE}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "=================================================="

python build_thesan_tree_cache.py \
  --tree-root "${TREE_ROOT}" \
  --output "${OUTPUT_FILE}"

echo "=================================================="
echo "THESAN tree cache build finished"
echo "Output: ${OUTPUT_FILE}"
echo "End: $(date)"
echo "=================================================="
