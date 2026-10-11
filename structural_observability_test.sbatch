#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --job-name=struct-obs
#SBATCH --output=logs/struct-obs-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --tmp=10G

set -u

export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
mkdir -p "$SLURM_SCRATCH_DIR/tmp"

export TMPDIR="$SLURM_SCRATCH_DIR/tmp"
export JOBLIB_TEMP_FOLDER="$SLURM_SCRATCH_DIR/tmp"

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"

echo "========================================================"
echo "STRUCTURAL OBSERVABILITY TEST"
echo "NODE: $(hostname)"
echo "START: $(date)"
echo "========================================================"

"$PY" structural_observability_test.py
STATUS=$?

echo
echo "========================================================"
echo "EXIT STATUS: $STATUS"
echo "END: $(date)"
echo "========================================================"

exit $STATUS
