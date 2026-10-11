#!/bin/bash

#SBATCH --partition=cpu
#SBATCH --job-name=curve-kill
#SBATCH --output=logs/curve-kill-%J.out
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
echo "CURVATURE KILL TEST"
echo "NODE: $(hostname)"
echo "START: $(date)"
echo "========================================================"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not available in this Slurm job."
    exit 2
fi

"$PY" curvature_killtest.py
STATUS=$?

echo
echo "========================================================"
echo "EXIT STATUS: $STATUS"
echo "END: $(date)"
echo "========================================================"

exit $STATUS
