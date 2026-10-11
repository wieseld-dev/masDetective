#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-00:30:00
#SBATCH --job-name=contrast-audit
#SBATCH --output=logs/contrast-audit-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --tmp=10G

export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
mkdir -p "$SLURM_SCRATCH_DIR/tmp"

export TMPDIR="$SLURM_SCRATCH_DIR/tmp"
export JOBLIB_TEMP_FOLDER="$SLURM_SCRATCH_DIR/tmp"

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"

echo "NODE: $(hostname)"
echo "START: $(date)"

"$PY" contrast_generalization_audit.py

echo "END: $(date)"
