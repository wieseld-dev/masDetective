#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-04:00:00
#SBATCH --job-name=exec-data
#SBATCH --output=logs/exec-data-%J.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --tmp=10G

export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
mkdir -p "$SLURM_SCRATCH_DIR/tmp"

export TMPDIR="$SLURM_SCRATCH_DIR/tmp"
export JOBLIB_TEMP_FOLDER="$SLURM_SCRATCH_DIR/tmp"

cd "$HOME/mas_rl_detective"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY missing"
    exit 1
fi

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" execution_operator_big_collect.py \
    --n-selected 160 \
    --concurrency 16
