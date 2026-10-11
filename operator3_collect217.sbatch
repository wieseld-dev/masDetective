#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-03:00:00
#SBATCH --job-name=op3-data
#SBATCH --output=logs/op3-data-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
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

"$PY" operator3_poc.py \
    --n-prompts 217 \
    --concurrency 16 \
    --cache artifacts/operator3_poc/api_runs.jsonl
