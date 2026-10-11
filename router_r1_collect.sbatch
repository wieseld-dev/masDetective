#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-04:00:00
#SBATCH --job-name=r1-data
#SBATCH --output=logs/r1-data-%J.out
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

"$PY" router_r1.py \
  --n-prompts 217 \
  --cache artifacts/router_r1/api_runs.jsonl \
  --concurrency 16
