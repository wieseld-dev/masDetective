#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-01:30:00
#SBATCH --job-name=op3-poc
#SBATCH --output=logs/op3-poc-%J.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"

echo "JOB=$SLURM_JOB_ID"
echo "NODE=$SLURM_JOB_NODELIST"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY missing"
    exit 1
fi

"$PY" operator3_poc.py \
    --n-prompts 60 \
    --concurrency 12
