#!/bin/bash
#SBATCH --job-name=q38lora-util
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --array=0-19%5
#SBATCH --output=logs/q38lora-util-%A_%a.out
#SBATCH --error=logs/q38lora-util-%A_%a.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

# Load OpenAI credentials exactly as the working MAS execution jobs do.
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing (.env not loaded?)}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

"$PY" -u \
    analysis/q38_utility_lora_worker_v1.py \
    "$SLURM_ARRAY_TASK_ID"
