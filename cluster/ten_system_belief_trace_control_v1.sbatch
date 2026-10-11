#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --job-name=belief10
#SBATCH --output=logs/belief10-%j.out
#SBATCH --error=logs/belief10-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY missing"
    exit 1
fi

rm -f \
  artifacts/ten_system_belief_trace_control_v1/systems.jsonl

"$PY" -u \
  analysis/ten_system_belief_trace_control_v1.py
