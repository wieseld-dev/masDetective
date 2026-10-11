#!/bin/bash
#SBATCH --job-name=presence-v0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/presence-v0-%j.out
#SBATCH --error=logs/presence-v0-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

[[ -n "${OPENAI_API_KEY:-}" ]] || {
    echo "OPENAI_API_KEY is not exported" >&2
    exit 2
}

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u presence_v0/run_presence_v0.py \
    --model gpt-4o-mini \
    --out artifacts/presence_v0
