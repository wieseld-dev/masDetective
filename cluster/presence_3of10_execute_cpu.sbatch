#!/bin/bash
#SBATCH --job-name=p310-exec
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-00:45:00
#SBATCH --output=logs/p310-exec-%j.out
#SBATCH --error=logs/p310-exec-%j.err

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

"$PY" -u presence_3of10/execute_round.py \
    --root "$P310_ROOT" \
    --round "$P310_ROUND" \
    --model "${P310_MODEL:-gpt-4o-mini}"
