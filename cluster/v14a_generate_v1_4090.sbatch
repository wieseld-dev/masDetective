#!/bin/bash
#SBATCH --job-name=v14a-gen
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/v14a-gen-%j.out
#SBATCH --error=logs/v14a-gen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${V14A_ROOT:?}"
: "${V14A_ROUND:?}"
: "${V14A_SPLIT:?}"
: "${V14A_GROUP_SIZE:?}"

SCRIPT="scripts/v14_adaptive_generate_round_v1.py"

ARGS=(
  --round "$V14A_ROUND"
  --split "$V14A_SPLIT"
  --group-size "$V14A_GROUP_SIZE"
  --root "$V14A_ROOT"
)

HELP="$("$PY" "$SCRIPT" --help 2>&1)"

if grep -q -- '--v11-adapter' <<<"$HELP"; then
    ARGS+=(--v11-adapter artifacts/v14_benign_policy)
fi

if grep -q -- '--max-seq-length' <<<"$HELP"; then
    ARGS+=(--max-seq-length 3072)
fi

if grep -q -- '--max-new-tokens' <<<"$HELP"; then
    ARGS+=(--max-new-tokens 128)
fi

if grep -q -- '--temperature' <<<"$HELP"; then
    ARGS+=(--temperature 0.85)
fi

if grep -q -- '--top-p' <<<"$HELP"; then
    ARGS+=(--top-p 0.95)
fi

# Full BF16 inference matched the successful smoke test.
if grep -q -- '--no-4bit' <<<"$HELP"; then
    ARGS+=(--no-4bit)
fi

if [ -n "${V14A_ADAPTER:-}" ]; then
    if grep -q -- '--adapter' <<<"$HELP"; then
        ARGS+=(--adapter "$V14A_ADAPTER")
    else
        echo "ERROR: generator has no --adapter override"
        exit 2
    fi
fi

"$PY" -u "$SCRIPT" "${ARGS[@]}"
