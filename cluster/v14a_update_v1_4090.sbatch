#!/bin/bash
#SBATCH --job-name=v14a-upd
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/v14a-upd-%j.out
#SBATCH --error=logs/v14a-upd-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$HOME/.conda/envs/masdetective/bin/python"

: "${V14A_ROOT:?}"
: "${V14A_ROUND:?}"

SCRIPT="scripts/v14_adaptive_grpo_update_v1.py"

ARGS=(
  --round "$V14A_ROUND"
  --root "$V14A_ROOT"
)

HELP="$("$PY" "$SCRIPT" --help 2>&1)"

if grep -q -- '--max-seq-length' <<<"$HELP"; then
    ARGS+=(--max-seq-length 3072)
fi

if grep -q -- '--lr' <<<"$HELP"; then
    ARGS+=(--lr 3e-6)
fi

if grep -q -- '--clip-eps' <<<"$HELP"; then
    ARGS+=(--clip-eps 0.20)
elif grep -q -- '--clip' <<<"$HELP"; then
    ARGS+=(--clip 0.20)
fi

if grep -q -- '--kl-beta' <<<"$HELP"; then
    ARGS+=(--kl-beta 0.05)
fi

if grep -q -- '--grpo-epochs' <<<"$HELP"; then
    ARGS+=(--grpo-epochs 2)
elif grep -q -- '--epochs' <<<"$HELP"; then
    ARGS+=(--epochs 2)
fi

if grep -q -- '--grad-accum' <<<"$HELP"; then
    ARGS+=(--grad-accum 8)
fi

if grep -q -- '--v11-adapter' <<<"$HELP"; then
    ARGS+=(--v11-adapter artifacts/v14_benign_policy)
fi

"$PY" -u "$SCRIPT" "${ARGS[@]}"
