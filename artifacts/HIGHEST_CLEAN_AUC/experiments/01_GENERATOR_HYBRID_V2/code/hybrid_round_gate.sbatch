#!/bin/bash
#SBATCH --job-name=hyb-v2-gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --output=logs/hyb-v2-gate-%j.out
#SBATCH --error=logs/hyb-v2-gate-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

: "${V14A_ROOT:?}"
: "${V14A_ROUND:?}"
: "${GATE_STAGE:?}"

"$HOME/.conda/envs/masdetective/bin/python" \
  artifacts/HIGHEST_CLEAN_AUC/experiments/01_GENERATOR_HYBRID_V2/code/hybrid_round_gate.py \
  --root "$V14A_ROOT" \
  --round "$V14A_ROUND" \
  --stage "$GATE_STAGE"
