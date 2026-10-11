#!/bin/bash
#SBATCH --job-name=kb-exec
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/kb-exec-%j.out
#SBATCH --error=logs/kb-exec-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${KBOOT_ROOT:?set KBOOT_ROOT}"
: "${KBOOT_ROUND:?set KBOOT_ROUND}"
: "${KBOOT_SPLIT:?set KBOOT_SPLIT}"
"$PY" -u scripts/adaptive_v11_execute_round.py \
  --round "$KBOOT_ROUND" \
  --split "$KBOOT_SPLIT" \
  --root "$KBOOT_ROOT" \
  --manifest "$KBOOT_ROOT/manifest_privileged.json" \
  --workers "${KBOOT_WORKERS:-4}" \
  --retries 3
