#!/bin/bash
#SBATCH --job-name=chain-audit
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/chain-audit-%j.out
#SBATCH --error=logs/chain-audit-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

TMPBASE="${SLURM_SCRATCH_DIR:-/tmp}"
mkdir -p "$TMPBASE/tmp"
export TMPDIR="$TMPBASE/tmp"
export JOBLIB_TEMP_FOLDER="$TMPBASE/tmp"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -m py_compile \
  scripts/chain_mechanism_audit_v2.py

"$PY" -u \
  scripts/chain_mechanism_audit_v2.py \
  --steps 2500 \
  --restarts 4 \
  --null-reps 1500 \
  --bootstrap-reps 1500 \
  --dims 8,16,32,64 \
  --dim-steps 1600 \
  --dim-restarts 2
