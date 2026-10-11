#!/bin/bash
#SBATCH --job-name=rc-aug
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rc-aug-%j.out
#SBATCH --error=logs/rc-aug-%j.err

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
  scripts/role_count_augmented_audit_v1.py

"$PY" -u \
  scripts/role_count_augmented_audit_v1.py
