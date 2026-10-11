#!/bin/bash
#SBATCH --job-name=v2-audit
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-00:20:00
#SBATCH --output=logs/v2-audit-%j.out
#SBATCH --error=logs/v2-audit-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" -u scripts/audit_curriculum_v2.py \
  --root artifacts/adaptive_v11_curriculum_v2
