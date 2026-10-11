#!/bin/bash
#SBATCH --job-name=av11-rrep
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/av11-rrep-%j.out
#SBATCH --error=logs/av11-rrep-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
ROOT="${AV11_ROLE_ROOT:-artifacts/adaptive_v11_role_only_v1}"
"$PY" -u scripts/adaptive_v11_role_report.py --rounds "${AV11_ROLE_ROUNDS:-3}" --root "$ROOT"
