#!/bin/bash
#SBATCH --job-name=av11-upd
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/av11-upd-%j.out
#SBATCH --error=logs/av11-upd-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
: "${AV11_ROUND:?set AV11_ROUND}"
"$PY" -u scripts/adaptive_v11_grpo_update.py --round "$AV11_ROUND" --no-4bit
