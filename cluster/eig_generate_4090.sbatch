#!/bin/bash
#SBATCH --job-name=eig-gen
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eig-gen-%j.out
#SBATCH --error=logs/eig-gen-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1; export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/av11_eig_generate.py --round "$EIG_ROUND" --split dev --root "$EIG_ROOT" --controller "$EIG_MODE" --utility-model "$EIG_UTILITY" --candidate-pool "${EIG_POOL:-8}" --max-seq-length 4096 --max-new-tokens 192 --local-retries 20 --no-4bit
