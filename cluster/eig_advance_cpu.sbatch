#!/bin/bash
#SBATCH --job-name=eig-adv
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eig-adv-%j.out
#SBATCH --error=logs/eig-adv-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"; export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"; export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"; export PYTHONNOUSERSITE=1; export PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u scripts/kboot_advance_role_only.py --round "$EIG_ROUND" --split dev --root "$EIG_ROOT" --belief "$EIG_BELIEF" --manifest "$EIG_ROOT/manifest_privileged.json"
