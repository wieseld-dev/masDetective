#!/usr/bin/env bash
#SBATCH --job-name=v43-hmhy
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v43-hmhy-%j.out
#SBATCH --error=logs/v43-hmhy-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ROOT="${V43_ROOT:?V43_ROOT not set}"
"$HOME/.conda/envs/masdetective/bin/python" -u analysis/role_direct_within_v4/v43_elicitation.py --root "$ROOT" --round 1
