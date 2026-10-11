#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/v4u-%x-%j.out
#SBATCH --error=logs/v4u-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
"$HOME/.conda/envs/masdetective/bin/python" -u analysis/role_direct_within_v4/role_upgrade_v1.py report --root "$V4_ROOT"
