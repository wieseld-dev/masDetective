#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v42-%x-%j.out
#SBATCH --error=logs/v42-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
"$HOME/.conda/envs/masdetective/bin/python" -u analysis/role_direct_within_v4/v42_probes.py --root "$V42_ROOT" --round 1
