#!/bin/bash
#SBATCH --job-name=gr-v14p-v2
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gr-v14p-v2-%j.out
#SBATCH --error=logs/gr-v14p-v2-%j.err

set -euo pipefail

cd /home/wieseld/mas_rl_detective
mkdir -p logs artifacts/general_role_bank_v1

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

echo "============================================================"
echo "GENERAL ROLE V14 PROBES V2"
echo "host=$(hostname)"
echo "start=$(date)"
echo "============================================================"

"$PY" -u \
  analysis/generate_general_role_v14_probes_v2.py

echo "============================================================"
echo "done=$(date)"
echo "============================================================"
