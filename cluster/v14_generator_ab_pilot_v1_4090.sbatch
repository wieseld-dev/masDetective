#!/bin/bash
#SBATCH --job-name=v14-ab-gen
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v14-ab-gen-%j.out
#SBATCH --error=logs/v14-ab-gen-%j.err

set -euo pipefail

cd /home/wieseld/mas_rl_detective
mkdir -p logs artifacts/v14_generator_ab_pilot_v1

set -a
if [ -f .env ]; then
    source .env
fi
set +a

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

echo "============================================================"
echo "V14 GENERATOR A/B PILOT"
echo "host=$(hostname)"
echo "start=$(date)"
echo "============================================================"

"$PY" analysis/v14_generator_ab_pilot_v1.py

echo "============================================================"
echo "done=$(date)"
echo "============================================================"
