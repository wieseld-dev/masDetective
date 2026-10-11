#!/bin/bash
#SBATCH --job-name=v14-sweep
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/v14-sweep-%j.out
#SBATCH --error=logs/v14-sweep-%j.err

set -euo pipefail

cd /home/wieseld/mas_rl_detective

export PYTHONPATH="/home/wieseld/mas_rl_detective:${PYTHONPATH:-}"

PY=/home/wieseld/.conda/envs/masdetective/bin/python

test -x "$PY"

set -a
source .env
set +a

test -n "${OPENAI_API_KEY:-}"

echo "============================================================"
echo "V14 COMPLETE ROLE SWEEP + TOP-6 CONFIRMATION"
echo "job=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "start=$(date)"
echo "============================================================"
echo "20 screening queries / MAS"
echo "6 confirmation queries / MAS"
echo "26 queries / MAS"
echo "260 maximum real MAS calls"
echo "NO historical MI selector"
echo "NO historical Bayes update"
echo "NO K feedback"
echo "Historical visible Q+O Sensor retained"
echo "============================================================"

nvidia-smi \
  --query-gpu=name,memory.total \
  --format=csv,noheader

"$PY" -u \
  analysis/v14_full_role_sweep_v1.py
