#!/bin/bash
#SBATCH --job-name=v14-deep3
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/v14-deep3-%j.out
#SBATCH --error=logs/v14-deep3-%j.err

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
echo "V14 DEEP ROLE NEW CHALLENGE V1"
echo "job=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "start=$(date)"
echo "============================================================"
echo "3 NEW CHALLENGE MAS"
echo "20 candidate roles"
echo "10 distinct V14 probes per role"
echo "200 queries / MAS"
echo "600 maximum MAS queries"
echo "PRIMARY aggregation = mean target sensor score"
echo "NO selector"
echo "NO Bayes update"
echo "NO K feedback"
echo "routing trace = evaluation only"
echo "============================================================"

nvidia-smi \
  --query-gpu=name,memory.total \
  --format=csv,noheader

"$PY" -u \
  analysis/v14_deep_role_challenge_v1.py
