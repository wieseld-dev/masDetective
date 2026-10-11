#!/bin/bash
#SBATCH --job-name=belief10-v14
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/belief10-v14-dropin-%j.out
#SBATCH --error=logs/belief10-v14-dropin-%j.err

set -euo pipefail

cd /home/wieseld/mas_rl_detective

PY=/home/wieseld/.conda/envs/masdetective/bin/python

test -x "$PY" || {
    echo "ERROR: Python env not found: $PY"
    exit 1
}

set -a
source .env
set +a

test -n "${OPENAI_API_KEY:-}" || {
    echo "ERROR: OPENAI_API_KEY missing"
    exit 1
}

echo "============================================================"
echo "10-MAS V14 GENERATOR TEST"
echo "============================================================"
echo "job=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "start=$(date)"
echo "python=$PY"
echo
echo "Historical attack pipeline preserved."
echo "Only generator implementation changed to V14."
echo
echo "Systems:    same 10"
echo "Rounds:     8"
echo "Candidates: 8"
echo "Selector:   historical MI"
echo "Sensor:     historical visible detector"
echo "Posterior:  historical particle Bayes"
echo "============================================================"

nvidia-smi \
  --query-gpu=name,memory.total \
  --format=csv,noheader

"$PY" -u \
  analysis/ten_system_belief_trace_control_v14_dropin.py
