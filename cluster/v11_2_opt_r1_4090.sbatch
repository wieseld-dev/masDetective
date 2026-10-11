#!/usr/bin/env bash
#SBATCH --job-name=v112-o1
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --output=logs/v112-o1-%j.out
#SBATCH --error=logs/v112-o1-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"; mkdir -p logs artifacts/v11_2_split
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
heartbeat(){ while true; do echo "[heartbeat] $(date -Is) job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null | sed 's/^/[gpu] /' || true; sleep 120; done; }
heartbeat & HB=$!
trap 'kill "$HB" 2>/dev/null || true' EXIT
python -u -m training.v11_2_optimize_round --round 1 --evaluated artifacts/v11_2_split/evaluated_r1.jsonl --adapter artifacts/v11_policy --no-4bit
