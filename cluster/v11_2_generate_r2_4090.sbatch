#!/usr/bin/env bash
#SBATCH --job-name=v112-g2
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=24G
#SBATCH --output=logs/v112-g2-%j.out
#SBATCH --error=logs/v112-g2-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"; mkdir -p logs artifacts/v11_2_split
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
heartbeat(){ while true; do echo "[heartbeat] $(date -Is) job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>/dev/null | sed 's/^/[gpu] /' || true; sleep 120; done; }
heartbeat & HB=$!
trap 'kill "$HB" 2>/dev/null || true' EXIT
python -u -m training.v11_2_generate_round --round 2 --adapter artifacts/v11_2_split/checkpoints/round_01/policy --previous-eval artifacts/v11_2_split/evaluated_r1.jsonl --output artifacts/v11_2_split/generated_r2.jsonl --no-4bit
