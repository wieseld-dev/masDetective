#!/bin/bash
#SBATCH --job-name=v112-ev-g
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=32G
#SBATCH --output=logs/v112-ev-g-%j.out
#SBATCH --error=logs/v112-ev-g-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
mkdir -p logs artifacts/v11_2_strong_eval
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
heartbeat(){ while true; do echo "[heartbeat] $(date -Is) job=$SLURM_JOB_ID host=$(hostname)"; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader || true; sleep 120; done; }
heartbeat & HB=$!; trap 'kill $HB 2>/dev/null || true' EXIT
python -u evaluation/v11_2_strong_generate.py --no-4bit
