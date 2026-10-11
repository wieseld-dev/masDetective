#!/bin/bash
#SBATCH --job-name=v112-ev-x
#SBATCH --partition=main
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/v112-ev-x-%j.out
#SBATCH --error=logs/v112-ev-x-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
mkdir -p logs artifacts/v11_2_strong_eval
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
if [ -f .env ]; then set -a; source .env; set +a; fi
heartbeat(){ while true; do echo "[heartbeat] $(date -Is) job=$SLURM_JOB_ID host=$(hostname) phase=CPU_API_EVAL"; sleep 120; done; }
heartbeat & HB=$!; trap 'kill $HB 2>/dev/null || true' EXIT
python -u evaluation/v11_2_strong_execute.py
