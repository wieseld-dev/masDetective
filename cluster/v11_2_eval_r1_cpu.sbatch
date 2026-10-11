#!/usr/bin/env bash
#SBATCH --job-name=v112-e1
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --output=logs/v112-e1-%j.out
#SBATCH --error=logs/v112-e1-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"; mkdir -p logs artifacts/v11_2_split
if [ -f .env ]; then set -a; source .env; set +a; fi
source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
heartbeat(){ while true; do echo "[heartbeat] $(date -Is) job=$SLURM_JOB_ID host=$(hostname) phase=CPU_API_EVAL"; sleep 120; done; }
heartbeat & HB=$!
trap 'kill "$HB" 2>/dev/null || true' EXIT
python -u -m training.v11_2_evaluate_round --round 1 --generated artifacts/v11_2_split/generated_r1.jsonl --output artifacts/v11_2_split/evaluated_r1.jsonl
