#!/bin/bash
#SBATCH --job-name=gr-sens-col
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/gr-sens-col-%j.out
#SBATCH --error=logs/gr-sens-col-%j.err

set -euo pipefail

cd /home/wieseld/mas_rl_detective

mkdir -p logs artifacts/general_role_sensor_v1

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing}"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

ARGS=(--workers "${WORKERS:-4}" --retries 3)

if [ "${PILOT:-0}" = "1" ]; then
    ARGS+=(--pilot)
fi

echo "============================================================"
echo "GENERAL SENSOR COLLECTION"
echo "job=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "pilot=${PILOT:-0}"
echo "workers=${WORKERS:-4}"
echo "start=$(date)"
echo "============================================================"

"$PY" -u analysis/collect_general_sensor_v1.py "${ARGS[@]}"

echo "============================================================"
echo "done=$(date)"
echo "============================================================"
