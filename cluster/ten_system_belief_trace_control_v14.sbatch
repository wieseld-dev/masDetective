#!/bin/bash
#SBATCH --job-name=belief10-v14
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/belief10-v14-%j.out
#SBATCH --error=logs/belief10-v14-%j.err

set -euo pipefail

cd /home/wieseld/mas_rl_detective

export TOKENIZERS_PARALLELISM=false

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing}"

echo "============================================================"
echo "10-MAS V14 GENERATOR A/B"
echo "job=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "start=$(date)"
echo "============================================================"
echo "Generator: V14 benign / DisambiguationProbePolicyV13"
echo "Sensor: historical visible fingerprint detector"
echo "Selector: historical visible-MI selector"
echo "Systems: same 10 K=3 MAS"
echo "Rounds: 8"
echo "Candidates/round: 8"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total \
  --format=csv,noheader

/home/wieseld/.conda/envs/masdetective/bin/python \
  analysis/ten_system_belief_trace_control_v14.py

echo "============================================================"
echo "done=$(date)"
echo "============================================================"
