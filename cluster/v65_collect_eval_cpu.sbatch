#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:30:00
#SBATCH --job-name v65-cand
#SBATCH --output logs/v65-cand-%J.out

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
PY="$HOME/.conda/envs/masdetective/bin/python"

set -a
source .env
set +a

# Collect only NEW diverse systems. V6.3 data is reused automatically by evaluation.
"$PY" -u -m training.collect_candidate_data_v65 --split both --execute

# CPU-only local embedding + held-out evaluation; no OpenAI calls here.
"$PY" -u -m evaluation.eval_candidate_presence_v65
