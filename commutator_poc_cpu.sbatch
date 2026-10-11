#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-01:00:00
#SBATCH --job-name=comm-poc
#SBATCH --output=logs/comm-poc-cpu-%J.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"

echo "JOB=$SLURM_JOB_ID"
echo "NODE=$SLURM_JOB_NODELIST"
echo "PYTHON=$PY"

if [[ ! -x "$PY" ]]; then
    echo "ERROR: masdetective Python not found at $PY"
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY missing"
    exit 1
fi

echo "OPENAI_API_KEY detected"

"$PY" - <<'PY'
import numpy
import torch
import scipy
import sklearn
import sentence_transformers
import openai

print("Imports OK")
print("torch:", torch.__version__)
PY

"$PY" blackbox_commutator_poc.py \
    --n-prompts 60 \
    --concurrency 8
