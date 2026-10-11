#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-01:00:00
#SBATCH --job-name=comm-v11
#SBATCH --output=logs/comm-v11-%J.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"

echo "JOB=$SLURM_JOB_ID"
echo "NODE=$SLURM_JOB_NODELIST"
echo "PYTHON=$PY"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY missing"
    exit 1
fi

echo "OPENAI_API_KEY detected"

"$PY" blackbox_commutator_v11_poc.py
