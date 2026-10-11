#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=belief5-bg
#SBATCH --output=logs/belief5-bg-%j.out
#SBATCH --error=logs/belief5-bg-%j.err
#SBATCH --exclude=cs-4090-09

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY=/home/wieseld/.conda/envs/masdetective/bin/python

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

if [ -f .env ]; then
    set +u
    set -a
    source .env
    set +a
    set -u
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY missing" >&2
    exit 2
fi

echo "============================================================"
echo "EXACT HISTORICAL FIVE-SYSTEM CONTROL + BENIGN JUDGE ONLY"
echo
echo "Historical reference:"
echo "  mean belief AUC   = 0.8471"
echo "  median belief AUC = 0.8824"
echo
echo "ONLY CHANGE:"
echo "  + learned Benign Judge V4 hard gate"
echo
echo "UNCHANGED:"
echo "  + CurriculumAdaptiveProbePolicyV11"
echo "  + historical filters/retries"
echo "  + candidate generation structure"
echo "  + ObservationNet"
echo "  + MI selector"
echo "  + MAS systems"
echo "  + Q+O visible fingerprint"
echo "  + calibration"
echo "  + particle Bayesian update"
echo "  + 8 rounds"
echo "============================================================"

"$PY" -u \
  analysis/five_system_belief_auc_control_judge_v1.py
