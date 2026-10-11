#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=hist10-bg
#SBATCH --output=logs/hist10-bg-%j.out
#SBATCH --error=logs/hist10-bg-%j.err
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

test -n "${OPENAI_API_KEY:-}" || {
    echo "OPENAI_API_KEY missing" >&2
    exit 2
}

echo "======================================================================"
echo "TEN-SYSTEM HISTORICAL GENERATOR + BENIGN JUDGE"
echo "======================================================================"
echo "Historical policy: adaptive_v11_curriculum/policy_historical_0847.py"
echo "Generator weights : artifacts/robust_visible_generator_v1/policy"
echo "Systems           : 10"
echo "Rounds/system     : 8"
echo "MAS calls         : 80"
echo "======================================================================"

"$PY" -u \
  analysis/ten_system_belief_auc_historical_judge_v1.py

echo
echo "===================== FINAL SUMMARY ====================="

cat \
  artifacts/ten_system_belief_auc_historical_judge_v1/summary.json
