#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=hist5-bg
#SBATCH --output=logs/hist5-bg-%j.out
#SBATCH --error=logs/hist5-bg-%j.err
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

echo "======================================================================"
echo "HISTORICAL 0.8471 PIPELINE + BENIGN JUDGE"
echo "======================================================================"
echo
echo "Generator implementation:"
echo "  adaptive_v11_curriculum/policy_historical_0847.py"
echo
echo "Generator weights:"
echo "  artifacts/robust_visible_generator_v1/policy"
echo
echo "Judge:"
echo "  standalone_benign_critic_binary_v4_hardening"
echo
echo "Everything else:"
echo "  historical five_system_belief_auc_control_v1 pipeline"
echo
echo "======================================================================"

"$PY" -u \
  analysis/five_system_belief_auc_historical_judge_v1.py

echo
echo "===== FINAL SUMMARY ====="

cat \
  artifacts/five_system_belief_auc_historical_judge_v1/summary.json
