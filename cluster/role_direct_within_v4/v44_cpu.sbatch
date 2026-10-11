#!/usr/bin/env bash
#SBATCH --job-name=v44-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v44-%x-%j.out
#SBATCH --error=logs/v44-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
if [[ "${V44_STAGE}" == execute_train || "${V44_STAGE}" == test_execute1 || "${V44_STAGE}" == test_execute2 ]]; then
  set -a
  source .env
  set +a
fi
exec "$HOME/.conda/envs/masdetective/bin/python" -u analysis/role_direct_within_v4/v44_night.py "$V44_STAGE" --root "$V44_ROOT"
