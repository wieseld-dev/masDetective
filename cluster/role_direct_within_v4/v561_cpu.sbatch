#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v561-%x-%j.out
#SBATCH --error=logs/v561-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${V561_ROOT:?missing root}"
: "${V561_STAGE:?missing stage}"
if [[ "$V561_STAGE" == execute1 || "$V561_STAGE" == execute2 ]]; then
  set -a
  source .env
  set +a
fi
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v561_live_auc "$V561_STAGE" --root "$V561_ROOT"
