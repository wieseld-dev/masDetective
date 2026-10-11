#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v59-%x-%j.out
#SBATCH --error=logs/v59-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${V59_ROOT:?missing root}" "${V59_STAGE:?missing stage}"
if [[ "$V59_STAGE" == execute* ]]; then
  set -a
  source .env
  set +a
fi
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v59_fresh_same_mas "$V59_STAGE" --root "$V59_ROOT"
