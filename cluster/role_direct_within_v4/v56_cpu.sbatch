#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v56-%x-%j.out
#SBATCH --error=logs/v56-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${V56_ROOT:?missing run root}"
: "${V56_STAGE:?missing stage}"
if [[ "$V56_STAGE" == execute1 || "$V56_STAGE" == execute2 ]]; then
  set -a
  source .env
  set +a
fi
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v56_nonadaptive "$V56_STAGE" --root "$V56_ROOT"
