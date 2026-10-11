#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v58-%x-%j.out
#SBATCH --error=logs/v58-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${V58_ROOT:?missing run root}"
: "${V58_STAGE:?missing stage}"
: "${V58_ROUND:?missing round}"
[[ "$V58_STAGE" == execute ]] || { echo 'CPU stage only execute' >&2; exit 2; }
set -a
source .env
set +a
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v58_adaptive_eight \
  execute --root "$V58_ROOT" --round "$V58_ROUND"
