#!/usr/bin/env bash
#SBATCH --job-name=mas5-v55-sensor
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v55-%x-%j.out
#SBATCH --error=logs/v55-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
: "${V55_SOURCE:?Set V55_SOURCE to the V5.4 root}"
: "${V55_OUT:?Set V55_OUT to a new V5.5 results directory}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v55_sensor_r2_diagnostic score --root "$V55_SOURCE" --out "$V55_OUT"
