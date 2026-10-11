#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v510-%x-%j.out
#SBATCH --error=logs/v510-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
: "${V510_RUN:?V510_RUN required}"
echo "V510: GENERALIZATION TRAIN, frozen role composition split; no API calls."
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v510_generalizing_sensor train \
 --run "$V510_RUN" --hours "${V510_HOURS:-8.75}" --max-steps "${V510_STEPS:-6000}" \
 --max-tokens "${V510_TOKENS:-1280}" --eval-every "${V510_EVAL_EVERY:-150}" --patience "${V510_PATIENCE:-12}"
