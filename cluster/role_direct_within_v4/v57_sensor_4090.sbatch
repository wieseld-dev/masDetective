#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v57-%x-%j.out
#SBATCH --error=logs/v57-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
: "${V57_RUN:?V57_RUN missing}" "${V57_V56:?V57_V56 missing}"
echo "MAS5 V5.7: Sensor only; NO target API requests; untouched V56 jobs"
echo "Parent adapter: artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/checkpoints/sensor_epoch02"
echo "Training max ${V57_HOURS:-8.75} hours; source $V57_V56"
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m \
 analysis.role_direct_within_v4.v57_sensor_night train \
 --run "$V57_RUN" --v56 "$V57_V56" --hours "${V57_HOURS:-8.75}" \
 --max-tokens "${V57_TOKENS:-1280}" --max-steps "${V57_MAX_STEPS:-6000}" \
 --eval-every "${V57_EVAL_EVERY:-150}" --patience "${V57_PATIENCE:-10}"
