#!/usr/bin/env bash
#SBATCH --job-name=mas5-v47-gpu
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v47-%x-%j.out
#SBATCH --error=logs/v47-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
case "$V46_STAGE" in
  train_sensor)
    exec "$PY" -u analysis/role_direct_within_v4/v46_sensor_train.py --root "$V46_ROOT";;
  train_generator)
    exec "$PY" -u analysis/role_direct_within_v4/v46_generator_train.py --root "$V46_ROOT" --stage dpo;;
  test_generate1|test_generate2|test_score1|test_score2)
    export V46_GEN_ADAPTER="$V46_ROOT/train/generator_adapter"
    export V46_SEED=20261011
    exec "$PY" -u analysis/role_direct_within_v4/v47_night.py "$V46_STAGE" --root "$V46_ROOT";;
  *) echo "Unknown V47 GPU stage: $V46_STAGE" >&2;exit 2;;
esac
