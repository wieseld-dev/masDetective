#!/usr/bin/env bash
#SBATCH --job-name=v44-gpu
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v44-%x-%j.out
#SBATCH --error=logs/v44-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY="$HOME/.conda/envs/masdetective/bin/python"
case "${V44_STAGE}" in
  train_probes)
    export V44_GEN_ADAPTER="$PWD/artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/checkpoints/generator_activation_sft"
    export V44_SEED=20261017
    mkdir -p "$V44_ROOT/train/probes"
    exec "$PY" -u analysis/role_direct_within_v4/v44_generator.py --root "$V44_ROOT/train/probes" --round 1 \
      --repair-bank "$PWD/artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/role_upgrade_v3/hmhy_20261009_215152_1793839/v43_hmhy_candidate_probes_round_01.jsonl"
    ;;
  train_sensor)
    exec "$PY" -u analysis/role_direct_within_v4/v44_sensor_train.py --root "$V44_ROOT"
    ;;
  train_generator)
    exec "$PY" -u analysis/role_direct_within_v4/v44_generator_train.py --root "$V44_ROOT"
    ;;
  test_generate1|test_generate2)
    export V44_GEN_ADAPTER="$V44_ROOT/train/generator_adapter"
    export V44_SEED=20261011
    exec "$PY" -u analysis/role_direct_within_v4/v44_night.py "$V44_STAGE" --root "$V44_ROOT"
    ;;
  test_score1|test_score2)
    exec "$PY" -u analysis/role_direct_within_v4/v44_night.py "$V44_STAGE" --root "$V44_ROOT"
    ;;
  *) echo "Invalid V44 GPU stage: $V44_STAGE" >&2;exit 1;;
esac
