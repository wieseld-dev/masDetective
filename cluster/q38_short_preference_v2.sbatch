#!/bin/bash
#SBATCH --job-name=q38-shortpref
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/q38-shortpref-%j.out
#SBATCH --error=logs/q38-shortpref-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="$HOME/.venvs/masdetective-q38/bin/python"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export START_ADAPTER="artifacts/q38_online_r7match_v1/checkpoints/round_05/policy"
export DATA_ROOT="artifacts/q38_shortfingerprint_v2_strict"
export OUT_DIR="artifacts/q38_shortfingerprint_pref_v2/policy"

export MAX_LEN=1024
export LR=5e-6
export BETA=0.10
export GRAD_ACC=4
export EPOCHS=1

"$PY" - <<'PY'
import torch

assert torch.cuda.is_available()

p = torch.cuda.get_device_properties(0)

print("GPU =", p.name)
print(
    "VRAM_GIB =",
    round(p.total_memory / 1024**3, 2),
)

if p.total_memory < 85 * 1024**3:
    raise SystemExit(
        "REFUSING: need Blackwell ~96GB"
    )

print("BLACKWELL_RESOURCE_OK")
PY

"$PY" -u \
  analysis/train_q38_short_preference_v2.py
