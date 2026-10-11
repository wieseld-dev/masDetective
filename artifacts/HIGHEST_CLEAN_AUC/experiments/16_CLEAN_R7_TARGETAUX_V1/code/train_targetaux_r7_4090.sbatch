#!/bin/bash
#SBATCH --job-name=r7-taux
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gpus=1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=artifacts/HIGHEST_CLEAN_AUC/experiments/16_CLEAN_R7_TARGETAUX_V1/logs/train-%j.out
#SBATCH --error=artifacts/HIGHEST_CLEAN_AUC/experiments/16_CLEAN_R7_TARGETAUX_V1/logs/train-%j.out

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

EXP="artifacts/HIGHEST_CLEAN_AUC/experiments/16_CLEAN_R7_TARGETAUX_V1"
PARENT="artifacts/v14_adaptive_role_v1/checkpoints/round_06/policy"
V11="artifacts/v14_benign_policy"

RUNNER="$EXP/code/run_seeded_r7_update.py"
UPDATER="scripts/v14_adaptive_grpo_update_v1.py"

train_one () {
    NAME="$1"
    ROOT="$EXP/$NAME"

    echo
    echo "======================================================================"
    echo "TRAINING: $NAME"
    echo "ROOT:     $ROOT"
    echo "PARENT:   $PARENT"
    echo "======================================================================"

    rm -rf "$ROOT/checkpoints/round_07"

    V14A_START_ADAPTER="$PARENT" \
    "$PY" "$RUNNER" "$UPDATER" \
      --round 7 \
      --root "$ROOT" \
      --v11-adapter "$V11" \
      --epochs 1 \
      --lr 1.5e-6 \
      --clip 0.20 \
      --kl-beta 0.05 \
      --max-seq-length 2048

    echo
    echo "===== $NAME SUMMARY ====="
    cat "$ROOT/checkpoints/round_07/update_summary.json"
}

echo "===== GPU ====="
nvidia-smi

echo
echo "===== PRECHECK DATASET SIZES ====="
wc -l \
  "$EXP/control/grpo_train_round_07.jsonl" \
  "$EXP/global_sign/grpo_train_round_07.jsonl" \
  "$EXP/composite_sign/grpo_train_round_07.jsonl"

train_one "control"
train_one "global_sign"
train_one "composite_sign"

echo
echo "======================================================================"
echo "ALL THREE R7 UPDATES COMPLETED"
echo "======================================================================"

"$PY" - <<'PY'
import json
from pathlib import Path

base = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/16_CLEAN_R7_TARGETAUX_V1"
)

for name in [
    "control",
    "global_sign",
    "composite_sign",
]:
    p = (
        base /
        name /
        "checkpoints/round_07/update_summary.json"
    )

    x = json.loads(
        p.read_text()
    )

    st = x["stats"][-1]

    print(
        f"{name:16s} "
        f"samples={x['samples']:4d} "
        f"loss={st['loss']:+.6f} "
        f"pg={st['pg']:+.6f} "
        f"kl={st['kl']:.6f} "
        f"n={st['n']}"
    )
PY
