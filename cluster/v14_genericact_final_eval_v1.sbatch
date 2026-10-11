#!/bin/bash
#SBATCH --job-name=genact-eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/genact-eval-%j.out
#SBATCH --error=logs/genact-eval-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing (.env not loaded?)}"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

PY="$HOME/.conda/envs/masdetective/bin/python"

ROOT="artifacts/v14a_genericact_r1_eval_v1"
ADAPTER="artifacts/v14_adaptive_g50_genericact_r1_v1/checkpoints/round_01/policy"

for R in $(seq 1 6); do

    echo
    echo "================================================================"
    echo "ROUND $R / 6"
    echo "================================================================"

    echo "=== GENERATE: GenericAct fixed adapter ==="

    "$PY" scripts/v14_adaptive_generate_round_v1.py \
      --round "$R" \
      --split dev \
      --group-size 1 \
      --root "$ROOT" \
      --adapter "$ADAPTER" \
      --v11-adapter artifacts/v14_benign_policy \
      --temperature 0.85 \
      --top-p 0.95 \
      --max-new-tokens 128 \
      --max-seq-length 3072

    echo "=== EXECUTE ==="

    "$PY" scripts/adaptive_v11_execute_round.py \
      --round "$R" \
      --split dev \
      --root "$ROOT" \
      --manifest "$ROOT/manifest_privileged.json" \
      --workers 4

    echo "=== SENSOR ==="

    "$PY" scripts/adaptive_v11_score_round.py \
      --round "$R" \
      --split dev \
      --root "$ROOT"

    echo "=== BELIEF UPDATE ==="

    "$PY" scripts/adaptive_v11_role_advance_round.py \
      --round "$R" \
      --split dev \
      --root "$ROOT" \
      --belief "$ROOT/frozen_belief.joblib" \
      --manifest "$ROOT/manifest_privileged.json"

    echo "=== ROUND $R RESULT ==="

    cat "$ROOT/advance_dev_round_$(printf '%02d' "$R")_summary.json"

done

echo
echo "================================================================"
echo "FINAL COMPARISON"
echo "================================================================"

"$PY" - <<'PY'
import json
from pathlib import Path

base = json.loads(
    Path(
        "artifacts/v14a_ckpt_eval_r07_v1/"
        "advance_dev_round_06_summary.json"
    ).read_text()
)

new = json.loads(
    Path(
        "artifacts/v14a_genericact_r1_eval_v1/"
        "advance_dev_round_06_summary.json"
    ).read_text()
)

b = float(base["mean_role_auc"])
n = float(new["mean_role_auc"])

print(f"Clean R7 AUC      = {b:.6f}")
print(f"GenericAct R1 AUC = {n:.6f}")
print(f"Delta             = {n-b:+.6f}")
print()
print(
    "RESULT =",
    "GENERICACT BETTER" if n > b else "GENERICACT NOT BETTER"
)
PY
