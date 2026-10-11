#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --job-name=v32-clpilot
#SBATCH --output=logs/v32-clpilot-%J.out
#SBATCH --error=logs/v32-clpilot-%J.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p \
  logs \
  artifacts/eqs_v3_2_closed_loop

PKG_DIR="$(
  find "$PWD" \
    -maxdepth 5 \
    -type d \
    -name adaptive_v11_active \
    -print \
    -quit
)"

if [ -z "$PKG_DIR" ]; then
    echo "ERROR: adaptive_v11_active not found"
    exit 2
fi

PKG_PARENT="$(
  dirname "$PKG_DIR"
)"

export PYTHONPATH="$PKG_PARENT:$PWD:${PYTHONPATH:-}"

PYTHON=""

for p in "$HOME"/.conda/envs/*/bin/python; do
    [ -x "$p" ] || continue

    if "$p" - <<'PY' >/dev/null 2>&1
import torch
import transformers
import sklearn
import adaptive_v11_active

assert torch.cuda.is_available()
PY
    then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: no existing compatible CUDA environment"
    exit 3
fi

echo "PYTHON=$PYTHON"
echo "PYTHONPATH=$PYTHONPATH"

"$PYTHON" - <<'PY'
import os
import torch

print(
    "cuda:",
    torch.cuda.is_available()
)

print(
    "gpu:",
    torch.cuda.get_device_name(0)
)

print(
    "OPENAI_API_KEY present:",
    bool(
        os.environ.get(
            "OPENAI_API_KEY"
        )
    )
)

if not os.environ.get(
    "OPENAI_API_KEY"
):
    raise SystemExit(
        "OPENAI_API_KEY is not set"
    )
PY

test -s scripts/v32_closed_loop_auc_pilot.py
test -s scripts/v3_1_mas_auc_semantic.py
test -s scripts/adaptive_v11_execute_round.py
test -s scripts/adaptive_v11_score_round.py
test -s scripts/kboot_advance_role_only.py

echo
echo "START=$(date)"
echo "Nominal maximum MAS executions: 144"
echo

"$PYTHON" -u \
  scripts/v32_closed_loop_auc_pilot.py \
  --source-root artifacts/v14_adaptive_role_v1 \
  --manifest artifacts/v14_adaptive_role_v1/manifest_privileged.json \
  --belief artifacts/v14_adaptive_role_v1/frozen_belief.joblib \
  --v31-script scripts/v3_1_mas_auc_semantic.py \
  --v31-ckpts artifacts/eqs_v3_1_mas_auc \
  --outdir artifacts/eqs_v3_2_closed_loop/pilot \
  --rounds 6 \
  --bank-size 512 \
  --workers 4 \
  --reset

echo
echo "END=$(date)"
