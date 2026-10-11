#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --job-name=v31mas-auc
#SBATCH --output=logs/v31mas-auc-%J.out
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# V31_PYTHONPATH_FIX
PKG_DIR="$(find "$PWD" -maxdepth 5 -type d -name adaptive_v11_active -print -quit)"
if [ -z "$PKG_DIR" ]; then
    echo "ERROR: adaptive_v11_active directory not found under $PWD"
    exit 2
fi
PKG_PARENT="$(dirname "$PKG_DIR")"
export PYTHONPATH="$PKG_PARENT:$PWD:${PYTHONPATH:-}"
echo "adaptive_v11_active=$PKG_DIR"
echo "PYTHONPATH=$PYTHONPATH"

mkdir -p logs artifacts/eqs_v3_1_mas_auc

echo "============================================================"
echo "V3.1-MAS SEMANTIC AUC"
echo "============================================================"
echo "JOB=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "START=$(date)"
echo

# Find an EXISTING conda Python that already contains the V14 stack.
# This runs on the allocated compute node, not on the login node.
PYTHON=""
for p in "$HOME"/.conda/envs/*/bin/python; do
    [ -x "$p" ] || continue
    if "$p" - <<'PY' >/dev/null 2>&1
import torch
import transformers
import sklearn
import joblib
import adaptive_v11_active
import v11
assert torch.cuda.is_available()
PY
    then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: no existing CUDA Python env with torch+transformers+sklearn+project modules"
    echo "Available envs:"
    ls -1 "$HOME/.conda/envs" || true
    exit 2
fi

echo "PYTHON=$PYTHON"
"$PYTHON" - <<'PY'
import torch, transformers, sklearn
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("sklearn", sklearn.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY

test -s scripts/v3_1_mas_auc_semantic.py
test -s artifacts/v14_adaptive_role_v1/manifest_privileged.json
test -s artifacts/v14_adaptive_role_v1/frozen_belief.joblib
test -s artifacts/v14_adaptive_role_v1/checkpoints/round_06/policy/adapter_config.json
test -s artifacts/v14_adaptive_role_v1/states_train_round_00.jsonl
test -s artifacts/v14_adaptive_role_v1/candidates_train_round_01.jsonl
test -s artifacts/v14_adaptive_role_v1/responses_train_round_01.jsonl
test -s artifacts/v14_adaptive_role_v1/scored_train_round_01.jsonl

echo "PRECHECK: PASS"
echo

"$PYTHON" -u scripts/v3_1_mas_auc_semantic.py \
  --root artifacts/v14_adaptive_role_v1 \
  --manifest artifacts/v14_adaptive_role_v1/manifest_privileged.json \
  --belief artifacts/v14_adaptive_role_v1/frozen_belief.joblib \
  --outdir artifacts/eqs_v3_1_mas_auc \
  --rounds 1,2,3,4,5,6

echo
echo "============================================================"
echo "V3.1-MAS COMPLETE"
echo "END=$(date)"
echo "RESULT=artifacts/eqs_v3_1_mas_auc/results.json"
echo "============================================================"
