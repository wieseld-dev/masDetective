#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: v137_overnight.sh audit|train}"

cd ~/mas_rl_detective

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export TOKENIZERS_PARALLELISM=false

PY=$(which python)
OUT="artifacts/v13_7_fp3b_overnight"

COMMON=(
  --v13-discovery artifacts/v13_probe_disambiguation.jsonl
  --v11-discovery artifacts/v11_probe_discovery.jsonl
  --output-dir "$OUT"

  --seed 1331
  --n-splits 5
  --fold 0

  --v11-source-weight 0.70

  --neg-ratio 3.0
  --self-target-neg-weight 3.0
  --hardneg-weight 2.0

  --base-model Qwen/Qwen2.5-3B-Instruct

  --lora-r 32
  --lora-alpha 64
  --lora-dropout 0.05

  --max-length 1024

  --batch-size 2
  --pair-batch-size 2
  --eval-batch-size 4
  --eval-pair-batch-size 2
  --grad-accum 8

  --epochs 6
  --lr 5e-5

  --pair-loss-weight 0.50
  --pair-margin 1.00

  --early-stop-patience 2
  --min-delta 1e-4
)

mkdir -p "$OUT"

if [[ "$MODE" == "audit" ]]; then
  echo "============================================================"
  echo "V13.7 OVERNIGHT DATA AUDIT"
  echo "============================================================"

  "$PY" analysis/train_v13_6b_activation_judge.py \
    "${COMMON[@]}" \
    --audit-only

  test -s "$OUT/data_audit.json"
  test -s "$OUT/split_report.json"

  echo
  echo "AUDIT JOB PASS"

elif [[ "$MODE" == "train" ]]; then
  echo "============================================================"
  echo "V13.7 FP3B OVERNIGHT TRAINING"
  echo "============================================================"

  test -s "$OUT/data_audit.json" || {
    echo "FATAL: audit artifact missing"
    exit 20
  }

  echo "===== GPU ====="
  nvidia-smi \
    --query-gpu=name,memory.total \
    --format=csv,noheader

  MEM=$(
    nvidia-smi \
      --query-gpu=memory.total \
      --format=csv,noheader,nounits \
      | head -1
  )

  if [[ "$MEM" -lt 20000 ]]; then
    echo "FATAL: refusing 3B training on GPU with ${MEM} MiB"
    exit 21
  fi

  "$PY" analysis/train_v13_6b_activation_judge.py \
    "${COMMON[@]}"

  test -d "$OUT/best_adapter"
  test -s "$OUT/validation_metrics.json"
  test -s "$OUT/per_role_metrics.json"
  test -s "$OUT/training_history.json"

  cp analysis/train_v13_6b_activation_judge.py \
     "$OUT/training_script_snapshot.py"

  echo
  echo "============================================================"
  echo "FINAL V13.7 SUMMARY"
  echo "============================================================"

  "$PY" - <<'PY'
import json
from pathlib import Path

NEW = Path("artifacts/v13_7_fp3b_overnight")
OLD = Path("artifacts/v13_6b_activation_judge")

keys = [
    "presence_micro_auc",
    "presence_macro_auc",
    "balanced_accuracy_at_0_5",
    "self_target_present_vs_absent_auc",
    "self_target_absent_fpr_at_0_5",
    "counterfactual_accuracy",
    "selection_score",
]

new = json.loads(
    (NEW / "validation_metrics.json").read_text()
)

old = None
op = OLD / "validation_metrics.json"
if op.exists():
    old = json.loads(op.read_text())

print("\nMETRIC COMPARISON")
print("-" * 84)

for k in keys:
    nv = new.get(k)
    ov = None if old is None else old.get(k)

    if nv is None:
        print(f"{k:40s} NEW=NA")
        continue

    if ov is None:
        print(f"{k:40s} NEW={nv:.6f}")
    else:
        print(
            f"{k:40s} "
            f"OLD={ov:.6f} "
            f"NEW={nv:.6f} "
            f"DELTA={nv-ov:+.6f}"
        )

audit = json.loads(
    (NEW / "data_audit.json").read_text()
)

print("\nDATA")
print("total_cases              :", audit.get("total_cases"))
print("v13_cases                :", audit.get("v13_cases"))
print("v11_cases                :", audit.get("v11_cases"))
print(
    "train_counterfactual_pairs:",
    audit.get("train_counterfactual_pairs")
)
print(
    "val_counterfactual_pairs  :",
    audit.get("val_counterfactual_pairs")
)

hist = json.loads(
    (NEW / "training_history.json").read_text()
)

print("\nEPOCHS")
for x in hist:
    print(
        "epoch", x.get("epoch"),
        "macro_auc=", x.get("presence_macro_auc"),
        "self_abs_fpr=", x.get("self_target_absent_fpr_at_0_5"),
        "cf_acc=", x.get("counterfactual_accuracy"),
        "select=", x.get("selection_score"),
    )
PY

  echo
  echo "===== BEST ADAPTER HASH ====="
  find "$OUT/best_adapter" -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sha256sum

  echo
  echo "TRAINING JOB PASS"

else
  echo "unknown mode: $MODE"
  exit 2
fi
