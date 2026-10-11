#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"

EXP_ROOT="${CTRL_EXPERIMENT_ROOT:-artifacts/adaptive_v11_controller_v1}"
SOURCE_MANIFEST="${CTRL_SOURCE_MANIFEST:-artifacts/adaptive_v11_k_bootstrap_v1/manifest_privileged.json}"
BELIEF="${CTRL_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"
ADAPTER="${CTRL_ADAPTER:-artifacts/v11_policy}"

[ -s "$SOURCE_MANIFEST" ] || { echo "missing source manifest: $SOURCE_MANIFEST"; exit 1; }
[ -s "$BELIEF" ] || { echo "missing frozen role belief: $BELIEF"; exit 1; }
[ -d "$ADAPTER" ] || { echo "missing V11 adapter: $ADAPTER"; exit 1; }

if [ -e "$EXP_ROOT" ] && find "$EXP_ROOT" -type f -name 'responses_*' -print -quit | grep -q .; then
  echo "REFUSING: $EXP_ROOT already contains executed responses. Set a new CTRL_EXPERIMENT_ROOT."
  exit 1
fi

mkdir -p "$EXP_ROOT/common" "$EXP_ROOT/static" "$EXP_ROOT/adaptive" "$EXP_ROOT/report"
for arm in common static adaptive; do
  cp "$SOURCE_MANIFEST" "$EXP_ROOT/$arm/manifest_privileged.json"
  "$PY" -u scripts/adaptive_v11_init_states.py \
    --manifest "$EXP_ROOT/$arm/manifest_privileged.json" \
    --belief "$BELIEF" \
    --output-dir "$EXP_ROOT/$arm"
done

"$PY" -u scripts/av11_controller_preflight.py \
  --root "$EXP_ROOT" \
  --manifest "$SOURCE_MANIFEST" \
  --belief "$BELIEF" \
  --adapter "$ADAPTER"

echo
echo "PREPARED ONLY. NOTHING EXPENSIVE WAS SUBMITTED."
echo "EXP_ROOT=$EXP_ROOT"
echo "MAX TARGET QUERIES = 32 shared R1 + 2*32*5 = 352"
