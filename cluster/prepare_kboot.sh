#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"

ROOT="${KBOOT_ROOT:-artifacts/adaptive_v11_k_bootstrap_v1}"
BELIEF="${KBOOT_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"

test -s "$BELIEF" || { echo "missing frozen role belief: $BELIEF" >&2; exit 2; }
test -d artifacts/v11_policy || { echo "missing artifacts/v11_policy" >&2; exit 2; }
test -d artifacts/v13_4_presence_judge/best_adapter || { echo "missing V13.4 adapter" >&2; exit 2; }
test -d artifacts/v13_5_evidence_judge/best_adapter || { echo "missing V13.5 adapter" >&2; exit 2; }

mkdir -p "$ROOT" logs
"$PY" -u scripts/kboot_make_manifest.py \
  --output "$ROOT/manifest_privileged.json" \
  --train-per-cell 6 --dev-per-cell 2

"$PY" -u scripts/adaptive_v11_init_states.py \
  --manifest "$ROOT/manifest_privileged.json" \
  --belief "$BELIEF" \
  --output-dir "$ROOT"

"$PY" -u scripts/kboot_preflight.py --root "$ROOT" --belief "$BELIEF"

echo
echo "PREPARED ONLY. NOTHING EXPENSIVE WAS SUBMITTED."
echo "ROOT=$ROOT"
