#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
PY="$HOME/.conda/envs/masdetective/bin/python"
ROOT="${CURR_EXP_ROOT:-artifacts/adaptive_v11_curriculum_v1}"
BELIEF="${CURR_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"
V11="${CURR_V11:-artifacts/v11_policy}"
if [[ -d "$ROOT/checkpoints" && "${CURR_ALLOW_REPREP:-NO}" != "YES" ]]; then
  echo "Refusing to prepare over existing checkpoints: $ROOT/checkpoints" >&2
  echo "Set CURR_ALLOW_REPREP=YES only if you intentionally want to regenerate manifests around existing files." >&2
  exit 2
fi
mkdir -p "$ROOT" logs
"$PY" -u scripts/curriculum_make_manifests.py --root "$ROOT" --search-root artifacts --seed "${CURR_SEED:-20260918}"
"$PY" -u scripts/curriculum_preflight.py --root "$ROOT" --v11 "$V11" --belief "$BELIEF"
echo
echo "PREPARED ONLY. NO GPU/API JOBS SUBMITTED."
echo "ROOT=$ROOT"
