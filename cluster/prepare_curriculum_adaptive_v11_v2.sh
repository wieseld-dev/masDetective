#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"

ROOT="${CURR_EXP_ROOT:-artifacts/adaptive_v11_curriculum_v2}"
V1ROOT="${CURR_V1_ROOT:-artifacts/adaptive_v11_curriculum_v1}"
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
mkdir -p "$ROOT" "$ROOT/checkpoints" "$ROOT/selector" logs

export CURR_EXP_ROOT="$ROOT"
"$PY" scripts/curriculum_v2_make_manifests.py

if [[ ! -s "$ROOT/v11_source_sha256.txt" ]]; then
  if [[ -s "$V1ROOT/v11_source_sha256.txt" ]]; then
    cp "$V1ROOT/v11_source_sha256.txt" "$ROOT/v11_source_sha256.txt"
  else
    echo "missing $ROOT/v11_source_sha256.txt and no V1 hash to copy" >&2
    exit 2
  fi
fi

test -s "$ROOT/manifest_report.json" || { echo "manifest_report.json was not created" >&2; exit 2; }
"$PY" scripts/curriculum_v2_preflight.py

echo
echo "V2 curriculum prepared successfully."
echo "ROOT=$ROOT"
echo "Plan:"
"$PY" scripts/curriculum_v2_plan.py
