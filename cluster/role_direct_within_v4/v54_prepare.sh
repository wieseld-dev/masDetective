#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -m py_compile analysis/role_direct_within_v4/v54_r2_paired_pilot.py
V51="${V54_V51_ROOT:-$PWD/artifacts/role_direct_within_v4/adaptive_v51/r2preview_20261010_181053_1596563}"
V49="${V54_V49_ROOT:-$PWD/artifacts/role_direct_within_v4/pilot_v49/pilot_20261010_172830_2543921}"
N="${V54_PAIRS_PER_ROLE:-1}"
case "$N" in 1|2|3);; *) echo 'V54_PAIRS_PER_ROLE must be 1,2,3' >&2; exit 2;; esac
mkdir -p logs artifacts/role_direct_within_v4/pilot_v54
RUN="$PWD/artifacts/role_direct_within_v4/pilot_v54/r2paired_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -u -m analysis.role_direct_within_v4.v54_r2_paired_pilot prepare \
    --root "$RUN" --v51-root "$V51" --v49-pilot "$V49" --pairs-per-role "$N"
echo ""
echo "RUN=$RUN"
echo "REVIEW=$RUN/human_review.tsv"
echo "NO MAS CALLS HAVE BEEN MADE. Inspect the preflight and preview first."
