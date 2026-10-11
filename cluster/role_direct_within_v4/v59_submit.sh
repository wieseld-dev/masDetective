#!/usr/bin/env bash
# V5.9: one-command CPU preflight + 2x100 MAS execution + two FROZEN Sensors + heldout report.
# Paid jobs are never submitted unless V59_APPROVE_MAS_EXECUTIONS=200 is explicitly set.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Python env missing" >&2;exit 2; }
mkdir -p logs artifacts/role_direct_within_v4/fresh_test_v59
RUN="$PWD/artifacts/role_direct_within_v4/fresh_test_v59/fresh_$(date +%Y%m%d_%H%M%S)_$$"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
"$PY" -m analysis.role_direct_within_v4.v59_fresh_same_mas selftest
# This stage MUST fail if any of the 40 questions had been used in V5.7 training.
"$PY" -u -m analysis.role_direct_within_v4.v59_fresh_same_mas prepare --root "$RUN"
echo "V59_RUN=$RUN"
echo "REVIEW=$RUN/review_40_fresh_questions.tsv"
if [[ "${V59_APPROVE_MAS_EXECUTIONS:-}" != '200' ]]; then
  echo "PREPARED_ONLY: no MAS/API calls. To start: V59_APPROVE_MAS_EXECUTIONS=200 bash cluster/role_direct_within_v4/v59_launch.sh '$RUN'"
  exit 0
fi
V59_APPROVE_MAS_EXECUTIONS=200 bash cluster/role_direct_within_v4/v59_launch.sh "$RUN"
