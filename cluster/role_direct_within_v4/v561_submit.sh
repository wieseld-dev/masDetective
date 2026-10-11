#!/usr/bin/env bash
# Prepares a fresh V5.6 run, then stops unless explicit paid approval is set.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
mkdir -p logs artifacts/role_direct_within_v4/nonadaptive_v56
RUN="$PWD/artifacts/role_direct_within_v4/nonadaptive_v56/mas5_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -u -m analysis.role_direct_within_v4.v56_nonadaptive prepare --root "$RUN"
printf '\nV561_RUN=%s\n' "$RUN"
if [[ "${V56_APPROVE_MAS_EXECUTIONS:-}" != 200 ]]; then
  printf 'PREPARE ONLY: no billable MAS calls. Review %s/review_40_questions.tsv\n' "$RUN"
  printf 'To launch: V56_APPROVE_MAS_EXECUTIONS=200 bash cluster/role_direct_within_v4/v561_launch.sh "%s"\n' "$RUN"
  exit 0
fi
V56_APPROVE_MAS_EXECUTIONS=200 bash cluster/role_direct_within_v4/v561_launch.sh "$RUN"
