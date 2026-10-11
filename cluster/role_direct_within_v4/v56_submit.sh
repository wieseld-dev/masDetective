#!/usr/bin/env bash
# Only the named 200-approval submits charged jobs. Without opt-in prepare-only.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
mkdir -p logs artifacts/role_direct_within_v4/nonadaptive_v56
RUN="$PWD/artifacts/role_direct_within_v4/nonadaptive_v56/mas5_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -u -m analysis.role_direct_within_v4.v56_nonadaptive prepare --root "$RUN"
printf '\nV56_RUN=%s\n' "$RUN"
if [[ "${V56_APPROVE_MAS_EXECUTIONS:-}" != 200 ]]; then
  echo 'Prepared ONLY: no API, no Slurm jobs.'
  echo "Review: $RUN/review_40_questions.tsv"
  echo "To approve 200 new billable MAS executions: V56_APPROVE_MAS_EXECUTIONS=200 bash cluster/role_direct_within_v4/v56_launch.sh '$RUN'"
  exit 0
fi
V56_APPROVE_MAS_EXECUTIONS=200 bash cluster/role_direct_within_v4/v56_launch.sh "$RUN"
