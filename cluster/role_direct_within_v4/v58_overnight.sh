#!/usr/bin/env bash
# One command: prepare new run (if no RUN supplied), then submit the entire
# 8-round afterok dependency chain. An existing R1 Preview is reused/waited for.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
[[ "${V58_APPROVE_MAS_EXECUTIONS:-}" == '160' ]] || {
  echo 'Set V58_APPROVE_MAS_EXECUTIONS=160 to authorize up to 160 paid MAS executions.' >&2
  exit 2
}
if [[ $# -gt 1 ]]; then echo 'Usage: V58_APPROVE_MAS_EXECUTIONS=160 bash .../v58_overnight.sh [RUN]' >&2;exit 2; fi
if [[ $# -eq 1 ]]; then
  RUN="$1"
else
  output="$(bash cluster/role_direct_within_v4/v58_submit.sh)"
  printf '%s\n' "$output"
  RUN="$(printf '%s\n' "$output" | sed -n 's/^V58_RUN=//p' | tail -1)"
  [[ -n "$RUN" ]] || { echo 'Could not determine prepared RUN path' >&2;exit 2; }
fi
printf 'OVERNIGHT_RUN=%s\n' "$RUN"
V58_APPROVE_MAS_EXECUTIONS=160 bash cluster/role_direct_within_v4/v58_launch.sh "$RUN"
