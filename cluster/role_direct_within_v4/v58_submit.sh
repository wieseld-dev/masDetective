#!/usr/bin/env bash
# Offline-only preparation. Does not submit any jobs or charge API.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Missing $PY" >&2;exit 2; }
mkdir -p logs artifacts/role_direct_within_v4/adaptive_v58
RUN="$PWD/artifacts/role_direct_within_v4/adaptive_v58/mas5_$(date +%Y%m%d_%H%M%S)_$$"
ARGS=()
if [[ -n "${V58_SESSION_ID:-}" ]]; then ARGS+=(--session-id "$V58_SESSION_ID"); fi
"$PY" -u -m analysis.role_direct_within_v4.v58_adaptive_eight prepare \
  --root "$RUN" "${ARGS[@]}"
echo
printf 'V58_RUN=%s\n' "$RUN"
printf 'NO API calls and NO SLURM jobs. Review %s/public_preflight.json\n' "$RUN"
printf 'To START billable one-MAS, 8-round trial (160 MAS queries):\n'
printf '  V58_APPROVE_MAS_EXECUTIONS=160 bash cluster/role_direct_within_v4/v58_launch.sh "%s"\n' "$RUN"
