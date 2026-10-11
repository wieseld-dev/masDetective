#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
RUN="${1:-$(ls -dt artifacts/role_direct_within_v4/sensor_v5112/rolewise_* 2>/dev/null | head -1 || true)}"
if [[ -n "$RUN" ]];then
  echo "RUN=$RUN"
  if [[ -f "$RUN/comparison.tsv" ]];then cat "$RUN/comparison.tsv";
  else echo "Result not ready";fi
  if [[ -f "$RUN/report.json" ]];then echo "Report ready: $RUN/report.json";fi
else
  echo "No rolewise run: check preflight audit."
  AUDIT=$(ls -t artifacts/role_direct_within_v4/sensor_v5112/preflight_overlap_*.json 2>/dev/null | head -1 || true)
  if [[ -n "$AUDIT" ]];then
    echo "AUDIT=$AUDIT"
    "$HOME/.conda/envs/masdetective/bin/python" - "$AUDIT" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print('status:',r.get('status'));print('reason:',r.get('reason'))
print('initial_overlap:',r.get('initial_overlap'))
print('dev_split_audit:',r.get('dev_split_audit'))
PY
  fi
fi
squeue -u "$USER" -o '%i %j %T %M' | grep -E 'JOBID|v5112' || true
LOG=$(ls -t logs/v5112-*.out 2>/dev/null | head -1 || true)
if [[ -n "$LOG" ]];then tail -n 14 "$LOG";fi
