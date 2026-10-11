#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-}"
[[ -d "$RUN" ]] || { echo 'Usage: bash .../v58_watch.sh RUN' >&2;exit 2; }
RUN="$(realpath "$RUN")"
[[ "$RUN" == "$PWD"/artifacts/role_direct_within_v4/adaptive_v58/mas5_* ]] || { echo 'Wrong V58 path' >&2;exit 2; }
echo '===== JOBS ====='
if [[ -f "$RUN/jobs.tsv" ]];then
 IDS="$(cut -f1 "$RUN/jobs.tsv" | paste -sd, -)"
 sacct -j "$IDS" -X --format=JobID,JobName%22,State,Elapsed 2>/dev/null | tail -n 27 || true
else echo 'Not submitted yet';fi
echo '===== AUC PER COMPLETED ROUND ====='
if [[ -s "$RUN/live_auc_history.jsonl" ]];then
 cat "$RUN/live_auc_history.jsonl"
else echo 'Waiting for first round and Sensor';fi
echo '===== LAST REPORT ====='
if [[ -s "$RUN/live_auc_latest.json" ]];then
 python - "$RUN/live_auc_latest.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(f"round={r['round_completed']}/8 AUC={r['within_mas_auc']:.6f} Top3={r['top3_correct']}/3 Exact={r['exact_top3']} Queries={r['total_executed_mas_queries']}")
PY
fi
