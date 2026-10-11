#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-}"
if [[ -z "$RUN" ]];then RUN="$(ls -dt artifacts/role_direct_within_v4/sensor_v510/generalize_* 2>/dev/null | head -1)";fi
[[ -n "$RUN" && -f "$RUN/jobs.tsv" ]] || { echo 'V510 run not found' >&2;exit 2; }
echo "===== RUN: $RUN ====="
cat "$RUN/jobs.tsv"
IDS="$(cut -f1 "$RUN/jobs.tsv" | paste -sd, -)"
sacct -j "$IDS" -X --format=JobID,JobName%27,State,ExitCode,Elapsed || true
echo '===== LAST TRAIN METRICS ====='
if [[ -f "$RUN/metrics_history.jsonl" ]];then tail -n 3 "$RUN/metrics_history.jsonl";else echo 'Waiting for first validation...';fi
echo '===== COMPARISON ====='
if [[ -f "$RUN/generalization_comparison.tsv" ]];then cat "$RUN/generalization_comparison.tsv";else echo 'Not ready; runs after train completed.';fi
