#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
OUT="${1:-$(ls -dt artifacts/role_direct_within_v4/sensor_v57/night_* 2>/dev/null | head -1)}"
[[ -f "$OUT/jobs.tsv" ]] || { echo 'No submitted V57 run';exit 2; }
OUT="$(realpath "$OUT")"
JOB="$(awk '$2=="v57_sensor" {print $1}' "$OUT/jobs.tsv")"
sacct -j "$JOB" -X --format=JobID,JobName%22,State,Elapsed,ExitCode
printf '\n===== LATEST STATUS =====\n'
if [[ -f "$OUT/current_status.json" ]];then cat "$OUT/current_status.json";else echo 'Awaiting start / model load';fi
printf '\n===== LAST LOG =====\n'
tail -n 25 "logs/v57-mas5-v57-sensor-${JOB}.out" 2>/dev/null || true
printf '\n===== ERRORS =====\n'
tail -n 15 "logs/v57-mas5-v57-sensor-${JOB}.err" 2>/dev/null || true
