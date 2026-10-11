#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
BASE=artifacts/role_direct_within_v4/sensor_v5113
RUN="${1:-$(cat "$BASE/latest_run_path.txt" 2>/dev/null || true)}"
if [[ -z "$RUN" ]];then echo "No submitted V5.11.3 job found";exit 0;fi
echo "RUN=$RUN"
JOBS="$BASE/submitted_jobs.tsv"
if [[ -f "$JOBS" ]];then
  JOB=$(tail -1 "$JOBS" | cut -f1)
  sacct -j "$JOB" -X --format=JobID,JobName%28,State,ExitCode,Elapsed
  for TYPE in out err;do
     FILE="logs/v5113-mas5-v5113-rolewise-${JOB}.${TYPE}"
     [[ ! -f "$FILE" ]] || { echo "---- ${TYPE} ----";tail -n 15 "$FILE"; }
  done
fi
if [[ -f "$RUN/comparison.tsv" ]];then
 echo "==== METRICS ====";cat "$RUN/comparison.tsv"
 [[ ! -f "$RUN/v58_offline_replay.json" ]] || echo "==== V58 REPLAY: $RUN/v58_offline_replay.json ===="
fi
