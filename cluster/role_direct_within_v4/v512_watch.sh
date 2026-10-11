#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
BASE="$PWD/artifacts/role_direct_within_v4/sensor_v512"
RUN="${1:-$(cat "$BASE/latest_run_path.txt" 2>/dev/null || true)}"
[[ -n "$RUN" ]] || { echo 'No V5.12 run submitted';exit 0; }
echo "RUN=$RUN"
if [[ -f "$BASE/submitted_jobs.tsv" ]];then
  JOB="$(awk -F '\t' -v r="$RUN" '$2==r {id=$1} END {print id}' "$BASE/submitted_jobs.tsv")"
  if [[ -n "$JOB" ]];then
    sacct -j "$JOB" -X --format=JobID,JobName%24,State,ExitCode,Elapsed
    for ext in out err;do
      path="logs/v512-mas5-v512-cal-$JOB.$ext"
      if [[ -s "$path" ]];then echo "===== $ext =====";tail -n 15 "$path";fi
    done
  fi
fi
if [[ -f "$RUN/comparison.tsv" ]];then
  echo '===== CALIBRATION COMPARISON =====';column -t -s $'\t' "$RUN/comparison.tsv" || cat "$RUN/comparison.tsv"
else
  echo 'Waiting for comparison.tsv'
fi
[[ ! -f "$RUN/report.json" ]] || echo "Full report: $RUN/report.json"
[[ ! -f "$RUN/v58_fusion_replay.json" ]] || echo "V58 round-by-round replay: $RUN/v58_fusion_replay.json"
