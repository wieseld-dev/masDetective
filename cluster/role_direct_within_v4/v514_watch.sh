#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
BASE="$PWD/artifacts/role_direct_within_v4/matched_v514"
[[ -f "$BASE/latest_run_path.txt" ]] || { echo 'No prepared V514 run' >&2; exit 2; }
RUN="$(cat "$BASE/latest_run_path.txt")"
echo "RUN=$RUN"
if [[ -f "$RUN/submitted.txt" ]]; then
  JOB="$(cat "$RUN/submitted.txt")"
  sacct -j "$JOB" -X --format=JobID,JobName%25,State,ExitCode,Elapsed 2>/dev/null || true
  echo '===== recent output ====='
  tail -n 28 "logs/v514-mas5-v514-match-$JOB.out" 2>/dev/null || true
  echo '===== recent errors ====='
  tail -n 12 "logs/v514-mas5-v514-match-$JOB.err" 2>/dev/null || true
else
  echo 'PREPARED ONLY; no billable job submitted'
fi
if [[ -f "$RUN/comparison.tsv" ]]; then
  echo '===== MATCHED ROLE COMPARISON ====='
  cat "$RUN/comparison.tsv"
  echo '===== REPORT ====='
  cat "$RUN/report.md"
else
  echo 'Comparison not available yet'
fi
