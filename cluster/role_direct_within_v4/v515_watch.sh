#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
BASE="$PWD/artifacts/role_direct_within_v4/matched_v515"
[[ -f "$BASE/latest_run_path.txt" ]] || { echo 'V515 not prepared'; exit 1; }
RUN="$(cat "$BASE/latest_run_path.txt")"
echo "===== RUN ===== $RUN"
if [[ -f "$RUN/submitted.txt" ]]; then
  JOB="$(cat "$RUN/submitted.txt")"
  sacct -j "$JOB" --format=JobID,JobName%24,State,ExitCode,Elapsed || true
  echo '===== OUT ====='
  [[ -f "logs/v515-mas5-v515-mbg-$JOB.out" ]] && tail -n 35 "logs/v515-mas5-v515-mbg-$JOB.out" || true
  ls "logs/v515-"*"-$JOB.out" 2>/dev/null | xargs -r tail -n 35 || true
  echo '===== ERR ====='
  ls "logs/v515-"*"-$JOB.err" 2>/dev/null | xargs -r tail -n 20 || true
else
  echo 'PREPARED_ONLY — no job submitted, no paid MAS calls'
fi
if [[ -f "$RUN/comparison.tsv" ]]; then
  echo '===== ROLE SUMMARY ====='
  column -s $'\t' -t "$RUN/comparison.tsv" 2>/dev/null || cat "$RUN/comparison.tsv"
  echo '===== BY BACKGROUND ====='
  column -s $'\t' -t "$RUN/by_background.tsv" 2>/dev/null || cat "$RUN/by_background.tsv"
else
  echo 'No finished report yet'
fi
