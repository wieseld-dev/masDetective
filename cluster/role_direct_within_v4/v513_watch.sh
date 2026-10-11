#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
BASE="$PWD/artifacts/role_direct_within_v4/sensor_v513"
RUN="${1:-$(cat "$BASE/latest_run_path.txt" 2>/dev/null || true)}"
[[ -n "$RUN" ]] || { echo 'No V5.13 run submitted';exit 0; }
echo "RUN=$RUN"
if [[ -f "$BASE/submitted_jobs.tsv" ]];then
  JOB="$(awk -F '\t' -v r="$RUN" '$2==r {id=$1} END {print id}' "$BASE/submitted_jobs.tsv")"
  if [[ -n "$JOB" ]];then
    sacct -j "$JOB" -X --format=JobID,JobName%24,State,ExitCode,Elapsed
    for ext in out err;do
      path="logs/v513-mas5-v513-audit-$JOB.$ext"
      if [[ -s "$path" ]];then echo "===== $ext =====";tail -n 18 "$path";fi
    done
  fi
fi
if [[ -f "$RUN/v58_round_summary.tsv" ]];then
  echo '===== V58 ANSWER ONLY ====='
  awk -F '\t' 'NR==1 || ($2=="answer_only")' "$RUN/v58_round_summary.tsv" | column -t -s $'\t' 2>/dev/null || awk -F '\t' 'NR==1 || ($2=="answer_only")' "$RUN/v58_round_summary.tsv"
  echo '===== THREE INSTALLED ROLES BY ROUND ====='
  awk -F '\t' 'NR==1 || $3==1' "$RUN/v58_role_scores.tsv" | column -t -s $'\t' 2>/dev/null || awk -F '\t' 'NR==1 || $3==1' "$RUN/v58_role_scores.tsv"
  echo "Full human-readable report: $RUN/role_rankings_report.md"
else
  echo 'Waiting for diagnostics or job failure'
fi
