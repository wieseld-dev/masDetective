#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-$(ls -dt artifacts/role_direct_within_v4/fresh_test_v59/fresh_* 2>/dev/null | head -1)}"
[[ -f "$RUN/public_preflight.json" ]] || { echo 'Missing V59 run';exit 2; }
echo "===== RUN: $RUN ====="
if [[ -f "$RUN/jobs.tsv" ]];then
  cat "$RUN/jobs.tsv"
  IDS="$(cut -f1 "$RUN/jobs.tsv" | paste -sd, -)"
  sacct -j "$IDS" -X --format=JobID,JobName%26,State,ExitCode,Elapsed
fi
echo '===== RESPONSE FILES ====='
ls -lh "$RUN"/responses_public_dev_round_*.jsonl 2>/dev/null || true
echo '===== SENSOR SCORES ====='
ls -lh "$RUN"/fresh_scores_*.jsonl 2>/dev/null || true
echo '===== FINAL REPORT ====='
if [[ -f "$RUN/fresh_MAS5_comparison.json" ]];then
  cat "$RUN/fresh_comparison_summary.tsv"
else echo 'Waiting for report; no new run needed.';fi
