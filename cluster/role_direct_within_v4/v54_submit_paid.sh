#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-}"
if [[ -z "$RUN" || ! -f "$RUN/preflight.json" ]]; then
 echo 'Usage: V54_APPROVE_BILLABLE_CALLS=18 bash cluster/role_direct_within_v4/v54_submit_paid.sh "$RUN"' >&2;exit 2
fi
RUN=$(realpath "$RUN")
if [[ "$RUN" != "$PWD"/artifacts/role_direct_within_v4/pilot_v54/r2paired_* ]]; then
 echo 'Unexpected output root' >&2;exit 2
fi
PY="$HOME/.conda/envs/masdetective/bin/python"
N=$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["mas_executions_max"])' "$RUN/preflight.json")
if [[ "${V54_APPROVE_BILLABLE_CALLS:-}" != "$N" ]]; then
  echo "Refusing paid API calls. To opt in explicitly set V54_APPROVE_BILLABLE_CALLS=$N" >&2
  echo "This is $N MAS queries (up to $((5*N)) LLM provider calls before retries), cost unknown." >&2
  exit 2
fi
if [[ -f "$RUN/responses_train_round_02.jsonl" ]]; then echo 'Already executed - no paid rerun' >&2;exit 2;fi
JOB=$(sbatch --parsable --export="ALL,V54_ROOT=$RUN,V54_APPROVE_BILLABLE_CALLS=$N" cluster/role_direct_within_v4/v54_execute_cpu.sbatch)
JOB="${JOB%%;*}"
printf '%s\n' "$JOB" > "$RUN/job_id.txt"
echo "V54_SUBMITTED JOB=$JOB RUN=$RUN"
echo "Check: sacct -j $JOB -X --format=JobID,JobName%26,State,ExitCode,Elapsed"
echo "Log: logs/v54-mas5-v54-r2-pilot-$JOB.out"
echo "Report after completion: $RUN/report_diagnostic.json"
