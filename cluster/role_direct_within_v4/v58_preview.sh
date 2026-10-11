#!/usr/bin/env bash
# First-round generation ONLY on GPU. ZERO MAS API calls.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-}"
[[ -n "$RUN" && -f "$RUN/public_preflight.json" ]] || { echo 'Usage: bash .../v58_preview.sh RUN' >&2;exit 2; }
RUN="$(realpath "$RUN")"
[[ "$RUN" == "$PWD"/artifacts/role_direct_within_v4/adaptive_v58/mas5_* ]] || { echo 'Unexpected path' >&2;exit 2; }
[[ ! -e "$RUN/preview_job_id.txt" && ! -e "$RUN/jobs.tsv" ]] || { echo 'Preview already submitted or paid run active';exit 2; }
[[ ! -e "$RUN/candidates_dev_round_01.jsonl" ]] || { echo 'Round 1 already generated';exit 2; }
mkdir -p logs
JOB="$(sbatch --parsable --job-name=mas5-v58-preview --export="ALL,V58_ROOT=$RUN,V58_STAGE=generate,V58_ROUND=1,V58_TRIES=${V58_TRIES:-3}" cluster/role_direct_within_v4/v58_gpu.sbatch)"
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo 'Bad Slurm job ID' >&2;exit 2; }
printf '%s\n' "$JOB" > "$RUN/preview_job_id.txt"
printf 'V58_PREVIEW_SUBMITTED RUN=%s JOB=%s\n' "$RUN" "$JOB"
printf 'When COMPLETED: inspect %s/review_selected_round_01.tsv before approval.\n' "$RUN"
