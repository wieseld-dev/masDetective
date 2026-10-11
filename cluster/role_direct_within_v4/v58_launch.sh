#!/usr/bin/env bash
# 8 x (Generate -> Execute -> Score/Live) = 24 sequential dependency jobs.
# Exactly 160 target calls IF every stage completes. No hidden paid preflight.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-}"
[[ -n "$RUN" && -f "$RUN/public_preflight.json" ]] || { echo 'Usage: V58_APPROVE_MAS_EXECUTIONS=160 bash .../v58_launch.sh RUN' >&2;exit 2; }
RUN="$(realpath "$RUN")"
[[ "$RUN" == "$PWD"/artifacts/role_direct_within_v4/adaptive_v58/mas5_* ]] || { echo 'Unexpected V58 path' >&2;exit 2; }
[[ "${V58_APPROVE_MAS_EXECUTIONS:-}" == '160' ]] || { echo 'Explicit approval for 160 billable MAS executions REQUIRED' >&2;exit 2; }
[[ ! -e "$RUN/jobs.tsv" ]] || { echo 'V58 already submitted. Refusing duplicate paid jobs.' >&2;exit 2; }
[[ -f .env ]] || { echo '.env missing; no jobs started' >&2;exit 2; }
set -a
source .env
set +a
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo 'API key missing; no jobs started' >&2;exit 2; }
command -v sbatch >/dev/null || { echo 'sbatch missing' >&2;exit 2; }
mkdir -p logs
# If R1 Preview already exists, wait for it using Slurm afterok.
# This also works if Preview is still RUNNING/PENDING: no manual wait needed.
REUSE_PREVIEW=0
PREV=''
if [[ -e "$RUN/preview_job_id.txt" ]]; then
  PREVIEW_ID="$(cat "$RUN/preview_job_id.txt")"
  [[ "$PREVIEW_ID" =~ ^[0-9]+$ ]] || { echo 'Invalid preview job ID' >&2;exit 2; }
  # For an already-completed Preview, fail fast if generation was incomplete.
  # For a pending/running Preview, the next stage is held behind afterok.
  if [[ -e "$RUN/generator_report_round_01.json" ]]; then
    [[ -s "$RUN/pool_public_dev_round_01.jsonl" && -s "$RUN/candidates_dev_round_01.jsonl" ]] || {
      echo 'Incomplete R1 Preview outputs; refusing paid submission' >&2;exit 2;
    }
  fi
  REUSE_PREVIEW=1
  PREV="$PREVIEW_ID"
  echo "R1 Preview job $PREVIEW_ID found: all paid stages wait for its successful completion (afterok)."
fi
JOBS=()
trap '[[ ${#JOBS[@]} -eq 0 ]] || scancel "${JOBS[@]}" 2>/dev/null || true' ERR
stage(){
  local rd="$1" action="$2" kind="$3" answer jid job
  job="mas5-v58-${action:0:1}$(printf '%02d' "$rd")"
  local depends=();[[ -z "$PREV" ]] || depends=(--dependency="afterok:$PREV")
  answer="$(sbatch --parsable "${depends[@]}" --job-name="$job" \
    --export="ALL,V58_ROOT=$RUN,V58_STAGE=$action,V58_ROUND=$rd,V58_APPROVE_MAS_EXECUTIONS=160,V58_TRIES=${V58_TRIES:-3},V58_SENSOR_MAXLEN=${V58_SENSOR_MAXLEN:-1536}" \
    "cluster/role_direct_within_v4/v58_${kind}.sbatch")"
  jid="${answer%%;*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response $answer" >&2;return 1; }
  JOBS+=("$jid");PREV="$jid"
  printf '%s\t%s\t%s\n' "$jid" "$action" "$rd" | tee -a "$RUN/jobs.tsv"
}
for rd in {1..8}; do
  if [[ "$rd" == 1 && "$REUSE_PREVIEW" == 1 ]]; then
    printf 'R1 Preview dependency set; execute1 starts automatically after successful Preview.\n'
  else
    stage "$rd" generate gpu
  fi
  stage "$rd" execute cpu
  stage "$rd" score gpu
done
trap - ERR
printf '\nV58_SUBMITTED RUN=%s\nNEW_JOBS=%s\nMAS_EXECUTIONS_IF_COMPLETE=160\nFINAL_JOB=%s\n' "$RUN" "${#JOBS[@]}" "$PREV"
printf 'Live status: bash cluster/role_direct_within_v4/v58_watch.sh "%s"\n' "$RUN"
printf 'No V5.6/V5.7 jobs cancelled, no checkpoints modified. API cost unknown.\n'
