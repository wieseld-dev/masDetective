#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-}"
if [[ -z "$RUN" || ! -f "$RUN/billable_preflight.json" ]]; then
  echo 'Usage: V56_APPROVE_MAS_EXECUTIONS=200 bash cluster/role_direct_within_v4/v56_launch.sh RUN' >&2;exit 2
fi
RUN="$(realpath "$RUN")"
[[ "$RUN" == "$PWD"/artifacts/role_direct_within_v4/nonadaptive_v56/mas5_* ]] || { echo 'Unexpected run path' >&2;exit 2; }
[[ "${V56_APPROVE_MAS_EXECUTIONS:-}" == 200 ]] || { echo 'Explicit billable approval for 200 MAS calls required' >&2;exit 2; }
[[ ! -f "$RUN/jobs.tsv" ]] || { echo 'Run already submitted: no duplicate jobs' >&2;exit 2; }
[[ -f .env ]] || { echo '.env missing' >&2;exit 2; }
# Check credentials without displaying secret. Jobs source .env independently.
set -a
source .env
set +a
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo 'OPENAI_API_KEY missing' >&2;exit 2; }
JOBS=();PREV=''
trap '[[ ${#JOBS[@]} -eq 0 ]] || scancel "${JOBS[@]}" 2>/dev/null || true' ERR
stage(){
  local name="$1" kind="$2" answer jid
  local dep=();[[ -z "$PREV" ]] || dep=(--dependency="afterok:$PREV")
  answer="$(sbatch --parsable "${dep[@]}" --job-name="mas5-v56-$name" --export="ALL,V56_ROOT=$RUN,V56_STAGE=$name,V56_APPROVE_MAS_EXECUTIONS=200" "cluster/role_direct_within_v4/v56_$kind.sbatch")"
  jid="${answer%%;*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Bad sbatch response $answer" >&2;return 1; }
  JOBS+=("$jid");PREV="$jid";printf '%s\t%s\n' "$jid" "$name" | tee -a "$RUN/jobs.tsv"
}
stage execute1 cpu
stage execute2 cpu
stage score640 gpu
stage score1536 gpu
stage report cpu
trap - ERR
printf '\nV56_SUBMITTED RUN=%s\nFINAL_JOB=%s\nRESULT=%s/final_MAS5_report.json\n' "$RUN" "$PREV" "$RUN"
printf 'Expected: tens of minutes to several hours + queue; each Slurm stage limit 20h. Actual provider cost unknown.\n'
