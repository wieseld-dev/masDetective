#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-}"
[[ -n "$RUN" && -f "$RUN/public_preflight.json" && -f "$RUN/billable_preflight.json" ]] || { echo 'Usage: V59_APPROVE_MAS_EXECUTIONS=200 bash .../v59_launch.sh RUN' >&2;exit 2; }
RUN="$(realpath "$RUN")"
[[ "$RUN" == "$PWD"/artifacts/role_direct_within_v4/fresh_test_v59/fresh_* ]] || { echo 'Run path not allowed' >&2;exit 2; }
[[ "${V59_APPROVE_MAS_EXECUTIONS:-}" == '200' ]] || { echo 'Explicit approval for 200 billable MAS executions required' >&2;exit 2; }
[[ ! -e "$RUN/jobs.tsv" ]] || { echo 'ALREADY SUBMITTED. Refusing duplicate paid jobs.' >&2;exit 2; }
[[ -f .env ]] || { echo 'Missing .env' >&2;exit 2; }
set -a;source .env;set +a
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo 'Missing OPENAI_API_KEY' >&2;exit 2; }
V57_ROOT="$("$HOME/.conda/envs/masdetective/bin/python" - "$RUN" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]+'/public_preflight.json'))['v57_root'])
PY
)"
[[ -f "$V57_ROOT/jobs.tsv" ]] || { echo 'V57 training job record missing' >&2;exit 2; }
TRAIN_ID="$(awk '$2=="v57_sensor" {print $1}' "$V57_ROOT/jobs.tsv" | tail -1)"
[[ "$TRAIN_ID" =~ ^[0-9]+$ ]] || { echo 'V57 training job ID missing' >&2;exit 2; }
JOBS=();PREV='';trap '[[ ${#JOBS[@]} -eq 0 ]] || scancel "${JOBS[@]}" 2>/dev/null || true' ERR
stage(){
  local stage="$1" kind="$2" dep=() ans jid
  if [[ -n "$PREV" ]];then dep=(--dependency="afterok:$PREV");fi
  # Never spend on paid queries unless the selected V57 training run completed successfully.
  if [[ "$stage" == 'execute1' ]];then dep=(--dependency="afterok:$TRAIN_ID");fi
  # All subsequent stages follow strictly by afterok via PREV.
  ans="$(sbatch --parsable "${dep[@]}" --job-name="mas5-v59-$stage" \
    --export="ALL,V59_ROOT=$RUN,V59_STAGE=$stage,V59_APPROVE_MAS_EXECUTIONS=200" \
    "cluster/role_direct_within_v4/v59_$kind.sbatch")"
  jid="${ans%%;*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch result: $ans" >&2;return 1; }
  JOBS+=("$jid");PREV="$jid"
  printf '%s\t%s\n' "$jid" "$stage" | tee -a "$RUN/jobs.tsv"
}
stage execute1 cpu
stage execute2 cpu
stage score_old gpu
stage score_v57 gpu
stage report cpu
trap - ERR
printf '\nV59_SUBMITTED=%s\nV59_TRAIN_DEPENDENCY=%s\nFINAL_JOB=%s\n' "$RUN" "$TRAIN_ID" "$PREV"
printf '200 new billable MAS executions; ~20-120 minutes API + Sensor scoring/Slurm waits; actual cost unknown.\n'
