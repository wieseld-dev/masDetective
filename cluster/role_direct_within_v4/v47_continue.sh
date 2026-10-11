#!/usr/bin/env bash
# Continue AFTER a reviewed 160-probe bank has been finalized. Up to 1160 MAS executions.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
if [[ "${V46_APPROVE_API_MAS_EXECUTIONS:-}" != 1160 ]]; then
  echo 'Explicit API budget acknowledgement required: V46_APPROVE_API_MAS_EXECUTIONS=1160' >&2
  exit 2
fi
: "${V46_ROOT:?V46_ROOT required}"
V46_ROOT=$(realpath -e "$V46_ROOT")
[[ "$V46_ROOT" == "$PWD"/artifacts/role_direct_within_v4/overnight_v46/mas5_night_* ]] || { echo 'Bad MAS5 root' >&2;exit 2; }
[[ -f "$V46_ROOT/train/v47_quality/MAS5_V47_APPROVED.json" ]] || { echo 'Probes have not been manually finalized. Stop.' >&2;exit 2; }
[[ ! -f "$V46_ROOT/train/candidates_train_round_01.jsonl" ]] || { echo 'Training already prepared. Refusing duplicate executions.' >&2;exit 2; }
[[ ! -f "$V46_ROOT/train/responses_train_round_01.jsonl" ]] || { echo 'Training already executed. Refusing duplicate executions.' >&2;exit 2; }
[[ ! -e "$V46_ROOT/jobs_v47.tsv" ]] || { echo 'V47 DAG already submitted.' >&2;exit 2; }
if squeue -h -u "$USER" -o '%j' | grep -qE '^mas5-'; then
  echo 'Another MAS5 job is running or queued. Refusing concurrent pipeline.' >&2
  squeue -u "$USER"
  exit 2
fi
PY="$HOME/.conda/envs/masdetective/bin/python"
for f in analysis/role_direct_within_v4/v47_{quality,curate,generator,repair,night}.py; do "$PY" -m py_compile "$f";done
mkdir -p logs
manifest="$V46_ROOT/jobs_v47.tsv"; PREV=''; JOBS=()
trap 'if (( ${#JOBS[@]} > 0 )); then scancel "${JOBS[@]}" 2>/dev/null || true; fi' ERR
stage(){
  local stage="$1" type="$2" answer id
  local dep=()
  [[ -z "$PREV" ]] || dep=(--dependency="afterok:$PREV")
  answer=$(sbatch --parsable "${dep[@]}" --job-name="mas5-$stage" \
    --export="ALL,V46_ROOT=$V46_ROOT,V46_STAGE=$stage" \
    "cluster/role_direct_within_v4/v47_${type}.sbatch")
  id="${answer%%;*}"; [[ "$id" =~ ^[0-9]+$ ]] || { echo "Bad ID $answer" >&2;return 1; }
  printf '%s\t%s\t%s\n' "$id" "$stage" "$type" | tee -a "$manifest"
  JOBS+=("$id"); PREV="$id"
}
stage prepare_train cpu
stage execute_train cpu
stage evidence cpu
stage selector cpu
stage train_sensor gpu_4090
stage train_generator gpu_4090
stage gate cpu
stage test_prepare cpu
stage test_generate1 gpu_4090
stage test_select1 cpu
stage test_execute1 cpu
stage test_score1 gpu_4090
stage test_generate2 gpu_4090
stage test_select2 cpu
stage test_execute2 cpu
stage test_score2 gpu_4090
stage report cpu
trap - ERR
echo "MAS5 V47 DAG submitted. LAST JOB: $PREV"
echo "Check: cat '$manifest'; squeue --me"
echo "Final report: $V46_ROOT/final_MAS5_report.json"
echo 'Expected: several hours or longer, dependent on API and queue; each stage max 20h.'
