#!/usr/bin/env bash
# RESUME existing V4.6 MAS5 run: reuse successful bootstrap and 158 accepted questions.
# GPU repair (NO API) -> controlled evidence train (UP TO 960 MAS) -> gates -> MAS5 (UP TO 200 MAS).
set -euo pipefail
cd /home/wieseld/mas_rl_detective
if [[ "${V46_APPROVE_API_MAS_EXECUTIONS:-}" != "1160" ]]; then
  echo 'Explicit consent required for up to 960 training + 200 MAS5 executions' >&2
  echo 'Run: V46_APPROVE_API_MAS_EXECUTIONS=1160 V46_ROOT=/absolute/previous/run bash cluster/role_direct_within_v4/v462_resume.sh' >&2
  exit 2
fi
if [[ -z "${V46_ROOT:-}" ]]; then echo 'V46_ROOT missing: point to existing failed run' >&2;exit 2;fi
V46_ROOT=$(realpath -e "$V46_ROOT")
[[ "$V46_ROOT" == "$PWD"/artifacts/role_direct_within_v4/overnight_v46/mas5_night_* ]] || { echo 'Invalid run root' >&2;exit 2; }
[[ -f "$V46_ROOT/train/generator_bootstrap_report.json" ]] || { echo 'Bootstrap report absent' >&2;exit 2; }
[[ -f "$V46_ROOT/train/generator_bootstrap_adapter/adapter_config.json" ]] || { echo 'Bootstrap adapter absent' >&2;exit 2; }
[[ ! -f "$V46_ROOT/train/candidates_train_round_01.jsonl" ]] || { echo 'Existing prepared training data: not safe to restart chain from this stage' >&2;exit 2; }
[[ ! -f "$V46_ROOT/train/probes/MAS5_V462_REPAIRED.json" ]] || { echo 'Existing repair marker: do not resubmit duplicate chain' >&2;exit 2; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
for f in analysis/role_direct_within_v4/{v46_generator,v462_repair_bank,v46_night,v46_generator_train,v46_sensor_train}.py; do "$PY" -m py_compile "$f";done
mkdir -p logs
manifest="$V46_ROOT/jobs_v462.tsv"
[[ ! -e "$manifest" ]] || { echo 'Resume job manifest already exists: refusing duplicate submission' >&2;exit 2; }
JOBS=();PREV=''
trap 'if [[ "${#JOBS[@]}" -gt 0 ]]; then scancel "${JOBS[@]}" 2>/dev/null || true;fi' ERR
stage(){
  local ST="$1" TYPE="$2" ANS J
  local DEP=()
  if [[ -n "$PREV" ]]; then DEP=(--dependency="afterok:$PREV");fi
  ANS=$(sbatch --parsable "${DEP[@]}" --job-name="mas5-$ST" \
   --export="ALL,V46_ROOT=$V46_ROOT,V46_STAGE=$ST" \
   "cluster/role_direct_within_v4/v46_${TYPE}.sbatch")
  J="${ANS%%;*}"
  [[ "$J" =~ ^[0-9]+$ ]] || { echo "Bad job ID: $ANS" >&2;return 1; }
  JOBS+=("$J");PREV="$J"
  printf '%s\t%s\t%s\n' "$J" "$ST" "$TYPE" | tee -a "$manifest"
}
stage repair_probes gpu_4090
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
printf '\nV4.6.2 RESUME SUBMITTED: %s\nFINAL JOB: %s\n' "$V46_ROOT" "$PREV"
printf 'Check: squeue --me; cat "%s"\n' "$manifest"
