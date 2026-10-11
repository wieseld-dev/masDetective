#!/usr/bin/env bash
# V4.4 Role Directed: TRAIN -> gates -> FRESH MAS5 (afterok Slurm DAG).
# Never launch target API execution without explicit approval of the MAS execution cap.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
if [[ "${V44_APPROVE_API_MAS_EXECUTIONS:-}" != 1160 ]]; then
  echo "V44 REQUIRES EXPLICIT APPROVAL: 960 TRAIN + 200 TEST = up to 1160 MAS executions." >&2
  echo "Each MAS execution may make MULTIPLE billable LLM API calls. Actual monetary cost is not known." >&2
  echo "To consent: V44_APPROVE_API_MAS_EXECUTIONS=1160 bash cluster/role_direct_within_v4/v44_submit.sh" >&2
  exit 2
fi
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
for f in analysis/role_direct_within_v4/{v44_night,v44_generator,v44_sensor_train,v44_generator_train}.py; do
  "$PY" -m py_compile "$f"
done
mkdir -p logs artifacts/role_direct_within_v4/overnight
RUN="$PWD/artifacts/role_direct_within_v4/overnight/night_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -u analysis/role_direct_within_v4/v44_night.py preflight --root "$RUN"
JOBS=();PREV=''
trap 'if [[ "${#JOBS[@]}" -gt 0 ]]; then scancel "${JOBS[@]}" 2>/dev/null || true; fi' ERR
stage(){
 local ST="$1" TYPE="$2" ANS J
 local DEP=()
 if [[ -n "$PREV" ]]; then DEP=(--dependency="afterok:$PREV"); fi
 ANS=$(sbatch --parsable "${DEP[@]}" \
   --job-name="mas5-$ST" \
   --export="ALL,V44_ROOT=$RUN,V44_STAGE=$ST" \
   "cluster/role_direct_within_v4/v44_${TYPE}.sbatch")
 J="${ANS%%;*}"
 [[ "$J" =~ ^[0-9]+$ ]] || { echo "Bad job ID: $ANS" >&2;return 1; }
 JOBS+=("$J");PREV="$J"
 printf '%s\t%s\t%s\n' "$J" "$ST" "$TYPE" | tee -a "$RUN/jobs.tsv"
}
stage train_probes gpu_4090
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
printf '\nV44 DAG SUBMITTED: %s\nFINAL JOB: %s\n' "$RUN" "$PREV"
printf 'RESULTS: %s/final_MAS5_report.json\n' "$RUN"
printf 'STATUS: squeue --me ; sacct -j %s -X --format=JobID,JobName%%22,State,ExitCode,Elapsed\n' "${JOBS[0]}"
printf 'EXPECTED: hours (API latency & Slurm queues dominate); each job max 20h.\n'
