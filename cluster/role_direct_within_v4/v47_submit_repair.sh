#!/usr/bin/env bash
# GPU probe repair ONLY. Does not authorize or perform any target/API calls.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
: "${V46_ROOT:?V46_ROOT must point to the original MAS5 V4.6.1 run}"
V46_ROOT=$(realpath -e "$V46_ROOT")
[[ "$V46_ROOT" == "$PWD"/artifacts/role_direct_within_v4/overnight_v46/mas5_night_* ]] || { echo 'Bad MAS5 run path' >&2;exit 2; }
[[ -f "$V46_ROOT/train/v47_quality/curation.json" ]] || { echo 'First audit and curate' >&2;exit 2; }
if [[ -e "$V46_ROOT/train/v47_quality/gpu_candidate" && "${V47_RETRY:-0}" != 1 ]]; then
  echo 'Repair already attempted. Inspect candidate or set V47_RETRY=1 for partial continuation.' >&2;exit 2
fi
[[ ! -f "$V46_ROOT/train/candidates_train_round_01.jsonl" ]] || { echo 'Training already prepared, abort' >&2;exit 2; }
if squeue -h -u "$USER" -o '%j' | grep -qE '^mas5-'; then
  echo 'Another MAS5 job is already active or queued. Do not race the same training files.' >&2
  squeue -u "$USER"
  exit 2
fi
PY="$HOME/.conda/envs/masdetective/bin/python"
for f in analysis/role_direct_within_v4/v47_{quality,curate,generator,repair}.py; do "$PY" -m py_compile "$f";done
mkdir -p logs
JOB=$(sbatch --parsable --export="ALL,V46_ROOT=$V46_ROOT,V47_RETRY=${V47_RETRY:-0}" \
  cluster/role_direct_within_v4/v47_repair_4090.sbatch)
echo "MAS5 V47 PROBE REPAIR JOB: $JOB"
echo "No MAS/API calls. Expected time: 10-90 minutes GPU plus queue (depends on rejected count)."
echo "Check: sacct -j $JOB -X --format=JobID,State,ExitCode,Elapsed"
echo "Log: tail -n 50 logs/v47-mas5-v47-repair-$JOB.out"
