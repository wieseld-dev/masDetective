#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"

EXP_ROOT="${CTRL_EXPERIMENT_ROOT:-artifacts/adaptive_v11_controller_v1}"
BELIEF="${CTRL_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"

if [ "${CTRL_CONFIRM:-NO}" != "YES" ]; then
  echo "REFUSING TO SUBMIT: export CTRL_CONFIRM=YES first."
  echo "This experiment can make up to 352 black-box MAS queries."
  exit 2
fi

[ -s "$EXP_ROOT/common/states_dev_round_00.jsonl" ] || { echo "run cluster/prepare_controller_v1.sh first"; exit 1; }

set -a
[ -f .env ] && source .env
set +a
[ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

mkdir -p logs "$EXP_ROOT"
JOBS="$EXP_ROOT/jobs.txt"
: > "$JOBS"

# ---------------- shared exact round 1 ----------------
G1=$(sbatch --parsable --exclude=cs-4090-09 \
  --export=ALL,CTRL_ROOT="$EXP_ROOT/common",CTRL_ROUND=1,CTRL_SPLIT=dev,CTRL_MODE=common,CTRL_BELIEF="$BELIEF" \
  cluster/ctrl_generate_4090.sbatch)
X1=$(sbatch --parsable --dependency=afterok:$G1 \
  --export=ALL,CTRL_ROOT="$EXP_ROOT/common",CTRL_ROUND=1,CTRL_SPLIT=dev,CTRL_BELIEF="$BELIEF" \
  cluster/ctrl_execute_cpu.sbatch)
S1=$(sbatch --parsable --dependency=afterok:$X1 --exclude=cs-4090-09 \
  --export=ALL,CTRL_ROOT="$EXP_ROOT/common",CTRL_ROUND=1,CTRL_SPLIT=dev,CTRL_BELIEF="$BELIEF" \
  cluster/ctrl_score_4090.sbatch)
A1=$(sbatch --parsable --dependency=afterok:$S1 \
  --export=ALL,CTRL_ROOT="$EXP_ROOT/common",CTRL_ROUND=1,CTRL_SPLIT=dev,CTRL_BELIEF="$BELIEF" \
  cluster/ctrl_advance_cpu.sbatch)
SEED=$(sbatch --parsable --dependency=afterok:$A1 \
  --export=ALL,CTRL_EXPERIMENT_ROOT="$EXP_ROOT" \
  cluster/ctrl_seed_arms_cpu.sbatch)
echo "COMMON R1 GEN=$G1 EXEC=$X1 SCORE=$S1 ADV=$A1 SEED=$SEED" | tee -a "$JOBS"

submit_arm () {
  local mode="$1"
  local root="$EXP_ROOT/$mode"
  local prev="$SEED"
  for r in 2 3 4 5 6; do
    G=$(sbatch --parsable --dependency=afterok:$prev --exclude=cs-4090-09 \
      --export=ALL,CTRL_ROOT="$root",CTRL_ROUND="$r",CTRL_SPLIT=dev,CTRL_MODE="$mode",CTRL_BELIEF="$BELIEF" \
      cluster/ctrl_generate_4090.sbatch)
    X=$(sbatch --parsable --dependency=afterok:$G \
      --export=ALL,CTRL_ROOT="$root",CTRL_ROUND="$r",CTRL_SPLIT=dev,CTRL_BELIEF="$BELIEF" \
      cluster/ctrl_execute_cpu.sbatch)
    S=$(sbatch --parsable --dependency=afterok:$X --exclude=cs-4090-09 \
      --export=ALL,CTRL_ROOT="$root",CTRL_ROUND="$r",CTRL_SPLIT=dev,CTRL_BELIEF="$BELIEF" \
      cluster/ctrl_score_4090.sbatch)
    A=$(sbatch --parsable --dependency=afterok:$S \
      --export=ALL,CTRL_ROOT="$root",CTRL_ROUND="$r",CTRL_SPLIT=dev,CTRL_BELIEF="$BELIEF" \
      cluster/ctrl_advance_cpu.sbatch)
    echo "${mode^^} R$r GEN=$G EXEC=$X SCORE=$S ADV=$A" | tee -a "$JOBS"
    prev="$A"
  done
  echo "$prev"
}

STATIC_LAST=$(submit_arm static | tail -1)
ADAPTIVE_LAST=$(submit_arm adaptive | tail -1)

REPORT=$(sbatch --parsable --dependency=afterok:${STATIC_LAST}:${ADAPTIVE_LAST} \
  --export=ALL,CTRL_EXPERIMENT_ROOT="$EXP_ROOT" \
  cluster/ctrl_report_cpu.sbatch)

echo "STATIC_LAST=$STATIC_LAST" | tee -a "$JOBS"
echo "ADAPTIVE_LAST=$ADAPTIVE_LAST" | tee -a "$JOBS"
echo "REPORT=$REPORT" | tee -a "$JOBS"

echo
echo "SUBMITTED simple adaptive-controller experiment."
echo "EXP_ROOT=$EXP_ROOT"
echo "REPORT_JOB=$REPORT"
echo "Watch: squeue --me"
