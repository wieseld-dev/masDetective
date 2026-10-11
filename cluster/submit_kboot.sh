#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"

ROOT="${KBOOT_ROOT:-artifacts/adaptive_v11_k_bootstrap_v1}"
BELIEF="${KBOOT_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"
ROUNDS="${KBOOT_ROUNDS:-6}"

if [ "${KBOOT_CONFIRM:-}" != "YES" ]; then
  cat <<EOF
REFUSING TO SUBMIT EXPENSIVE JOBS.
This experiment will execute up to:
  TRAIN: 96 x ${ROUNDS} = $((96*ROUNDS)) MAS queries
  DEV:   32 x ${ROUNDS} = $((32*ROUNDS)) MAS queries
  TOTAL: $((128*ROUNDS)) MAS queries
(each MAS query itself may invoke router/workers/aggregator API calls)

To submit intentionally:
  export KBOOT_CONFIRM=YES
  bash cluster/submit_kboot.sh
EOF
  exit 3
fi

for p in \
  "$ROOT/manifest_privileged.json" \
  "$ROOT/states_train_round_00.jsonl" \
  "$ROOT/states_dev_round_00.jsonl" \
  "$BELIEF"
do
  test -s "$p" || { echo "missing required artifact: $p" >&2; exit 2; }
done

mkdir -p logs "$ROOT"
export KBOOT_ROOT="$ROOT"
export KBOOT_BELIEF="$BELIEF"
export KBOOT_ROUNDS="$ROUNDS"

# TRAIN acquisition: frozen original V11 adapter, K held uniform, role belief adapts.
prev=""
: > "$ROOT/train_jobs.txt"
for r in $(seq 1 "$ROUNDS"); do
  dep=()
  [ -n "$prev" ] && dep=(--dependency="afterok:$prev")
  G=$(sbatch --parsable "${dep[@]}" --exclude=cs-4090-09 \
      --export=ALL,KBOOT_ROOT="$ROOT",KBOOT_BELIEF="$BELIEF",KBOOT_ROUND="$r",KBOOT_SPLIT=train \
      cluster/kboot_generate_4090.sbatch)
  X=$(sbatch --parsable --dependency=afterok:$G \
      --export=ALL,KBOOT_ROOT="$ROOT",KBOOT_BELIEF="$BELIEF",KBOOT_ROUND="$r",KBOOT_SPLIT=train \
      cluster/kboot_execute_cpu.sbatch)
  S=$(sbatch --parsable --dependency=afterok:$X --exclude=cs-4090-09 \
      --export=ALL,KBOOT_ROOT="$ROOT",KBOOT_BELIEF="$BELIEF",KBOOT_ROUND="$r",KBOOT_SPLIT=train \
      cluster/kboot_score_4090.sbatch)
  A=$(sbatch --parsable --dependency=afterok:$S \
      --export=ALL,KBOOT_ROOT="$ROOT",KBOOT_BELIEF="$BELIEF",KBOOT_ROUND="$r",KBOOT_SPLIT=train \
      cluster/kboot_advance_cpu.sbatch)
  echo "TRAIN round=$r GEN=$G EXEC=$X SCORE=$S ADV=$A" | tee -a "$ROOT/train_jobs.txt"
  prev="$A"
done

# Freeze the new K estimator BEFORE any DEV acquisition/inspection.
F=$(sbatch --parsable --dependency=afterok:$prev \
    --export=ALL,KBOOT_ROOT="$ROOT",KBOOT_BELIEF="$BELIEF",KBOOT_ROUNDS="$ROUNDS" \
    cluster/kboot_fit_k_cpu.sbatch)
echo "FIT_K=$F" | tee "$ROOT/k_fit_job.txt"

# Untouched DEV acquisition uses the same fixed acquisition protocol.
prev="$F"
: > "$ROOT/dev_jobs.txt"
for r in $(seq 1 "$ROUNDS"); do
  G=$(sbatch --parsable --dependency=afterok:$prev --exclude=cs-4090-09 \
      --export=ALL,KBOOT_ROOT="$ROOT",KBOOT_BELIEF="$BELIEF",KBOOT_ROUND="$r",KBOOT_SPLIT=dev \
      cluster/kboot_generate_4090.sbatch)
  X=$(sbatch --parsable --dependency=afterok:$G \
      --export=ALL,KBOOT_ROOT="$ROOT",KBOOT_BELIEF="$BELIEF",KBOOT_ROUND="$r",KBOOT_SPLIT=dev \
      cluster/kboot_execute_cpu.sbatch)
  S=$(sbatch --parsable --dependency=afterok:$X --exclude=cs-4090-09 \
      --export=ALL,KBOOT_ROOT="$ROOT",KBOOT_BELIEF="$BELIEF",KBOOT_ROUND="$r",KBOOT_SPLIT=dev \
      cluster/kboot_score_4090.sbatch)
  A=$(sbatch --parsable --dependency=afterok:$S \
      --export=ALL,KBOOT_ROOT="$ROOT",KBOOT_BELIEF="$BELIEF",KBOOT_ROUND="$r",KBOOT_SPLIT=dev \
      cluster/kboot_advance_cpu.sbatch)
  echo "DEV round=$r GEN=$G EXEC=$X SCORE=$S ADV=$A" | tee -a "$ROOT/dev_jobs.txt"
  prev="$A"
done

E=$(sbatch --parsable --dependency=afterok:$prev \
    --export=ALL,KBOOT_ROOT="$ROOT",KBOOT_BELIEF="$BELIEF",KBOOT_ROUNDS="$ROUNDS" \
    cluster/kboot_eval_k_cpu.sbatch)
echo "EVAL_K=$E" | tee "$ROOT/k_eval_job.txt"

echo
echo "SUBMITTED."
echo "TRAIN_FINAL=$F"
echo "DEV_EVAL=$E"
echo "Watch: squeue --me"
echo "Final report: $ROOT/k_head_dev_report.json"
