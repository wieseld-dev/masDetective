#!/bin/bash
# Expensive experiment. Run only after inspecting the fresh manifest and preparation logs.
set -euo pipefail
cd "$HOME/mas_rl_detective"
ROUNDS="${AV11_TOTAL_ROUNDS:-4}"
ROOT=artifacts/adaptive_v11_active_v1
for p in "$ROOT/frozen_belief.joblib" "$ROOT/manifest_privileged.json" "$ROOT/states_train_round_00.jsonl" "$ROOT/states_dev_round_00.jsonl"; do
  test -s "$p" || { echo "missing required prepared artifact: $p" >&2; exit 2; }
done
mkdir -p logs "$ROOT"

prev=""
: > "$ROOT/train_jobs.txt"
for r in $(seq 1 "$ROUNDS"); do
  dep=()
  [ -n "$prev" ] && dep=(--dependency="afterok:$prev")
  G=$(sbatch --parsable "${dep[@]}" --export=ALL,AV11_ROUND=$r,AV11_SPLIT=train,AV11_GROUP_SIZE=4 cluster/adaptive_v11_generate_4090.sbatch)
  X=$(sbatch --parsable --dependency=afterok:$G --export=ALL,AV11_ROUND=$r,AV11_SPLIT=train cluster/adaptive_v11_execute_cpu.sbatch)
  S=$(sbatch --parsable --dependency=afterok:$X --export=ALL,AV11_ROUND=$r,AV11_SPLIT=train cluster/adaptive_v11_score_4090.sbatch)
  A=$(sbatch --parsable --dependency=afterok:$S --export=ALL,AV11_ROUND=$r,AV11_SPLIT=train cluster/adaptive_v11_advance_cpu.sbatch)
  U=$(sbatch --parsable --dependency=afterok:$A --export=ALL,AV11_ROUND=$r cluster/adaptive_v11_update_4090.sbatch)
  echo "ROUND=$r GEN=$G EXEC=$X SCORE=$S ADVANCE=$A UPDATE=$U" | tee -a "$ROOT/train_jobs.txt"
  prev="$U"
done

FINAL_ADAPTER="$ROOT/checkpoints/round_$(printf '%02d' "$ROUNDS")/policy"
: > "$ROOT/dev_jobs.txt"
prev_dev="$prev"
for r in $(seq 1 "$ROUNDS"); do
  G=$(sbatch --parsable --dependency=afterok:$prev_dev --export=ALL,AV11_ROUND=$r,AV11_SPLIT=dev,AV11_GROUP_SIZE=1,AV11_ADAPTER=$FINAL_ADAPTER cluster/adaptive_v11_generate_4090.sbatch)
  X=$(sbatch --parsable --dependency=afterok:$G --export=ALL,AV11_ROUND=$r,AV11_SPLIT=dev cluster/adaptive_v11_execute_cpu.sbatch)
  S=$(sbatch --parsable --dependency=afterok:$X --export=ALL,AV11_ROUND=$r,AV11_SPLIT=dev cluster/adaptive_v11_score_4090.sbatch)
  A=$(sbatch --parsable --dependency=afterok:$S --export=ALL,AV11_ROUND=$r,AV11_SPLIT=dev cluster/adaptive_v11_advance_cpu.sbatch)
  echo "DEV_ROUND=$r GEN=$G EXEC=$X SCORE=$S ADVANCE=$A" | tee -a "$ROOT/dev_jobs.txt"
  prev_dev="$A"
done
R=$(sbatch --parsable --dependency=afterok:$prev_dev --export=ALL,AV11_TOTAL_ROUNDS=$ROUNDS cluster/adaptive_v11_report_cpu.sbatch)
echo "REPORT=$R" | tee "$ROOT/report_job.txt"
echo "Final train update=$prev ; final dev advance=$prev_dev ; report=$R"
