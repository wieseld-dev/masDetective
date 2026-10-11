#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"

ROUNDS="${AV11_ROLE_ROUNDS:-3}"
ROOT="${AV11_ROLE_ROOT:-artifacts/adaptive_v11_role_only_v1}"
GROUP="${AV11_ROLE_GROUP_SIZE:-4}"

mkdir -p logs "$ROOT"
test -s artifacts/v11_summary.json || { echo "missing artifacts/v11_summary.json" >&2; exit 2; }
test -d artifacts/v11_policy || { echo "missing artifacts/v11_policy" >&2; exit 2; }
test -s artifacts/adaptive_v11_active_v1/frozen_belief.joblib || { echo "missing frozen belief v2" >&2; exit 2; }
test -s artifacts/adaptive_v11_active_v1/manifest_privileged.json || { echo "missing prepared 24/8 manifest" >&2; exit 2; }

P=$(sbatch --parsable cluster/adaptive_v11_role_preflight_cpu.sbatch)
I=$(sbatch --parsable --dependency=afterok:$P --export=ALL,AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_init_cpu.sbatch)

echo "PREFLIGHT=$P INIT=$I" | tee "$ROOT/jobs.txt"
prev="$I"

for r in $(seq 1 "$ROUNDS"); do
  G=$(sbatch --parsable --dependency=afterok:$prev --export=ALL,AV11_ROUND=$r,AV11_SPLIT=train,AV11_GROUP_SIZE="$GROUP",AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_generate_4090.sbatch)
  X=$(sbatch --parsable --dependency=afterok:$G --export=ALL,AV11_ROUND=$r,AV11_SPLIT=train,AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_execute_cpu.sbatch)
  S=$(sbatch --parsable --dependency=afterok:$X --export=ALL,AV11_ROUND=$r,AV11_SPLIT=train,AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_score_4090.sbatch)
  A=$(sbatch --parsable --dependency=afterok:$S --export=ALL,AV11_ROUND=$r,AV11_SPLIT=train,AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_advance_cpu.sbatch)
  U=$(sbatch --parsable --dependency=afterok:$A --export=ALL,AV11_ROUND=$r,AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_update_4090.sbatch)
  echo "TRAIN_ROUND=$r GEN=$G EXEC=$X SCORE=$S ADV=$A UPDATE=$U" | tee -a "$ROOT/jobs.txt"
  prev="$U"
done

FINAL_ADAPTER="$ROOT/checkpoints/round_$(printf '%02d' "$ROUNDS")/policy"
prev_dev="$prev"
for r in $(seq 1 "$ROUNDS"); do
  G=$(sbatch --parsable --dependency=afterok:$prev_dev --export=ALL,AV11_ROUND=$r,AV11_SPLIT=dev,AV11_GROUP_SIZE=1,AV11_ADAPTER="$FINAL_ADAPTER",AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_generate_4090.sbatch)
  X=$(sbatch --parsable --dependency=afterok:$G --export=ALL,AV11_ROUND=$r,AV11_SPLIT=dev,AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_execute_cpu.sbatch)
  S=$(sbatch --parsable --dependency=afterok:$X --export=ALL,AV11_ROUND=$r,AV11_SPLIT=dev,AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_score_4090.sbatch)
  A=$(sbatch --parsable --dependency=afterok:$S --export=ALL,AV11_ROUND=$r,AV11_SPLIT=dev,AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_advance_cpu.sbatch)
  echo "DEV_ROUND=$r GEN=$G EXEC=$X SCORE=$S ADV=$A" | tee -a "$ROOT/jobs.txt"
  prev_dev="$A"
done

R=$(sbatch --parsable --dependency=afterok:$prev_dev --export=ALL,AV11_ROLE_ROUNDS="$ROUNDS",AV11_ROLE_ROOT="$ROOT" cluster/adaptive_v11_role_report_cpu.sbatch)
echo "REPORT=$R" | tee -a "$ROOT/jobs.txt"
echo "Submitted role-adaptive V11: rounds=$ROUNDS group=$GROUP root=$ROOT report=$R"
