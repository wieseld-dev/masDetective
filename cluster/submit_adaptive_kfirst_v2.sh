#!/bin/bash
set -euo pipefail

cd "$HOME/mas_rl_detective"
mkdir -p logs

test -s artifacts/adaptive_v1/action_library.jsonl || {
  echo "Missing artifacts/adaptive_v1/action_library.jsonl"
  exit 2
}

for f in \
  cluster/adaptive_collect_cpu.sbatch \
  cluster/adaptive_score_v135_4090.sbatch \
  cluster/adaptive_kfirst_fit_eval_cpu.sbatch
do
  test -f "$f" || { echo "Missing $f"; exit 2; }
done

# Manifest is intentionally NOT submitted here.
# The user should create and inspect the locked 32-system manifest first.
C=$(sbatch --parsable cluster/adaptive_collect_cpu.sbatch)
S=$(sbatch --parsable --dependency=afterok:${C} cluster/adaptive_score_v135_4090.sbatch)
E=$(sbatch --parsable --dependency=afterok:${S} cluster/adaptive_kfirst_fit_eval_cpu.sbatch)

echo "ADAPT_COLLECT=$C"
echo "ADAPT_DUAL_SCORE=$S"
echo "ADAPT_KFIRST_EVAL=$E"
