#!/usr/bin/env bash
# Submit only offline SFT + real-public-context format assessment. NO API calls.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
mkdir -p logs
V51_ROOT=$(ls -dt "$PWD"/artifacts/role_direct_within_v4/adaptive_v51/r2preview_* 2>/dev/null | head -1 || true)
if [ -z "$V51_ROOT" ] || [ ! -f "$V51_ROOT/public_scored_r1_states.jsonl" ]; then
  echo 'ERROR: V5.1 public R1 scored states not found. Run V5.1 scoring first.' >&2
  exit 1
fi
RUN="$PWD/artifacts/role_direct_within_v4/adaptive_v53/r2sft_$(date +%Y%m%d_%H%M%S)_$$"
"$HOME/.conda/envs/masdetective/bin/python" -m analysis.role_direct_within_v4.v53_r2_sft prepare \
  --root "$RUN" --v51-root "$V51_ROOT"
TRAIN=$(sbatch --parsable --export="ALL,V53_ROOT=$RUN,V51_ROOT=$V51_ROOT" \
  cluster/role_direct_within_v4/v53_train_4090.sbatch)
EVAL=$(sbatch --parsable --dependency="afterok:${TRAIN}" \
  --export="ALL,V53_ROOT=$RUN,V51_ROOT=$V51_ROOT" \
  cluster/role_direct_within_v4/v53_eval_4090.sbatch)
printf 'job_id\tstage\n%s\ttrain_r2_sft\n%s\tcompare_real_r1\n' "$TRAIN" "$EVAL" > "$RUN/jobs.tsv"
echo "MAS5_V53_SUBMITTED RUN=$RUN TRAIN_JOB=$TRAIN EVAL_JOB=$EVAL"
echo "After completion: cat '$RUN/r2_sft_training_report.json' '$RUN/r2_real_context_comparison.json'"
echo 'No MAS/API calls; checkpoints from previous versions remain untouched.'
