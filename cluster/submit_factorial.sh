#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"

FINAL="artifacts/adaptive_v11_factorial_heldout32"
ORIG="artifacts/v11_policy"
GRPO="artifacts/adaptive_v11_role_only_v1/checkpoints/round_01/policy"
BELIEF="artifacts/adaptive_v11_active_v1/frozen_belief.joblib"
PY="$HOME/.conda/envs/masdetective/bin/python"
MANIFEST="$FINAL/manifest_privileged.json"
JOBS="$FINAL/jobs.txt"

mkdir -p "$FINAL" logs
test -d "$ORIG"; test -d "$GRPO"; test -f "$BELIEF"

# We rely on the patched carry-forward semantics: invalid/no-viable probe = no belief update, not a crash.
grep -q "no_viable_candidate" scripts/adaptive_v11_role_advance_round.py || {
  echo "ERROR: no-viable carry-forward patch missing" >&2; exit 1;
}

set -a
source .env
set +a
: "${OPENAI_API_KEY:?OPENAI_API_KEY missing}"

: > "$JOBS"

PREP=$(sbatch --parsable \
  --job-name=av11-fprep --partition=cpu --cpus-per-task=2 --mem=8G --time=00:20:00 \
  --output=logs/av11-fprep-%j.out --error=logs/av11-fprep-%j.err \
  --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/av11_factorial_eval.py prepare --root $FINAL --belief $BELIEF --per-cell 2 --run-seeds 101,202,303")
echo "PREP=$PREP" | tee -a "$JOBS"

submit_arm () {
  local seed="$1" arm="$2" adapter="$3" static="$4" startdep="$5"
  local rr="$FINAL/runs/seed_$(printf '%03d' "$seed")/$arm"
  local sf=""; [[ "$static" == "1" ]] && sf="--static"
  local g1 x1 s1 a1 g2 x2 s2 a2 g3 x3 s3 a3

  g1=$(sbatch --parsable --exclude=cs-4090-09 --dependency=afterok:$startdep \
    --job-name=av11-fgen --partition=main --gpus=1 --constraint=rtx_4090 --mem=24G --time=01:30:00 \
    --output=logs/av11-fgen-%j.out --error=logs/av11-fgen-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/av11_factorial_eval.py generate --root $rr --adapter $adapter --seed $seed --round 1 $sf")
  x1=$(sbatch --parsable --dependency=afterok:$g1 \
    --job-name=av11-fexec --partition=cpu --cpus-per-task=2 --mem=12G --time=06:00:00 \
    --output=logs/av11-fexec-%j.out --error=logs/av11-fexec-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/adaptive_v11_execute_round.py --round 1 --split dev --root $rr --manifest $MANIFEST --workers 2 --retries 3")
  s1=$(sbatch --parsable --exclude=cs-4090-09 --dependency=afterok:$x1 \
    --job-name=av11-fscore --partition=main --gpus=1 --constraint=rtx_4090 --mem=32G --time=03:00:00 \
    --output=logs/av11-fscore-%j.out --error=logs/av11-fscore-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/adaptive_v11_score_round.py --round 1 --split dev --root $rr")
  a1=$(sbatch --parsable --dependency=afterok:$s1 \
    --job-name=av11-fadv --partition=cpu --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --output=logs/av11-fadv-%j.out --error=logs/av11-fadv-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/adaptive_v11_role_advance_round.py --round 1 --split dev --root $rr --manifest $MANIFEST --belief $BELIEF")

  g2=$(sbatch --parsable --exclude=cs-4090-09 --dependency=afterok:$a1 \
    --job-name=av11-fgen --partition=main --gpus=1 --constraint=rtx_4090 --mem=24G --time=01:30:00 \
    --output=logs/av11-fgen-%j.out --error=logs/av11-fgen-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/av11_factorial_eval.py generate --root $rr --adapter $adapter --seed $seed --round 2 $sf")
  x2=$(sbatch --parsable --dependency=afterok:$g2 \
    --job-name=av11-fexec --partition=cpu --cpus-per-task=2 --mem=12G --time=06:00:00 \
    --output=logs/av11-fexec-%j.out --error=logs/av11-fexec-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/adaptive_v11_execute_round.py --round 2 --split dev --root $rr --manifest $MANIFEST --workers 2 --retries 3")
  s2=$(sbatch --parsable --exclude=cs-4090-09 --dependency=afterok:$x2 \
    --job-name=av11-fscore --partition=main --gpus=1 --constraint=rtx_4090 --mem=32G --time=03:00:00 \
    --output=logs/av11-fscore-%j.out --error=logs/av11-fscore-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/adaptive_v11_score_round.py --round 2 --split dev --root $rr")
  a2=$(sbatch --parsable --dependency=afterok:$s2 \
    --job-name=av11-fadv --partition=cpu --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --output=logs/av11-fadv-%j.out --error=logs/av11-fadv-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/adaptive_v11_role_advance_round.py --round 2 --split dev --root $rr --manifest $MANIFEST --belief $BELIEF")

  g3=$(sbatch --parsable --exclude=cs-4090-09 --dependency=afterok:$a2 \
    --job-name=av11-fgen --partition=main --gpus=1 --constraint=rtx_4090 --mem=24G --time=01:30:00 \
    --output=logs/av11-fgen-%j.out --error=logs/av11-fgen-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/av11_factorial_eval.py generate --root $rr --adapter $adapter --seed $seed --round 3 $sf")
  x3=$(sbatch --parsable --dependency=afterok:$g3 \
    --job-name=av11-fexec --partition=cpu --cpus-per-task=2 --mem=12G --time=06:00:00 \
    --output=logs/av11-fexec-%j.out --error=logs/av11-fexec-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/adaptive_v11_execute_round.py --round 3 --split dev --root $rr --manifest $MANIFEST --workers 2 --retries 3")
  s3=$(sbatch --parsable --exclude=cs-4090-09 --dependency=afterok:$x3 \
    --job-name=av11-fscore --partition=main --gpus=1 --constraint=rtx_4090 --mem=32G --time=03:00:00 \
    --output=logs/av11-fscore-%j.out --error=logs/av11-fscore-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/adaptive_v11_score_round.py --round 3 --split dev --root $rr")
  a3=$(sbatch --parsable --dependency=afterok:$s3 \
    --job-name=av11-fadv --partition=cpu --cpus-per-task=2 --mem=8G --time=00:30:00 \
    --output=logs/av11-fadv-%j.out --error=logs/av11-fadv-%j.err \
    --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/adaptive_v11_role_advance_round.py --round 3 --split dev --root $rr --manifest $MANIFEST --belief $BELIEF")

  echo "seed=$seed arm=$arm G1=$g1 X1=$x1 S1=$s1 A1=$a1 G2=$g2 X2=$x2 S2=$s2 A2=$a2 G3=$g3 X3=$x3 S3=$s3 A3=$a3" | tee -a "$JOBS" >&2
  ARM_END="$a3"
}

START="$PREP"
for seed in 101 202 303; do
  ENDS=()
  submit_arm "$seed" A_original_static    "$ORIG" 1 "$START"; ENDS+=("$ARM_END")
  submit_arm "$seed" B_original_adaptive  "$ORIG" 0 "$START"; ENDS+=("$ARM_END")
  submit_arm "$seed" C_grpo_static        "$GRPO" 1 "$START"; ENDS+=("$ARM_END")
  submit_arm "$seed" D_grpo_adaptive      "$GRPO" 0 "$START"; ENDS+=("$ARM_END")
  START=$(IFS=:; echo "${ENDS[*]}")
done

ANALYSIS=$(sbatch --parsable --dependency=afterok:$START \
  --job-name=av11-fanal --partition=cpu --cpus-per-task=2 --mem=4G --time=00:30:00 \
  --output=logs/av11-factorial-analysis-%j.out --error=logs/av11-factorial-analysis-%j.err \
  --wrap="cd $HOME/mas_rl_detective && export PYTHONPATH=$HOME/mas_rl_detective:\${PYTHONPATH:-} && $PY -u scripts/av11_factorial_eval.py analyze --root $FINAL")

echo "ANALYSIS_JOB=$ANALYSIS" | tee -a "$JOBS"
echo "Analysis: cat logs/av11-factorial-analysis-${ANALYSIS}.out"
echo "Max external MAS calls = 1152"
squeue --me
