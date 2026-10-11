#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Python missing: $PY" >&2;exit 2; }
mkdir -p logs artifacts/role_direct_within_v4/sensor_v510
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
"$PY" -m analysis.role_direct_within_v4.v510_generalizing_sensor selftest
RUN="$PWD/artifacts/role_direct_within_v4/sensor_v510/generalize_$(date +%Y%m%d_%H%M%S)_$$"
# Preflight extracts actual existing data, checks composition/test exclusion, and
# stops BEFORE submission if there are not enough real examples.
"$PY" -u -m analysis.role_direct_within_v4.v510_generalizing_sensor prepare --run "$RUN"
TRAIN="$(sbatch --parsable --job-name=mas5-v510-train --export="ALL,V510_RUN=$RUN,V510_HOURS=${V510_HOURS:-8.75},V510_STEPS=${V510_STEPS:-6000},V510_TOKENS=${V510_TOKENS:-1280},V510_EVAL_EVERY=${V510_EVAL_EVERY:-150},V510_PATIENCE=${V510_PATIENCE:-12}" cluster/role_direct_within_v4/v510_train.sbatch)"
TRAIN="${TRAIN%%;*}"
[[ "$TRAIN" =~ ^[0-9]+$ ]] || { echo "Invalid train JOB ID" >&2;exit 2; }
printf '%s\ttrain\n' "$TRAIN" > "$RUN/jobs.tsv"
TEST="$(sbatch --parsable --dependency="afterok:$TRAIN" --job-name=mas5-v510-test --export="ALL,V510_RUN=$RUN,V510_TOKENS=${V510_TOKENS:-1280}" cluster/role_direct_within_v4/v510_compare.sbatch)"
TEST="${TEST%%;*}"
[[ "$TEST" =~ ^[0-9]+$ ]] || { echo "Invalid test JOB ID; train already submitted as $TRAIN" >&2;exit 2; }
printf '%s\ttest\n' "$TEST" >> "$RUN/jobs.tsv"
cat <<EOF
V510_RUN=$RUN
TRAIN_JOB=$TRAIN
TEST_JOB=$TEST
TEST_DEPENDENCY=afterok:$TRAIN
NO_MAS_API_CALLS=true
No existing Slurm jobs or adapters were modified.
EOF
