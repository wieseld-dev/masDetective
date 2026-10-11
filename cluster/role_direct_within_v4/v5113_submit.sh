#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Python env not found at $PY" >&2; exit 2; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p artifacts/role_direct_within_v4/sensor_v5113 logs
"$PY" -m analysis.role_direct_within_v4.v5113_dev_heldout selftest
AUDIT="$PWD/artifacts/role_direct_within_v4/sensor_v5113/preflight_$(date +%Y%m%d_%H%M%S)_$$.json"
export V5113_AUDIT_FILE="$AUDIT"
echo "AUDIT=$AUDIT"
"$PY" -m analysis.role_direct_within_v4.v5113_dev_heldout preflight
RUN="$PWD/artifacts/role_direct_within_v4/sensor_v5113/rolewise_$(date +%Y%m%d_%H%M%S)_$$"
JOB=$(sbatch --parsable --job-name=mas5-v5113-rolewise \
  --export="ALL,V5113_RUN=$RUN,V5113_AUDIT_FILE=$AUDIT" \
  cluster/role_direct_within_v4/v5113_cpu.sbatch)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Invalid Job ID: $JOB" >&2; exit 3; }
printf '%s\t%s\t%s\n' "$JOB" "rolewise" "$RUN" >> "$PWD/artifacts/role_direct_within_v4/sensor_v5113/submitted_jobs.tsv"
printf '%s\n' "$RUN" > "$PWD/artifacts/role_direct_within_v4/sensor_v5113/latest_run_path.txt"
echo "V5113_JOB=$JOB"
echo "V5113_RUN=$RUN"
echo "ESTIMATED_CPU_AFTER_START=3_TO_30_MINUTES; NO_MAS_API_CALLS"
