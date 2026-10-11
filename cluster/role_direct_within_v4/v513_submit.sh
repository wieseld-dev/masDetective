#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Missing Python environment: $PY" >&2; exit 2; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
BASE="$PWD/artifacts/role_direct_within_v4/sensor_v513"
mkdir -p "$BASE" logs
"$PY" -m analysis.role_direct_within_v4.v513_role_error_attribution selftest
"$PY" -m analysis.role_direct_within_v4.v513_role_error_attribution preflight
SOURCE="$("$PY" - <<'PY'
from analysis.role_direct_within_v4.v512_calibration_audit import latest_source
print(latest_source().resolve())
PY
)"
RUN="$BASE/role_audit_$(date +%Y%m%d_%H%M%S)_$$"
[[ ! -e "$RUN" ]] || { echo "Output exists: $RUN" >&2; exit 3; }
JOB="$(sbatch --parsable --job-name=mas5-v513-audit \
  --export="ALL,V513_RUN=$RUN,V513_SOURCE=$SOURCE" \
  cluster/role_direct_within_v4/v513_cpu.sbatch)"
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Invalid Job ID: $JOB" >&2; exit 4; }
printf '%s\t%s\n' "$JOB" "$RUN" >> "$BASE/submitted_jobs.tsv"
printf '%s\n' "$RUN" > "$BASE/latest_run_path.txt"
echo "V513_JOB=$JOB"
echo "V513_RUN=$RUN"
echo 'EXPECTED_CPU_RUNTIME=1_TO_5_MINUTES_AFTER_START; NO_MAS_CALLS; NO_MODEL_TRAINING'
