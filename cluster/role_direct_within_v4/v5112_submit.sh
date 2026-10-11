#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Missing python at $PY" >&2; exit 2; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p artifacts/role_direct_within_v4/sensor_v5112 logs
"$PY" -m analysis.role_direct_within_v4.v5112_overlap_safe selftest
export V5112_AUDIT_FILE="$PWD/artifacts/role_direct_within_v4/sensor_v5112/preflight_overlap_$(date +%Y%m%d_%H%M%S)_$$.json"
echo "LEAKAGE_AUDIT=$V5112_AUDIT_FILE"
# If no clean Val/Test grouping exists, fail here: no Slurm job is submitted.
"$PY" -m analysis.role_direct_within_v4.v5112_overlap_safe preflight
RUN="$PWD/artifacts/role_direct_within_v4/sensor_v5112/rolewise_$(date +%Y%m%d_%H%M%S)_$$"
JOB=$(sbatch --parsable --job-name=mas5-v5112-rolewise \
  --export="ALL,V5112_RUN=$RUN,V5112_AUDIT_FILE=$V5112_AUDIT_FILE" \
  cluster/role_direct_within_v4/v5112_cpu.sbatch)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Bad Slurm job ID $JOB" >&2; exit 3; }
echo "V5112_JOB=$JOB"
echo "V5112_RUN=$RUN"
echo "EXPECTED_CPU_AFTER_ALLOCATION=3_TO_20_MINUTES"
echo "NO_NEW_MAS_API_CALLS=true"
