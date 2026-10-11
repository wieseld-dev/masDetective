#!/usr/bin/env bash
# MAS5 V4.7.5: only resubmit regeneration, never execute MAS/API.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
: "${V46_ROOT:?Export V46_ROOT first}"
V46_ROOT=$(realpath -e "$V46_ROOT")
export V46_ROOT
[[ "$V46_ROOT" == "$PWD"/artifacts/role_direct_within_v4/overnight_v46/mas5_night_* ]] || { echo 'Invalid MAS5 RUN path' >&2; exit 2; }
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -m py_compile analysis/role_direct_within_v4/v473_generator.py analysis/role_direct_within_v4/v473_regenerate.py analysis/role_direct_within_v4/v475_preflight.py
"$PY" -m analysis.role_direct_within_v4.v475_preflight --root "$V46_ROOT"
if squeue -h -u "$USER" -o '%j' | grep -Eq '^(mas5-|mas5-v473-regen$)'; then
  echo 'MAS5 job queued/running; refusing duplicate submission' >&2;exit 2
fi
mkdir -p logs
export V475_SAFE_INFERENCE=1
export V473_ATTEMPTS="${V473_ATTEMPTS:-36}"
EXCLUDE=()
if [[ -n "${V475_EXCLUDE_NODES:-}" ]]; then EXCLUDE=(--exclude="$V475_EXCLUDE_NODES"); fi
JOB=$(sbatch --parsable "${EXCLUDE[@]}"       --export="ALL,V46_ROOT=$V46_ROOT,V475_SAFE_INFERENCE=1,V473_ATTEMPTS=$V473_ATTEMPTS"       cluster/role_direct_within_v4/v473_regenerate_4090.sbatch)
echo "MAS5_V475_JOB=$JOB"
echo "LOG=logs/v473-mas5-v473-regen-$JOB.out"
echo "CHECK=sacct -j $JOB -X --format=JobID,State,ExitCode,Elapsed"
echo 'No MAS/API calls; no new model training. CUDA context starts clean in the new job.'
