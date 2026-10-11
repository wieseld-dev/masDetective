#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Missing python: $PY" >&2; exit 2; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
"$PY" -m analysis.role_direct_within_v4.v511_rolewise_verifier selftest
# Preflight source/split before submitting a job, no API calls.
"$PY" - <<'PY'
from analysis.role_direct_within_v4.v511_rolewise_verifier import prepare_sources
roles,train,val,test,stat,priority=prepare_sources()
print('V511_PREFLIGHT',{'roles':len(roles),'train_observations':len(train),'validation_systems':len(val),'test_systems':len(test)})
for role,c in stat['training_by_role'].items():
    if min(c['positive'],c['negative'])<3:
        raise RuntimeError(f'Insufficient PRESENT/ABSENT examples for {role}: {c}')
PY
mkdir -p logs artifacts/role_direct_within_v4/sensor_v511
RUN="$PWD/artifacts/role_direct_within_v4/sensor_v511/rolewise_$(date +%Y%m%d_%H%M%S)_$$"
JOB=$(sbatch --parsable --job-name=mas5-v511-rolewise --export="ALL,V511_RUN=$RUN" cluster/role_direct_within_v4/v511_cpu.sbatch)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Invalid job ID: $JOB" >&2; exit 2; }
echo "V511_JOB=$JOB"
echo "V511_RUN=$RUN"
echo "V511_COST=NO_API_CALLS_CPU_ONLY"
echo "V511_EXPECTED=3_TO_20_MINUTES_AFTER_ALLOCATION"
