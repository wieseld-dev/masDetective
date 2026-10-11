#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Missing python: $PY" >&2; exit 2; }
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
"$PY" -m analysis.role_direct_within_v4.v5111_rolewise_verifier selftest
# Preflight source/split before submitting a job, no API calls.
mkdir -p artifacts/role_direct_within_v4/sensor_v5111
export V5111_AUDIT_FILE="$PWD/artifacts/role_direct_within_v4/sensor_v5111/preflight_conflicts_$(date +%Y%m%d_%H%M%S)_$$.json"
"$PY" - <<'PY'
import os
from analysis.role_direct_within_v4.v5111_rolewise_verifier import prepare_sources, dump
roles,train,val,test,stat,priority=prepare_sources()
dump(os.environ['V5111_AUDIT_FILE'],stat['visible_label_conflicts'])
print('V5111_CONFLICT_REPORT='+os.environ['V5111_AUDIT_FILE'],flush=True)
print('V5111_PREFLIGHT',{'roles':len(roles),'train_observations':len(train),'validation_systems':len(val),'test_systems':len(test),'ambiguous_unique_excluded':stat['visible_label_conflicts']['ambiguous_unique_observations_excluded'], 'ambiguous_by_role':stat['visible_label_conflicts']['ambiguous_unique_by_role']},flush=True)
for role,c in stat['training_by_role'].items():
    if min(c['positive'],c['negative'])<3:
        raise RuntimeError(f'Insufficient PRESENT/ABSENT examples for {role}: {c}')
PY
mkdir -p logs artifacts/role_direct_within_v4/sensor_v5111
RUN="$PWD/artifacts/role_direct_within_v4/sensor_v5111/rolewise_$(date +%Y%m%d_%H%M%S)_$$"
JOB=$(sbatch --parsable --job-name=mas5-v5111-rolewise --export="ALL,V5111_RUN=$RUN" cluster/role_direct_within_v4/v5111_cpu.sbatch)
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Invalid job ID: $JOB" >&2; exit 2; }
echo "V5111_JOB=$JOB"
echo "V5111_RUN=$RUN"
echo "V5111_COST=NO_API_CALLS_CPU_ONLY"
echo "V5111_EXPECTED=3_TO_20_MINUTES_AFTER_ALLOCATION"
