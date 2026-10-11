#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
APPROVAL=408
if [[ "${V50_APPROVE_MAS_EXECUTIONS:-}" != "$APPROVAL" ]]; then
 echo "STOP: requires V50_APPROVE_MAS_EXECUTIONS=408 bash cluster/role_direct_within_v4/v50_submit.sh" >&2
 echo "Up to 408 MAS executions, multiple paid API requests per execution; dollar cost unknown." >&2
 exit 2
fi
"$PY" -m py_compile analysis/role_direct_within_v4/v50_matched_expansion.py
mkdir -p logs artifacts/role_direct_within_v4/expansion_v50
RUN="$PWD/artifacts/role_direct_within_v4/expansion_v50/exp_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -u -m analysis.role_direct_within_v4.v50_matched_expansion prepare --root "$RUN"
# Review run preflight, no automatic training/test; budget must be explicitly approved on submit.
JOB=$(sbatch --parsable --export="ALL,V50_ROOT=$RUN,V50_APPROVE_MAS_EXECUTIONS=408" cluster/role_direct_within_v4/v50_expansion_cpu.sbatch)
JOB="${JOB%%;*}"
printf 'V50_SUBMITTED JOB=%s RUN=%s\n' "$JOB" "$RUN"
printf 'Check: sacct -j %s -X --format=JobID,State,ExitCode,Elapsed\n' "$JOB"
printf 'Report: %s/evidence_report.json\n' "$RUN"
