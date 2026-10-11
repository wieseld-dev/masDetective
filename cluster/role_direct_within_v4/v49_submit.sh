#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
if [[ "${V49_APPROVE_MAS_EXECUTIONS:-}" != 54 ]]; then
 echo "Refusing API stage. Explicit opt-in needed: V49_APPROVE_MAS_EXECUTIONS=54 bash cluster/role_direct_within_v4/v49_submit.sh" >&2
 echo "A MAS execution may trigger multiple billable LLM requests; financial cost is unknown." >&2
 exit 2
fi
"$PY" -m py_compile analysis/role_direct_within_v4/v49_pilot.py
mkdir -p logs artifacts/role_direct_within_v4/pilot_v49
RUN="$PWD/artifacts/role_direct_within_v4/pilot_v49/pilot_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" -u -m analysis.role_direct_within_v4.v49_pilot prepare --root "$RUN"
JOB=$(sbatch --parsable --export="ALL,V49_ROOT=$RUN" cluster/role_direct_within_v4/v49_pilot_cpu.sbatch)
JOB="${JOB%%;*}"
echo "MAS5_V49_SUBMITTED JOB=$JOB RUN=$RUN"
echo "Check: sacct -j $JOB -X --format=JobID,State,ExitCode,Elapsed; tail -n 40 logs/v49-mas5-v49-pilot-$JOB.out"
echo "Result: $RUN/pilot_report.json"
