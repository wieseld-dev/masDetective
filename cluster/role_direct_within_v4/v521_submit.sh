#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
mkdir -p logs
if [[ -z "${V51_ROOT:-}" ]]; then
  shopt -s nullglob
  dirs=("$PWD"/artifacts/role_direct_within_v4/adaptive_v51/r2preview_*)
  ((${#dirs[@]})) || { echo 'No existing V5.1 R2 preview found' >&2; exit 2; }
  V51_ROOT=$(ls -dt "${dirs[@]}" | head -1)
fi
V51_ROOT=$(realpath "$V51_ROOT")
[[ -f "$V51_ROOT/public_scored_r1_states.jsonl" ]] || { echo 'V5.1 public scored states missing' >&2; exit 2; }
[[ -f "$V51_ROOT/r2_candidates_public.jsonl" ]] || { echo 'Original V5.1 candidates missing' >&2; exit 2; }
"$PY" -m py_compile analysis/role_direct_within_v4/v521_adaptive_r2.py
"$PY" -m analysis.role_direct_within_v4.v521_adaptive_r2 selftest
"$PY" -m analysis.role_direct_within_v4.v521_adaptive_r2 preflight --root "$V51_ROOT"
# Prevent concurrent V5.2.1 writers to same checkpoint.
if squeue -h -u "$USER" -o '%j' | grep -Eq '^mas5-v5(2|21)-r2$'; then
  echo 'MAS5 V5.2.1 job already running/queued. Refusing concurrent write.' >&2; exit 3
fi
JOB=$(sbatch --parsable --export="ALL,V51_ROOT=$V51_ROOT,V521_TRIES=${V521_TRIES:-8}" cluster/role_direct_within_v4/v521_generate_4090.sbatch)
JOB=${JOB%%;*}
echo "MAS5_V521_SUBMITTED job=$JOB root=$V51_ROOT (NO MAS/API calls)"
echo "CHECK: sacct -j $JOB -X --format=JobID,State,ExitCode,Elapsed"
echo "REPORT: $V51_ROOT/r2_v521_preview/r2_audit_report.json"
