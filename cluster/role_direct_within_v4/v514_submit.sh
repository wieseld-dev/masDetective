#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Python not found: $PY" >&2; exit 2; }
SCRIPT=analysis/role_direct_within_v4/v514_matched_discrimination.py
BASE="$PWD/artifacts/role_direct_within_v4/matched_v514"
mkdir -p "$BASE" logs
"$PY" -m py_compile "$SCRIPT"
"$PY" -u -m analysis.role_direct_within_v4.v514_matched_discrimination selftest
ACTION="${1:-prepare}"
case "$ACTION" in
  prepare)
    RUN="$BASE/pilot_$(date +%Y%m%d_%H%M%S)_$$"
    "$PY" -u -m analysis.role_direct_within_v4.v514_matched_discrimination prepare --root "$RUN"
    printf '%s\n' "$RUN" > "$BASE/latest_run_path.txt"
    echo "V514_PREPARED_RUN=$RUN"
    echo "REVIEW_QUESTIONS=$RUN/public_probe_preview.tsv"
    echo "NO_BILLABLE_EXECUTIONS_SUBMITTED"
    echo "To execute after reviewing, run: V514_APPROVE_MAS_EXECUTIONS=30 bash cluster/role_direct_within_v4/v514_submit.sh execute"
    ;;
  execute)
    if [[ "${V514_APPROVE_MAS_EXECUTIONS:-}" != 30 ]]; then
      echo 'Explicit billable approval required: V514_APPROVE_MAS_EXECUTIONS=30 bash cluster/role_direct_within_v4/v514_submit.sh execute' >&2
      exit 3
    fi
    [[ -f "$BASE/latest_run_path.txt" ]] || { echo 'Prepare first' >&2; exit 4; }
    RUN="$(cat "$BASE/latest_run_path.txt")"
    [[ -f "$RUN/preflight.json" && -f "$RUN/candidates_train_round_01.jsonl" ]] || { echo 'Prepared run not found' >&2; exit 4; }
    [[ ! -e "$RUN/responses_train_round_01.jsonl" ]] || { echo 'Already executed; refusing repeat' >&2; exit 5; }
    [[ ! -e "$RUN/submitted.txt" ]] || { echo 'Job already submitted; refusing duplicate scheduling' >&2; exit 5; }
    JOB="$(sbatch --parsable --export="ALL,V514_RUN=$RUN,V514_APPROVE_MAS_EXECUTIONS=30" cluster/role_direct_within_v4/v514_cpu.sbatch)"
    JOB="${JOB%%;*}"
    [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected Slurm id: $JOB" >&2; exit 6; }
    printf '%s\n' "$JOB" > "$RUN/submitted.txt"
    echo "V514_SUBMITTED JOB=$JOB RUN=$RUN 30 MAX MAS calls"
    echo 'Expected runtime after queue: roughly 5-20 minutes; backend latency/rate limits may vary.'
    ;;
  *) echo 'Usage: v514_submit.sh [prepare|execute]' >&2; exit 2;;
esac
