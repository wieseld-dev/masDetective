#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Python missing: $PY" >&2; exit 2; }
SCRIPT=analysis/role_direct_within_v4/v515_multi_background.py
BASE="$PWD/artifacts/role_direct_within_v4/matched_v515"
mkdir -p "$BASE" logs
"$PY" -m py_compile "$SCRIPT"
"$PY" -u -m analysis.role_direct_within_v4.v515_multi_background selftest
ACTION="${1:-prepare}"
case "$ACTION" in
  prepare)
    RUN="$BASE/pilot_$(date +%Y%m%d_%H%M%S)_$$"
    "$PY" -u -m analysis.role_direct_within_v4.v515_multi_background prepare --root "$RUN"
    printf '%s\n' "$RUN" > "$BASE/latest_run_path.txt"
    echo "V515_PREPARED_RUN=$RUN"
    echo "REVIEW=$RUN/public_probe_preview.tsv"
    echo 'NO_PAID_EXECUTIONS_SCHEDULED'
    echo 'To authorize: V515_APPROVE_MAS_EXECUTIONS=60 bash cluster/role_direct_within_v4/v515_submit.sh execute'
    ;;
  execute)
    [[ "${V515_APPROVE_MAS_EXECUTIONS:-}" == 60 ]] || { echo 'Set V515_APPROVE_MAS_EXECUTIONS=60 to authorize 60 paid MAS executions' >&2; exit 3; }
    [[ -f "$BASE/latest_run_path.txt" ]] || { echo 'Prepare first' >&2; exit 4; }
    RUN="$(cat "$BASE/latest_run_path.txt")"
    [[ -f "$RUN/preflight.json" && -f "$RUN/candidates_train_round_01.jsonl" ]] || { echo 'Missing prepared run' >&2; exit 4; }
    [[ ! -e "$RUN/responses_train_round_01.jsonl" && ! -e "$RUN/submitted.txt" ]] || { echo 'Already submitted/executed; refusing duplicate' >&2; exit 5; }
    JOB="$(sbatch --parsable --export="ALL,V515_RUN=$RUN,V515_APPROVE_MAS_EXECUTIONS=60" cluster/role_direct_within_v4/v515_cpu.sbatch)"
    JOB="${JOB%%;*}"
    [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "Unexpected Slurm JobID: $JOB" >&2; exit 6; }
    printf '%s\n' "$JOB" > "$RUN/submitted.txt"
    echo "V515_SUBMITTED JOB=$JOB RUN=$RUN MAX=60 MAS executions"
    echo 'Estimated runtime after queue ~10-40 minutes; may vary due to remote model latency.'
    ;;
  *) echo 'Usage: v515_submit.sh [prepare|execute]' >&2; exit 2 ;;
esac
