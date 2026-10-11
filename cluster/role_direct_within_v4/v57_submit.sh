#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
PY="$HOME/.conda/envs/masdetective/bin/python"
[[ -x "$PY" ]] || { echo "Missing python at $PY" >&2;exit 2; }
V56_RUN="${1:-}"
if [[ -z "$V56_RUN" ]];then
  V56_RUN="$(ls -dt artifacts/role_direct_within_v4/nonadaptive_v56/mas5_* 2>/dev/null | head -1)"
fi
[[ -f "$V56_RUN/jobs.tsv" ]] || { echo 'Provide a submitted V56 run containing jobs.tsv' >&2;exit 2; }
V56_RUN="$(realpath "$V56_RUN")"
[[ "$V56_RUN" == "$PWD"/artifacts/role_direct_within_v4/nonadaptive_v56/mas5_* ]] || { echo 'Unsafe V56 path' >&2;exit 2; }
EXEC2="$(awk '$2=="execute2" {print $1}' "$V56_RUN/jobs.tsv" | tail -1)"
[[ "$EXEC2" =~ ^[0-9]+$ ]] || { echo 'No V56 execute2 job ID; will not run concurrently with incomplete data' >&2;exit 2; }
V57_RUN="$PWD/artifacts/role_direct_within_v4/sensor_v57/night_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$V57_RUN" logs
# Login-node preflight requires only standard Python (no GPU, no API).
"$PY" -u -m analysis.role_direct_within_v4.v57_sensor_night preflight \
 --run "$V57_RUN" --v56 "$V56_RUN"
# Require a dependency even if R2 completed. If failed, the training will not run.
JOB="$(sbatch --parsable --dependency="afterok:$EXEC2" \
  --job-name=mas5-v57-sensor --export="ALL,V57_RUN=$V57_RUN,V57_V56=$V56_RUN,V57_HOURS=${V57_HOURS:-8.75},V57_TOKENS=${V57_TOKENS:-1280},V57_MAX_STEPS=${V57_MAX_STEPS:-6000},V57_EVAL_EVERY=${V57_EVAL_EVERY:-150},V57_PATIENCE=${V57_PATIENCE:-10}" \
  cluster/role_direct_within_v4/v57_sensor_4090.sbatch)"
JOB="${JOB%%;*}"
[[ "$JOB" =~ ^[0-9]+$ ]] || { echo 'Bad SLURM job output' >&2;exit 2; }
printf '%s\tv57_sensor\n' "$JOB" > "$V57_RUN/jobs.tsv"
printf '%s\n' "$V56_RUN" > "$V57_RUN/source_v56_path.txt"
printf '%s\n' "$EXEC2" > "$V57_RUN/dependency_execute2_id.txt"
printf 'V57_RUN=%s\nV57_JOB=%s\nWAITING_FOR_V56_EXECUTE2=%s\n' "$V57_RUN" "$JOB" "$EXEC2"
printf 'No V56 cancel/requeue. No MAS/API calls. Training up to 8.75 hours and early stopping.\n'
