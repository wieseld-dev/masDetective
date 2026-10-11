#!/usr/bin/env bash
# Safely add live Sensor/AUC snapshots to an already-submitted *original* V5.6 job chain.
# Adds GPU scoring, no MAS executions or provider/API calls.
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-}"
[[ -n "$RUN" && -f "$RUN/jobs.tsv" ]] || { echo 'Usage: bash cluster/role_direct_within_v4/v561_attach_live.sh EXISTING_V56_RUN' >&2;exit 2; }
RUN="$(realpath "$RUN")"
[[ "$RUN" == "$PWD"/artifacts/role_direct_within_v4/nonadaptive_v56/mas5_* ]] || { echo 'Wrong root' >&2;exit 2; }
[[ ! -e "$RUN/live_jobs.tsv" ]] || { echo 'Already attached. Refusing duplicate GPU jobs.' >&2;exit 2; }
[[ ! -e "$RUN/live_auc_round_01.json" && ! -e "$RUN/sensor_public_len640_round_01.jsonl" ]] || {
  echo 'Existing live scores or snapshots detected. Inspect before attaching.' >&2;exit 2;
}
EXEC1="$(awk -F '\t' '$2=="execute1"{print $1}' "$RUN/jobs.tsv")"
EXEC2="$(awk -F '\t' '$2=="execute2"{print $1}' "$RUN/jobs.tsv")"
[[ "$EXEC1" =~ ^[0-9]+$ && "$EXEC2" =~ ^[0-9]+$ ]] || { echo 'Expected original execute1/execute2 jobs' >&2;exit 2; }
if awk -F '\t' '$2=="score1" || $2=="score2" {bad=1} END{exit !bad}' "$RUN/jobs.tsv";then
  echo 'Looks like V5.6.1 already; refuse attaching duplicate scoring' >&2;exit 2
fi
JOBS=()
trap '[[ ${#JOBS[@]} -eq 0 ]] || scancel "${JOBS[@]}" 2>/dev/null || true' ERR
add_job() {
  local name="$1" kind="$2" dependency="$3" answer jid
  answer="$(sbatch --parsable --dependency="afterok:$dependency" --job-name="mas5-v561-$name" \
    --export="ALL,V561_ROOT=$RUN,V561_STAGE=$name" \
    "cluster/role_direct_within_v4/v561_$kind.sbatch")"
  jid="${answer%%;*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo 'Bad sbatch job id' >&2;return 1; }
  JOBS+=("$jid")
  printf '%s\t%s\n' "$jid" "$name" | tee -a "$RUN/live_jobs.tsv"
  RET_JID="$jid"
}
add_job score1 gpu "$EXEC1"; SCORE1="$RET_JID"
add_job live1 cpu "$SCORE1"; LIVE1="$RET_JID"
add_job score2 gpu "$EXEC2:$LIVE1"; SCORE2="$RET_JID"
add_job live2 cpu "$SCORE2"; LIVE2="$RET_JID"
trap - ERR
printf 'V561_ATTACHED: Original paid jobs untouched. Four diagnostic jobs added, NO additional MAS API calls.\n'
printf 'Live scoreboard: %s/live_auc_history.jsonl\n' "$RUN"
