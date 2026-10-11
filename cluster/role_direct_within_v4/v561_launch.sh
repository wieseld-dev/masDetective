#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-}"
if [[ -z "$RUN" || ! -f "$RUN/billable_preflight.json" ]]; then
  echo 'Usage: V56_APPROVE_MAS_EXECUTIONS=200 bash cluster/role_direct_within_v4/v561_launch.sh RUN' >&2;exit 2
fi
RUN="$(realpath "$RUN")"
[[ "$RUN" == "$PWD"/artifacts/role_direct_within_v4/nonadaptive_v56/mas5_* ]] || { echo 'Unexpected run path' >&2;exit 2; }
[[ "${V56_APPROVE_MAS_EXECUTIONS:-}" == 200 ]] || { echo 'Explicit billable approval for 200 MAS calls required' >&2;exit 2; }
[[ ! -e "$RUN/jobs.tsv" ]] || { echo 'Run already submitted. Do NOT submit a second copy.' >&2;exit 2; }
[[ ! -e "$RUN/responses_dev_round_01.jsonl" && ! -e "$RUN/responses_dev_round_02.jsonl" ]] || { echo 'Existing MAS responses: refusing duplicate paid calls' >&2;exit 2; }
[[ -f .env ]] || { echo '.env missing' >&2;exit 2; }
set -a
source .env
set +a
[[ -n "${OPENAI_API_KEY:-}" ]] || { echo 'OPENAI_API_KEY missing' >&2;exit 2; }
STAGES=(execute1 score1 live1 execute2 score2 live2 final)
JOBS=();PREV=''
trap '[[ ${#JOBS[@]} -eq 0 ]] || scancel "${JOBS[@]}" 2>/dev/null || true' ERR
for name in "${STAGES[@]}"; do
  kind=cpu
  [[ "$name" == score1 || "$name" == score2 ]] && kind=gpu
  dep=();[[ -z "$PREV" ]] || dep=(--dependency="afterok:$PREV")
  answer="$(sbatch --parsable "${dep[@]}" --job-name="mas5-v561-$name" \
    --export="ALL,V561_ROOT=$RUN,V561_STAGE=$name,V56_APPROVE_MAS_EXECUTIONS=200" \
    "cluster/role_direct_within_v4/v561_$kind.sbatch")"
  jid="${answer%%;*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Bad sbatch response $answer" >&2;exit 1; }
  JOBS+=("$jid");PREV="$jid"
  printf '%s\t%s\n' "$jid" "$name" | tee -a "$RUN/jobs.tsv"
done
trap - ERR
printf '\nV561_SUBMITTED RUN=%s\nLIVE=%s/live_auc_latest.json\nHISTORY=%s/live_auc_history.jsonl\n' "$RUN" "$RUN" "$RUN"
printf 'AUC snapshots after R1 and R2; approx 20-120min+queue total; unknown API costs. Each Slurm stage limit 20h.\n'
