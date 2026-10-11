#!/usr/bin/env bash
set -euo pipefail

cd /home/wieseld/mas_rl_detective

CODE=analysis/role_direct_adaptive_v3/attack.py
PY=/home/wieseld/.conda/envs/masdetective/bin/python
MANIFEST=artifacts/role_direct_routing_pilot_v1/manifest_privileged.json

if squeue -h -u "$USER" -o '%j' |
   grep -Eq '^rda2-(generate|select|execute|score|report)-r[12]$'; then
    echo "ERROR: V2 jobs still active. Cancel V2 first."
    exit 1
fi

ROOT="artifacts/role_direct_adaptive_v3_$(date +%Y%m%d_%H%M%S)_$$"

"$PY" "$CODE" preflight \
    --root "$ROOT" \
    --manifest "$MANIFEST"

PREVIOUS=''
JOBS=()

trap 'if [ "${#JOBS[@]}" -gt 0 ]; then scancel "${JOBS[@]}" 2>/dev/null || true; fi' ERR

for task in \
    'generate 1 gpu' \
    'select 1 cpu' \
    'execute 1 cpu' \
    'score 1 gpu' \
    'generate 2 gpu' \
    'select 2 cpu' \
    'execute 2 cpu' \
    'score 2 gpu' \
    'report 2 cpu'
do
    read -r STAGE RND RESOURCE <<< "$task"

    DEP=()
    if [ -n "$PREVIOUS" ]; then
        DEP=("--dependency=afterok:${PREVIOUS}")
    fi

    ANS=$(sbatch --parsable "${DEP[@]}" \
        --job-name="rda3-${STAGE}-r${RND}" \
        --export="ALL,ROLE_STAGE=$STAGE,ROLE_ROUND=$RND,ROLE_ROOT=$ROOT,ROLE_MANIFEST=$MANIFEST" \
        "cluster/role_direct_adaptive_v3/${RESOURCE}.sbatch")

    JOB="${ANS%%;*}"
    [[ "$JOB" =~ ^[0-9]+$ ]] || exit 1

    JOBS+=("$JOB")
    printf '%s\t%s\t%s\n' "$JOB" "$STAGE" "$RND" |
        tee -a "$ROOT/job_ids.tsv"

    PREVIOUS="$JOB"
done

trap - ERR

echo "V3_SUBMITTED ROOT=$ROOT FINAL_JOB=$PREVIOUS"
echo "MONITOR: squeue --me"
echo "RESULTS: $ROOT/final_report.json"
