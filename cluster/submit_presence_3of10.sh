#!/bin/bash
set -euo pipefail

cd "$HOME/mas_rl_detective"

ROOT="${P310_ROOT:-artifacts/presence_3of10_v1}"
BELIEF="${P310_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"
ADAPTER="${P310_ADAPTER:-artifacts/adaptive_v11_curriculum_v2/checkpoints/stage_10_cycle_02/policy}"
V11="${P310_V11:-artifacts/v11_policy}"
H="${P310_HORIZON:-7}"
MODEL="${P310_MODEL:-gpt-4o-mini}"

mkdir -p "$ROOT" logs

test -s "$BELIEF" || {
    echo "missing belief: $BELIEF" >&2
    exit 2
}

test -d "$ADAPTER" || {
    echo "missing adapter: $ADAPTER" >&2
    exit 2
}

test -d "$V11" || {
    echo "missing V11: $V11" >&2
    exit 2
}

JOBS="$ROOT/jobs.tsv"
: > "$JOBS"

EXCLUDE="${P310_EXCLUDE_GPU_NODE:-cs-4090-09}"
GPUEX=()

if [[ -n "$EXCLUDE" ]]; then
    GPUEX=(--exclude="$EXCLUDE")
fi

submit_cpu () {
    local dep="$1"
    local exports="$2"
    local script="$3"

    local D=()

    if [[ -n "$dep" ]]; then
        D=(--dependency="afterok:$dep")
    fi

    sbatch --parsable \
        "${D[@]}" \
        --export="ALL,$exports" \
        "$script"
}

submit_gpu () {
    local dep="$1"
    local exports="$2"
    local script="$3"

    local D=()

    if [[ -n "$dep" ]]; then
        D=(--dependency="afterok:$dep")
    fi

    sbatch --parsable \
        "${D[@]}" \
        "${GPUEX[@]}" \
        --mem=24G \
        --export="ALL,$exports" \
        "$script"
}

I=$(submit_cpu "" \
    "P310_ROOT=$ROOT,P310_BELIEF=$BELIEF" \
    cluster/presence_3of10_init_cpu.sbatch)

echo -e "init\t$I" | tee -a "$JOBS"

PREV="$I"

for R in $(seq 1 "$H"); do

    G=$(submit_gpu "$PREV" \
        "CURR_ROOT=$ROOT,CURR_ADAPTER=$ADAPTER,CURR_V11=$V11,CURR_ROUND=$R,CURR_SPLIT=dev,CURR_GROUP_SIZE=1,CURR_LOCAL_RETRIES=12" \
        cluster/curriculum_generate_4090.sbatch)

    X=$(submit_cpu "$G" \
        "P310_ROOT=$ROOT,P310_ROUND=$R,P310_MODEL=$MODEL" \
        cluster/presence_3of10_execute_cpu.sbatch)

    S=$(submit_gpu "$X" \
        "CURR_ROOT=$ROOT,CURR_ROUND=$R,CURR_SPLIT=dev" \
        cluster/curriculum_score_4090.sbatch)

    A=$(submit_cpu "$S" \
        "P310_ROOT=$ROOT,P310_ROUND=$R,P310_BELIEF=$BELIEF" \
        cluster/presence_3of10_advance_cpu.sbatch)

    echo -e "r${R}_generate\t$G" | tee -a "$JOBS"
    echo -e "r${R}_execute\t$X" | tee -a "$JOBS"
    echo -e "r${R}_score\t$S" | tee -a "$JOBS"
    echo -e "r${R}_advance\t$A" | tee -a "$JOBS"

    PREV="$A"
done

F=$(submit_cpu "$PREV" \
    "P310_ROOT=$ROOT,P310_HORIZON=$H" \
    cluster/presence_3of10_report_cpu.sbatch)

echo -e "report\t$F" | tee -a "$JOBS"

echo
echo "=============================================="
echo "PRESENCE 3-of-10 SUBMITTED"
echo "=============================================="
echo "ROOT=$ROOT"
echo "FINAL REPORT JOB=$F"
echo "JOBS=$JOBS"
echo
echo "Monitor:"
echo "  squeue --me"
echo
echo "After completion:"
echo "  cat $ROOT/REPORT.txt"
