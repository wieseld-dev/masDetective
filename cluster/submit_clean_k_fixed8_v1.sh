#!/bin/bash
set -euo pipefail

cd "$HOME/mas_rl_detective"

ROOT="artifacts/clean_k_fixed8_v1"
MAN="$ROOT/manifest_privileged.json"

BELIEF="artifacts/adaptive_v11_active_v1/frozen_belief.joblib"

ADAPTER="artifacts/adaptive_v11_curriculum_v2/checkpoints/stage_10_cycle_02/policy"

SEL="artifacts/adaptive_v11_curriculum_v2/selector/utility_ridge.npz"

mkdir -p logs "$ROOT"

echo "=================================================="
echo "CLEAN K FIXED-8 CONFIG"
echo "=================================================="
echo "ROOT     = $ROOT"
echo "MANIFEST = $MAN"
echo "BELIEF   = $BELIEF"
echo "ADAPTER  = $ADAPTER"
echo "SELECTOR = $SEL"
echo

for x in \
    "$MAN" \
    "$BELIEF" \
    "$ADAPTER/adapter_model.safetensors" \
    "$ADAPTER/adapter_config.json" \
    "$SEL"
do
    test -s "$x" || {
        echo "MISSING/EMPTY: $x"
        exit 2
    }
done

# Initial public state must already exist from preparation.
test -s "$ROOT/states_dev_round_00.jsonl" || {
    echo "Missing initial state file"
    exit 2
}

JOBS="$ROOT/rollout_jobs.tsv"
: > "$JOBS"

PREV=""

for R in $(seq 1 8); do

    echo
    echo "================ ROUND $R ================"

    DEP=()
    if [[ -n "$PREV" ]]; then
        DEP=(--dependency="afterok:$PREV")
    fi

    # ---------------------------------------------------------
    # GENERATE:
    # 8 benign candidates/system.
    # V2 selector chooses exactly one.
    # dev split => epsilon forced to zero.
    # ---------------------------------------------------------
    G=$(
        sbatch --parsable \
          "${DEP[@]}" \
          --export="ALL,CK_ROOT=$ROOT,CK_ROUND=$R,CK_ADAPTER=$ADAPTER,CK_SELECTOR=$SEL" \
          cluster/clean_k_generate_4090.sbatch
    )
    G="${G%%;*}"

    # ---------------------------------------------------------
    # EXECUTE:
    # candidates_dev contains only the selected candidate,
    # therefore exactly one real black-box query/system.
    # ---------------------------------------------------------
    X=$(
        sbatch --parsable \
          --dependency="afterok:$G" \
          --export="ALL,CURR_ROOT=$ROOT,CURR_MANIFEST=$MAN,CURR_ROUND=$R,CURR_SPLIT=dev,CURR_WORKERS=6" \
          cluster/curriculum_execute_cpu.sbatch
    )
    X="${X%%;*}"

    # ---------------------------------------------------------
    # SCORE visible query/response.
    # ---------------------------------------------------------
    S=$(
        sbatch --parsable \
          --dependency="afterok:$X" \
          --export="ALL,CURR_ROOT=$ROOT,CURR_ROUND=$R,CURR_SPLIT=dev" \
          cluster/curriculum_score_4090.sbatch
    )
    S="${S%%;*}"

    # ---------------------------------------------------------
    # V2 ADVANCE:
    # updates PUBLIC role belief/history for next round.
    # selector stays frozen because split=dev.
    # ---------------------------------------------------------
    A=$(
        sbatch --parsable \
          --dependency="afterok:$S" \
          --export="ALL,CK_ROOT=$ROOT,CK_MANIFEST=$MAN,CK_BELIEF=$BELIEF,CK_SELECTOR=$SEL,CK_ROUND=$R" \
          cluster/clean_k_advance_cpu.sbatch
    )
    A="${A%%;*}"

    printf "R%02d\tGEN=%s\tEXEC=%s\tSCORE=%s\tADV=%s\n" \
      "$R" "$G" "$X" "$S" "$A" \
      | tee -a "$JOBS"

    PREV="$A"
done

echo "$PREV" > "$ROOT/final_rollout_job.txt"

echo
echo "=================================================="
echo "CLEAN K FIXED-8 SUBMITTED"
echo "=================================================="
echo "FINAL_JOB=$PREV"
echo "JOB_TABLE=$JOBS"
echo
echo "Systems                  : 252"
echo "Rounds/system            : 8"
echo "Candidate pool/round     : 8"
echo "Real MAS queries/system  : 8"
echo "Total real MAS queries   : 2016"
echo "Selector exploration     : 0"
echo "Stage                    : 10"
