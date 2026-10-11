#!/bin/bash

set -euo pipefail

cd "$HOME/mas_rl_detective"

ROOT="artifacts/belief_targeted_simple5_clean_v1"

test -s "$ROOT/states_dev_round_00.jsonl"
test -s "$ROOT/manifest_privileged.json"
test -s "$ROOT/frozen_belief.joblib"
test -s "$ROOT/prospective_contract.json"

: > "$ROOT/jobs.tsv"

prev=""

for R in $(seq 1 8); do

    dep=()

    if [ -n "$prev" ]; then
        dep=(
            --dependency="afterok:$prev"
        )
    fi

    G=$(
        sbatch --parsable \
          "${dep[@]}" \
          --export=ALL,S5_ROOT="$ROOT",S5_ROUND="$R" \
          cluster/bt_simple5_gen_select_4090.sbatch
    )

    G="${G%%;*}"

    X=$(
        sbatch --parsable \
          --dependency="afterok:$G" \
          --export=ALL,S5_ROOT="$ROOT",S5_ROUND="$R" \
          cluster/bt_simple5_execute_cpu.sbatch
    )

    X="${X%%;*}"

    S=$(
        sbatch --parsable \
          --dependency="afterok:$X" \
          --export=ALL,S5_ROOT="$ROOT",S5_ROUND="$R" \
          cluster/bt_simple5_score_4090.sbatch
    )

    S="${S%%;*}"

    A=$(
        sbatch --parsable \
          --dependency="afterok:$S" \
          --export=ALL,S5_ROOT="$ROOT",S5_ROUND="$R" \
          cluster/bt_simple5_advance_cpu.sbatch
    )

    A="${A%%;*}"

    printf \
      "%s\t%s\t%s\t%s\t%s\n" \
      "$R" "$G" "$X" "$S" "$A" \
      | tee -a "$ROOT/jobs.tsv"

    prev="$A"
done

REPORT=$(
    sbatch --parsable \
      --dependency="afterok:$prev" \
      --export=ALL,S5_ROOT="$ROOT" \
      cluster/bt_simple5_report_cpu.sbatch
)

REPORT="${REPORT%%;*}"

echo "$REPORT" \
  > "$ROOT/report_job_id"

echo
echo "FINAL_ADVANCE=$prev"
echo "REPORT=$REPORT"
