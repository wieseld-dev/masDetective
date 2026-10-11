#!/bin/bash

set -euo pipefail

cd "$HOME/mas_rl_detective"

ROOT="artifacts/belief_targeted_simple5_clean_v1"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY="$HOME/.conda/envs/masdetective/bin/python"

# ------------------------------------------------------------
# R1 MUST ALREADY BE COMPLETE.
# ------------------------------------------------------------

"$PY" -u analysis/bt_simple5_runtime_guard_v1.py \
  --root "$ROOT" \
  --round 1 \
  --phase post_advance

test -s "$ROOT/states_dev_round_01.jsonl"

# ------------------------------------------------------------
# Fail closed if R2+ somehow already contains actual outputs.
# ------------------------------------------------------------

for R in $(seq 2 8); do

    RR=$(printf '%02d' "$R")

    for F in \
      "$ROOT/candidate_pool_dev_round_${RR}.jsonl" \
      "$ROOT/candidates_dev_round_${RR}.jsonl" \
      "$ROOT/responses_dev_round_${RR}.jsonl" \
      "$ROOT/scored_dev_round_${RR}.jsonl" \
      "$ROOT/states_dev_round_${RR}.jsonl"
    do
        if [ -e "$F" ]; then
            echo "ABORT: unexpected pre-existing artifact: $F" >&2
            exit 2
        fi
    done
done

: > "$ROOT/jobs_resume_r2.tsv"

prev=""

for R in $(seq 2 8); do

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
      | tee -a "$ROOT/jobs_resume_r2.tsv"

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
  > "$ROOT/report_job_id_resume"

echo
echo "FINAL_ADVANCE=$prev"
echo "REPORT=$REPORT"
