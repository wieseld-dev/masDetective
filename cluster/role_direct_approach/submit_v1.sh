#!/bin/bash

set -euo pipefail

cd "$HOME/mas_rl_detective"

SOURCE="artifacts/belief_targeted_simple5_clean_v1"
ROOT="artifacts/role_direct_approach_v1"

if [ -e "$ROOT" ]; then
    echo "ABORT: $ROOT already exists." >&2
    echo "ROLE DIRECT V1 requires a fresh root." >&2
    exit 2
fi

mkdir -p \
  "$ROOT" \
  logs

cp \
  "$SOURCE/manifest_privileged.json" \
  "$ROOT/manifest_privileged.json"

cp \
  "$SOURCE/states_dev_round_00.jsonl" \
  "$ROOT/states_dev_round_00.jsonl"

export RD_ROOT="$ROOT"
export RD_PROBES_PER_ROLE="${RD_PROBES_PER_ROLE:-6}"

G=$(
    sbatch --parsable \
      --export=ALL,RD_ROOT="$ROOT",RD_PROBES_PER_ROLE="$RD_PROBES_PER_ROLE" \
      cluster/role_direct_approach/generate_4090.sbatch
)

G="${G%%;*}"

E=$(
    sbatch --parsable \
      --dependency="afterok:$G" \
      --export=ALL,RD_ROOT="$ROOT" \
      cluster/role_direct_approach/execute_cpu.sbatch
)

E="${E%%;*}"

S=$(
    sbatch --parsable \
      --dependency="afterok:$E" \
      --export=ALL,RD_ROOT="$ROOT" \
      cluster/role_direct_approach/score_4090.sbatch
)

S="${S%%;*}"

R=$(
    sbatch --parsable \
      --dependency="afterok:$S" \
      --export=ALL,RD_ROOT="$ROOT" \
      cluster/role_direct_approach/report_cpu.sbatch
)

R="${R%%;*}"

{
    printf "generate\t%s\n" "$G"
    printf "execute\t%s\n" "$E"
    printf "score\t%s\n" "$S"
    printf "report\t%s\n" "$R"
} \
> "$ROOT/jobs.tsv"

echo "ROOT=$ROOT"
echo "PROBES_PER_ROLE=$RD_PROBES_PER_ROLE"
echo "GENERATE=$G"
echo "EXECUTE=$E"
echo "SCORE=$S"
echo "REPORT=$R"
