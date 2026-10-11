#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"

[[ "${V3_CONFIRM:-NO}" == "YES" ]] || {
  echo "This pilot can make up to 384 MAS calls. Re-run with: V3_CONFIRM=YES bash cluster/submit_v3_coverage_elimination.sh" >&2
  exit 2
}

ROOT="${V3_ROOT:-artifacts/v3_coverage_elimination_pilot}"
MAN="$ROOT/manifest_privileged.json"
BASE="${V3_BASE_ADAPTER:-artifacts/robust_visible_generator_v1_epoch2/policy}"
V11="${V3_V11:-artifacts/v11_policy}"
H="${V3_HORIZON:-6}"
G="${V3_GROUP_SIZE:-4}"
WORKERS="${V3_WORKERS:-6}"

HASH="${V3_V11_HASH_FILE:-artifacts/adaptive_v11_curriculum_v2/v11_source_sha256.txt}"
RET="${V3_RETENTION:-artifacts/adaptive_v11_curriculum_v2/v11_retention.jsonl}"

mkdir -p "$ROOT" logs
for p in "$BASE" "$V11" "$HASH" "$RET" cluster/curriculum_update_4090.sbatch; do
  [[ -e "$p" ]] || { echo "Missing required path: $p" >&2; exit 2; }
done

JOBS="$ROOT/jobs.tsv"
: > "$JOBS"

PREP=$(sbatch --parsable \
  --export=ALL,V3_ROOT="$ROOT" \
  cluster/v3_prepare_cpu.sbatch)
PREP="${PREP%%;*}"
echo -e "prepare\t$PREP" | tee -a "$JOBS"

P="$PREP"
for R in $(seq 1 "$H"); do
  GEN=$(sbatch --parsable \
    --dependency="afterok:$P" \
    --export=ALL,V3_ROOT="$ROOT",V3_MANIFEST="$MAN",V3_ADAPTER="$BASE",V3_V11="$V11",V3_ROUND="$R",V3_SPLIT=train,V3_GROUP_SIZE="$G" \
    cluster/v3_generate_4090.sbatch)
  GEN="${GEN%%;*}"

  EXE=$(sbatch --parsable \
    --dependency="afterok:$GEN" \
    --export=ALL,V3_ROOT="$ROOT",V3_MANIFEST="$MAN",V3_ROUND="$R",V3_SPLIT=train,V3_WORKERS="$WORKERS" \
    cluster/v3_execute_cpu.sbatch)
  EXE="${EXE%%;*}"

  ADV=$(sbatch --parsable \
    --dependency="afterok:$EXE" \
    --export=ALL,V3_ROOT="$ROOT",V3_MANIFEST="$MAN",V3_ROUND="$R",V3_SPLIT=train \
    cluster/v3_advance_cpu.sbatch)
  ADV="${ADV%%;*}"

  echo -e "r${R}_gen\t$GEN\nr${R}_exec\t$EXE\nr${R}_advance\t$ADV" | tee -a "$JOBS"
  P="$ADV"
done

OUTAD="$ROOT/policy"

UPD=$(sbatch --parsable \
  --dependency="afterok:$P" \
  --export=ALL,CURR_ROOT="$ROOT",CURR_HORIZON="$H",CURR_INPUT_ADAPTER="$BASE",CURR_OUTPUT_ADAPTER="$OUTAD",CURR_V11="$V11",CURR_V11_HASH_FILE="$HASH",CURR_RETENTION="$RET",CURR_LR="${V3_LR:-2e-6}",CURR_KL_BETA="${V3_KL_BETA:-0.03}",CURR_RETENTION_WEIGHT="${V3_RETENTION_WEIGHT:-0.10}",CURR_MC_WEIGHT="${V3_MC_WEIGHT:-0.30}",CURR_EPOCHS="${V3_EPOCHS:-1}",CURR_GRAD_ACCUM="${V3_GRAD_ACCUM:-8}" \
  cluster/curriculum_update_4090.sbatch)
UPD="${UPD%%;*}"
echo -e "update\t$UPD" | tee -a "$JOBS"
echo "$UPD" > "$ROOT/update_job.txt"

echo
echo "V3 COVERAGE+ELIMINATION PILOT SUBMITTED"
echo "ROOT=$ROOT"
echo "TRAIN SYSTEMS=16"
echo "HORIZON=$H"
echo "GROUP=$G"
echo "MAX MAS CALLS=$((16 * H * G))"
echo "BASE ADAPTER=$BASE"
echo "UPDATE JOB=$UPD"
echo
echo "Reserved dev systems are created by PREP but are NOT touched by this training submitter."
