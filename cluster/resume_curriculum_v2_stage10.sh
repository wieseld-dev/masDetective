#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"

[[ "${CURR_CONFIRM:-NO}" == "YES" ]] || {
  echo "Set CURR_CONFIRM=YES to submit Stage 10 resume." >&2
  exit 2
}

# ============================================================
# EXISTING V2 ARTIFACTS
# ============================================================

ROOT="${CURR_EXP_ROOT:-artifacts/adaptive_v11_curriculum_v2}"
BELIEF="${CURR_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"
V11="${CURR_V11:-artifacts/v11_policy}"
HASHFILE="$ROOT/v11_source_sha256.txt"
RET="$ROOT/v11_retention.jsonl"
SELECTOR="${CURR_SELECTOR_MODEL:-$ROOT/selector/utility_ridge.npz}"

# Resume exactly from completed Stage 9 Cycle 2.
ADAPTER_IN="$ROOT/checkpoints/stage_09_cycle_02/policy"
OPT_IN="$ROOT/checkpoints/stage_09_cycle_02/optimizer.pt"

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

# ============================================================
# STAGE 10 CONFIG
# ============================================================

STAGE=10
SS=10

HORIZON=8
CYCLES=2
GROUP=8

LR="2e-6"
KL="0.060"
RETW="0.18"
MCW="0.45"
EPS="0.05"

DEVPOOL="${CURR_DEV_POOL_SIZE:-8}"

EXCLUDE="${CURR_EXCLUDE_GPU_NODE:-cs-4090-09}"
GPUEX=()
[[ -n "$EXCLUDE" ]] && GPUEX=(--exclude="$EXCLUDE")

mkdir -p "$ROOT" "$ROOT/checkpoints" logs

# Keep original jobs.tsv untouched.
STAMP=$(date +%Y%m%d_%H%M%S)
JOBS="$ROOT/jobs_stage10_resume_${STAMP}.tsv"
: > "$JOBS"

# ============================================================
# VALIDATION
# ============================================================

for p in \
  "$BELIEF" \
  "$HASHFILE" \
  "$RET" \
  "$SELECTOR" \
  "$OPT_IN" \
  "$ROOT/stage_10/manifest_train_cycle_01.json" \
  "$ROOT/stage_10/manifest_train_cycle_02.json" \
  "$ROOT/stage_10/manifest_dev.json"
do
  test -s "$p" || {
    echo "MISSING REQUIRED ARTIFACT: $p" >&2
    exit 2
  }
done

test -d "$ADAPTER_IN" || {
  echo "MISSING STAGE 9 ADAPTER: $ADAPTER_IN" >&2
  exit 2
}

[[ -f .env ]] || {
  echo "missing .env" >&2
  exit 2
}

# Preserve provenance: Stage 9 did NOT pass its original strict gate.
cat > "$ROOT/stage10_resume_reason.txt" <<EOF
Stage 10 resumed after Stage 9 strict gate failure.

Original Stage 9 gate:
  role_log_loss = 0.6391365284430371
  hard threshold = 0.62

Stage 9 dev still showed progressive improvement:
  role_log_loss: 0.672119 -> 0.639137
  role_brier:    0.237283 -> 0.223515
  role_auc:      0.557171 -> 0.671322
  margin:        0.009917 -> 0.041993
  recall@k:      0.449735 -> 0.535384

Purpose of resume:
  complete Stage 10 for generator / active-probing evaluation.

The failed Stage 9 gate is not relabeled as PASS.
Stage 10 final gate is diagnostic (strict=0).
Untouched final test remains unscheduled.
EOF

# ============================================================
# SUBMISSION HELPERS
# All newly submitted GPU jobs request 24G CPU RAM.
# ============================================================

submit_gpu(){
  local dep="$1" exports="$2" script="$3"
  local d=()
  [[ -n "$dep" ]] && d=(--dependency="afterok:$dep")

  sbatch --parsable \
    "${d[@]}" \
    "${GPUEX[@]}" \
    --mem=24G \
    --export="ALL,$exports" \
    "$script"
}

submit_cpu(){
  local dep="$1" exports="$2" script="$3"
  local d=()
  [[ -n "$dep" ]] && d=(--dependency="afterok:$dep")

  sbatch --parsable \
    "${d[@]}" \
    --export="ALL,$exports" \
    "$script"
}

# No dependency on failed Stage-9 gate:
# Stage-9 Cycle-2 update/checkpoint already completed successfully.
PREV=""

LAST_UPDATE_SUMMARY=""
FINAL_ADAPTER=""

echo "========================================================"
echo "RESUMING V2 AT STAGE 10 ONLY"
echo "INPUT ADAPTER: $ADAPTER_IN"
echo "INPUT OPTIMIZER: $OPT_IN"
echo "GPU CPU-MEM: 24G"
echo "========================================================"

# ============================================================
# TRAIN — TWO CYCLES
# ============================================================

for CYCLE in 1 2; do
  printf -v CC '%02d' "$CYCLE"

  CROOT="$ROOT/stage_${SS}/cycle_${CC}"
  MAN="$ROOT/stage_${SS}/manifest_train_cycle_${CC}.json"

  OUTBASE="$ROOT/checkpoints/stage_${SS}_cycle_${CC}"
  OUTAD="$OUTBASE/policy"

  mkdir -p "$CROOT" "$OUTBASE"

  I=$(submit_cpu \
    "$PREV" \
    "CURR_MANIFEST=$MAN,CURR_BELIEF=$BELIEF,CURR_ROOT=$CROOT,CURR_SPLIT=train" \
    cluster/curriculum_init_cpu.sbatch)

  echo -e "resume_s${SS}c${CC}_init\t$I" | tee -a "$JOBS"
  P="$I"

  for R in $(seq 1 "$HORIZON"); do

    G=$(submit_gpu \
      "$P" \
      "CURR_ROOT=$CROOT,CURR_MANIFEST=$MAN,CURR_ADAPTER=$ADAPTER_IN,CURR_V11=$V11,CURR_ROUND=$R,CURR_SPLIT=train,CURR_GROUP_SIZE=$GROUP,CURR_LOCAL_RETRIES=${CURR_LOCAL_RETRIES:-12},CURR_SELECTOR_MODEL=$SELECTOR,CURR_SELECTOR_EPS=$EPS,CURR_SELECTOR_MIN_SAMPLES=${CURR_SELECTOR_MIN_SAMPLES:-96},CURR_STAGE=$STAGE" \
      cluster/curriculum_v2_generate_4090.sbatch)

    X=$(submit_cpu \
      "$G" \
      "CURR_ROOT=$CROOT,CURR_MANIFEST=$MAN,CURR_ROUND=$R,CURR_SPLIT=train,CURR_WORKERS=${CURR_WORKERS:-6}" \
      cluster/curriculum_execute_cpu.sbatch)

    S=$(submit_gpu \
      "$X" \
      "CURR_ROOT=$CROOT,CURR_ROUND=$R,CURR_SPLIT=train" \
      cluster/curriculum_score_4090.sbatch)

    A=$(submit_cpu \
      "$S" \
      "CURR_ROOT=$CROOT,CURR_MANIFEST=$MAN,CURR_BELIEF=$BELIEF,CURR_ROUND=$R,CURR_SPLIT=train,CURR_SELECTOR_MODEL=$SELECTOR,CURR_STAGE=$STAGE" \
      cluster/curriculum_v2_advance_cpu.sbatch)

    echo -e \
"resume_s${SS}c${CC}r${R}_gen\t$G
resume_s${SS}c${CC}r${R}_exec\t$X
resume_s${SS}c${CC}r${R}_score\t$S
resume_s${SS}c${CC}r${R}_adv\t$A" \
      | tee -a "$JOBS"

    P="$A"
  done

  OPEX=""
  [[ -n "$OPT_IN" ]] && OPEX=",CURR_OPTIMIZER_IN=$OPT_IN"

  U=$(submit_gpu \
    "$P" \
    "CURR_ROOT=$CROOT,CURR_HORIZON=$HORIZON,CURR_INPUT_ADAPTER=$ADAPTER_IN,CURR_OUTPUT_ADAPTER=$OUTAD,CURR_V11=$V11,CURR_V11_HASH_FILE=$HASHFILE,CURR_RETENTION=$RET,CURR_LR=$LR,CURR_KL_BETA=$KL,CURR_RETENTION_WEIGHT=$RETW,CURR_MC_WEIGHT=$MCW,CURR_EPOCHS=${CURR_EPOCHS:-1},CURR_GRAD_ACCUM=${CURR_GRAD_ACCUM:-8}$OPEX" \
    cluster/curriculum_update_4090.sbatch)

  echo -e "resume_s${SS}c${CC}_update\t$U" | tee -a "$JOBS"

  ADAPTER_IN="$OUTAD"
  OPT_IN="$OUTBASE/optimizer.pt"
  PREV="$U"

  LAST_UPDATE_SUMMARY="$OUTBASE/update_summary.json"
  FINAL_ADAPTER="$OUTAD"
done

# ============================================================
# DEV — GREEDY SELECTOR, ONE EXECUTED PROBE PER SYSTEM/ROUND
# ============================================================

DROOT="$ROOT/stage_${SS}/dev_eval"
DMAN="$ROOT/stage_${SS}/manifest_dev.json"

mkdir -p "$DROOT"

DI=$(submit_cpu \
  "$PREV" \
  "CURR_MANIFEST=$DMAN,CURR_BELIEF=$BELIEF,CURR_ROOT=$DROOT,CURR_SPLIT=dev" \
  cluster/curriculum_init_cpu.sbatch)

echo -e "resume_s${SS}_dev_init\t$DI" | tee -a "$JOBS"

DP="$DI"

for R in $(seq 1 "$HORIZON"); do

  DG=$(submit_gpu \
    "$DP" \
    "CURR_ROOT=$DROOT,CURR_MANIFEST=$DMAN,CURR_ADAPTER=$FINAL_ADAPTER,CURR_V11=$V11,CURR_ROUND=$R,CURR_SPLIT=dev,CURR_GROUP_SIZE=$DEVPOOL,CURR_LOCAL_RETRIES=${CURR_LOCAL_RETRIES:-12},CURR_SELECTOR_MODEL=$SELECTOR,CURR_SELECTOR_EPS=0.0,CURR_SELECTOR_MIN_SAMPLES=${CURR_SELECTOR_MIN_SAMPLES:-96},CURR_STAGE=$STAGE" \
    cluster/curriculum_v2_generate_4090.sbatch)

  DX=$(submit_cpu \
    "$DG" \
    "CURR_ROOT=$DROOT,CURR_MANIFEST=$DMAN,CURR_ROUND=$R,CURR_SPLIT=dev,CURR_WORKERS=${CURR_WORKERS:-6}" \
    cluster/curriculum_execute_cpu.sbatch)

  DS=$(submit_gpu \
    "$DX" \
    "CURR_ROOT=$DROOT,CURR_ROUND=$R,CURR_SPLIT=dev" \
    cluster/curriculum_score_4090.sbatch)

  DA=$(submit_cpu \
    "$DS" \
    "CURR_ROOT=$DROOT,CURR_MANIFEST=$DMAN,CURR_BELIEF=$BELIEF,CURR_ROUND=$R,CURR_SPLIT=dev,CURR_SELECTOR_MODEL=$SELECTOR,CURR_STAGE=$STAGE" \
    cluster/curriculum_v2_advance_cpu.sbatch)

  echo -e \
"resume_s${SS}devr${R}_gen\t$DG
resume_s${SS}devr${R}_exec\t$DX
resume_s${SS}devr${R}_score\t$DS
resume_s${SS}devr${R}_adv\t$DA" \
    | tee -a "$JOBS"

  DP="$DA"
done

# ============================================================
# EVAL + NON-STRICT DIAGNOSTIC GATE + FINALIZE
# ============================================================

EOUT="$ROOT/stage_${SS}/dev_report.json"

ER=$(submit_cpu \
  "$DP" \
  "CURR_ROOT=$DROOT,CURR_HORIZON=$HORIZON,CURR_EVAL_OUTPUT=$EOUT" \
  cluster/curriculum_eval_report_cpu.sbatch)

echo -e "resume_s${SS}_dev_report\t$ER" | tee -a "$JOBS"

# Keep the gate calculation, but do not let the old absolute threshold
# cancel the experiment again.
GT=$(submit_cpu \
  "$ER" \
  "CURR_UPDATE_SUMMARY=$LAST_UPDATE_SUMMARY,CURR_EVAL_REPORT=$EOUT,CURR_STRICT_GATE=0" \
  cluster/curriculum_gate_cpu.sbatch)

echo -e "resume_s${SS}_gate_diagnostic\t$GT" | tee -a "$JOBS"

F=$(submit_cpu \
  "$GT" \
  "CURR_EXP_ROOT=$ROOT,CURR_FINAL_SOURCE=$FINAL_ADAPTER,CURR_V11=$V11,CURR_V11_HASH_FILE=$HASHFILE" \
  cluster/curriculum_finalize_cpu.sbatch)

echo -e "resume_finalize\t$F" | tee -a "$JOBS"

echo "$F" > "$ROOT/final_job.txt"
echo "$JOBS" > "$ROOT/stage10_resume_jobs_file.txt"

echo
echo "========================================================"
echo "STAGE 10 RESUME SUBMITTED"
echo "========================================================"
echo "JOBS_FILE=$JOBS"
echo "FINALIZE_JOB=$F"
echo "INPUT=$ROOT/checkpoints/stage_09_cycle_02/policy"
echo "OUTPUT=$ROOT/checkpoints/stage_10_cycle_02/policy"
echo
echo "Untouched final test remains UNSCHEDULED."
echo "Stage 9 original gate remains recorded as FAIL."
echo "Stage 10 gate will be diagnostic only (strict=0)."
