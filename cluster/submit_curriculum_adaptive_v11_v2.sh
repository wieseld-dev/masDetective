#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"

[[ "${CURR_CONFIRM:-NO}" == "YES" ]] || {
  echo "Set CURR_CONFIRM=YES to submit the expensive V2 curriculum." >&2
  exit 2
}

ROOT="${CURR_EXP_ROOT:-artifacts/adaptive_v11_curriculum_v2}"
BELIEF="${CURR_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"
V11="${CURR_V11:-artifacts/v11_policy}"
HASHFILE="$ROOT/v11_source_sha256.txt"
RET="$ROOT/v11_retention.jsonl"
SELECTOR="${CURR_SELECTOR_MODEL:-$ROOT/selector/utility_ridge.npz}"
BOOT_SOURCE="${CURR_BOOTSTRAP_SOURCE:-artifacts/adaptive_v11_curriculum_v1}"
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
mkdir -p "$ROOT" "$ROOT/checkpoints" "$ROOT/selector" logs

for p in "$BELIEF" "$HASHFILE" "$ROOT/manifest_report.json"; do
  test -s "$p" || { echo "missing prepared artifact $p" >&2; exit 2; }
done
[[ -f .env ]] || { echo "missing .env" >&2; exit 2; }

"$PY" scripts/curriculum_v2_preflight.py >/tmp/curriculum_v2_preflight.json
cat /tmp/curriculum_v2_preflight.json

EXCLUDE="${CURR_EXCLUDE_GPU_NODE:-cs-4090-09}"
GPUEX=(); [[ -n "$EXCLUDE" ]] && GPUEX=(--exclude="$EXCLUDE")
JOBS="$ROOT/jobs.tsv"; : > "$JOBS"

submit_gpu(){
  local dep="$1" exports="$2" script="$3"
  local d=(); [[ -n "$dep" ]] && d=(--dependency="afterok:$dep")
  sbatch --parsable "${d[@]}" "${GPUEX[@]}" --export="ALL,$exports" "$script"
}
submit_cpu(){
  local dep="$1" exports="$2" script="$3"
  local d=(); [[ -n "$dep" ]] && d=(--dependency="afterok:$dep")
  sbatch --parsable "${d[@]}" --export="ALL,$exports" "$script"
}

epsilon_for_stage(){
  case "$1" in
    1) echo 0.30 ;;
    2) echo 0.27 ;;
    3) echo 0.23 ;;
    4) echo 0.20 ;;
    5) echo 0.16 ;;
    6) echo 0.13 ;;
    7) echo 0.10 ;;
    8) echo 0.08 ;;
    9) echo 0.06 ;;
    *) echo 0.05 ;;
  esac
}

PREV=""
if [[ ! -s "$SELECTOR" && -d "$BOOT_SOURCE/stage_01" ]]; then
  BOOT=$(submit_cpu "" "CURR_SELECTOR_MODEL=$SELECTOR,CURR_BOOTSTRAP_SOURCE=$BOOT_SOURCE,CURR_BOOTSTRAP_MAX_STAGE=${CURR_BOOTSTRAP_MAX_STAGE:-2}" cluster/curriculum_v2_bootstrap_cpu.sbatch)
  echo -e "selector_bootstrap\t$BOOT" | tee -a "$JOBS"
  PREV="$BOOT"
else
  echo "Selector bootstrap skipped (existing selector or no V1 source)."
fi

RETJOB=$(submit_gpu "$PREV" "CURR_V11=$V11,CURR_RETENTION=$RET,CURR_RETENTION_VARIANTS=${CURR_RETENTION_VARIANTS:-3}" cluster/curriculum_retention_4090.sbatch)
echo -e "retention\t$RETJOB" | tee -a "$JOBS"
PREV="$RETJOB"
ADAPTER_IN="$V11"
OPT_IN=""
FINAL_ADAPTER=""

while IFS=$'\t' read -r STAGE HORIZON TRAINN DEVN CYCLES GROUP LR KL RETW MCW NAME; do
  printf -v SS '%02d' "$STAGE"
  EPS=$(epsilon_for_stage "$STAGE")
  echo "===== V2 STAGE $STAGE $NAME horizon=$HORIZON cycles=$CYCLES G=$GROUP eps=$EPS ====="
  LAST_UPDATE_SUMMARY=""

  for CYCLE in $(seq 1 "$CYCLES"); do
    printf -v CC '%02d' "$CYCLE"
    CROOT="$ROOT/stage_${SS}/cycle_${CC}"
    MAN="$ROOT/stage_${SS}/manifest_train_cycle_${CC}.json"
    OUTBASE="$ROOT/checkpoints/stage_${SS}_cycle_${CC}"
    OUTAD="$OUTBASE/policy"
    mkdir -p "$CROOT" "$OUTBASE"

    I=$(submit_cpu "$PREV" "CURR_MANIFEST=$MAN,CURR_BELIEF=$BELIEF,CURR_ROOT=$CROOT,CURR_SPLIT=train" cluster/curriculum_init_cpu.sbatch)
    echo -e "s${SS}c${CC}_init\t$I" | tee -a "$JOBS"
    P="$I"

    for R in $(seq 1 "$HORIZON"); do
      G=$(submit_gpu "$P" "CURR_ROOT=$CROOT,CURR_MANIFEST=$MAN,CURR_ADAPTER=$ADAPTER_IN,CURR_V11=$V11,CURR_ROUND=$R,CURR_SPLIT=train,CURR_GROUP_SIZE=$GROUP,CURR_LOCAL_RETRIES=${CURR_LOCAL_RETRIES:-12},CURR_SELECTOR_MODEL=$SELECTOR,CURR_SELECTOR_EPS=$EPS,CURR_SELECTOR_MIN_SAMPLES=${CURR_SELECTOR_MIN_SAMPLES:-96},CURR_STAGE=$STAGE" cluster/curriculum_v2_generate_4090.sbatch)
      X=$(submit_cpu "$G" "CURR_ROOT=$CROOT,CURR_MANIFEST=$MAN,CURR_ROUND=$R,CURR_SPLIT=train,CURR_WORKERS=${CURR_WORKERS:-6}" cluster/curriculum_execute_cpu.sbatch)
      S=$(submit_gpu "$X" "CURR_ROOT=$CROOT,CURR_ROUND=$R,CURR_SPLIT=train" cluster/curriculum_score_4090.sbatch)
      A=$(submit_cpu "$S" "CURR_ROOT=$CROOT,CURR_MANIFEST=$MAN,CURR_BELIEF=$BELIEF,CURR_ROUND=$R,CURR_SPLIT=train,CURR_SELECTOR_MODEL=$SELECTOR,CURR_STAGE=$STAGE" cluster/curriculum_v2_advance_cpu.sbatch)
      echo -e "s${SS}c${CC}r${R}_gen\t$G\ns${SS}c${CC}r${R}_exec\t$X\ns${SS}c${CC}r${R}_score\t$S\ns${SS}c${CC}r${R}_adv\t$A" | tee -a "$JOBS"
      P="$A"
    done

    OPEX=""
    [[ -n "$OPT_IN" ]] && OPEX=",CURR_OPTIMIZER_IN=$OPT_IN"
    U=$(submit_gpu "$P" "CURR_ROOT=$CROOT,CURR_HORIZON=$HORIZON,CURR_INPUT_ADAPTER=$ADAPTER_IN,CURR_OUTPUT_ADAPTER=$OUTAD,CURR_V11=$V11,CURR_V11_HASH_FILE=$HASHFILE,CURR_RETENTION=$RET,CURR_LR=$LR,CURR_KL_BETA=$KL,CURR_RETENTION_WEIGHT=$RETW,CURR_MC_WEIGHT=$MCW,CURR_EPOCHS=${CURR_EPOCHS:-1},CURR_GRAD_ACCUM=${CURR_GRAD_ACCUM:-8}$OPEX" cluster/curriculum_update_4090.sbatch)
    echo -e "s${SS}c${CC}_update\t$U" | tee -a "$JOBS"
    ADAPTER_IN="$OUTAD"
    OPT_IN="$OUTBASE/optimizer.pt"
    PREV="$U"
    LAST_UPDATE_SUMMARY="$OUTBASE/update_summary.json"
    FINAL_ADAPTER="$OUTAD"
  done

  # DEV: create a local candidate pool, let the learned selector choose ONE, execute only that one.
  DROOT="$ROOT/stage_${SS}/dev_eval"
  DMAN="$ROOT/stage_${SS}/manifest_dev.json"
  mkdir -p "$DROOT"
  DI=$(submit_cpu "$PREV" "CURR_MANIFEST=$DMAN,CURR_BELIEF=$BELIEF,CURR_ROOT=$DROOT,CURR_SPLIT=dev" cluster/curriculum_init_cpu.sbatch)
  DP="$DI"
  DEVPOOL="${CURR_DEV_POOL_SIZE:-8}"
  for R in $(seq 1 "$HORIZON"); do
    DG=$(submit_gpu "$DP" "CURR_ROOT=$DROOT,CURR_MANIFEST=$DMAN,CURR_ADAPTER=$FINAL_ADAPTER,CURR_V11=$V11,CURR_ROUND=$R,CURR_SPLIT=dev,CURR_GROUP_SIZE=$DEVPOOL,CURR_LOCAL_RETRIES=${CURR_LOCAL_RETRIES:-12},CURR_SELECTOR_MODEL=$SELECTOR,CURR_SELECTOR_EPS=0.0,CURR_SELECTOR_MIN_SAMPLES=${CURR_SELECTOR_MIN_SAMPLES:-96},CURR_STAGE=$STAGE" cluster/curriculum_v2_generate_4090.sbatch)
    DX=$(submit_cpu "$DG" "CURR_ROOT=$DROOT,CURR_MANIFEST=$DMAN,CURR_ROUND=$R,CURR_SPLIT=dev,CURR_WORKERS=${CURR_WORKERS:-6}" cluster/curriculum_execute_cpu.sbatch)
    DS=$(submit_gpu "$DX" "CURR_ROOT=$DROOT,CURR_ROUND=$R,CURR_SPLIT=dev" cluster/curriculum_score_4090.sbatch)
    DA=$(submit_cpu "$DS" "CURR_ROOT=$DROOT,CURR_MANIFEST=$DMAN,CURR_BELIEF=$BELIEF,CURR_ROUND=$R,CURR_SPLIT=dev,CURR_SELECTOR_MODEL=$SELECTOR,CURR_STAGE=$STAGE" cluster/curriculum_v2_advance_cpu.sbatch)
    DP="$DA"
  done

  EOUT="$ROOT/stage_${SS}/dev_report.json"
  ER=$(submit_cpu "$DP" "CURR_ROOT=$DROOT,CURR_HORIZON=$HORIZON,CURR_EVAL_OUTPUT=$EOUT" cluster/curriculum_eval_report_cpu.sbatch)
  GT=$(submit_cpu "$ER" "CURR_UPDATE_SUMMARY=$LAST_UPDATE_SUMMARY,CURR_EVAL_REPORT=$EOUT,CURR_STRICT_GATE=${CURR_STRICT_GATE:-1}" cluster/curriculum_gate_cpu.sbatch)
  echo -e "s${SS}_dev_report\t$ER\ns${SS}_gate\t$GT" | tee -a "$JOBS"
  PREV="$GT"
done < <("$PY" scripts/curriculum_v2_plan.py)

F=$(submit_cpu "$PREV" "CURR_EXP_ROOT=$ROOT,CURR_FINAL_SOURCE=$FINAL_ADAPTER,CURR_V11=$V11,CURR_V11_HASH_FILE=$HASHFILE" cluster/curriculum_finalize_cpu.sbatch)
echo -e "finalize\t$F" | tee -a "$JOBS"
echo "$F" > "$ROOT/final_job.txt"

echo
echo "SUBMITTED ADAPTIVE-V11 V2 SEQUENTIAL UTILITY CURRICULUM"
echo "ROOT=$ROOT"
echo "SELECTOR=$SELECTOR"
echo "FINALIZE_JOB=$F"
echo "Untouched final test remains unscheduled."
