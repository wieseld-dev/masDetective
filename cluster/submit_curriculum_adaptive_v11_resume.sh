#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
[[ "${CURR_CONFIRM:-NO}" == "YES" ]] || { echo "Set CURR_CONFIRM=YES to submit the expensive curriculum." >&2; exit 2; }
ROOT="${CURR_EXP_ROOT:-artifacts/adaptive_v11_curriculum_v1}"
BELIEF="${CURR_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"
V11="${CURR_V11:-artifacts/v11_policy}"
HASHFILE="$ROOT/v11_source_sha256.txt"
RET="$ROOT/v11_retention.jsonl"
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
mkdir -p "$ROOT" "$ROOT/checkpoints" logs
for p in "$BELIEF" "$HASHFILE" "$ROOT/manifest_report.json"; do test -s "$p" || { echo "missing prepared artifact $p" >&2; exit 2; }; done
[[ -f .env ]] || { echo "missing .env" >&2; exit 2; }
EXCLUDE="${CURR_EXCLUDE_GPU_NODE:-cs-4090-09}"
GPUEX=(); [[ -n "$EXCLUDE" ]] && GPUEX=(--exclude="$EXCLUDE")
JOBS="$ROOT/jobs_resume.tsv"; : > "$JOBS"

submit_gpu(){ local dep="$1" exports="$2" script="$3"; local d=(); [[ -n "$dep" ]] && d=(--dependency="afterok:$dep"); sbatch --parsable "${d[@]}" "${GPUEX[@]}" --export="ALL,$exports" "$script"; }
submit_cpu(){ local dep="$1" exports="$2" script="$3"; local d=(); [[ -n "$dep" ]] && d=(--dependency="afterok:$dep"); sbatch --parsable "${d[@]}" --export="ALL,$exports" "$script"; }

# RESUME: retention already completed in original run.
RETJOB=""
PREV=""
ADAPTER_IN="$V11"
OPT_IN=""
FINAL_ADAPTER=""

while IFS=$'\t' read -r STAGE HORIZON TRAINN DEVN CYCLES GROUP LR KL RETW MCW NAME; do
  printf -v SS '%02d' "$STAGE"
  echo "===== PLAN STAGE $STAGE $NAME horizon=$HORIZON cycles=$CYCLES G=$GROUP ====="
  LAST_UPDATE_SUMMARY=""
  for CYCLE in $(seq 1 "$CYCLES"); do
    printf -v CC '%02d' "$CYCLE"
    CROOT="$ROOT/stage_${SS}/cycle_${CC}"
    MAN="$ROOT/stage_${SS}/manifest_train_cycle_${CC}.json"
    OUTBASE="$ROOT/checkpoints/stage_${SS}_cycle_${CC}"
    OUTAD="$OUTBASE/policy"
    mkdir -p "$CROOT" "$OUTBASE"

    # RESUME POINT:
    # Stage 1 / Cycle 1 rounds are complete.
    # Its update is already submitted as job 21462320.
    if [[ "$STAGE" == "1" && "$CYCLE" == "1" ]]; then
      PREV="21462320"
      ADAPTER_IN="$OUTAD"
      OPT_IN="$OUTBASE/optimizer.pt"
      LAST_UPDATE_SUMMARY="$OUTBASE/update_summary.json"
      FINAL_ADAPTER="$OUTAD"
      echo -e "s01c01_update_existing\t21462320" | tee -a "$JOBS"
      continue
    fi

    I=$(submit_cpu "$PREV" "CURR_MANIFEST=$MAN,CURR_BELIEF=$BELIEF,CURR_ROOT=$CROOT,CURR_SPLIT=train" cluster/curriculum_init_cpu.sbatch)
    echo -e "s${SS}c${CC}_init\t$I" | tee -a "$JOBS"
    P="$I"
    for R in $(seq 1 "$HORIZON"); do
      G=$(submit_gpu "$P" "CURR_ROOT=$CROOT,CURR_MANIFEST=$MAN,CURR_ADAPTER=$ADAPTER_IN,CURR_V11=$V11,CURR_ROUND=$R,CURR_SPLIT=train,CURR_GROUP_SIZE=$GROUP,CURR_LOCAL_RETRIES=${CURR_LOCAL_RETRIES:-12}" cluster/curriculum_generate_4090.sbatch)
      X=$(submit_cpu "$G" "CURR_ROOT=$CROOT,CURR_MANIFEST=$MAN,CURR_ROUND=$R,CURR_SPLIT=train,CURR_WORKERS=${CURR_WORKERS:-6}" cluster/curriculum_execute_cpu.sbatch)
      S=$(submit_gpu "$X" "CURR_ROOT=$CROOT,CURR_ROUND=$R,CURR_SPLIT=train" cluster/curriculum_score_4090.sbatch)
      A=$(submit_cpu "$S" "CURR_ROOT=$CROOT,CURR_MANIFEST=$MAN,CURR_BELIEF=$BELIEF,CURR_ROUND=$R,CURR_SPLIT=train" cluster/curriculum_advance_cpu.sbatch)
      echo -e "s${SS}c${CC}r${R}_gen\t$G\ns${SS}c${CC}r${R}_exec\t$X\ns${SS}c${CC}r${R}_score\t$S\ns${SS}c${CC}r${R}_adv\t$A" | tee -a "$JOBS"
      P="$A"
    done
    OPEX=""
    [[ -n "$OPT_IN" ]] && OPEX=",CURR_OPTIMIZER_IN=$OPT_IN"
    U=$(submit_gpu "$P" "CURR_ROOT=$CROOT,CURR_HORIZON=$HORIZON,CURR_INPUT_ADAPTER=$ADAPTER_IN,CURR_OUTPUT_ADAPTER=$OUTAD,CURR_V11=$V11,CURR_V11_HASH_FILE=$HASHFILE,CURR_RETENTION=$RET,CURR_LR=$LR,CURR_KL_BETA=$KL,CURR_RETENTION_WEIGHT=$RETW,CURR_MC_WEIGHT=$MCW,CURR_EPOCHS=${CURR_EPOCHS:-1},CURR_GRAD_ACCUM=${CURR_GRAD_ACCUM:-8}$OPEX" cluster/curriculum_update_4090.sbatch)
    echo -e "s${SS}c${CC}_update\t$U" | tee -a "$JOBS"
    ADAPTER_IN="$OUTAD"; OPT_IN="$OUTBASE/optimizer.pt"; PREV="$U"; LAST_UPDATE_SUMMARY="$OUTBASE/update_summary.json"; FINAL_ADAPTER="$OUTAD"
  done

  # Stage-held-out evaluation uses one real probe per system/round; no gradient updates.
  DROOT="$ROOT/stage_${SS}/dev_eval"; DMAN="$ROOT/stage_${SS}/manifest_dev.json"; mkdir -p "$DROOT"
  DI=$(submit_cpu "$PREV" "CURR_MANIFEST=$DMAN,CURR_BELIEF=$BELIEF,CURR_ROOT=$DROOT,CURR_SPLIT=dev" cluster/curriculum_init_cpu.sbatch)
  DP="$DI"
  for R in $(seq 1 "$HORIZON"); do
    DG=$(submit_gpu "$DP" "CURR_ROOT=$DROOT,CURR_MANIFEST=$DMAN,CURR_ADAPTER=$FINAL_ADAPTER,CURR_V11=$V11,CURR_ROUND=$R,CURR_SPLIT=dev,CURR_GROUP_SIZE=1,CURR_LOCAL_RETRIES=${CURR_LOCAL_RETRIES:-12}" cluster/curriculum_generate_4090.sbatch)
    DX=$(submit_cpu "$DG" "CURR_ROOT=$DROOT,CURR_MANIFEST=$DMAN,CURR_ROUND=$R,CURR_SPLIT=dev,CURR_WORKERS=${CURR_WORKERS:-6}" cluster/curriculum_execute_cpu.sbatch)
    DS=$(submit_gpu "$DX" "CURR_ROOT=$DROOT,CURR_ROUND=$R,CURR_SPLIT=dev" cluster/curriculum_score_4090.sbatch)
    DA=$(submit_cpu "$DS" "CURR_ROOT=$DROOT,CURR_MANIFEST=$DMAN,CURR_BELIEF=$BELIEF,CURR_ROUND=$R,CURR_SPLIT=dev" cluster/curriculum_advance_cpu.sbatch)
    DP="$DA"
  done
  EOUT="$ROOT/stage_${SS}/dev_report.json"
  ER=$(submit_cpu "$DP" "CURR_ROOT=$DROOT,CURR_HORIZON=$HORIZON,CURR_EVAL_OUTPUT=$EOUT" cluster/curriculum_eval_report_cpu.sbatch)
  GT=$(submit_cpu "$ER" "CURR_UPDATE_SUMMARY=$LAST_UPDATE_SUMMARY,CURR_EVAL_REPORT=$EOUT,CURR_STRICT_GATE=${CURR_STRICT_GATE:-1}" cluster/curriculum_gate_cpu.sbatch)
  echo -e "s${SS}_dev_report\t$ER\ns${SS}_gate\t$GT" | tee -a "$JOBS"
  PREV="$GT"
done < <("$PY" scripts/curriculum_plan.py)

F=$(submit_cpu "$PREV" "CURR_EXP_ROOT=$ROOT,CURR_FINAL_SOURCE=$FINAL_ADAPTER,CURR_V11=$V11,CURR_V11_HASH_FILE=$HASHFILE" cluster/curriculum_finalize_cpu.sbatch)
echo -e "finalize\t$F" | tee -a "$JOBS"
echo "$F" > "$ROOT/final_job.txt"
echo
echo "SUBMITTED STRONG CURRICULUM ADAPTIVE-V11 TRAINING"
echo "ROOT=$ROOT"
echo "FINALIZE_JOB=$F"
echo "Frozen original V11 remains at $V11 and is hash-guarded on every optimization cycle."
echo "Untouched final test is NOT scheduled: $ROOT/final_test_manifest_UNTOUCHED.json"
