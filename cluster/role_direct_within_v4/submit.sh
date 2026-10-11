#!/bin/bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
PY="$HOME/.conda/envs/masdetective/bin/python"
CODE=analysis/role_direct_within_v4/pipeline.py
[[ -f "$CODE" ]] || { echo 'Installer missing'; exit 1; }
mkdir -p artifacts/role_direct_within_v4/runs logs
ROOT="artifacts/role_direct_within_v4/runs/v4_$(date +%Y%m%d_%H%M%S)_$$"
# Preflight is synchronous and refuses to start if train/dev split, activation set, or adapters are invalid.
"$PY" -u "$CODE" prepare --root "$PWD/$ROOT"
JOBS=()
submit () {
  local name="$1" stage="$2" kind="$3" deps="$4" res jid
  local dep=()
  if [[ -n "$deps" ]]; then dep=(--dependency="afterok:$deps"); fi
  res=$(sbatch --parsable --kill-on-invalid-dep=yes "${dep[@]}" --job-name="$name" \
    --export="ALL,V4_ROOT=$PWD/$ROOT,V4_STAGE=$stage" "cluster/role_direct_within_v4/stage_${kind}.sbatch")
  jid="${res%%;*}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch: $res"; exit 1; }
  JOBS+=("$jid")
  printf '%s\t%s\n' "$jid" "$stage" >> "$ROOT/job_ids.tsv"
  echo "SUBMITTED $stage $jid" >&2
  printf '%s' "$jid"
}
cleanup() { if [[ -f "$ROOT/job_ids.tsv" ]]; then mapfile -t ids < <(cut -f1 "$ROOT/job_ids.tsv"); if [[ ${#ids[@]} -gt 0 ]]; then scancel "${ids[@]}" 2>/dev/null || true; fi; fi; }
trap cleanup ERR
SEL=$(submit v4-selector selector cpu '')
SEN=$(submit v4-sensor sensor gpu '')
GEN=$(submit v4-generator generator gpu '')
AUD=$(submit v4-gen-audit probe_audit gpu "$GEN")
# Report always runs after the stage jobs terminate; missing stages are recorded explicitly.
RES=$(sbatch --parsable --dependency="afterany:$SEL:$SEN:$AUD" \
  --job-name=v4-report --export="ALL,V4_ROOT=$PWD/$ROOT,V4_STAGE=report" \
  cluster/role_direct_within_v4/stage_cpu.sbatch)
FJ="${RES%%;*}"
[[ "$FJ" =~ ^[0-9]+$ ]]
JOBS+=("$FJ");printf '%s\treport\n' "$FJ" >> "$ROOT/job_ids.tsv"
trap - ERR
printf '\nV4_SUBMITTED ROOT=%s FINAL=%s\n' "$ROOT" "$FJ"
echo "LIVE: squeue -u $USER"
echo "RESULTS: $ROOT/reports/final.json"
