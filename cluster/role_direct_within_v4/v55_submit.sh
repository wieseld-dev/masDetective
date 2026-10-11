#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
SOURCE="${1:-$(ls -dt artifacts/role_direct_within_v4/pilot_v54/r2paired_* 2>/dev/null | head -1)}"
if [[ -z "$SOURCE" || ! -f "$SOURCE/responses_train_round_02.jsonl" ]]; then
  echo "Could not identify V5.4 source. Supply its directory: bash cluster/role_direct_within_v4/v55_submit.sh PATH" >&2
  exit 2
fi
SOURCE="$(realpath "$SOURCE")"
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -u -m analysis.role_direct_within_v4.v55_sensor_r2_diagnostic check --root "$SOURCE"
"$PY" -u -m analysis.role_direct_within_v4.v55_sensor_r2_diagnostic test --root "$SOURCE"
ADAPTER="${V55_SENSOR_ADAPTER:-$PWD/artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/checkpoints/sensor_epoch02}"
if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  echo "Missing frozen Sensor checkpoint: $ADAPTER" >&2
  echo 'Set V55_SENSOR_ADAPTER=/path/to/checkpoint if needed.' >&2
  exit 2
fi
mkdir -p artifacts/role_direct_within_v4/sensor_v55 logs
OUT="$PWD/artifacts/role_direct_within_v4/sensor_v55/sensor_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$OUT"
printf '%s\n' "$SOURCE" > "$OUT/source_root.txt"
ID=$(sbatch --parsable --export="ALL,V55_SOURCE=$SOURCE,V55_OUT=$OUT,V55_SENSOR_ADAPTER=$ADAPTER" cluster/role_direct_within_v4/v55_sensor_4090.sbatch)
printf '%s\n' "$ID" > "$OUT/job_id.txt"
printf 'V55_SOURCE=%s\nV55_OUT=%s\nJOB_ID=%s\n' "$SOURCE" "$OUT" "$ID"
printf 'CHECK: sacct -j %s -X --format=JobID,JobName%%25,State,ExitCode,Elapsed\n' "$ID"
printf 'LOG: tail -n 30 logs/v55-mas5-v55-sensor-%s.out\n' "$ID"
