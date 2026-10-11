#!/bin/bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
PY="$HOME/.conda/envs/masdetective/bin/python"
ROOT="$PWD/artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154"
[[ -f "$ROOT/manifest.json" ]] || { echo "Missing original V4 run"; exit 1; }
[[ -f "$ROOT/checkpoints/generator_activation_sft/adapter_config.json" ]] || { echo "Generator warm-start missing"; exit 1; }
mkdir -p logs
"$PY" -u analysis/role_direct_within_v4/role_upgrade_v1.py prepare --root "$ROOT"
SEL=$(sbatch --parsable --kill-on-invalid-dep=yes --job-name=v4u-selector \
  --export="ALL,V4_ROOT=$ROOT" cluster/role_direct_within_v4/upgrade_selector.sbatch)
GEN=$(sbatch --parsable --kill-on-invalid-dep=yes --job-name=v4u-generator \
  --export="ALL,V4_ROOT=$ROOT" cluster/role_direct_within_v4/upgrade_generator_4090.sbatch)
REP=$(sbatch --parsable --dependency="afterany:${SEL%%;*}:${GEN%%;*}" --job-name=v4u-report \
  --export="ALL,V4_ROOT=$ROOT" cluster/role_direct_within_v4/upgrade_report.sbatch)
printf 'selector\t%s\ngenerator\t%s\nreport\t%s\n' "$SEL" "$GEN" "$REP" | tee "$ROOT/role_upgrade_v1/jobs.tsv"
echo 'No existing Sensor job modified.'
echo "Results: $ROOT/role_upgrade_v1/reports/final.json"
