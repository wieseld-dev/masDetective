#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY="$HOME/.conda/envs/masdetective/bin/python"
for f in analysis/role_direct_within_v4/v42_{sensor,probes,matched}.py; do "$PY" -m py_compile "$f"; done
BASE="$PWD/artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154"
[[ -f "$BASE/manifest.json" ]] || { echo "MISSING original V4 manifest"; exit 1; }
[[ -f "$BASE/checkpoints/generator_activation_sft/adapter_config.json" ]] || { echo "MISSING previous trained generator adapter"; exit 1; }
for f in artifacts/v137_presence_v1/dataset_v1/{train,dev}_v137.jsonl; do [[ -s "$f" ]] || { echo "MISSING $f";exit 1; };done
mkdir -p logs "$BASE/role_upgrade_v2"
RUN="$BASE/role_upgrade_v2/run_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN"
[[ "$RUN" == "$BASE"/role_upgrade_v2/run_* ]] || exit 1
A=$(sbatch --parsable --job-name=v42-sensor --export="ALL,V42_ROOT=$RUN" cluster/role_direct_within_v4/v42_cpu.sbatch)
B=$(sbatch --parsable --job-name=v42-probes --export="ALL,V42_ROOT=$RUN" cluster/role_direct_within_v4/v42_generator_4090.sbatch)
A="${A%%;*}";B="${B%%;*}"
C=$(sbatch --parsable --dependency="afterany:$A:$B" --job-name=v42-report --export="ALL,V42_ROOT=$RUN" cluster/role_direct_within_v4/v42_report.sbatch)
C="${C%%;*}"
printf 'sensor\t%s\nprobes\t%s\nreport\t%s\n' "$A" "$B" "$C" > "$RUN/jobs.tsv"
printf 'V42_ROOT=%s\nV42_SENSOR_JOB=%s\nV42_PROBES_JOB=%s\nV42_REPORT_JOB=%s\n' "$RUN" "$A" "$B" "$C"
echo 'NO API CALLS, NO CHANGES TO PREVIOUS CHECKPOINTS.'
echo "VIEW: cat '$RUN/v42_final_summary.json'"
