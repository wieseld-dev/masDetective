#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
: "${V46_ROOT:?Export V46_ROOT as the existing overnight_v46/mas5_night run}"
export V46_ROOT
V46_ROOT=$(realpath -e "$V46_ROOT")
[[ "$V46_ROOT" == "$PWD"/artifacts/role_direct_within_v4/overnight_v46/mas5_night_* ]] || { echo 'Invalid RUN'; exit 2; }
if squeue -h -u "$USER" -o '%j' | grep -q '^mas5-'; then
  echo 'Another MAS5 job already queued/running. Will not launch two pipelines.' >&2
  exit 2
fi
PY="$HOME/.conda/envs/masdetective/bin/python"
"$PY" -m py_compile analysis/role_direct_within_v4/v473_generator.py analysis/role_direct_within_v4/v473_regenerate.py
mkdir -p logs
JOB=$(sbatch --parsable --export="ALL,V46_ROOT=$V46_ROOT,V473_ATTEMPTS=${V473_ATTEMPTS:-36}" \
    cluster/role_direct_within_v4/v473_regenerate_4090.sbatch)
echo "MAS5_V473_JOB=$JOB"
echo "Status: sacct -j $JOB -X --format=JobID,State,ExitCode,Elapsed"
echo "Log: tail -n 50 logs/v473-mas5-v473-regen-$JOB.out"
echo 'Generates only; no MAS/API calls. Estimated duration: 20 min to a few hours, plus Slurm queue.'
