#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
mkdir -p logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
"$HOME/.conda/envs/masdetective/bin/python" -m py_compile analysis/role_direct_within_v4/v43_elicitation.py
RUN="$PWD/artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/role_upgrade_v3/hmhy_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN"
JID=$(sbatch --parsable --export="ALL,V43_ROOT=$RUN" cluster/role_direct_within_v4/v43_hmhy_4090.sbatch)
JID="${JID%%;*}"
printf 'job\t%s\nrun\t%s\n' "$JID" "$RUN" | tee "$RUN/job.tsv"
echo "ESTIMATED GPU RUNTIME 10-35 MINUTES + QUEUE; no MAS/API calls; existing adapters never modified"
