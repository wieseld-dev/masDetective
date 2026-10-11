#!/usr/bin/env bash
set -euo pipefail
cd /home/wieseld/mas_rl_detective
RUN="${1:-$(ls -dt artifacts/role_direct_within_v4/sensor_v5111/rolewise_* 2>/dev/null | head -1 || true)}"
if [[ -z "$RUN" ]];then echo 'No V5111 run yet';exit 0;fi
echo "RUN=$RUN"
if [[ -f "$RUN/comparison.tsv" ]];then
  cat "$RUN/comparison.tsv"
else
  echo 'Evaluation not finished yet.'
fi
squeue -u "$USER" -o '%i %j %T %M' | grep -E 'JOBID|v5111' || true
ls -t logs/v5111-*.out 2>/dev/null | head -1 | xargs -r tail -n 12 || true
