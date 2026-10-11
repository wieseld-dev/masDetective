#!/bin/bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <V11_O2_JOBID> <V13_5_JOBID>"
  exit 2
fi
V11="$1"; JUDGE="$2"
mkdir -p logs
M=$(sbatch --parsable cluster/adaptive_manifest_cpu.sbatch)
G=$(sbatch --parsable --dependency=afterok:${V11}:${M} cluster/adaptive_generate_actions_4090.sbatch)
C=$(sbatch --parsable --dependency=afterok:${G} cluster/adaptive_collect_cpu.sbatch)
S=$(sbatch --parsable --dependency=afterok:${C}:${JUDGE} cluster/adaptive_score_v135_4090.sbatch)
E=$(sbatch --parsable --dependency=afterok:${S} cluster/adaptive_fit_eval_cpu.sbatch)
echo "ADAPT_MANIFEST=$M"
echo "ADAPT_GENERATE=$G"
echo "ADAPT_COLLECT=$C"
echo "ADAPT_SCORE=$S"
echo "ADAPT_EVAL=$E"
