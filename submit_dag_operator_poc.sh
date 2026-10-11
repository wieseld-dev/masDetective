#!/bin/bash
set -euo pipefail
mkdir -p logs
DATAJOB=$(sbatch --parsable dag_operator_collect.sbatch)
EVALJOB=$(sbatch --parsable --dependency=afterok:${DATAJOB} dag_operator_eval.sbatch)

echo "=============================================="
echo "FIXED-DAG OPERATOR PILOT SUBMITTED"
echo "DATA JOB: ${DATAJOB}"
echo "EVAL JOB: ${EVALJOB}"
echo "=============================================="
echo "Watch data: tail -f logs/dagop-data-${DATAJOB}.out"
echo "Watch eval: tail -f logs/dagop-eval-${EVALJOB}.out"
