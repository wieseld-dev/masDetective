#!/bin/bash
set -euo pipefail
cd "${1:-$HOME/mas_rl_detective}"
BOOT_ID=$(sbatch --parsable cluster/bootstrap_v63_cpu.sbatch)
echo "BOOTSTRAP JOB: $BOOT_ID"
TRAIN_ID=$(sbatch --parsable --dependency=afterok:$BOOT_ID cluster/train_v63_fanout48.sbatch)
echo "TRAINING JOB:  $TRAIN_ID"
echo "The GPU RL job starts only if the estimator bootstrap + system-level CV quality gate succeeds."
