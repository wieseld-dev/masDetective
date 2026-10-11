#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
O2_JOB="${1:-21252447}"
JUDGE_TRAIN_JOB="${2:-21251252}"
G=$(sbatch --parsable --dependency=afterok:${O2_JOB} cluster/v11_2_strong_eval_generate_4090.sbatch)
X=$(sbatch --parsable --dependency=afterok:${G} cluster/v11_2_strong_eval_execute_cpu.sbatch)
J=$(sbatch --parsable --dependency=afterok:${X}:${JUDGE_TRAIN_JOB} cluster/v11_2_strong_eval_judge_4090.sbatch)
R=$(sbatch --parsable --dependency=afterok:${J} cluster/v11_2_strong_eval_report_cpu.sbatch)
echo "EVAL_GENERATE=$G"
echo "EVAL_EXECUTE=$X"
echo "EVAL_JUDGE=$J"
echo "EVAL_REPORT=$R"
