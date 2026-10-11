#!/bin/bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
EXP="${EIG_EXPERIMENT_ROOT:-artifacts/adaptive_v11_query_eig_v2}"
BELIEF="${EIG_BELIEF:-artifacts/adaptive_v11_active_v1/frozen_belief.joblib}"
UTIL="$EXP/query_utility.joblib"
[ "${EIG_CONFIRM:-NO}" = YES ] || { echo 'REFUSING: export EIG_CONFIRM=YES'; exit 2; }
[ -s "$UTIL" ] || { echo 'run prepare_query_eig_v2.sh first'; exit 1; }
set -a; [ -f .env ] && source .env; set +a; [ -n "${OPENAI_API_KEY:-}" ] || { echo OPENAI_API_KEY_missing; exit 1; }
mkdir -p logs "$EXP"; JOBS="$EXP/jobs.txt"; :>"$JOBS"
G1=$(sbatch --parsable --exclude=cs-4090-09 --export=ALL,EIG_ROOT="$EXP/common",EIG_ROUND=1,EIG_MODE=common,EIG_UTILITY="$UTIL" cluster/eig_generate_4090.sbatch)
X1=$(sbatch --parsable --dependency=afterok:$G1 --export=ALL,EIG_ROOT="$EXP/common",EIG_ROUND=1 cluster/eig_execute_cpu.sbatch)
S1=$(sbatch --parsable --dependency=afterok:$X1 --exclude=cs-4090-09 --export=ALL,EIG_ROOT="$EXP/common",EIG_ROUND=1 cluster/eig_score_4090.sbatch)
A1=$(sbatch --parsable --dependency=afterok:$S1 --export=ALL,EIG_ROOT="$EXP/common",EIG_ROUND=1,EIG_BELIEF="$BELIEF" cluster/eig_advance_cpu.sbatch)
SEED=$(sbatch --parsable --dependency=afterok:$A1 --export=ALL,EIG_EXP="$EXP" cluster/eig_seed_cpu.sbatch)
echo "COMMON R1 GEN=$G1 EXEC=$X1 SCORE=$S1 ADV=$A1 SEED=$SEED" | tee -a "$JOBS"
submit_arm(){ mode=$1; root="$EXP/$mode"; prev=$SEED; for r in 2 3 4 5 6; do G=$(sbatch --parsable --dependency=afterok:$prev --exclude=cs-4090-09 --export=ALL,EIG_ROOT="$root",EIG_ROUND=$r,EIG_MODE=$mode,EIG_UTILITY="$UTIL",EIG_POOL="${EIG_POOL:-8}" cluster/eig_generate_4090.sbatch); X=$(sbatch --parsable --dependency=afterok:$G --export=ALL,EIG_ROOT="$root",EIG_ROUND=$r cluster/eig_execute_cpu.sbatch); S=$(sbatch --parsable --dependency=afterok:$X --exclude=cs-4090-09 --export=ALL,EIG_ROOT="$root",EIG_ROUND=$r cluster/eig_score_4090.sbatch); A=$(sbatch --parsable --dependency=afterok:$S --export=ALL,EIG_ROOT="$root",EIG_ROUND=$r,EIG_BELIEF="$BELIEF" cluster/eig_advance_cpu.sbatch); echo "${mode^^} R$r GEN=$G EXEC=$X SCORE=$S ADV=$A" | tee -a "$JOBS" >&2; prev=$A; done; echo $prev; }
SL=$(submit_arm static); ML=$(submit_arm maxent); UL=$(submit_arm utility)
R=$(sbatch --parsable --dependency=afterok:$SL:$ML:$UL --export=ALL,EIG_EXP="$EXP" cluster/eig_report_cpu.sbatch)
echo "STATIC_LAST=$SL"|tee -a "$JOBS"; echo "MAXENT_LAST=$ML"|tee -a "$JOBS"; echo "UTILITY_LAST=$UL"|tee -a "$JOBS"; echo "REPORT=$R"|tee -a "$JOBS"; echo "SUBMITTED. REPORT_JOB=$R"
