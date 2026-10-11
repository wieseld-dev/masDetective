#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/mas_rl_detective"
G1=$(sbatch --parsable cluster/v11_2_generate_r1_4090.sbatch)
E1=$(sbatch --parsable --dependency=afterok:$G1 cluster/v11_2_eval_r1_cpu.sbatch)
O1=$(sbatch --parsable --dependency=afterok:$E1 cluster/v11_2_opt_r1_4090.sbatch)
G2=$(sbatch --parsable --dependency=afterok:$O1 cluster/v11_2_generate_r2_4090.sbatch)
E2=$(sbatch --parsable --dependency=afterok:$G2 cluster/v11_2_eval_r2_cpu.sbatch)
O2=$(sbatch --parsable --dependency=afterok:$E2 cluster/v11_2_opt_r2_4090.sbatch)
printf 'G1=%s\nE1=%s\nO1=%s\nG2=%s\nE2=%s\nO2=%s\n' "$G1" "$E1" "$O1" "$G2" "$E2" "$O2"
