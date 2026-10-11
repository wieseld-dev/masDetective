#!/bin/bash
# Cheap preparation only: preflight -> fit frozen belief -> fresh manifest -> leak-safe initial states.
# If the fixed-probe dual-score job is still running, pass its job id as $1.
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/adaptive_v11_active_v1
BASE_DEP="${1:-}"
P=$(sbatch --parsable cluster/adaptive_v11_preflight_cpu.sbatch)
if [ -n "$BASE_DEP" ]; then
  B=$(sbatch --parsable --dependency=afterok:${P}:${BASE_DEP} cluster/adaptive_v11_fit_belief_cpu.sbatch)
else
  B=$(sbatch --parsable --dependency=afterok:${P} cluster/adaptive_v11_fit_belief_cpu.sbatch)
fi
M=$(sbatch --parsable --dependency=afterok:${B} cluster/adaptive_v11_manifest_cpu.sbatch)
I=$(sbatch --parsable --dependency=afterok:${M} cluster/adaptive_v11_init_cpu.sbatch)
P2=$(sbatch --parsable --dependency=afterok:${I} --export=ALL,AV11_REQUIRE_BELIEF=1 cluster/adaptive_v11_preflight_cpu.sbatch)
printf 'PREFLIGHT=%s\nBELIEF=%s\nMANIFEST=%s\nINIT=%s\nFINAL_PREFLIGHT=%s\n' "$P" "$B" "$M" "$I" "$P2"
cat > artifacts/adaptive_v11_active_v1/prepare_jobs.txt <<EOF
PREFLIGHT=$P
BELIEF=$B
MANIFEST=$M
INIT=$I
FINAL_PREFLIGHT=$P2
EOF
