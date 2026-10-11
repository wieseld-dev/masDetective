#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/mas_rl_detective}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p scripts cluster logs artifacts
cp -f "$HERE"/scripts/diagnose_v10_final_elites.py "$ROOT/scripts/"
cp -f "$HERE"/cluster/v10_final_elites_diag_cpu.sbatch "$ROOT/cluster/"
chmod +x "$ROOT/cluster/v10_final_elites_diag_cpu.sbatch"
echo "Applied V10 post-hoc elite diagnostic patch to $ROOT"
echo "No training code. Uses the already-consumed final systems only for diagnostics."
