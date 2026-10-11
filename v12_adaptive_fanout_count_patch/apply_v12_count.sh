#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/mas_rl_detective}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p analysis cluster logs
cp -f "$HERE/analysis/adaptive_fanout_count_v12.py" "$ROOT/analysis/"
cp -f "$HERE/cluster/v12_adaptive_fanout_count_gpu.sbatch" "$ROOT/cluster/"
chmod +x "$ROOT/cluster/v12_adaptive_fanout_count_gpu.sbatch"
printf 'Applied V12 adaptive FANOUT count patch to %s\n' "$ROOT"
printf 'Plan:   PYTHONPATH="$PWD" python analysis/adaptive_fanout_count_v12.py\n'
printf 'Run:    sbatch cluster/v12_adaptive_fanout_count_gpu.sbatch\n'
