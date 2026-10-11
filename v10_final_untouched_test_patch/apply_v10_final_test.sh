#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/mas_rl_detective}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p targets scripts cluster tests logs artifacts
cp -f "$HERE"/targets/*.py "$ROOT/targets/"
cp -f "$HERE"/scripts/*.py "$ROOT/scripts/"
cp -f "$HERE"/cluster/*.sbatch "$ROOT/cluster/"
cp -f "$HERE"/tests/*.py "$ROOT/tests/"
cp -f "$HERE/README_FINAL_TEST.md" "$ROOT/README_V10_FINAL_TEST.md"
chmod +x "$ROOT"/cluster/v10_final_test_gpu.sbatch

echo "Applied V10.1 final untouched test patch to $ROOT"
echo "This patch contains NO training code and does not modify artifacts/v10_policy."
