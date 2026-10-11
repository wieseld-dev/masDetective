#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/mas_rl_detective}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p v10 training scripts cluster tests
cp -f "$HERE"/v10/*.py "$ROOT/v10/"
cp -f "$HERE"/training/*.py "$ROOT/training/"
cp -f "$HERE"/scripts/*.py "$ROOT/scripts/"
cp -f "$HERE"/cluster/*.sbatch "$ROOT/cluster/"
cp -f "$HERE"/tests/*.py "$ROOT/tests/"
cp -f "$HERE/README_V10.md" "$ROOT/README_V10.md"
chmod +x "$ROOT"/cluster/v10_*.sbatch 2>/dev/null || true
echo "Applied V10 patch to $ROOT"
