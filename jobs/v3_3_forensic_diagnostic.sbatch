#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=v33-forensic
#SBATCH --output=logs/v33-forensic-%J.out
#SBATCH --error=logs/v33-forensic-%J.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PKG_DIR="$(
    find "$PWD" \
      -maxdepth 5 \
      -type d \
      -name adaptive_v11_active \
      -print \
      -quit
)"

if [ -n "$PKG_DIR" ]; then
    export PYTHONPATH="$(dirname "$PKG_DIR"):$PWD:${PYTHONPATH:-}"
else
    export PYTHONPATH="$PWD:${PYTHONPATH:-}"
fi

PYTHON="$HOME/.conda/envs/masdetective/bin/python"

test -x "$PYTHON"
test -s scripts/v3_3_forensic_diagnostic.py
test -s artifacts/eqs_v3_2_closed_loop/pilot/summary.json

echo "START=$(date)"
echo "NO OPENAI CALLS"
echo "NO MAS EXECUTIONS"
echo

"$PYTHON" -u \
    scripts/v3_3_forensic_diagnostic.py

echo
echo "END=$(date)"
