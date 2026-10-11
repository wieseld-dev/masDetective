#!/bin/bash
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
ROOT="${1:-$HOME/mas_rl_detective}"
cd "$ROOT"

required=(
  "artifacts/v11_policy"
  "artifacts/v11_summary.json"
  "artifacts/v11_probe_discovery.jsonl"
  "analysis/adaptive_fanout_count_v12.py"
  "training/train_cross_role_probe_discovery_v11.py"
  "v11/role_universe.py"
  "v10/policy.py"
)
for x in "${required[@]}"; do
  if [ ! -e "$x" ]; then
    echo "ERROR: required prior artifact/source missing: $x" >&2
    exit 1
  fi
done

mkdir -p v13 training analysis cluster tests
cp "$SRC/v13/"*.py v13/
cp "$SRC/training/train_disambiguation_v13.py" training/
cp "$SRC/analysis/adaptive_fanout_count_v13.py" analysis/
cp "$SRC/cluster/"*.sbatch cluster/
cp "$SRC/tests/"*.py tests/ 2>/dev/null || true

echo "Installed V13 without modifying artifacts/v11_policy."
echo "Plan training:"
echo "  PYTHONPATH=\"$ROOT\" python training/train_disambiguation_v13.py"
echo "Launch training:"
echo "  sbatch cluster/v13_disambiguation_train_gpu.sbatch"
echo "After training, plan eval:"
echo "  PYTHONPATH=\"$ROOT\" python analysis/adaptive_fanout_count_v13.py"
echo "Then launch eval:"
echo "  sbatch cluster/v13_fanout_eval_gpu.sbatch"
