#!/bin/bash
set -euo pipefail

PROJECT="${1:-$HOME/mas_rl_detective}"
cd "$PROJECT"

if [[ ! -f targets/role_presence_mas.py ]]; then
  echo "ERROR: $PROJECT does not look like the V4 mas_rl_detective project." >&2
  exit 1
fi

# These three tests belong to the removed offline V3 dataset and fail when that
# legacy dataset is intentionally absent. V5 does not use them.
rm -f tests/test_estimator.py tests/test_metrics.py tests/test_real_dataset.py

mkdir -p artifacts outputs logs
chmod 600 .env 2>/dev/null || true
chmod +x cluster/*.sbatch 2>/dev/null || true

echo "V5 overlay prepared in: $PROJECT"
echo "Next: conda activate masdetective && pip install -r requirements_v5.txt"
