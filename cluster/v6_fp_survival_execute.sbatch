#!/bin/bash
#SBATCH --job-name=v6fp-exec
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v6fp-exec-%j.out
#SBATCH --error=logs/v6fp-exec-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

set -a
source .env
set +a

: "${OPENAI_API_KEY:?OPENAI_API_KEY missing}"

ROOT="artifacts/v6_fp_survival_multisys_v1"

SEEDS=(
  20261011
  20261012
  20261013
  20261014
  20261015
)

for SEED in "${SEEDS[@]}"; do
    R="$ROOT/seed_${SEED}"

    echo
    echo "=================================================="
    echo "EXECUTING SEED=$SEED"
    echo "=================================================="

    python -u analysis/v3_execute.py \
      --round 1 \
      --split dev \
      --root "$R" \
      --manifest "$R/manifest_privileged.json" \
      --model gpt-4o-mini \
      --workers 6 \
      --retries 3

    test -s "$R/responses_dev_round_01.jsonl"
done

echo
echo "===== EXECUTION COUNTS ====="

python - <<'PY'
import json
from pathlib import Path
from collections import Counter

root = Path("artifacts/v6_fp_survival_multisys_v1")
n = 0

for p in sorted(root.glob(
    "seed_*/responses_dev_round_01.jsonl"
)):
    xs = [
        json.loads(x)
        for x in p.open()
        if x.strip()
    ]
    c = Counter(
        x.get("execution_status")
        for x in xs
    )
    n += c.get("ok", 0)
    print(p.parent.name, dict(c))

print("TOTAL OK:", n)
PY
