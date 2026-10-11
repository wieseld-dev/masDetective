#!/bin/bash
#SBATCH --job-name=v6fp-gen
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v6fp-gen-%j.out
#SBATCH --error=logs/v6fp-gen-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=false

BASE="artifacts/adaptive_v11_active_v1"
ROOT="artifacts/v6_fp_survival_multisys_v1"
ADAPTER="artifacts/v6_matched_sft_dpo/policy"

SEEDS=(
  20261011
  20261012
  20261013
  20261014
  20261015
)

test -s "$BASE/states_dev_round_00.jsonl"
test -s "$BASE/manifest_privileged.json"
test -s "$ADAPTER/adapter_config.json"

for SEED in "${SEEDS[@]}"; do
    R="$ROOT/seed_${SEED}"

    mkdir -p "$R"

    cp "$BASE/states_dev_round_00.jsonl" \
       "$R/states_dev_round_00.jsonl"

    cp "$BASE/manifest_privileged.json" \
       "$R/manifest_privileged.json"

    echo
    echo "=================================================="
    echo "GENERATING SEED=$SEED"
    echo "=================================================="

    python -u scripts/curriculum_generate.py \
      --round 1 \
      --split dev \
      --root "$R" \
      --adapter "$ADAPTER" \
      --v11-reference artifacts/v11_policy \
      --group-size 1 \
      --seed "$SEED" \
      --temperature 0.85 \
      --top-p 0.95 \
      --max-new-tokens 192 \
      --max-seq-length 4096 \
      --local-retries 12

    test -s "$R/candidates_dev_round_01.jsonl"
done

echo
echo "===== GENERATED COUNTS ====="

python - <<'PY'
import json
from pathlib import Path

root = Path("artifacts/v6_fp_survival_multisys_v1")
total = 0

for p in sorted(root.glob(
    "seed_*/candidates_dev_round_01.jsonl"
)):
    xs = [
        json.loads(x)
        for x in p.open()
        if x.strip()
    ]
    good = sum(bool(x.get("valid")) for x in xs)
    total += good
    print(p.parent.name, "rows=", len(xs), "valid=", good)

print("TOTAL VALID:", total)
PY
