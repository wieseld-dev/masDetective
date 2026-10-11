#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=0-00:30:00
#SBATCH --job-name=genv121
#SBATCH --output=logs/genv121-%J.out
#SBATCH --error=logs/genv121-%J.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

module load anaconda
source activate masdetective

echo "===== ENV ====="
which python
python --version

python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA NOT AVAILABLE")
PY

STATES="artifacts/v6_fp_survival_multisys_v1/seed_20261015/states_dev_round_00.jsonl"

test -f "$STATES"

rm -rf artifacts/generator_v12_1/smoke100

python -u \
  analysis/generator_v12_1_offline_audit.py \
  --states "$STATES" \
  --adapter artifacts/robust_visible_generator_v1_epoch2/policy \
  --n 100 \
  --group-size 8 \
  --temperature 0.90 \
  --top-p 0.96 \
  --max-new-tokens 256 \
  --out artifacts/generator_v12_1/smoke100

echo
echo "===== SUMMARY ====="
cat artifacts/generator_v12_1/smoke100/summary.json

echo
echo "DONE"
