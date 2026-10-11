#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=0-05:00:00
#SBATCH --job-name=rankv2-3
#SBATCH --output=logs/rankv2-3-%J.out
#SBATCH --error=logs/rankv2-3-%J.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

module load anaconda
source activate masdetective

if [[ "$(which python)" != "$HOME/.conda/envs/masdetective/bin/python" ]]; then
    echo "WRONG ENV: $(which python)"
    exit 10
fi

set -a
source .env
set +a

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY missing"
    exit 11
fi

python - <<'PY'
import torch
print("python OK")
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if not torch.cuda.is_available():
    raise SystemExit("NO CUDA")
PY

python -m py_compile \
    analysis/system_v2_rankfirst.py

python -u analysis/system_v2_rankfirst.py \
    --systems 3 \
    --seed 2026092701 \
    --focus-floor 0.72 \
    --min-activation 0.12 \
    --bf-clip 1.75 \
    --out artifacts/system_v2_rankfirst/dev3

cat artifacts/system_v2_rankfirst/dev3/summary.json
