#!/bin/bash
#SBATCH --partition=main
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=0-08:00:00
#SBATCH --job-name=mas10-night
#SBATCH --output=logs/mas10-night-%J.out
#SBATCH --error=logs/mas10-night-%J.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

echo "============================================================"
echo "10 MAS NIGHT TEST"
echo "============================================================"
date
hostname
echo "JOB=$SLURM_JOB_ID"

module load anaconda
source activate masdetective

echo "PYTHON=$(which python)"
python --version

if [[ "$(which python)" != "$HOME/.conda/envs/masdetective/bin/python" ]]; then
    echo "ERROR: wrong Python environment"
    exit 10
fi

if [[ ! -f .env ]]; then
    echo "ERROR: .env missing"
    exit 11
fi

set -a
source .env
set +a

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY missing after sourcing .env"
    exit 12
fi

echo "OPENAI_API_KEY: PRESENT"
echo "CUDA:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable on allocated node")
print("gpu:", torch.cuda.get_device_name(0))
PY

echo
echo "===== PY_COMPILE ====="
python -m py_compile \
    analysis/night_v7_stack.py \
    analysis/night_10mas_reconstruction.py

echo
echo "===== REAL END-TO-END SMOKE ====="
python analysis/night_10mas_reconstruction.py \
    --smoke \
    --sensor-batch-size 8

echo
echo "===== FULL 10 MAS TEST ====="
python -u analysis/night_10mas_reconstruction.py \
    --systems 10 \
    --max-probes 8 \
    --min-probes 5 \
    --calibration-rows 320 \
    --sensor-batch-size 8 \
    --seed 20260926 \
    --out artifacts/night_20260926/test10_current_stack

echo
echo "===== FINAL RESULT ====="
cat artifacts/night_20260926/test10_current_stack/summary.json

date
