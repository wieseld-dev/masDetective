#!/bin/bash
#SBATCH --job-name=k3demo
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:35:00
#SBATCH --output=logs/k3demo-%j.out
#SBATCH --error=logs/k3demo-%j.err
set -euo pipefail
cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective
if [ -f .env ]; then set -a; source .env; set +a; fi
python -u analysis/one_system_k3_demo.py \
  --adapter artifacts/robust_visible_generator_v1/policy \
  --rounds 5 --group-size 8 --particles-per-k 128 --mode mixed
