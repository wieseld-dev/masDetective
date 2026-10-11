#!/bin/bash
#SBATCH --job-name=k3-sweep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --output=logs/k3-sweep-%j.out
#SBATCH --error=logs/k3-sweep-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

python -u analysis/oracle_capability_sweep_k3.py \
  --beta 1.0 \
  --mode mixed
