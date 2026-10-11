#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-04:00:00
#SBATCH --job-name=dagop-data
#SBATCH --output=logs/dagop-data-%J.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --tmp=10G

set -eo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/dag_operator_active_poc

set +u
source ~/.bashrc
set -u
conda activate masdetective

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not exported into the job environment."
  exit 2
fi

python -u dag_operator_active_poc.py \
  --n-prompts 60 \
  --concurrency 12 \
  --max-tokens 180 \
  --collect-only
