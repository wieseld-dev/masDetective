#!/bin/bash
#SBATCH --job-name=mas-wb-smoke
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

cd "$HOME/mas_rl_detective"
mkdir -p logs

# Create/activate the cluster env before first use as described in README_V4.md.
source .venv/bin/activate

python -u -m scripts.smoke_live_whitebox \
  --system-id "${SYSTEM_ID:-C03_s1}" \
  --max-probes "${MAX_PROBES:-5}" \
  --model "${MAS_MODEL:-gpt-4o-mini}"
