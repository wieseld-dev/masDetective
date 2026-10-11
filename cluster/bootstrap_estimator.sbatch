#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-02:00:00
#SBATCH --job-name mas-k-bootstrap
#SBATCH --output logs/mas-k-bootstrap-%J.out
#SBATCH --gpus=0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
module load anaconda
source activate masdetective
set -a
source .env
set +a

python -u -m training.bootstrap_estimator --limit-systems 6 --max-seeds 5 --execute
