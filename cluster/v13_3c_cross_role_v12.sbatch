#!/bin/bash
#SBATCH --job-name=v13-3c-v12
#SBATCH --output=logs/v13-3c-v12-%j.out
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

echo "host=$(hostname)"
echo "python=$(which python)"
python --version
echo "API calls: 0"

python -u -m analysis.v13_3c_cross_role_matrix \
  --judge-dir artifacts/v13_2_dual_judge_resume \
  --sessions artifacts/v12_fanout_count/sessions.jsonl \
  --output-dir artifacts/v13_3c_cross_role_v12 \
  --eval-batch-size 6 \
  --request-chunk 160

echo "DONE"
