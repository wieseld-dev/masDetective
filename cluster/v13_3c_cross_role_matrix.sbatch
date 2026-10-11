#!/bin/bash
#SBATCH --job-name=v13-3c-xrole
#SBATCH --output=logs/v13-3c-xrole-%j.out
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

echo "===== V13.3c CROSS-ROLE MATRIX ====="
echo "host=$(hostname)"
echo "start=$(date)"
echo "python=$(which python)"
python --version
echo "API calls: 0"
echo

python -u -m analysis.v13_3c_cross_role_matrix \
  --judge-dir artifacts/v13_2_dual_judge_resume \
  --sessions artifacts/v13_fanout_count_pilot1/sessions.jsonl \
  --output-dir artifacts/v13_3c_cross_role \
  --eval-batch-size 6 \
  --request-chunk 160

echo
echo "end=$(date)"
echo "===== DONE ====="
