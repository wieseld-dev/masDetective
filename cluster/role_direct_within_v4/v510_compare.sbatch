#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v510-%x-%j.out
#SBATCH --error=logs/v510-%x-%j.err
set -euo pipefail
cd /home/wieseld/mas_rl_detective
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
: "${V510_RUN:?V510_RUN required}"
echo "V510: FROZEN checkpoint comparison on composition-disjoint TEST, never seen during training."
exec "$HOME/.conda/envs/masdetective/bin/python" -u -m analysis.role_direct_within_v4.v510_generalizing_sensor compare \
 --run "$V510_RUN" --max-tokens "${V510_TOKENS:-1280}"
