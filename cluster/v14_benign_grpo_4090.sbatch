#!/bin/bash
#SBATCH --job-name=gen-v14-ben
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/gen-v14-ben-%j.out
#SBATCH --error=logs/gen-v14-ben-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PY="/home/wieseld/.conda/envs/masdetective/bin/python"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

"$PY" -u \
  training/train_disambiguation_v14_benign.py \
  --execute \
  --resume-adapter artifacts/v11_policy \
  --rounds-per-role 2 \
  --group-size 4 \
  --bundle-size 3 \
  --confirmation-size 1 \
  --confirm-top-k 2 \
  --systems-per-cell 4 \
  --neighbor-k 5 \
  --balanced-minibatches 3 \
  --grpo-epochs 2 \
  --lr 8e-6 \
  --clip 0.20 \
  --kl-beta 0.0 \
  --elite-nll-weight 0.06 \
  --benign-lambda 0.10 \
  --benign-adapter \
    artifacts/standalone_benign_critic_binary_v4_hardening/best_adapter \
  --benign-temperature 0.5975856324790199 \
  --output-dir artifacts/v14_benign_policy \
  --log artifacts/v14_benign_probe_disambiguation.jsonl \
  --summary artifacts/v14_benign_summary.json \
  --response-public-cache artifacts/v14_benign_response_public.jsonl \
  --response-truth-cache artifacts/v14_benign_response_truth.jsonl \
  --checkpoint-state artifacts/v14_benign_checkpoint.json \
  --checkpoint-root artifacts/v14_benign_checkpoints \
  --optimizer-mirror artifacts/v14_benign_optimizer.pt \
  --optimizer-state artifacts/v14_benign_optimizer.pt
