#!/usr/bin/env bash
#SBATCH --job-name=v9f-resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/v9-full-resume-%j.out
set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts
if [ -f .env ]; then set -a; source .env; set +a; fi
PY="$HOME/.conda/envs/masdetective/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STATE=artifacts/v9_full_checkpoint.json
if [ ! -f "$STATE" ]; then echo "No checkpoint: $STATE"; exit 1; fi
START_GROUP=$("$PY" - <<'PY'
import json
print(int(json.load(open('artifacts/v9_full_checkpoint.json'))['next_group']))
PY
)
echo "Resuming V9-FULL from group $START_GROUP"
"$PY" -u -m training.train_seeded_probe_discovery_v9full \
  --rounds-per-role 6 --group-size 6 --bundle-size 2 --groups-per-update 2 \
  --grpo-epochs 2 --elite-nll-weight 0.10 --lr 2e-5 --kl-beta 0.02 \
  --syntax-retries 3 --gate-mode hybrid --holdout-eval --holdout-probes-per-role 3 \
  --resume-adapter artifacts/v9_full_policy --start-group "$START_GROUP" \
  --output-dir artifacts/v9_full_policy \
  --log artifacts/v9_full_probe_discovery.jsonl --summary artifacts/v9_full_summary.json \
  --response-public-cache artifacts/v9_full_response_public.jsonl \
  --response-truth-cache artifacts/v9_full_response_truth.jsonl \
  --checkpoint-state artifacts/v9_full_checkpoint.json --optimizer-state artifacts/v9_full_optimizer.pt \
  --execute
