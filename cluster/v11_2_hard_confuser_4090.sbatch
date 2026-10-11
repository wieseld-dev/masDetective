#!/usr/bin/env bash
#SBATCH --job-name=v11-2-hard
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=48G
#SBATCH --output=logs/v11-2-hard-%j.out
#SBATCH --error=logs/v11-2-hard-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== V11.2 HARD-CONFUSER CONTINUATION ====="
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
echo "python=$(which python)"
python --version
nvidia-smi

# Heartbeat goes into the normal .out log so long API/MAS evaluations never look frozen.
heartbeat() {
  while true; do
    echo "[heartbeat] $(date -Is) job=$SLURM_JOB_ID host=$(hostname)"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader 2>/dev/null | sed 's/^/[gpu] /' || true
    sleep 120
  done
}
heartbeat &
HB_PID=$!
trap 'kill "$HB_PID" 2>/dev/null || true' EXIT

python -m py_compile \
  v11_2/policy.py \
  v11_2/counterfactual.py \
  v11_2/hashcheck.py \
  training/train_hard_confuser_probe_discovery_v11_2.py

# Continues from the exact trained V11 adapter. The script logs every candidate,
# confirmation, group summary, minibatch update, checkpoint, and dev evaluation.
python -u -m training.train_hard_confuser_probe_discovery_v11_2 \
  --resume-adapter artifacts/v11_policy \
  --v11-frozen-summary artifacts/v11_summary.json \
  --initial-elite-summary artifacts/v11_summary.json \
  --baseline-summary artifacts/v11_summary.json \
  --rounds-per-role 2 \
  --group-size 6 \
  --bundle-size 4 \
  --confirmation-size 2 \
  --confirm-top-k 2 \
  --balanced-minibatches 4 \
  --grpo-epochs 2 \
  --elite-nll-weight 0.06 \
  --lr 5e-6 \
  --kl-beta 0.0 \
  --syntax-retries 3 \
  --holdout-generation-retries 8 \
  --gate-mode hybrid \
  --min-activation-rate 0.75 \
  --novelty-free-similarity 0.90 \
  --novelty-weight 0.25 \
  --semantic-hard-duplicate 0.97 \
  --holdout-max-similarity 0.94 \
  --holdout-eval \
  --holdout-probes-per-role 3 \
  --no-4bit \
  --output-dir artifacts/v11_2_policy \
  --log artifacts/v11_2_probe_discovery.jsonl \
  --summary artifacts/v11_2_summary.json \
  --response-public-cache artifacts/v11_2_response_public.jsonl \
  --response-truth-cache artifacts/v11_2_response_truth.jsonl \
  --checkpoint-state artifacts/v11_2_checkpoint.json \
  --checkpoint-root artifacts/v11_2_checkpoints \
  --optimizer-mirror artifacts/v11_2_optimizer.pt \
  --execute

echo "===== V11.2 FINISHED $(date -Is) ====="
