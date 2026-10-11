#!/usr/bin/env bash
#SBATCH --job-name=v13-5-evid
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --mem=48G
#SBATCH --output=logs/v13-5-evid-%j.out
#SBATCH --error=logs/v13-5-evid-%j.err

set -euo pipefail
cd "$HOME/mas_rl_detective"
mkdir -p logs artifacts/v13_5_evidence_judge

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "===== V13.5 EVIDENCE JUDGE ====="
echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is)"
echo "python=$(which python)"
python --version
nvidia-smi

# Heartbeat in the main .out log even during long validation/model operations.
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

python -m py_compile analysis/train_v13_5_evidence_judge.py

# Same Qwen2.5-1.5B, warm-starting the trained V13.4 adapter.
# No explicit SBATCH time request: scheduler/partition policy supplies its own limit.
python -u analysis/train_v13_5_evidence_judge.py \
  --resume-adapter artifacts/v13_4_presence_judge/best_adapter \
  --output-dir artifacts/v13_5_evidence_judge \
  --no-4bit \
  --gradient-checkpointing \
  --max-length 768 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --rank-batch-size 1 \
  --cf-batch-size 1 \
  --eval-pair-batch-size 2 \
  --grad-accum 4 \
  --epochs 4 \
  --neg-ratio 2.0 \
  --self-target-neg-weight 3.0 \
  --hardneg-weight 1.5 \
  --same-answer-negatives 2 \
  --rank-loss-weight 0.5 \
  --cf-loss-weight 0.5 \
  --rank-margin 0.75 \
  --cf-margin 0.75 \
  --aux-every 2 \
  --lr 5e-5 \
  --log-every 20

echo "===== V13.5 FINISHED $(date -Is) ====="
