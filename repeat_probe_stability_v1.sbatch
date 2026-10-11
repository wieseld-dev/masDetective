#!/bin/bash
#SBATCH --partition=main
#SBATCH --time=0-00:35:00
#SBATCH --job-name=repstab
#SBATCH --output=logs/repstab-%J.out
#SBATCH --error=logs/repstab-%J.err
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0

set -euo pipefail
cd ~/mas_rl_detective
export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"
mkdir -p logs artifacts/repeat_probe_stability_v1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate masdetective

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

[[ -n "${OPENAI_API_KEY:-}" ]] || {
    echo "ERROR: OPENAI_API_KEY unavailable" >&2
    exit 2
}

echo "===== PHASE A: 96-call fresh repeated-execution pilot ====="
python -u analysis/repeat_probe_stability_execute_v1.py \
  --root artifacts/adaptive_v11_curriculum_v2 \
  --obs-dir artifacts/observation_model_v1 \
  --rank-preds artifacts/learned_visible_probe_ranker_v1/test_predictions.jsonl \
  --out artifacts/repeat_probe_stability_v1 \
  --round 1 \
  --groups 8 \
  --repeats 3 \
  --workers 4

echo
echo "===== PHASE B: stability + expected-utility analysis ====="
python -u analysis/repeat_probe_stability_analyze_v1.py \
  --root artifacts/adaptive_v11_curriculum_v2 \
  --obs-dir artifacts/observation_model_v1 \
  --vis-dir artifacts/visible_output_fingerprint_v1 \
  --cal-dir artifacts/calibrated_bayes_diag_v1 \
  --repeat-dir artifacts/repeat_probe_stability_v1 \
  --particles-per-k 32
