#!/bin/bash

#SBATCH --partition=main
#SBATCH --time=0-02:00:00
#SBATCH --job-name=eqs-v1
#SBATCH --output=logs/eqs-v1-%J.out
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -euo pipefail

echo "============================================================"
echo "EQS HISTORY -> NEXT QUERY EXPERIMENT"
echo "============================================================"
echo "JOB_ID=$SLURM_JOB_ID"
echo "NODE=$SLURM_JOB_NODELIST"
echo "START=$(date)"
echo

cd "$SLURM_SUBMIT_DIR"

module load anaconda
source activate eqs

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"


# ==============================================================================
# SANITY TESTS
# ==============================================================================

echo "===== SANITY TESTS ====="

python - <<'PY'
import torch
from scripts.eqs_history_query_v1 import (
    SyntheticWorld,
    entropy
)

# 1. Posterior must remain a valid probability distribution.
w = SyntheticWorld(seed=101)

b = torch.full(
    (w.n_h,),
    1.0 / w.n_h
)

b2 = w.update_belief(
    b,
    query_id=3,
    response=1
)

assert torch.all(b2 >= 0)
assert abs(float(b2.sum()) - 1.0) < 1e-6


# 2. Expected Information Gain should not be negative
# except tiny floating point noise.
eig = w.eig(b)

assert float(eig.min()) > -1e-6


# 3. Highest-EIG query should reduce expected entropy.
q = int(eig.argmax())

p1 = float(
    (b * w.probs[:, q]).sum()
)

post1 = w.update_belief(
    b,
    q,
    1
)

post0 = w.update_belief(
    b,
    q,
    0
)

expected_entropy = (
    p1 * float(entropy(post1))
    + (1.0 - p1)
    * float(entropy(post0))
)

measured_information_gain = (
    float(entropy(b))
    - expected_entropy
)

assert expected_entropy <= float(entropy(b)) + 1e-6

assert abs(
    measured_information_gain
    - float(eig[q])
) < 1e-5


# 4. Query representation must contain finite values.
assert torch.isfinite(
    w.raw_q
).all()

print("ALL SANITY TESTS PASSED")
PY


# ==============================================================================
# TRAIN + TEST QUALITY
# ==============================================================================

echo
echo "===== TRAINING ====="

python -u scripts/eqs_history_query_v1.py \
    --epochs 8 \
    --episodes 2200 \
    --seed 7 \
    --save artifacts/eqs_v1/model_seed7.pt \
    --results artifacts/eqs_v1/results_seed7.json


echo
echo "============================================================"
echo "JOB FINISHED"
echo "END=$(date)"
echo
echo "RESULT FILE:"
echo "artifacts/eqs_v1/results_seed7.json"
echo
echo "MODEL:"
echo "artifacts/eqs_v1/model_seed7.pt"
echo "============================================================"
