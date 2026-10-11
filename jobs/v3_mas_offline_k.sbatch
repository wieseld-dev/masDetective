#!/bin/bash

#SBATCH --partition=main
#SBATCH --time=0-04:00:00
#SBATCH --job-name=v3mas-k
#SBATCH --output=logs/v3mas-k-%J.out
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="$HOME/.conda/envs/eqs/bin/python"

echo "============================================================"
echo "V3-MAS: REAL MAS AGENT-COUNT QUERY SELECTION"
echo "============================================================"
echo "JOB=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "START=$(date)"
echo "PYTHON=$PYTHON"
echo

test -x "$PYTHON"

test -s \
  artifacts/v14_adaptive_role_v1/manifest_privileged.json

test -s \
  artifacts/v14_adaptive_role_v1/states_train_round_00.jsonl

test -s \
  artifacts/v14_adaptive_role_v1/candidates_train_round_01.jsonl

test -s \
  artifacts/v14_adaptive_role_v1/responses_train_round_01.jsonl

test -s \
  artifacts/v14_adaptive_role_v1/states_dev_round_00.jsonl

test -s \
  scripts/v3_mas_offline_k.py

echo "PRECHECK: PASS"
echo

"$PYTHON" -u scripts/v3_mas_offline_k.py \
  --root artifacts/v14_adaptive_role_v1 \
  --manifest artifacts/v14_adaptive_role_v1/manifest_privileged.json \
  --outdir artifacts/eqs_v3_mas \
  --rounds 1,2,3,4,5,6 \
  --dim 2048

echo
echo "============================================================"
echo "V3-MAS COMPLETE"
echo "END=$(date)"
echo "============================================================"
echo
cat artifacts/eqs_v3_mas/results.json
