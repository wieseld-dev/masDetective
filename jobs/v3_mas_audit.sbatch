#!/bin/bash

#SBATCH --partition=main
#SBATCH --time=0-00:30:00
#SBATCH --job-name=v3mas-audit
#SBATCH --output=logs/v3mas-audit-%J.out
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

PYTHON="$HOME/.conda/envs/eqs/bin/python"

echo "============================================================"
echo "V3-MAS DATA AUDIT"
echo "============================================================"
echo "JOB=$SLURM_JOB_ID"
echo "NODE=$(hostname)"
echo "START=$(date)"
echo

test -x "$PYTHON"

test -s artifacts/v14_adaptive_role_v1/states_train_round_06.jsonl
test -s artifacts/v14_adaptive_role_v1/states_dev_round_00.jsonl
test -s artifacts/v14_adaptive_role_v1/manifest_privileged.json

echo "INPUT PRECHECK: PASS"
echo

"$PYTHON" -u scripts/v3_mas_audit.py

echo
echo "============================================================"
echo "AUDIT SUCCESS"
echo "END=$(date)"
echo "REPORT=artifacts/eqs_v3_mas/audit_report.txt"
echo "============================================================"
