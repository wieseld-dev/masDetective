#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-00:45:00
#SBATCH --job-name=order-spec
#SBATCH --output=logs/order-spec-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=10G

export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
mkdir -p "$SLURM_SCRATCH_DIR/tmp"

export TMPDIR="$SLURM_SCRATCH_DIR/tmp"
export JOBLIB_TEMP_FOLDER="$SLURM_SCRATCH_DIR/tmp"

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"

echo "NODE: $(hostname)"
echo "START: $(date)"

"$PY" pure_order_spectral_benchmark.py

echo "END: $(date)"
