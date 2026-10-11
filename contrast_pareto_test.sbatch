#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --time=0-03:00:00
#SBATCH --job-name=contrast-pareto
#SBATCH --output=logs/contrast-pareto-%J.out
#SBATCH --cpus-per-task=6
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" contrast_pareto_test.py

echo "END: $(date)"
