#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constraint=rtx_4090
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00:00
#SBATCH --job-name=comp-latent
#SBATCH --output=logs/comp-latent-%J.out
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" compositional_latent_test.py
