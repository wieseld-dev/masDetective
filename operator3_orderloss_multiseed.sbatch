#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-02:00:00
#SBATCH --job-name=op3-order
#SBATCH --output=logs/op3-order-%J.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --tmp=10G

export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
mkdir -p "$SLURM_SCRATCH_DIR/tmp"

export TMPDIR="$SLURM_SCRATCH_DIR/tmp"
export JOBLIB_TEMP_FOLDER="$SLURM_SCRATCH_DIR/tmp"

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" operator3_orderloss_multiseed.py
