#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-00:30:00
#SBATCH --job-name=op-drift
#SBATCH --output=logs/op-drift-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --tmp=10G

export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
mkdir -p "$SLURM_SCRATCH_DIR/tmp"

export TMPDIR="$SLURM_SCRATCH_DIR/tmp"
export JOBLIB_TEMP_FOLDER="$SLURM_SCRATCH_DIR/tmp"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd "$HOME/mas_rl_detective"

"$HOME/.conda/envs/masdetective/bin/python" operator_drift_test.py
