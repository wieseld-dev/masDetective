#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-02:00:00
#SBATCH --job-name=exec-eval
#SBATCH --output=logs/exec-eval-%J.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=10G

export SLURM_SCRATCH_DIR=/scratch/${SLURM_JOB_USER}/${SLURM_JOB_ID}
mkdir -p "$SLURM_SCRATCH_DIR/tmp"

export TMPDIR="$SLURM_SCRATCH_DIR/tmp"
export JOBLIB_TEMP_FOLDER="$SLURM_SCRATCH_DIR/tmp"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

cd "$HOME/mas_rl_detective"

PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" execution_operator_big_eval.py
