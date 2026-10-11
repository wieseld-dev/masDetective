#!/bin/bash

#SBATCH --partition=main
#SBATCH --constraint=rtx_4090
#SBATCH --gpus=1
#SBATCH --time=0-01:00:00
#SBATCH --job-name=comm-poc
#SBATCH --output=logs/comm-poc-%J.out

echo "JOB=$SLURM_JOB_ID"
echo "NODE=$SLURM_JOB_NODELIST"

module load anaconda
source activate masdetective

echo "Python:"
which python
python --version

echo "GPU:"
nvidia-smi

cd ~/mas_rl_detective

python blackbox_commutator_poc.py \
    --n-prompts 60 \
    --concurrency 8
