#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --time=0-01:00:00
#SBATCH --job-name=comm-ms
#SBATCH --output=logs/comm-ms-%J.out
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G

cd "$HOME/mas_rl_detective"
PY="$HOME/.conda/envs/masdetective/bin/python"

"$PY" v11_multiseed.py
