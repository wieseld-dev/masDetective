#!/bin/bash
#SBATCH --job-name=bw-check
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/bw-check-%j.out
#SBATCH --error=logs/bw-check-%j.err

echo "===== NVIDIA ====="
nvidia-smi

echo
echo "===== CURRENT TORCH ====="
/home/wieseld/.conda/envs/masdetective/bin/python - <<'PY'
import torch
print("torch =", torch.__version__)
print("torch CUDA =", torch.version.cuda)
print("GPU =", torch.cuda.get_device_name(0))
print("capability =", torch.cuda.get_device_capability(0))
print("compiled architectures =", torch.cuda.get_arch_list())
PY
