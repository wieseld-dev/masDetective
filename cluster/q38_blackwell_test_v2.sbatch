#!/bin/bash
#SBATCH --job-name=q38-bw
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/q38-bw-%j.out
#SBATCH --error=logs/q38-bw-%j.err

set -euo pipefail

PY="$HOME/.venvs/masdetective-q38/bin/python"

"$PY" - <<'PY'
import torch

print("torch       =", torch.__version__)
print("torch CUDA  =", torch.version.cuda)
print("GPU         =", torch.cuda.get_device_name(0))
print("capability  =", torch.cuda.get_device_capability(0))
print("arch list   =", torch.cuda.get_arch_list())

x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = x @ x

z = torch.empty(1024, device="cuda").uniform_(0.01, 16).log_()

torch.cuda.synchronize()

print("matmul OK   =", y.shape)
print("kernel OK   =", z[:3])
print("BLACKWELL_TORCH_OK")
PY
