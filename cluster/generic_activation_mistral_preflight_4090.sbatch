#!/bin/bash
#SBATCH --job-name=act-mis-pre
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=rtx_4090
#SBATCH --exclude=cs-4090-09
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/act-mis-pre-%j.out
#SBATCH --error=logs/act-mis-pre-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PATH="$HOME/.conda/envs/masdetective/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

PY="$HOME/.conda/envs/masdetective/bin/python"

echo "===== GPU ====="
nvidia-smi

"$PY" - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

print("torch =", torch.__version__)
print("cuda =", torch.version.cuda)
print("gpu =", torch.cuda.get_device_name(0))

tok = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

print("tokenizer OK")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

model.eval()
model.config.use_cache = False

print("model OK")
print("layers =", model.config.num_hidden_layers)
print("hidden =", model.config.hidden_size)

text = tok.apply_chat_template(
    [
        {
            "role": "user",
            "content": "Briefly compare two database indexing strategies."
        }
    ],
    tokenize=False,
    add_generation_prompt=True,
)

enc = tok(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

enc = {
    k: v.cuda()
    for k, v in enc.items()
}

with torch.inference_mode():
    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

print("hidden states =", len(out.hidden_states))
print("last shape =", tuple(out.hidden_states[-1].shape))

print(
    "peak VRAM GiB =",
    round(
        torch.cuda.max_memory_allocated()
        / 1024**3,
        2,
    )
)

print("MISTRAL_ACTIVATION_PREFLIGHT_OK")
PY
