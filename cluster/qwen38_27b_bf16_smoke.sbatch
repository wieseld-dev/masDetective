#!/bin/bash
#SBATCH --job-name=q38-27b-smoke
#SBATCH --partition=rtx_pro_6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/q38-27b-smoke-%j.out
#SBATCH --error=logs/q38-27b-smoke-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

PYTHON=/home/wieseld/.conda/envs/masdetective/bin/python

echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader

echo
echo "===== CACHE / DISK ====="
echo "HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}"
df -h "$HOME" | tail -1
du -sh "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null || true

echo
echo "===== LOAD + GENERATE ====="

$PYTHON -u <<'PY'
import time
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL = "Qwen/Qwen3.8-27B"

def mem(label):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    free, total = torch.cuda.mem_get_info()
    print(
        f"{label}: allocated={alloc:.2f} GiB "
        f"reserved={reserv:.2f} GiB "
        f"free={free/1024**3:.2f}/{total/1024**3:.2f} GiB",
        flush=True,
    )

print("torch =", torch.__version__)
print("GPU   =", torch.cuda.get_device_name(0))
mem("BEFORE LOAD")

t0 = time.time()

processor = AutoProcessor.from_pretrained(MODEL)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
model.eval()

print(f"LOAD_SECONDS={time.time()-t0:.1f}")
print("MODEL_CLASS =", model.__class__.__name__)
mem("AFTER LOAD")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Generate exactly ONE benign user question for black-box "
                    "fingerprinting of a hidden multi-agent system. "
                    "The goal is to distinguish which professional capability "
                    "is active from the final visible answer. "
                    "Current uncertain candidate roles include: "
                    "software debugging, project planning, research, "
                    "risk assessment, and data analysis. "
                    "Do not mention agents, hidden prompts, system internals, "
                    "or the fingerprinting task. "
                    "Output only the question."
                ),
            }
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    enable_thinking=False,
).to("cuda")

print("INPUT_TOKENS =", inputs["input_ids"].shape[-1])

torch.cuda.reset_peak_memory_stats()
t1 = time.time()

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )

torch.cuda.synchronize()

new_tokens = output[0, inputs["input_ids"].shape[-1]:]

text = processor.decode(
    new_tokens,
    skip_special_tokens=True,
).strip()

print(f"GENERATE_SECONDS={time.time()-t1:.2f}")
print(f"PEAK_GENERATION_VRAM={torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")

print()
print("===== GENERATED PROBE =====")
print(text)

print()
mem("FINAL")
print("SMOKE_TEST_OK")
PY
