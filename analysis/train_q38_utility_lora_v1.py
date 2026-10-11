from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)

from peft import (
    LoraConfig,
    get_peft_model,
)


MODEL = "Qwen/Qwen3.8-27B"

DATA = Path(
    os.environ.get(
        "DATA_ROOT",
        "artifacts/q38_utility_sft_v1",
    )
)

OUT = Path(
    os.environ.get(
        "OUT_DIR",
        "artifacts/q38_utility_lora_v1/policy",
    )
)

MAX_LEN = int(os.environ.get("MAX_LEN", "1024"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))

LR = float(os.environ.get("LR", "2e-5"))
GRAD_ACC = int(os.environ.get("GRAD_ACC", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))

DEVICE = "cuda"


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


print("=" * 100)
print("QWEN3.8-27B UTILITY LoRA V1")
print("=" * 100)
print("MODEL =", MODEL)
print("DATA  =", DATA)
print("OUT   =", OUT)
print("MAX_LEN =", MAX_LEN)
print("MAX_STEPS =", MAX_STEPS)
print("LR =", LR)
print("GRAD_ACC =", GRAD_ACC)


processor = AutoProcessor.from_pretrained(
    MODEL,
)

tokenizer = getattr(
    processor,
    "tokenizer",
    processor,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading Qwen3.8-27B BF16...")

t0 = time.time()

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

print(
    "LOAD_SECONDS =",
    round(time.time() - t0, 1),
)

if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

model.gradient_checkpointing_enable()

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()


wanted = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

available = {
    name.rsplit(".", 1)[-1]
    for name, module in model.named_modules()
}

targets = [
    x for x in wanted
    if x in available
]

print("LoRA targets:", targets)

if not targets:
    raise RuntimeError(
        "No expected Qwen linear modules found"
    )


lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=targets,
)

model = get_peft_model(
    model,
    lora,
)

model.print_trainable_parameters()


class ProbeDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = self.rows[i]

        prompt = str(x["prompt"])
        completion = str(x["completion"]).strip()

        pids = tokenizer.encode(
            prompt,
            add_special_tokens=True,
        )

        cids = tokenizer.encode(
            completion,
            add_special_tokens=False,
        )

        if tokenizer.eos_token_id is not None:
            cids = cids + [tokenizer.eos_token_id]

        # Never truncate completion.
        room = max(32, MAX_LEN - len(cids))
        pids = pids[-room:]

        ids = pids + cids

        if len(ids) > MAX_LEN:
            ids = ids[-MAX_LEN:]

            # safety fallback
            prompt_kept = max(
                0,
                len(ids) - len(cids),
            )
        else:
            prompt_kept = len(pids)

        labels = (
            [-100] * prompt_kept
            + ids[prompt_kept:]
        )

        labels = labels[:len(ids)]

        return {
            "input_ids": ids,
            "labels": labels,
            "reward": float(x["reward"]),
        }


def collate(batch):
    m = max(len(x["input_ids"]) for x in batch)

    input_ids = []
    attention = []
    labels = []

    for x in batch:
        n = len(x["input_ids"])
        pad = m - n

        input_ids.append(
            x["input_ids"]
            + [tokenizer.pad_token_id] * pad
        )

        attention.append(
            [1] * n + [0] * pad
        )

        labels.append(
            x["labels"]
            + [-100] * pad
        )

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
    }


train_rows = read_jsonl(
    DATA / "train.jsonl"
)

val_rows = read_jsonl(
    DATA / "val.jsonl"
)

print(
    f"TRAIN={len(train_rows)} "
    f"VAL={len(val_rows)}"
)

if not train_rows:
    raise RuntimeError("Empty training set")


loader = DataLoader(
    ProbeDataset(train_rows),
    batch_size=1,
    shuffle=True,
    collate_fn=collate,
)

optimizer = torch.optim.AdamW(
    [
        p for p in model.parameters()
        if p.requires_grad
    ],
    lr=LR,
    weight_decay=1e-4,
)

model.train()

torch.cuda.reset_peak_memory_stats()

micro = 0
opt_steps = 0
loss_sum = 0.0

optimizer.zero_grad(set_to_none=True)

start = time.time()

stop = False

for epoch in range(EPOCHS):

    for batch in loader:

        batch = {
            k: v.to(DEVICE)
            for k, v in batch.items()
        }

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        loss = out.loss / GRAD_ACC
        loss.backward()

        micro += 1
        loss_sum += float(out.loss.detach().cpu())

        if micro % GRAD_ACC == 0:

            torch.nn.utils.clip_grad_norm_(
                [
                    p for p in model.parameters()
                    if p.requires_grad
                ],
                1.0,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            opt_steps += 1

            if (
                opt_steps == 1
                or opt_steps % 10 == 0
            ):
                print(
                    f"step={opt_steps:04d} "
                    f"micro={micro:05d} "
                    f"loss={loss_sum/micro:.5f} "
                    f"peak_vram="
                    f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GiB",
                    flush=True,
                )

            if (
                MAX_STEPS > 0
                and opt_steps >= MAX_STEPS
            ):
                stop = True
                break

    if stop:
        break


# flush final partial accumulation
if (
    not stop
    and micro % GRAD_ACC != 0
):
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    opt_steps += 1


elapsed = time.time() - start

print()
print("TRAIN_DONE")
print("MICRO_STEPS =", micro)
print("OPT_STEPS   =", opt_steps)
print("SECONDS     =", round(elapsed, 1))
print(
    "PEAK_VRAM_GIB =",
    round(
        torch.cuda.max_memory_allocated()
        / 1024**3,
        2,
    ),
)


# Preflight intentionally does not overwrite real adapter.
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(OUT)
processor.save_pretrained(OUT)

meta = {
    "base_model": MODEL,
    "max_len": MAX_LEN,
    "lr": LR,
    "grad_acc": GRAD_ACC,
    "epochs": EPOCHS,
    "max_steps": MAX_STEPS,
    "micro_steps": micro,
    "optimizer_steps": opt_steps,
    "peak_vram_gib":
        torch.cuda.max_memory_allocated()
        / 1024**3,
    "elapsed_seconds": elapsed,
}

(
    OUT.parent / "train_summary.json"
).write_text(
    json.dumps(meta, indent=2)
    + "\n"
)

print("SAVED =", OUT)
print("Q38_UTILITY_LORA_OK")
