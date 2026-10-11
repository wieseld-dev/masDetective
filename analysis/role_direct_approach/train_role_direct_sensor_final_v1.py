#!/usr/bin/env python3

import gc
import json
import math
import random
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

import analysis.role_direct_approach.train_role_direct_sensor_v1 as tr


OUT = Path(
    "artifacts/role_direct_sensor_final_v1"
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SEED = 20261009
EPOCHS = 2
LR = 1e-4
BATCH_PAIRS = 2
GRAD_ACCUM = 4
MAX_LENGTH = 1024
PAIR_WEIGHT = 0.5


if (OUT / "adapter").exists():
    raise RuntimeError(
        "Final adapter already exists; refusing to overwrite."
    )

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")


examples = tr.build_examples()

assert len(examples) == 300

# 100 matched system pairs × 3 probes.
assert len({
    (x["role"], x["pair_id"])
    for x in examples
}) == 100


print("=" * 95)
print("ROLE DIRECT SENSOR — FINAL TRAIN")
print("=" * 95)
print("matched query pairs =", len(examples))
print("training rows       =", 2 * len(examples))
print("matched systems     = 100")
print("epochs              =", EPOCHS)
print("base                =", BASE_MODEL)


tok = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

tok.padding_side = "right"
tok.truncation_side = "right"


qconf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        BASE_MODEL,
        num_labels=1,
        quantization_config=qconf,
        device_map={"": 0},
        dtype=torch.bfloat16,
        local_files_only=True,
    )
)

model.config.pad_token_id = tok.pad_token_id
model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    modules_to_save=["score"],
)

model = get_peft_model(
    model,
    lora,
)

model.print_trainable_parameters()

trainable = [
    p
    for p in model.parameters()
    if p.requires_grad
]

opt = torch.optim.AdamW(
    trainable,
    lr=LR,
    weight_decay=0.01,
)

total_batches = math.ceil(
    len(examples) / BATCH_PAIRS
)


for epoch in range(1, EPOCHS + 1):

    model.train()

    rng = random.Random(
        SEED + epoch
    )

    order = list(range(len(examples)))
    rng.shuffle(order)

    opt.zero_grad(set_to_none=True)

    losses = []
    bces = []
    ranks = []

    for step, start in enumerate(
        range(
            0,
            len(order),
            BATCH_PAIRS,
        ),
        1,
    ):

        batch = [
            examples[i]
            for i in order[
                start:start + BATCH_PAIRS
            ]
        ]

        b = len(batch)

        texts = (
            [x["positive_text"] for x in batch]
            +
            [x["negative_text"] for x in batch]
        )

        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to("cuda")

        logits = (
            model(**enc)
            .logits
            .view(-1)
            .float()
        )

        pos = logits[:b]
        neg = logits[b:]

        labels = torch.cat([
            torch.ones_like(pos),
            torch.zeros_like(neg),
        ])

        bce = (
            F.binary_cross_entropy_with_logits(
                logits,
                labels,
            )
        )

        rank = (
            F.softplus(
                -(pos - neg)
            )
            .mean()
        )

        loss = (
            bce
            + PAIR_WEIGHT * rank
        )

        (
            loss / GRAD_ACCUM
        ).backward()

        if (
            step % GRAD_ACCUM == 0
            or step == total_batches
        ):
            torch.nn.utils.clip_grad_norm_(
                trainable,
                1.0,
            )

            opt.step()
            opt.zero_grad(
                set_to_none=True
            )

        losses.append(
            float(loss.detach().cpu())
        )

        bces.append(
            float(bce.detach().cpu())
        )

        ranks.append(
            float(rank.detach().cpu())
        )

        if (
            step % 25 == 0
            or step == total_batches
        ):
            print(
                f"epoch={epoch} "
                f"{step}/{total_batches} "
                f"loss={mean(losses[-25:]):.4f} "
                f"bce={mean(bces[-25:]):.4f} "
                f"rank={mean(ranks[-25:]):.4f}",
                flush=True,
            )


adapter = OUT / "adapter"

model.save_pretrained(adapter)
tok.save_pretrained(adapter)

summary = {
    "version":
        "role_direct_sensor_final_v1",

    "base_model":
        BASE_MODEL,

    "train_matched_queries":
        300,

    "train_rows":
        600,

    "train_matched_system_pairs":
        100,

    "epochs":
        EPOCHS,

    "objective":
        "BCE + 0.5 matched pairwise ranking",

    "input":
        "target role + description + query + visible response",

    "sealed_test_used":
        False,
}

(
    OUT / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

print()
print("FINAL_SENSOR_TRAIN_PASS")
print("adapter =", adapter)

del model
gc.collect()
torch.cuda.empty_cache()
