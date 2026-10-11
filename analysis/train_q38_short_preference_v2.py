from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)

from peft import PeftModel


MODEL = "Qwen/Qwen3.8-27B"

START = Path(
    os.environ.get(
        "START_ADAPTER",
        "artifacts/q38_online_r7match_v1/"
        "checkpoints/round_05/policy",
    )
)

DATA = Path(
    os.environ.get(
        "DATA_ROOT",
        "artifacts/q38_shortfingerprint_v2_strict",
    )
)

OUT = Path(
    os.environ.get(
        "OUT_DIR",
        "artifacts/q38_shortfingerprint_pref_v2/policy",
    )
)

MAX_LEN = int(
    os.environ.get("MAX_LEN", "1024")
)

LR = float(
    os.environ.get("LR", "5e-6")
)

BETA = float(
    os.environ.get("BETA", "0.10")
)

GRAD_ACC = int(
    os.environ.get("GRAD_ACC", "4")
)

EPOCHS = int(
    os.environ.get("EPOCHS", "1")
)

SEED = int(
    os.environ.get("SEED", "3817")
)


random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


train = read_jsonl(
    DATA / "preferences_train.jsonl"
)

val = read_jsonl(
    DATA / "preferences_val.jsonl"
)


if len(train) < 50:
    raise RuntimeError(
        f"Too few training pairs: {len(train)}"
    )


print("=" * 100)
print("Q38 SHORT-FINGERPRINT PREFERENCE V2")
print("=" * 100)
print("MODEL =", MODEL)
print("START =", START)
print("TRAIN =", len(train))
print("VAL   =", len(val))
print("LR    =", LR)
print("BETA  =", BETA)
print("GRAD_ACC =", GRAD_ACC)


processor = AutoProcessor.from_pretrained(
    MODEL
)

tokenizer = getattr(
    processor,
    "tokenizer",
    processor,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading 27B BF16 base...")

t0 = time.time()

base = AutoModelForMultimodalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

print(
    "BASE_LOAD_SECONDS =",
    round(time.time() - t0, 1),
)


if hasattr(base.config, "use_cache"):
    base.config.use_cache = False

base.gradient_checkpointing_enable()

if hasattr(
    base,
    "enable_input_require_grads",
):
    base.enable_input_require_grads()


print("Loading Q38 R5 adapter...")

model = PeftModel.from_pretrained(
    base,
    str(START),
    is_trainable=True,
)

model.print_trainable_parameters()


def encode(prompt, completion):
    pids = tokenizer.encode(
        str(prompt),
        add_special_tokens=True,
    )

    cids = tokenizer.encode(
        str(completion).strip(),
        add_special_tokens=False,
    )

    if tokenizer.eos_token_id is not None:
        cids.append(
            tokenizer.eos_token_id
        )

    if len(cids) >= MAX_LEN:
        cids = cids[:MAX_LEN - 1]

    room = MAX_LEN - len(cids)

    pids = pids[-room:]

    ids = pids + cids

    labels = (
        [-100] * len(pids)
        + cids
    )

    return (
        torch.tensor(
            [ids],
            dtype=torch.long,
            device="cuda",
        ),
        torch.tensor(
            [labels],
            dtype=torch.long,
            device="cuda",
        ),
    )


def mean_completion_logp(
    prompt,
    completion,
):
    ids, labels = encode(
        prompt,
        completion,
    )

    attention = torch.ones_like(ids)

    out = model(
        input_ids=ids,
        attention_mask=attention,
        use_cache=False,
    )

    logits = out.logits[:, :-1, :].float()
    targets = ids[:, 1:]
    label_shift = labels[:, 1:]

    mask = label_shift.ne(-100)

    lp = F.log_softmax(
        logits,
        dim=-1,
    )

    token_lp = lp.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)

    return token_lp[mask].mean()


# --------------------------------------------------
# Freeze R5 reference numerically BEFORE training.
# No second 27B model required.
# --------------------------------------------------

all_rows = train + val

print()
print(
    "Precomputing frozen R5 reference log-probs..."
)

model.eval()

torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    for i, x in enumerate(all_rows, 1):

        x["_ref_chosen"] = float(
            mean_completion_logp(
                x["prompt"],
                x["chosen"],
            )
            .detach()
            .cpu()
        )

        x["_ref_rejected"] = float(
            mean_completion_logp(
                x["prompt"],
                x["rejected"],
            )
            .detach()
            .cpu()
        )

        if (
            i == 1
            or i % 25 == 0
            or i == len(all_rows)
        ):
            print(
                f"reference {i}/{len(all_rows)}",
                flush=True,
            )


optimizer = torch.optim.AdamW(
    [
        p
        for p in model.parameters()
        if p.requires_grad
    ],
    lr=LR,
    weight_decay=1e-4,
)


model.train()

optimizer.zero_grad(
    set_to_none=True
)

micro = 0
steps = 0
loss_sum = 0.0

start = time.time()


for epoch in range(EPOCHS):

    random.shuffle(train)

    for x in train:

        chosen_lp = mean_completion_logp(
            x["prompt"],
            x["chosen"],
        )

        rejected_lp = mean_completion_logp(
            x["prompt"],
            x["rejected"],
        )

        policy_gap = (
            chosen_lp
            - rejected_lp
        )

        ref_gap = torch.tensor(
            x["_ref_chosen"]
            - x["_ref_rejected"],
            dtype=torch.float32,
            device="cuda",
        )

        preference_logit = (
            BETA
            * (
                policy_gap
                - ref_gap
            )
        )

        loss_raw = (
            -F.logsigmoid(
                preference_logit
            )
        )

        loss = (
            loss_raw
            / GRAD_ACC
        )

        loss.backward()

        micro += 1
        loss_sum += float(
            loss_raw.detach().cpu()
        )

        if micro % GRAD_ACC == 0:

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                1.0,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            steps += 1

            if (
                steps == 1
                or steps % 10 == 0
            ):
                print(
                    f"step={steps:04d} "
                    f"pairs={micro:04d} "
                    f"loss={loss_sum/micro:.5f} "
                    f"peak_vram="
                    f"{torch.cuda.max_memory_allocated()/1024**3:.2f}GiB",
                    flush=True,
                )


if micro % GRAD_ACC:

    torch.nn.utils.clip_grad_norm_(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        1.0,
    )

    optimizer.step()

    optimizer.zero_grad(
        set_to_none=True
    )

    steps += 1


train_seconds = (
    time.time()
    - start
)


# --------------------------------------------------
# Held-out preference evaluation
# --------------------------------------------------

model.eval()

correct = 0
margins = []

with torch.no_grad():

    for x in val:

        chosen_lp = float(
            mean_completion_logp(
                x["prompt"],
                x["chosen"],
            )
            .detach()
            .cpu()
        )

        rejected_lp = float(
            mean_completion_logp(
                x["prompt"],
                x["rejected"],
            )
            .detach()
            .cpu()
        )

        learned_delta = (
            (chosen_lp - rejected_lp)
            -
            (
                x["_ref_chosen"]
                - x["_ref_rejected"]
            )
        )

        margins.append(
            learned_delta
        )

        correct += int(
            learned_delta > 0
        )


val_acc = (
    correct / len(val)
    if val
    else float("nan")
)

val_margin = (
    sum(margins) / len(margins)
    if margins
    else float("nan")
)


print()
print("PREFERENCE_TRAIN_DONE")
print("OPT_STEPS =", steps)
print("SECONDS =", round(train_seconds, 1))
print("VAL_PREF_ACC =", round(val_acc, 4))
print(
    "VAL_MEAN_RELATIVE_MARGIN =",
    round(val_margin, 6),
)
print(
    "PEAK_VRAM_GIB =",
    round(
        torch.cuda.max_memory_allocated()
        / 1024**3,
        2,
    ),
)


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(
    OUT
)

processor.save_pretrained(
    OUT
)


summary = {
    "base_model": MODEL,
    "start_adapter": str(START),

    "train_pairs": len(train),
    "val_pairs": len(val),

    "lr": LR,
    "beta": BETA,
    "grad_acc": GRAD_ACC,
    "epochs": EPOCHS,

    "optimizer_steps": steps,
    "train_seconds": train_seconds,

    "val_preference_accuracy":
        val_acc,

    "val_mean_relative_margin":
        val_margin,

    "peak_vram_gib":
        torch.cuda.max_memory_allocated()
        / 1024**3,
}

(
    OUT.parent
    / "train_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)
