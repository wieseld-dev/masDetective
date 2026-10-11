from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


MODEL = "Qwen/Qwen3.8-27B"

DATA = Path(
    os.environ.get(
        "DATA_ROOT",
        "artifacts/q38_selector_v4_listwise_data_v1",
    )
)

OUT = Path(
    os.environ.get(
        "OUT_DIR",
        "artifacts/q38_selector_v4_listwise_v1",
    )
)

MAX_LEN = int(
    os.environ.get("MAX_LEN", "1024")
)

EPOCHS = int(
    os.environ.get("EPOCHS", "4")
)

MAX_MICRO = int(
    os.environ.get("MAX_MICRO", "15000")
)

GRAD_ACC = int(
    os.environ.get("GRAD_ACC", "8")
)

LR = float(
    os.environ.get("LR", "8e-6")
)

PATIENCE = int(
    os.environ.get("PATIENCE", "3")
)

EVAL_BATCH = int(
    os.environ.get("EVAL_BATCH", "2")
)

LETTERS = list("ABCDEFGH")

SYSTEM = """You are a selector for benign black-box behavioral fingerprinting.

Your job is NOT to answer the candidate probes.

Given the current public belief, previous visible interactions, and a set of benign candidate probes, choose the probe expected to provide the greatest diagnostic value for recovering the installed role set.

Reason comparatively across the whole candidate set. Prefer discriminative behavioral probes over topical or redundant probes.

Return only the requested candidate letter."""


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def dump(path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


print("=" * 100)
print("QWEN3.8-27B SELECTOR V4 — LISTWISE LoRA")
print("=" * 100)

print("MODEL      =", MODEL)
print("DATA       =", DATA)
print("OUT        =", OUT)
print("MAX_LEN    =", MAX_LEN)
print("EPOCHS     =", EPOCHS)
print("MAX_MICRO  =", MAX_MICRO)
print("GRAD_ACC   =", GRAD_ACC)
print("LR         =", LR)
print("PATIENCE   =", PATIENCE)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

train_rows = read_jsonl(
    DATA / "train_sft.jsonl"
)

val_groups = read_jsonl(
    DATA / "val_groups.jsonl"
)

challenge_groups = read_jsonl(
    DATA / "challenge_groups.jsonl"
)

print()
print(
    f"TRAIN={len(train_rows)} "
    f"VAL_GROUPS={len(val_groups)} "
    f"CHALLENGE_GROUPS={len(challenge_groups)}"
)


###############################################################################
# TOKENIZER
###############################################################################

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

tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"


def chat_prompt(user):
    messages = [
        {
            "role": "system",
            "content": SYSTEM,
        },
        {
            "role": "user",
            "content": user,
        },
    ]

    if hasattr(
        tokenizer,
        "apply_chat_template",
    ):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return (
        SYSTEM
        + "\n\n"
        + user
        + "\n\nASSISTANT:\n"
    )


# Find one-token representations for A..H.
label_ids = {}
label_texts = {}

for letter in LETTERS:
    found = None

    for txt in [
        letter,
        " " + letter,
    ]:
        ids = tokenizer.encode(
            txt,
            add_special_tokens=False,
        )

        if len(ids) == 1:
            found = (
                txt,
                int(ids[0]),
            )
            break

    if found is None:
        raise RuntimeError(
            f"No single-token representation for label {letter}"
        )

    label_texts[letter] = found[0]
    label_ids[letter] = found[1]

print("LABEL TOKENS =", label_ids)


###############################################################################
# PRETOKENIZE TRAINING
###############################################################################

class TrainDataset(Dataset):
    def __init__(self, rows):
        self.items = []

        t0 = time.time()

        for i, x in enumerate(rows):
            p = chat_prompt(
                str(x["prompt"])
            )

            pids = tokenizer.encode(
                p,
                add_special_tokens=False,
            )

            # Reserve exactly one supervised token.
            pids = pids[
                -(MAX_LEN - 1):
            ]

            y = label_ids[
                str(x["completion"])
            ]

            ids = pids + [y]

            labels = (
                [-100] * len(pids)
                + [y]
            )

            self.items.append({
                "input_ids": ids,
                "labels": labels,
            })

            if (
                (i + 1) % 1000 == 0
                or i + 1 == len(rows)
            ):
                print(
                    f"tokenized "
                    f"{i+1}/{len(rows)}",
                    flush=True,
                )

        print(
            "TOKENIZATION_SECONDS =",
            round(time.time() - t0, 1),
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    m = max(
        len(x["input_ids"])
        for x in batch
    )

    ids = []
    masks = []
    labels = []

    for x in batch:
        n = len(x["input_ids"])
        pad = m - n

        # left padding
        ids.append(
            [tokenizer.pad_token_id] * pad
            + x["input_ids"]
        )

        masks.append(
            [0] * pad
            + [1] * n
        )

        labels.append(
            [-100] * pad
            + x["labels"]
        )

    return {
        "input_ids":
            torch.tensor(
                ids,
                dtype=torch.long,
            ),

        "attention_mask":
            torch.tensor(
                masks,
                dtype=torch.long,
            ),

        "labels":
            torch.tensor(
                labels,
                dtype=torch.long,
            ),
    }


train_ds = TrainDataset(
    train_rows
)

loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    collate_fn=collate,
)


###############################################################################
# MODEL
###############################################################################

print()
print("Loading Qwen3.8-27B BF16...")

load_t0 = time.time()

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

print(
    "LOAD_SECONDS =",
    round(
        time.time() - load_t0,
        1,
    ),
)

if hasattr(
    model.config,
    "use_cache",
):
    model.config.use_cache = False

model.gradient_checkpointing_enable()

if hasattr(
    model,
    "enable_input_require_grads",
):
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
    for name, module
    in model.named_modules()
}

targets = [
    x
    for x in wanted
    if x in available
]

print("LoRA targets =", targets)

if not targets:
    raise RuntimeError(
        "No Qwen LoRA targets found"
    )

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=targets,
)

model = get_peft_model(
    model,
    lora,
)

model.print_trainable_parameters()


###############################################################################
# METRICS
###############################################################################

def rankdata(a):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    order = np.argsort(a)
    ranks = np.empty(
        len(a),
        dtype=np.float64,
    )

    i = 0

    while i < len(a):
        j = i + 1

        while (
            j < len(a)
            and a[order[j]]
            == a[order[i]]
        ):
            j += 1

        r = (
            i + j - 1
        ) / 2.0

        for k in range(i, j):
            ranks[order[k]] = r

        i = j

    return ranks


def spearman(x, y):
    rx = rankdata(x)
    ry = rankdata(y)

    if (
        np.std(rx) < 1e-12
        or np.std(ry) < 1e-12
    ):
        return None

    return float(
        np.corrcoef(
            rx,
            ry,
        )[0, 1]
    )


@torch.inference_mode()
def evaluate(groups, name):
    model.eval()

    selected = []
    random_gain = []
    oracle = []
    top1 = []
    top2 = []
    rhos = []
    pair_ok = []
    pair_n = []

    done = 0

    for start in range(
        0,
        len(groups),
        EVAL_BATCH,
    ):
        batch = groups[
            start:
            start + EVAL_BATCH
        ]

        texts = [
            chat_prompt(
                str(x["prompt"])
            )
            for x in batch
        ]

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            add_special_tokens=False,
            return_tensors="pt",
        )

        enc = {
            k: v.to("cuda")
            for k, v in enc.items()
        }

        out = model(
            **enc
        )

        next_logits = (
            out.logits[:, -1, :]
            .float()
            .cpu()
            .numpy()
        )

        for bi, g in enumerate(batch):
            gains = np.asarray(
                g["gains"],
                dtype=np.float64,
            )

            n = len(gains)

            scores = np.asarray([
                next_logits[
                    bi,
                    label_ids[LETTERS[j]],
                ]
                for j in range(n)
            ])

            pick = int(
                np.argmax(scores)
            )

            selected.append(
                float(gains[pick])
            )

            random_gain.append(
                float(np.mean(gains))
            )

            best = float(
                np.max(gains)
            )

            oracle.append(best)

            best_idx = {
                i
                for i, z
                in enumerate(gains)
                if abs(z - best) < 1e-12
            }

            score_order = list(
                np.argsort(scores)[::-1]
            )

            top1.append(
                float(
                    pick in best_idx
                )
            )

            top2.append(
                float(
                    any(
                        i in best_idx
                        for i
                        in score_order[:2]
                    )
                )
            )

            rho = spearman(
                scores,
                gains,
            )

            if rho is not None:
                rhos.append(rho)

            ok = 0
            tot = 0

            for i in range(n):
                for j in range(i + 1, n):
                    dg = (
                        gains[i]
                        - gains[j]
                    )

                    if abs(dg) < 1e-12:
                        continue

                    ds = (
                        scores[i]
                        - scores[j]
                    )

                    ok += int(
                        dg * ds > 0
                    )

                    tot += 1

            pair_ok.append(ok)
            pair_n.append(tot)

        done += len(batch)

        if (
            done % 100 == 0
            or done == len(groups)
        ):
            print(
                f"[{name}] "
                f"{done}/{len(groups)}",
                flush=True,
            )

    ms = float(np.mean(selected))
    mr = float(np.mean(random_gain))
    mo = float(np.mean(oracle))

    result = {
        "states":
            len(groups),

        "mean_selected_gain":
            ms,

        "mean_random_gain":
            mr,

        "mean_oracle_gain":
            mo,

        "mean_regret":
            mo - ms,

        "oracle_top1_hit":
            float(np.mean(top1)),

        "oracle_top2_hit":
            float(np.mean(top2)),

        "mean_spearman":
            (
                float(np.mean(rhos))
                if rhos
                else None
            ),

        "mean_pair_accuracy":
            (
                float(
                    sum(pair_ok)
                    / max(1, sum(pair_n))
                )
            ),

        "oracle_signal_captured":
            (
                (ms - mr)
                / (mo - mr)
                if mo > mr + 1e-12
                else None
            ),
    }

    print()
    print(
        name,
        json.dumps(
            result,
            indent=2,
        ),
        flush=True,
    )

    model.train()

    return result


###############################################################################
# ZERO-SHOT BASELINE
###############################################################################

history = []

print()
print("=" * 100)
print("EPOCH 0 — ZERO-SHOT 27B")
print("=" * 100)

base_val = evaluate(
    val_groups,
    "VAL_BASE",
)

history.append({
    "epoch": 0,
    "train_loss": None,
    "val": base_val,
})


def clone_adapter():
    s = get_peft_model_state_dict(
        model
    )

    return {
        k:
            v.detach()
            .cpu()
            .clone()
        for k, v
        in s.items()
    }


best_regret = float(
    base_val["mean_regret"]
)

best_epoch = 0
best_state = clone_adapter()

print(
    "[BEST] epoch=0 "
    f"regret={best_regret:.6f}",
    flush=True,
)


###############################################################################
# TRAIN
###############################################################################

trainable = [
    p
    for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable,
    lr=LR,
    weight_decay=0.01,
)

possible_micro = (
    len(loader) * EPOCHS
)

micro_budget = (
    min(
        possible_micro,
        MAX_MICRO,
    )
    if MAX_MICRO > 0
    else possible_micro
)

total_opt_steps = max(
    1,
    math.ceil(
        micro_budget
        / GRAD_ACC
    ),
)

warmup = max(
    10,
    int(
        0.03
        * total_opt_steps
    ),
)

scheduler = (
    get_cosine_schedule_with_warmup(
        optimizer,
        warmup,
        total_opt_steps,
    )
)

print()
print(
    "micro_budget =",
    micro_budget,
)

print(
    "optimizer_steps_target =",
    total_opt_steps,
)

print(
    "warmup_steps =",
    warmup,
)


torch.cuda.reset_peak_memory_stats()

optimizer.zero_grad(
    set_to_none=True
)

global_micro = 0
opt_steps = 0
non_improve = 0
train_start = time.time()

stop_all = False


for epoch in range(
    1,
    EPOCHS + 1,
):
    model.train()

    epoch_loss = 0.0
    epoch_n = 0
    accum = 0

    for batch in loader:
        batch = {
            k: v.to("cuda")
            for k, v in batch.items()
        }

        out = model(
            input_ids=
                batch["input_ids"],

            attention_mask=
                batch["attention_mask"],

            labels=
                batch["labels"],
        )

        raw_loss = out.loss

        (
            raw_loss
            / GRAD_ACC
        ).backward()

        epoch_loss += float(
            raw_loss.detach().cpu()
        )

        epoch_n += 1
        global_micro += 1
        accum += 1

        if accum >= GRAD_ACC:
            torch.nn.utils.clip_grad_norm_(
                trainable,
                1.0,
            )

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            opt_steps += 1
            accum = 0

        if global_micro % 50 == 0:
            elapsed = (
                time.time()
                - train_start
            ) / 3600.0

            print(
                f"epoch={epoch} "
                f"micro={global_micro}/"
                f"{micro_budget} "
                f"opt={opt_steps}/"
                f"{total_opt_steps} "
                f"loss="
                f"{epoch_loss/max(1,epoch_n):.4f} "
                f"hours={elapsed:.2f}",
                flush=True,
            )

        if (
            MAX_MICRO > 0
            and global_micro >= MAX_MICRO
        ):
            stop_all = True
            break

    if accum > 0:
        torch.nn.utils.clip_grad_norm_(
            trainable,
            1.0,
        )

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        opt_steps += 1

    tag = (
        f"epoch_{epoch:02d}"
        + (
            "_partial"
            if stop_all
            else ""
        )
    )

    ckpt = (
        OUT
        / "checkpoints"
        / tag
    )

    ckpt.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        ckpt
    )

    val = evaluate(
        val_groups,
        f"VAL_{tag}",
    )

    mean_loss = (
        epoch_loss
        / max(1, epoch_n)
    )

    history.append({
        "epoch": epoch,
        "partial": stop_all,
        "micro_seen": global_micro,
        "train_loss": mean_loss,
        "val": val,
        "elapsed_hours":
            (
                time.time()
                - train_start
            )
            / 3600.0,
    })

    dump(
        OUT / "training_history.json",
        history,
    )

    regret = float(
        val["mean_regret"]
    )

    if regret < best_regret - 1e-6:
        best_regret = regret
        best_epoch = epoch
        best_state = clone_adapter()
        non_improve = 0

        print(
            f"[BEST] epoch={epoch} "
            f"regret={regret:.6f}",
            flush=True,
        )

    else:
        non_improve += 1

        print(
            f"[NO IMPROVEMENT] "
            f"{non_improve}/{PATIENCE}",
            flush=True,
        )

    if stop_all:
        print(
            "MAX_MICRO reached",
            flush=True,
        )
        break

    if non_improve >= PATIENCE:
        print(
            "EARLY STOP",
            flush=True,
        )
        break


###############################################################################
# RESTORE BEST + FINAL EVAL
###############################################################################

set_peft_model_state_dict(
    model,
    best_state,
)

best_dir = (
    OUT / "best_adapter"
)

best_dir.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(
    best_dir
)

tokenizer.save_pretrained(
    best_dir
)

print()
print("=" * 100)
print(
    "FINAL BEST EPOCH =",
    best_epoch,
)
print("=" * 100)

final_val = evaluate(
    val_groups,
    "VAL_FINAL_BEST",
)

challenge = evaluate(
    challenge_groups,
    "CHALLENGE_FINAL_BEST",
)

runtime_hours = (
    time.time()
    - train_start
) / 3600.0

summary = {
    "version":
        "q38_selector_v4_listwise_v1",

    "model":
        MODEL,

    "best_epoch":
        best_epoch,

    "selection_metric":
        "VAL mean_regret",

    "training_objective":
        "listwise whole-pool best-candidate "
        "classification via one-token LM loss",

    "training_filter":
        json.loads(
            (
                DATA
                / "summary.json"
            ).read_text()
        ),

    "train_examples":
        len(train_rows),

    "val":
        final_val,

    "challenge":
        challenge,

    "epoch0_zero_shot_val":
        base_val,

    "global_micro":
        global_micro,

    "optimizer_steps":
        opt_steps,

    "runtime_hours":
        runtime_hours,

    "peak_gpu_gib":
        (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        ),
}

dump(
    OUT / "summary.json",
    summary,
)

dump(
    OUT / "training_history.json",
    history,
)

print()
print(json.dumps(
    summary,
    indent=2,
))

print()
print("TRAINING COMPLETE")
