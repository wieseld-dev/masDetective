#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SRC = Path("artifacts/v11_probe_discovery.jsonl")
OUT = Path("artifacts/generic_neural_ranker_v1")
OUT.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

SEED = 20260927

EPOCHS = 3
LR = 5e-5

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

MAX_LENGTH = 1024

PAIR_BATCH_SIZE = 2
GRAD_ACCUM = 8

PAIR_MARGIN = 1.0
BCE_WEIGHT = 0.5
PAIR_WEIGHT = 1.0

EARLY_STOP_PATIENCE = 2


from v11.role_universe import (
    TRAIN_ROLES,
    DEV_UNSEEN_ROLES,
    ROLE_DESCRIPTIONS,
)


TRAIN_ROLES = tuple(map(str, TRAIN_ROLES))
DEV_ROLES = tuple(map(str, DEV_UNSEEN_ROLES))


# ============================================================
# 1. DETERMINISTIC ROLE-LEVEL INTERNAL SPLIT
#
# Absolutely no DEV role is used for model selection.
# Four of the original 20 TRAIN roles are selected by SHA256.
# ============================================================

role_hash_order = sorted(
    TRAIN_ROLES,
    key=lambda r:
        hashlib.sha256(
            r.encode("utf-8")
        ).hexdigest(),
)

INTERNAL_VAL_ROLES = tuple(
    role_hash_order[:4]
)

NEURAL_TRAIN_ROLES = tuple(
    r
    for r in TRAIN_ROLES
    if r not in INTERNAL_VAL_ROLES
)


print("=" * 120)
print("GENERIC NEURAL FINGERPRINT RANKER V1")
print("=" * 120)

print("base model:", BASE_MODEL)

print(
    "neural train roles:",
    len(NEURAL_TRAIN_ROLES),
    list(NEURAL_TRAIN_ROLES),
)

print(
    "internal unseen validation roles:",
    len(INTERNAL_VAL_ROLES),
    list(INTERNAL_VAL_ROLES),
)

print(
    "external DEV roles — NEVER used for checkpoint selection:",
    len(DEV_ROLES),
    list(DEV_ROLES),
)


assert not (
    set(NEURAL_TRAIN_ROLES)
    & set(INTERNAL_VAL_ROLES)
)

assert not (
    set(TRAIN_ROLES)
    & set(DEV_ROLES)
)


# ============================================================
# 2. LOAD MATCHED COUNTERFACTUAL BRANCHES
#
# V11 counterfactual semantics:
#
#   PLUS  = target role installed
#   SHAM  = target role still installed; unrelated composition change
#   MINUS = target role removed/replaced
#
# Therefore each V11 branch yields two matched ranking pairs:
#
#   PLUS > MINUS
#   SHAM > MINUS
#
# This is installed-role supervision, not selected-role supervision.
# ============================================================

def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


raw = read_jsonl(
    SRC
)

branches = []
seen = set()
audit = defaultdict(int)

all_allowed = (
    set(TRAIN_ROLES)
    | set(DEV_ROLES)
)


for rec_i, rec in enumerate(raw):
    target = str(
        rec.get(
            "target_role",
            "",
        )
    ).strip()

    phase = str(
        rec.get(
            "phase",
            "",
        )
    ).strip()

    query = str(
        rec.get(
            "query",
            "",
        )
    ).strip()

    if target not in all_allowed:
        audit["unknown_or_unused_target"] += 1
        continue

    if phase not in {
        "train",
        "holdout",
    }:
        audit["unused_phase"] += 1
        continue

    if not query:
        audit["missing_query"] += 1
        continue


    bb = list(
        rec.get(
            "branches_training_only"
        )
        or []
    )

    bb += list(
        rec.get(
            "confirmation_branches_training_only"
        )
        or []
    )


    for branch_i, b in enumerate(bb):
        plus = str(
            b.get(
                "plus_output",
                "",
            )
            or ""
        ).strip()

        sham = str(
            b.get(
                "sham_output",
                "",
            )
            or ""
        ).strip()

        minus = str(
            b.get(
                "minus_output",
                "",
            )
            or ""
        ).strip()


        if not (
            plus
            and sham
            and minus
        ):
            audit[
                "missing_output"
            ] += 1
            continue


        base_key = (
            target,
            query,
            plus,
            sham,
            minus,
        )

        if base_key in seen:
            audit[
                "duplicate_branch"
            ] += 1
            continue

        seen.add(
            base_key
        )


        split = (
            "external_dev"
            if target in DEV_ROLES
            else (
                "internal_val"
                if target
                in INTERNAL_VAL_ROLES
                else
                "train"
            )
        )


        branch_id = (
            f"{phase}:"
            f"{rec_i}:"
            f"{branch_i}"
        )


        # Same target role, same benign query.
        # Only the hidden MAS composition changes.
        for pos_kind, pos_output in (
            (
                "plus",
                plus,
            ),
            (
                "sham",
                sham,
            ),
        ):
            branches.append({
                "pair_id":
                    (
                        f"{branch_id}:"
                        f"{pos_kind}:minus"
                    ),

                "split":
                    split,

                "target_role":
                    target,

                "query":
                    query,

                "positive":
                    pos_output,

                "negative":
                    minus,

                "positive_kind":
                    pos_kind,

                "negative_kind":
                    "minus",
            })

            audit[
                f"pair:{split}"
            ] += 1


print()
print("===== DATA AUDIT =====")

print(
    json.dumps(
        dict(audit),
        indent=2,
        sort_keys=True,
    )
)


train_pairs = [
    x
    for x in branches
    if x["split"] == "train"
]

val_pairs = [
    x
    for x in branches
    if x["split"] == "internal_val"
]

dev_pairs = [
    x
    for x in branches
    if x["split"] == "external_dev"
]


print()
print(
    "train pairs:",
    len(train_pairs),
)

print(
    "internal-val pairs:",
    len(val_pairs),
)

print(
    "external-dev pairs:",
    len(dev_pairs),
)


for split_name, rr, roles in (
    (
        "TRAIN",
        train_pairs,
        NEURAL_TRAIN_ROLES,
    ),
    (
        "INTERNAL_VAL",
        val_pairs,
        INTERNAL_VAL_ROLES,
    ),
    (
        "EXTERNAL_DEV",
        dev_pairs,
        DEV_ROLES,
    ),
):
    print()
    print(split_name)

    for role in roles:
        n = sum(
            x["target_role"] == role
            for x in rr
        )

        print(
            f"  {role:<28} pairs={n}"
        )


if not train_pairs:
    raise RuntimeError(
        "No training pairs"
    )

if not val_pairs:
    raise RuntimeError(
        "No internal validation pairs"
    )

if not dev_pairs:
    raise RuntimeError(
        "No external DEV pairs"
    )


# ============================================================
# 3. INPUT TEXT
# ============================================================

def model_text(
    role,
    query,
    response,
):
    desc = ROLE_DESCRIPTIONS.get(
        role,
        role,
    )

    return (
        "[TASK]\n"
        "Estimate whether the candidate specialist capability is actually "
        "present in the hidden multi-agent system. "
        "The user query is deliberately related to the candidate capability, "
        "so topic similarity alone is NOT evidence of presence. "
        "Use behavioral evidence in how the final response transforms, "
        "prioritizes, verifies, constrains, or reasons about the request.\n\n"

        f"[CANDIDATE ROLE]\n{role}\n\n"

        f"[PUBLIC ROLE DESCRIPTION]\n{desc}\n\n"

        f"[BENIGN PROBE]\n{query}\n\n"

        f"[VISIBLE FINAL RESPONSE]\n{response}\n\n"

        "[DECISION]\n"
        "Is the candidate specialist capability present?"
    )


@dataclass(frozen=True)
class PairExample:
    role: str
    pos_text: str
    neg_text: str
    pair_id: str


def convert_pairs(rows):
    return [
        PairExample(
            role=x[
                "target_role"
            ],

            pos_text=model_text(
                x[
                    "target_role"
                ],
                x[
                    "query"
                ],
                x[
                    "positive"
                ],
            ),

            neg_text=model_text(
                x[
                    "target_role"
                ],
                x[
                    "query"
                ],
                x[
                    "negative"
                ],
            ),

            pair_id=x[
                "pair_id"
            ],
        )
        for x in rows
    ]


train_data = convert_pairs(
    train_pairs
)

val_data = convert_pairs(
    val_pairs
)

dev_data = convert_pairs(
    dev_pairs
)


# ============================================================
# 4. TORCH DATA
# ============================================================

class PairDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(
            self.rows
        )

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    return {
        "role":
            [
                x.role
                for x in batch
            ],

        "pos_text":
            [
                x.pos_text
                for x in batch
            ],

        "neg_text":
            [
                x.neg_text
                for x in batch
            ],

        "pair_id":
            [
                x.pair_id
                for x in batch
            ],
    }


# ============================================================
# 5. MODEL
# ============================================================

def build_model():
    import torch

    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"


    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )


    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


    base = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL,
            num_labels=1,
            quantization_config=quant,
            device_map={
                "": 0
            },
            torch_dtype=dtype,
        )
    )


    base.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    base.config.use_cache = False


    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
    )


    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        modules_to_save=[
            "score",
        ],
    )


    model = get_peft_model(
        base,
        lora,
    )


    print()
    print(
        "===== TRAINABLE PARAMETERS ====="
    )

    model.print_trainable_parameters()


    return (
        tokenizer,
        model,
        dtype,
    )


def logits(
    model,
    tokenizer,
    texts,
    device,
):
    x = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(
        device
    )

    return (
        model(
            **x
        )
        .logits[
            :,
            0
        ]
        .float()
    )


# ============================================================
# 6. METRICS
# ============================================================

def safe_auc(
    y,
    s,
):
    from sklearn.metrics import (
        roc_auc_score,
    )

    y = np.asarray(
        y,
        dtype=int,
    )

    s = np.asarray(
        s,
        dtype=float,
    )

    if len(
        np.unique(y)
    ) != 2:
        return None

    return float(
        roc_auc_score(
            y,
            s,
        )
    )


def evaluate(
    model,
    tokenizer,
    rows,
    device,
    dtype,
):
    import torch

    from torch.utils.data import (
        DataLoader,
    )


    loader = DataLoader(
        PairDataset(rows),
        batch_size=4,
        shuffle=False,
        collate_fn=collate,
    )


    by_role = defaultdict(
        lambda: {
            "pos": [],
            "neg": [],
            "margin": [],
        }
    )


    all_y = []
    all_s = []
    margins = []


    model.eval()


    with torch.no_grad():
        for b in loader:
            texts = (
                b["pos_text"]
                + b["neg_text"]
            )

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                z = logits(
                    model,
                    tokenizer,
                    texts,
                    device,
                )


            n = len(
                b["pos_text"]
            )

            zp = (
                z[:n]
                .cpu()
                .numpy()
            )

            zn = (
                z[n:]
                .cpu()
                .numpy()
            )


            for role, p, m in zip(
                b["role"],
                zp,
                zn,
            ):
                p = float(p)
                m = float(m)

                by_role[
                    role
                ][
                    "pos"
                ].append(
                    p
                )

                by_role[
                    role
                ][
                    "neg"
                ].append(
                    m
                )

                by_role[
                    role
                ][
                    "margin"
                ].append(
                    p - m
                )


            all_y.extend(
                [1] * n
                + [0] * n
            )

            all_s.extend(
                zp.tolist()
                + zn.tolist()
            )

            margins.extend(
                (
                    zp
                    - zn
                ).tolist()
            )


    per_role = {}


    for role in sorted(
        by_role
    ):
        d = by_role[
            role
        ]

        yy = (
            [1] * len(
                d["pos"]
            )
            + [0] * len(
                d["neg"]
            )
        )

        ss = (
            d["pos"]
            + d["neg"]
        )

        per_role[
            role
        ] = {
            "auc":
                safe_auc(
                    yy,
                    ss,
                ),

            "pair_accuracy":
                float(
                    np.mean(
                        np.asarray(
                            d["margin"]
                        )
                        > 0
                    )
                ),

            "mean_margin":
                float(
                    np.mean(
                        d["margin"]
                    )
                ),

            "n_pairs":
                len(
                    d["margin"]
                ),
        }


    role_aucs = [
        x["auc"]
        for x in per_role.values()
        if x["auc"] is not None
    ]

    role_pair = [
        x["pair_accuracy"]
        for x in per_role.values()
    ]


    out = {
        "global_auc":
            safe_auc(
                all_y,
                all_s,
            ),

        "mean_role_auc":
            float(
                np.mean(
                    role_aucs
                )
            ),

        "median_role_auc":
            float(
                np.median(
                    role_aucs
                )
            ),

        "pair_accuracy":
            float(
                np.mean(
                    np.asarray(
                        margins
                    )
                    > 0
                )
            ),

        "mean_role_pair_accuracy":
            float(
                np.mean(
                    role_pair
                )
            ),

        "mean_margin":
            float(
                np.mean(
                    margins
                )
            ),

        "per_role":
            per_role,
    }


    out[
        "selection_score"
    ] = (
        0.60
        * out[
            "mean_role_auc"
        ]
        + 0.40
        * out[
            "mean_role_pair_accuracy"
        ]
    )


    return out


# ============================================================
# 7. TRAINING
#
# Each forward contains matched POSITIVE and NEGATIVE responses.
#
# BCE and pairwise ranking use the SAME forward pass:
#
#   L = 0.5 * balanced_BCE
#       + 1.0 * softplus(margin - (s+ - s-))
#
# ============================================================

def train():
    import torch
    import torch.nn.functional as F

    from torch.utils.data import (
        DataLoader,
    )

    from transformers import (
        get_cosine_schedule_with_warmup,
    )


    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )


    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )


    device = torch.device(
        "cuda:0"
    )


    tokenizer, model, dtype = (
        build_model()
    )


    loader = DataLoader(
        PairDataset(
            train_data
        ),
        batch_size=PAIR_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )


    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]


    optimizer = torch.optim.AdamW(
        params,
        lr=LR,
        weight_decay=0.01,
    )


    optimizer_steps_per_epoch = (
        math.ceil(
            len(loader)
            / GRAD_ACCUM
        )
    )

    total_steps = (
        optimizer_steps_per_epoch
        * EPOCHS
    )

    warmup_steps = max(
        10,
        int(
            0.05
            * total_steps
        ),
    )


    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
        )
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            dtype
            == torch.float16
        ),
    )


    best_score = -1.0
    best_epoch = None
    bad_epochs = 0

    history = []


    print()
    print("===== TRAINING =====")

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

    print(
        "train pairs:",
        len(train_data),
    )

    print(
        "steps/epoch:",
        len(loader),
    )

    print(
        "optimizer steps/epoch:",
        optimizer_steps_per_epoch,
    )

    print(
        "planned epochs:",
        EPOCHS,
    )

    print(
        "max length:",
        MAX_LENGTH,
    )

    print(
        "pair batch:",
        PAIR_BATCH_SIZE,
    )

    print(
        "grad accum:",
        GRAD_ACCUM,
    )


    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        loss_sum = 0.0
        bce_sum = 0.0
        pair_sum = 0.0
        seen = 0


        for step, b in enumerate(
            loader,
            1,
        ):
            texts = (
                b["pos_text"]
                + b["neg_text"]
            )


            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                z = logits(
                    model,
                    tokenizer,
                    texts,
                    device,
                )


                n = len(
                    b["pos_text"]
                )

                zp = z[:n]
                zn = z[n:]


                zz = torch.cat(
                    [
                        zp,
                        zn,
                    ]
                )


                yy = torch.cat(
                    [
                        torch.ones_like(
                            zp
                        ),

                        torch.zeros_like(
                            zn
                        ),
                    ]
                )


                # Balanced because every batch has
                # exactly one positive and one negative
                # response per matched pair.
                bce = (
                    F.binary_cross_entropy_with_logits(
                        zz,
                        yy,
                    )
                )


                pair = (
                    F.softplus(
                        PAIR_MARGIN
                        - (
                            zp
                            - zn
                        )
                    )
                    .mean()
                )


                loss = (
                    BCE_WEIGHT
                    * bce
                    + PAIR_WEIGHT
                    * pair
                )


            scaler.scale(
                loss
                / GRAD_ACCUM
            ).backward()


            bs = n

            loss_sum += (
                float(
                    loss.detach().cpu()
                )
                * bs
            )

            bce_sum += (
                float(
                    bce.detach().cpu()
                )
                * bs
            )

            pair_sum += (
                float(
                    pair.detach().cpu()
                )
                * bs
            )

            seen += bs


            if (
                step % GRAD_ACCUM == 0
                or
                step == len(loader)
            ):
                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    params,
                    1.0,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                scheduler.step()


            if step % 100 == 0:
                print(
                    f"[epoch {epoch}] "
                    f"step={step}/{len(loader)} "
                    f"loss={loss_sum/max(seen,1):.4f} "
                    f"bce={bce_sum/max(seen,1):.4f} "
                    f"pair={pair_sum/max(seen,1):.4f}",
                    flush=True,
                )


        print()
        print(
            f"===== EPOCH {epoch} INTERNAL UNSEEN-ROLE VALIDATION ====="
        )


        val = evaluate(
            model,
            tokenizer,
            val_data,
            device,
            dtype,
        )


        print(
            json.dumps(
                val,
                indent=2,
                sort_keys=True,
            )
        )


        row = {
            "epoch":
                epoch,

            "train_loss":
                loss_sum
                / max(
                    seen,
                    1,
                ),

            "train_bce":
                bce_sum
                / max(
                    seen,
                    1,
                ),

            "train_pair":
                pair_sum
                / max(
                    seen,
                    1,
                ),

            "internal_val":
                val,
        }


        history.append(
            row
        )


        (
            OUT
            / "training_history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
            )
            + "\n"
        )


        last = (
            OUT
            / "last_adapter"
        )

        shutil.rmtree(
            last,
            ignore_errors=True,
        )

        model.save_pretrained(
            last
        )

        tokenizer.save_pretrained(
            last
        )


        if (
            val[
                "selection_score"
            ]
            > best_score
            + 1e-4
        ):
            best_score = (
                val[
                    "selection_score"
                ]
            )

            best_epoch = epoch

            bad_epochs = 0


            best = (
                OUT
                / "best_adapter"
            )

            shutil.rmtree(
                best,
                ignore_errors=True,
            )

            model.save_pretrained(
                best
            )

            tokenizer.save_pretrained(
                best
            )


            (
                OUT
                / "best_internal_val.json"
            ).write_text(
                json.dumps(
                    val,
                    indent=2,
                )
                + "\n"
            )


            print(
                f"[BEST] epoch={epoch} "
                f"selection={best_score:.4f}",
                flush=True,
            )

        else:
            bad_epochs += 1

            print(
                f"[NO IMPROVEMENT] "
                f"bad_epochs={bad_epochs}",
                flush=True,
            )


            if (
                bad_epochs
                >= EARLY_STOP_PATIENCE
            ):
                print(
                    "[EARLY STOP]",
                    flush=True,
                )
                break


    print()
    print(
        "BEST INTERNAL-VAL EPOCH:",
        best_epoch,
    )

    print(
        "BEST INTERNAL-VAL SCORE:",
        best_score,
    )


    # --------------------------------------------------------
    # FINAL EXTERNAL DEV EVALUATION
    #
    # Only now do we evaluate the six completely unseen
    # development roles.
    # --------------------------------------------------------

    del model

    torch.cuda.empty_cache()


    from peft import PeftModel

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )


    best_dir = (
        OUT
        / "best_adapter"
    )


    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            best_dir,
            use_fast=True,
        )
    )


    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )


    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


    base = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL,
            num_labels=1,
            quantization_config=quant,
            device_map={
                "": 0
            },
            torch_dtype=dtype,
        )
    )


    base.config.pad_token_id = (
        tokenizer.pad_token_id
    )


    model = PeftModel.from_pretrained(
        base,
        best_dir,
    )


    print()
    print("=" * 120)
    print(
        "FINAL EXTERNAL 6-ROLE DEV EVALUATION"
    )

    print(
        "THESE ROLES WERE NOT USED FOR TRAINING OR CHECKPOINT SELECTION"
    )

    print("=" * 120)


    dev = evaluate(
        model,
        tokenizer,
        dev_data,
        device,
        dtype,
    )


    print(
        json.dumps(
            dev,
            indent=2,
            sort_keys=True,
        )
    )


    summary = {
        "experiment":
            "generic_neural_ranker_v1",

        "base_model":
            BASE_MODEL,

        "training_objective":
            (
                "matched same-role same-query "
                "BCE + pairwise ranking"
            ),

        "neural_train_roles":
            list(
                NEURAL_TRAIN_ROLES
            ),

        "internal_val_roles":
            list(
                INTERNAL_VAL_ROLES
            ),

        "external_dev_roles":
            list(
                DEV_ROLES
            ),

        "train_pairs":
            len(
                train_data
            ),

        "internal_val_pairs":
            len(
                val_data
            ),

        "external_dev_pairs":
            len(
                dev_data
            ),

        "best_epoch":
            best_epoch,

        "best_internal_val_score":
            best_score,

        "external_dev":
            dev,
    }


    (
        OUT
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


    print()
    print(
        "wrote:",
        OUT / "summary.json"
    )


if __name__ == "__main__":
    train()
