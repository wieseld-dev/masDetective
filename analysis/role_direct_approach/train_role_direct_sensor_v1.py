#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

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

from analysis.role_direct_approach.generate_direct_panel_v1 import (
    DESCRIPTIONS,
)


DATA_ROOT = Path(
    "artifacts/role_direct_matched_cf_v1"
)

OUT_ROOT = Path(
    "artifacts/role_direct_sensor_v1"
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def sid(x):
    return str(
        x.get(
            "system_id",
            x.get("session_id", ""),
        )
    )


def render_text(
    role,
    query,
    answer,
):
    return f"""You are detecting one candidate specialist capability inside a black-box multi-agent application.

TARGET CAPABILITY:
{role}

TARGET CAPABILITY DESCRIPTION:
{DESCRIPTIONS[role]}

BENIGN DIRECT PROBE:
{query}

VISIBLE FINAL RESPONSE:
{answer}

TASK:
Estimate whether the TARGET CAPABILITY was installed in the hidden application when this response was produced.

Use the behavioral evidence in the visible response relative to the probe.
Do not infer presence merely because the probe topic matches the target capability.
"""


def build_examples():

    candidates = read_jsonl(
        DATA_ROOT
        / "candidates_dev_round_01.jsonl"
    )

    responses = read_jsonl(
        DATA_ROOT
        / "responses_dev_round_01.jsonl"
    )

    assert len(candidates) == 600
    assert len(responses) == 600

    rb = {
        (
            sid(x),
            str(x["query"]),
        ): x
        for x in responses
    }

    grouped = defaultdict(dict)

    for c in candidates:

        system = str(
            c["session_id"]
        )

        query = str(
            c["query"]
        )

        r = rb[
            (
                system,
                query,
            )
        ]

        role = str(
            c["target_role"]
        )

        condition = str(
            c["condition"]
        )

        key = (
            role,
            str(c["pair_id"]),
            int(c["direct_probe_rank"]),
        )

        grouped[key][
            condition
        ] = {
            "role":
                role,

            "pair_id":
                str(
                    c["pair_id"]
                ),

            "pair_index":
                int(
                    c["pair_index"]
                ),

            "probe_rank":
                int(
                    c["direct_probe_rank"]
                ),

            "strategy":
                str(
                    c["probe_strategy"]
                ),

            "query":
                query,

            "answer":
                str(
                    r[
                        "final_output"
                    ]
                ),

            "system_id":
                system,
        }

    examples = []

    for key, d in grouped.items():

        assert set(d) == {
            "TARGET_PRESENT",
            "TARGET_ABSENT",
        }, (
            key,
            d.keys(),
        )

        pos = d[
            "TARGET_PRESENT"
        ]

        neg = d[
            "TARGET_ABSENT"
        ]

        assert (
            pos["role"]
            == neg["role"]
        )

        assert (
            pos["query"]
            == neg["query"]
        )

        assert (
            pos["pair_id"]
            == neg["pair_id"]
        )

        assert (
            pos["pair_index"]
            == neg["pair_index"]
        )

        examples.append({
            "role":
                pos["role"],

            "pair_id":
                pos["pair_id"],

            "pair_index":
                pos["pair_index"],

            "probe_rank":
                pos["probe_rank"],

            "strategy":
                pos["strategy"],

            "query":
                pos["query"],

            "positive_system":
                pos["system_id"],

            "negative_system":
                neg["system_id"],

            "positive_answer":
                pos["answer"],

            "negative_answer":
                neg["answer"],

            "positive_text":
                render_text(
                    pos["role"],
                    pos["query"],
                    pos["answer"],
                ),

            "negative_text":
                render_text(
                    neg["role"],
                    neg["query"],
                    neg["answer"],
                ),
        })

    assert len(examples) == 300

    by_pair = defaultdict(list)

    for x in examples:
        by_pair[
            (
                x["role"],
                x["pair_id"],
            )
        ].append(x)

    assert len(by_pair) == 100

    assert all(
        len(v) == 3
        for v in by_pair.values()
    )

    return examples


def win(a, b):

    if a > b:
        return 1.0

    if a < b:
        return 0.0

    return 0.5


def score_examples(
    model,
    tokenizer,
    examples,
    batch_size=8,
    max_length=1024,
):

    model.eval()

    texts = []

    metadata = []

    for x in examples:

        texts.append(
            x[
                "positive_text"
            ]
        )

        metadata.append(
            (
                x,
                1,
            )
        )

        texts.append(
            x[
                "negative_text"
            ]
        )

        metadata.append(
            (
                x,
                0,
            )
        )

    logits = []

    with torch.inference_mode():

        for start in range(
            0,
            len(texts),
            batch_size,
        ):

            batch = texts[
                start:
                start + batch_size
            ]

            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(
                "cuda"
            )

            out = model(
                **enc
            )

            z = (
                out.logits
                .view(-1)
                .float()
                .cpu()
                .tolist()
            )

            logits.extend(
                z
            )

    assert (
        len(logits)
        == len(metadata)
    )

    rows = []

    for (
        x,
        y,
    ), z in zip(
        metadata,
        logits,
    ):

        rows.append({
            "role":
                x["role"],

            "pair_id":
                x["pair_id"],

            "pair_index":
                x["pair_index"],

            "probe_rank":
                x["probe_rank"],

            "strategy":
                x["strategy"],

            "query":
                x["query"],

            "label":
                int(y),

            "logit":
                float(z),

            "prob":
                float(
                    torch.sigmoid(
                        torch.tensor(
                            z
                        )
                    ).item()
                ),
        })

    return rows


def metrics(rows):

    y = [
        x["label"]
        for x in rows
    ]

    z = [
        x["logit"]
        for x in rows
    ]

    global_auc = float(
        roc_auc_score(
            y,
            z,
        )
    )

    by_probe = defaultdict(dict)

    for x in rows:

        key = (
            x["role"],
            x["pair_id"],
            x["probe_rank"],
        )

        by_probe[key][
            x["label"]
        ] = x["logit"]

    probe_wins = []

    for key, d in by_probe.items():

        assert set(d) == {
            0,
            1,
        }

        probe_wins.append(
            win(
                d[1],
                d[0],
            )
        )

    by_pair = defaultdict(list)

    for (
        role,
        pair_id,
        probe_rank,
    ), d in by_probe.items():

        by_pair[
            (
                role,
                pair_id,
            )
        ].append(
            (
                probe_rank,
                d[1],
                d[0],
            )
        )

    pair_wins = []

    for key, rr in by_pair.items():

        rr = sorted(
            rr
        )

        assert len(rr) == 3

        p = mean(
            x[1]
            for x in rr
        )

        n = mean(
            x[2]
            for x in rr
        )

        pair_wins.append(
            win(
                p,
                n,
            )
        )

    return {
        "global_auc":
            global_auc,

        "matched_probe_win":
            float(
                mean(
                    probe_wins
                )
            ),

        "matched_pair3_win":
            float(
                mean(
                    pair_wins
                )
            ),
    }


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--fold",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20261008,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    ap.add_argument(
        "--batch-pairs",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--grad-accum",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--pair-weight",
        type=float,
        default=0.5,
    )

    args = ap.parse_args()

    fold = int(
        args.fold
    )

    if fold not in {
        1,
        2,
        3,
        4,
        5,
    }:
        raise ValueError(
            "fold must be 1..5"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    random.seed(
        args.seed + fold
    )

    np.random.seed(
        args.seed + fold
    )

    torch.manual_seed(
        args.seed + fold
    )

    torch.cuda.manual_seed_all(
        args.seed + fold
    )

    outdir = (
        OUT_ROOT
        / f"fold_{fold}"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples = build_examples()

    train = [
        x
        for x in examples
        if int(
            x["pair_index"]
        )
        != fold
    ]

    test = [
        x
        for x in examples
        if int(
            x["pair_index"]
        )
        == fold
    ]

    assert len(train) == 240
    assert len(test) == 60

    assert {
        x["pair_index"]
        for x in train
    } == (
        {
            1,
            2,
            3,
            4,
            5,
        }
        - {
            fold
        }
    )

    assert {
        x["pair_index"]
        for x in test
    } == {
        fold
    }

    # Exact system-pair isolation.
    train_pairs = {
        (
            x["role"],
            x["pair_id"],
        )
        for x in train
    }

    test_pairs = {
        (
            x["role"],
            x["pair_id"],
        )
        for x in test
    }

    assert not (
        train_pairs
        & test_pairs
    )

    print(
        "=" * 95,
        flush=True,
    )

    print(
        "ROLE DIRECT SENSOR V1",
        flush=True,
    )

    print(
        f"FOLD={fold}",
        flush=True,
    )

    print(
        "=" * 95,
        flush=True,
    )

    print(
        "TRAIN matched query pairs =",
        len(train),
        flush=True,
    )

    print(
        "TRAIN rows =",
        2 * len(train),
        flush=True,
    )

    print(
        "TEST matched query pairs =",
        len(test),
        flush=True,
    )

    print(
        "TEST rows =",
        2 * len(test),
        flush=True,
    )

    print(
        "TEST pair_index =",
        fold,
        flush=True,
    )

    print(
        "BASE MODEL =",
        BASE_MODEL,
        flush=True,
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL,
            local_files_only=True,
        )
    )

    if (
        tokenizer.pad_token_id
        is None
    ):
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = (
        "right"
    )

    tokenizer.truncation_side = (
        "right"
    )

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
            device_map={
                "": 0
            },
            dtype=torch.bfloat16,
            local_files_only=True,
        )
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.config.use_cache = False

    model = (
        prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
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
        modules_to_save=[
            "score",
        ],
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

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(
            args.lr
        ),
        weight_decay=0.01,
    )

    total_batches = (
        math.ceil(
            len(train)
            / args.batch_pairs
        )
    )

    print(
        "BATCHES/EPOCH =",
        total_batches,
        flush=True,
    )

    print(
        "EPOCHS =",
        args.epochs,
        flush=True,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        rng = random.Random(
            args.seed
            + fold * 1000
            + epoch
        )

        order = list(
            range(
                len(train)
            )
        )

        rng.shuffle(
            order
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        losses = []
        bces = []
        ranks = []

        for step, start in enumerate(
            range(
                0,
                len(order),
                args.batch_pairs,
            ),
            1,
        ):

            idx = order[
                start:
                start
                + args.batch_pairs
            ]

            batch = [
                train[i]
                for i in idx
            ]

            b = len(batch)

            texts = (
                [
                    x[
                        "positive_text"
                    ]
                    for x in batch
                ]
                +
                [
                    x[
                        "negative_text"
                    ]
                    for x in batch
                ]
            )

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(
                "cuda"
            )

            out = model(
                **enc
            )

            logits = (
                out.logits
                .view(-1)
                .float()
            )

            pos = logits[
                :b
            ]

            neg = logits[
                b:
            ]

            labels = torch.cat(
                [
                    torch.ones_like(
                        pos
                    ),
                    torch.zeros_like(
                        neg
                    ),
                ]
            )

            bce = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                )
            )

            rank = (
                F.softplus(
                    -(
                        pos
                        - neg
                    )
                )
                .mean()
            )

            loss = (
                bce
                + float(
                    args.pair_weight
                )
                * rank
            )

            (
                loss
                / args.grad_accum
            ).backward()

            if (
                step
                % args.grad_accum
                == 0
                or step
                == total_batches
            ):

                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    1.0,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            losses.append(
                float(
                    loss.detach().cpu()
                )
            )

            bces.append(
                float(
                    bce.detach().cpu()
                )
            )

            ranks.append(
                float(
                    rank.detach().cpu()
                )
            )

            if (
                step % 20 == 0
                or step
                == total_batches
            ):

                print(
                    f"[epoch {epoch}] "
                    f"{step}/{total_batches} "
                    f"loss={mean(losses[-20:]):.4f} "
                    f"bce={mean(bces[-20:]):.4f} "
                    f"rank={mean(ranks[-20:]):.4f}",
                    flush=True,
                )

        print(
            f"EPOCH {epoch} DONE "
            f"loss={mean(losses):.4f}",
            flush=True,
        )

    adapter_dir = (
        outdir
        / "adapter"
    )

    model.save_pretrained(
        adapter_dir
    )

    tokenizer.save_pretrained(
        adapter_dir
    )

    # Test only AFTER training is fully complete.
    test_rows = score_examples(
        model,
        tokenizer,
        test,
        batch_size=8,
        max_length=args.max_length,
    )

    m = metrics(
        test_rows
    )

    pred_path = (
        outdir
        / "test_predictions.jsonl"
    )

    with pred_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for x in test_rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "version":
            "role_direct_sensor_v1",

        "fold":
            fold,

        "test_pair_index":
            fold,

        "base_model":
            BASE_MODEL,

        "epochs":
            args.epochs,

        "train_matched_queries":
            len(train),

        "train_rows":
            2 * len(train),

        "test_matched_queries":
            len(test),

        "test_rows":
            len(test_rows),

        "objective":
            (
                "BCE present/absent + "
                "matched pairwise ranking"
            ),

        "input":
            (
                "target role + target description "
                "+ direct query + visible final response"
            ),

        "test":
            m,
    }

    (
        outdir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 95,
        flush=True,
    )

    print(
        "FOLD TEST RESULT",
        flush=True,
    )

    print(
        "=" * 95,
        flush=True,
    )

    print(
        json.dumps(
            m,
            indent=2,
        ),
        flush=True,
    )

    print(
        "ROLE_DIRECT_SENSOR_FOLD_PASS",
        flush=True,
    )

    del model

    gc.collect()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
