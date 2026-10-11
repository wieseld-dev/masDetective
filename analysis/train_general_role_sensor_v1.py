#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

DATA = Path(
    "artifacts/general_role_sensor_trainset_v1"
)

TRAIN_PATH = DATA / "train.jsonl"
VAL_PATH = DATA / "val.jsonl"

OUT = Path(
    "artifacts/general_role_sensor_model_v1"
)

MAX_LENGTH = 1024
PAIR_BATCH_SIZE = 2
GRAD_ACCUM = 8

EPOCHS = 3

LR = 1.0e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05

PAIR_WEIGHT = 0.5
PAIR_MARGIN = 1.0

SEED = 20260929


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path):
    out = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                out.append(
                    json.loads(line)
                )

    return out


def format_input(
    x: dict,
    output_override=None,
):
    role = (
        str(x["target_role"])
        .replace("_", " ")
    )

    family = (
        str(x["target_family"])
        .replace("_", " ")
    )

    query = str(x["query"])

    if output_override is None:
        output = str(
            x["final_output"]
        )
    else:
        output = str(
            output_override
        )

    return (
        "[TARGET CAPABILITY]\n"
        f"{role}\n\n"

        "[TARGET FAMILY]\n"
        f"{family}\n\n"

        "[BENIGN USER PROBE]\n"
        f"{query}\n\n"

        "[VISIBLE FINAL MAS RESPONSE]\n"
        f"{output}\n\n"

        "[TASK]\n"
        "Predict whether the target capability "
        "is installed in the hidden system."
    )


def build_pairs(rows):
    groups = defaultdict(list)

    for x in rows:
        key = (
            str(x["target_role"]),
            int(x["pair_index"]),
            str(x["probe_id"]),
        )

        groups[key].append(x)

    pairs = []

    for key in sorted(groups):
        g = groups[key]

        if len(g) != 2:
            raise RuntimeError(
                f"bad pair {key}: {len(g)}"
            )

        plus = [
            x for x in g
            if int(x["label"]) == 1
        ]

        minus = [
            x for x in g
            if int(x["label"]) == 0
        ]

        if (
            len(plus) != 1
            or len(minus) != 1
        ):
            raise RuntimeError(
                f"bad labels for {key}"
            )

        p = plus[0]
        m = minus[0]

        if p["query"] != m["query"]:
            raise RuntimeError(
                f"query mismatch {key}"
            )

        pairs.append({
            "key": key,
            "plus": p,
            "minus": m,
        })

    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        x = self.pairs[i]

        return {
            "plus_text":
                format_input(
                    x["plus"]
                ),

            "minus_text":
                format_input(
                    x["minus"]
                ),

            "key":
                x["key"],
        }


def collate_pairs(batch):
    return {
        "plus_text": [
            x["plus_text"]
            for x in batch
        ],

        "minus_text": [
            x["minus_text"]
            for x in batch
        ],

        "key": [
            x["key"]
            for x in batch
        ],
    }


def make_model(
    model_name,
    trainable,
    adapter=None,
):
    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=
            torch.bfloat16,
    )

    base = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_name,
            num_labels=1,
            quantization_config=quant,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )
    )

    base.config.pad_token_id = (
        tok.pad_token_id
    )

    base.config.use_cache = False

    if adapter is not None:
        model = PeftModel.from_pretrained(
            base,
            str(adapter),
            is_trainable=trainable,
        )

        return tok, model

    if not trainable:
        raise RuntimeError(
            "fresh model requested "
            "with trainable=False"
        )

    base = prepare_model_for_kbit_training(
        base
    )

    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        modules_to_save=[
            "score",
        ],

        task_type=TaskType.SEQ_CLS,
        bias="none",
    )

    model = get_peft_model(
        base,
        lora,
    )

    model.gradient_checkpointing_enable()

    if hasattr(
        model,
        "enable_input_require_grads",
    ):
        model.enable_input_require_grads()

    return tok, model


def tokenize(tok, texts):
    x = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    return {
        k: v.cuda(
            non_blocking=True
        )
        for k, v in x.items()
    }


@torch.no_grad()
def predict_texts(
    model,
    tok,
    texts,
    batch_size=4,
):
    model.eval()

    scores = []

    for i in range(
        0,
        len(texts),
        batch_size,
    ):
        batch = texts[
            i:i + batch_size
        ]

        enc = tokenize(
            tok,
            batch,
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            out = model(**enc)

        z = (
            out.logits
            .float()
            .view(-1)
            .detach()
            .cpu()
            .numpy()
        )

        scores.extend(
            map(float, z)
        )

    return np.asarray(
        scores,
        dtype=np.float64,
    )


def auc_safe(y, z):
    y = np.asarray(y)

    if len(np.unique(y)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y,
            z,
        )
    )


def evaluate_scores(
    rows,
    logits,
):
    y = np.asarray(
        [
            int(x["label"])
            for x in rows
        ],
        dtype=np.int64,
    )

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    global_auc = auc_safe(
        y,
        logits,
    )

    by_role = defaultdict(list)

    for i, x in enumerate(rows):
        by_role[
            str(x["target_role"])
        ].append(i)

    role_auc = {}

    for role, idx in sorted(
        by_role.items()
    ):
        idx = np.asarray(
            idx,
            dtype=int,
        )

        role_auc[role] = auc_safe(
            y[idx],
            logits[idx],
        )

    macro_role_auc = float(
        np.mean(
            list(role_auc.values())
        )
    )

    by_source = defaultdict(list)

    for i, x in enumerate(rows):
        by_source[
            str(x["probe_source"])
        ].append(i)

    source_auc = {}

    for source, idx in sorted(
        by_source.items()
    ):
        idx = np.asarray(
            idx,
            dtype=int,
        )

        source_auc[source] = auc_safe(
            y[idx],
            logits[idx],
        )

    probs = (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    logits,
                    -40,
                    40,
                )
            )
        )
    )

    best_thr = None
    best_bal = -1.0

    for thr in np.linspace(
        0.02,
        0.98,
        97,
    ):
        pred = (
            probs >= thr
        ).astype(int)

        bal = float(
            balanced_accuracy_score(
                y,
                pred,
            )
        )

        if bal > best_bal:
            best_bal = bal
            best_thr = float(thr)

    return {
        "global_auc":
            global_auc,

        "macro_role_auc":
            macro_role_auc,

        "per_role_auc":
            role_auc,

        "probe_source_auc":
            source_auc,

        "best_balanced_accuracy":
            best_bal,

        "best_threshold":
            best_thr,
    }


def pair_metrics(
    pairs,
    plus_logits,
    minus_logits,
):
    plus_logits = np.asarray(
        plus_logits,
        dtype=np.float64,
    )

    minus_logits = np.asarray(
        minus_logits,
        dtype=np.float64,
    )

    d = (
        plus_logits
        - minus_logits
    )

    acc = np.mean(
        np.where(
            d > 0,
            1.0,
            np.where(
                d < 0,
                0.0,
                0.5,
            ),
        )
    )

    return {
        "pair_accuracy":
            float(acc),

        "mean_pair_margin":
            float(np.mean(d)),

        "median_pair_margin":
            float(np.median(d)),

        "positive_margin_rate":
            float(np.mean(d > 0)),
    }


def eval_full(
    model,
    tok,
    rows,
    pairs,
):
    texts = [
        format_input(x)
        for x in rows
    ]

    logits = predict_texts(
        model,
        tok,
        texts,
    )

    row_metrics = evaluate_scores(
        rows,
        logits,
    )

    logit_by_id = {
        str(x["execution_id"]):
            float(z)
        for x, z in zip(
            rows,
            logits,
        )
    }

    plus = []
    minus = []

    for pair in pairs:
        plus.append(
            logit_by_id[
                str(
                    pair["plus"]
                    ["execution_id"]
                )
            ]
        )

        minus.append(
            logit_by_id[
                str(
                    pair["minus"]
                    ["execution_id"]
                )
            ]
        )

    pm = pair_metrics(
        pairs,
        plus,
        minus,
    )

    return (
        {
            **row_metrics,
            **pm,
        },
        logits,
    )


def query_only_control(
    model,
    tok,
    rows,
):
    texts = [
        format_input(
            x,
            output_override="",
        )
        for x in rows
    ]

    logits = predict_texts(
        model,
        tok,
        texts,
    )

    return evaluate_scores(
        rows,
        logits,
    )


def shuffled_output_control(
    model,
    tok,
    rows,
    seed,
):
    rng = random.Random(seed)

    groups = defaultdict(list)

    for i, x in enumerate(rows):
        groups[
            (
                x["target_role"],
                x["probe_id"],
            )
        ].append(i)

    shuffled = [
        str(x["final_output"])
        for x in rows
    ]

    for idx in groups.values():
        outputs = [
            shuffled[i]
            for i in idx
        ]

        rng.shuffle(outputs)

        for i, o in zip(
            idx,
            outputs,
        ):
            shuffled[i] = o

    texts = [
        format_input(
            x,
            output_override=shuffled[i],
        )
        for i, x in enumerate(rows)
    ]

    logits = predict_texts(
        model,
        tok,
        texts,
    )

    return evaluate_scores(
        rows,
        logits,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out",
        default=str(OUT),
    )

    args = ap.parse_args()

    out = Path(args.out)

    if out.exists() and any(
        out.iterdir()
    ):
        raise SystemExit(
            f"{out} is non-empty; "
            "refusing overwrite"
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_all(SEED)

    train_rows = read_jsonl(
        TRAIN_PATH
    )

    val_rows = read_jsonl(
        VAL_PATH
    )

    assert len(train_rows) == 3240
    assert len(val_rows) == 648

    train_roles = {
        x["target_role"]
        for x in train_rows
    }

    val_roles = {
        x["target_role"]
        for x in val_rows
    }

    assert len(train_roles) == 60
    assert len(val_roles) == 12
    assert not (
        train_roles
        & val_roles
    )

    train_pairs = build_pairs(
        train_rows
    )

    val_pairs = build_pairs(
        val_rows
    )

    assert len(train_pairs) == 1620
    assert len(val_pairs) == 324

    print("=" * 100)
    print("GENERAL ROLE SENSOR V1")
    print("=" * 100)
    print("model:", MODEL_NAME)
    print(
        "TRAIN:",
        len(train_rows),
        "rows /",
        len(train_pairs),
        "pairs /",
        len(train_roles),
        "roles",
    )
    print(
        "VAL:",
        len(val_rows),
        "rows /",
        len(val_pairs),
        "pairs /",
        len(val_roles),
        "UNSEEN roles",
    )
    print(
        "objective: "
        "BCE(installed) + "
        "0.5 * pairwise PLUS>MINUS"
    )
    print(
        "internal TEST: NOT LOADED"
    )
    print()

    tok, model = make_model(
        MODEL_NAME,
        trainable=True,
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "trainable params:",
        trainable,
        "/",
        total,
    )

    ds = PairDataset(
        train_pairs
    )

    gen = torch.Generator()
    gen.manual_seed(SEED)

    loader = DataLoader(
        ds,
        batch_size=PAIR_BATCH_SIZE,
        shuffle=True,
        generator=gen,
        num_workers=0,
        collate_fn=collate_pairs,
        drop_last=False,
    )

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    opt = torch.optim.AdamW(
        params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = math.ceil(
        len(loader)
        / GRAD_ACCUM
    )

    total_steps = (
        steps_per_epoch
        * EPOCHS
    )

    warmup = int(
        WARMUP_RATIO
        * total_steps
    )

    sched = (
        get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )
    )

    history = []

    best_score = -1e9
    best_epoch = None

    best_dir = (
        out / "best_adapter"
    )

    global_step = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        opt.zero_grad(
            set_to_none=True
        )

        sum_loss = 0.0
        sum_bce = 0.0
        sum_rank = 0.0
        n_batches = 0

        for bi, batch in enumerate(
            loader,
            1,
        ):
            texts = (
                batch["plus_text"]
                + batch["minus_text"]
            )

            b = len(
                batch["plus_text"]
            )

            enc = tokenize(
                tok,
                texts,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                outm = model(**enc)

                logits = (
                    outm.logits
                    .float()
                    .view(-1)
                )

                plus = logits[:b]
                minus = logits[b:]

                labels = torch.cat([
                    torch.ones_like(
                        plus
                    ),
                    torch.zeros_like(
                        minus
                    ),
                ])

                bce = (
                    F.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                    )
                )

                rank = (
                    F.softplus(
                        PAIR_MARGIN
                        - (
                            plus
                            - minus
                        )
                    )
                    .mean()
                )

                loss = (
                    bce
                    + PAIR_WEIGHT
                    * rank
                )

                scaled = (
                    loss
                    / GRAD_ACCUM
                )

            scaled.backward()

            sum_loss += float(
                loss.detach().cpu()
            )

            sum_bce += float(
                bce.detach().cpu()
            )

            sum_rank += float(
                rank.detach().cpu()
            )

            n_batches += 1

            if (
                bi % GRAD_ACCUM == 0
                or bi == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    params,
                    1.0,
                )

                opt.step()
                sched.step()

                opt.zero_grad(
                    set_to_none=True
                )

                global_step += 1

        train_stats = {
            "loss":
                sum_loss / n_batches,

            "bce":
                sum_bce / n_batches,

            "pair_rank":
                sum_rank / n_batches,
        }

        val_metrics, _ = eval_full(
            model,
            tok,
            val_rows,
            val_pairs,
        )

        selection_score = (
            0.65
            * val_metrics[
                "macro_role_auc"
            ]
            +
            0.35
            * val_metrics[
                "pair_accuracy"
            ]
        )

        record = {
            "epoch":
                epoch,

            "train":
                train_stats,

            "val":
                val_metrics,

            "selection_score":
                float(
                    selection_score
                ),

            "global_step":
                global_step,
        }

        history.append(record)

        print()
        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )
        print(
            "train_loss =",
            f"{train_stats['loss']:.5f}",
            "bce =",
            f"{train_stats['bce']:.5f}",
            "rank =",
            f"{train_stats['pair_rank']:.5f}",
        )
        print(
            "VAL global AUC      =",
            f"{val_metrics['global_auc']:.4f}",
        )
        print(
            "VAL macro-role AUC  =",
            f"{val_metrics['macro_role_auc']:.4f}",
        )
        print(
            "VAL pair accuracy   =",
            f"{val_metrics['pair_accuracy']:.4f}",
        )
        print(
            "VAL pair margin     =",
            f"{val_metrics['mean_pair_margin']:+.4f}",
        )
        print(
            "VAL selection score =",
            f"{selection_score:.4f}",
        )

        (
            out
            / "history.json"
        ).write_text(
            json.dumps(
                history,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        if (
            selection_score
            > best_score
        ):
            best_score = float(
                selection_score
            )

            best_epoch = epoch

            shutil.rmtree(
                best_dir,
                ignore_errors=True,
            )

            model.save_pretrained(
                best_dir
            )

            tok.save_pretrained(
                best_dir
            )

            print(
                "NEW BEST CHECKPOINT:",
                best_dir,
            )

    print()
    print("=" * 100)
    print(
        "RELOAD BEST CHECKPOINT"
    )
    print("=" * 100)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    tok, model = make_model(
        MODEL_NAME,
        trainable=False,
        adapter=best_dir,
    )

    best_val, best_logits = eval_full(
        model,
        tok,
        val_rows,
        val_pairs,
    )

    print()
    print(
        "QUERY-ONLY CONTROL...",
        flush=True,
    )

    query_only = query_only_control(
        model,
        tok,
        val_rows,
    )

    print(
        "SHUFFLED-OUTPUT CONTROLS...",
        flush=True,
    )

    shuffled = []

    for seed in [
        1001,
        2003,
        3001,
        4001,
        5003,
    ]:
        m = shuffled_output_control(
            model,
            tok,
            val_rows,
            seed,
        )

        shuffled.append({
            "seed":
                seed,

            **m,
        })

    shuffled_summary = {
        "repeats":
            len(shuffled),

        "mean_global_auc":
            float(np.mean([
                x["global_auc"]
                for x in shuffled
            ])),

        "mean_macro_role_auc":
            float(np.mean([
                x["macro_role_auc"]
                for x in shuffled
            ])),

        "runs":
            shuffled,
    }

    # Save row-level VAL scores for later analysis.
    scored = []

    for x, z in zip(
        val_rows,
        best_logits,
    ):
        z = float(z)

        scored.append({
            "execution_id":
                x["execution_id"],

            "target_role":
                x["target_role"],

            "target_family":
                x["target_family"],

            "probe_id":
                x["probe_id"],

            "probe_source":
                x["probe_source"],

            "pair_index":
                x["pair_index"],

            "label_eval_only":
                int(x["label"]),

            "logit":
                z,

            "probability":
                float(
                    1.0
                    / (
                        1.0
                        + math.exp(
                            -max(
                                -40,
                                min(
                                    40,
                                    z,
                                ),
                            )
                        )
                    )
                ),
        })

    with (
        out / "val_scores.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in scored:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "version":
            "general_role_sensor_v1",

        "model":
            MODEL_NAME,

        "objective":
            (
                "BCE installed presence + "
                "0.5 pairwise PLUS>MINUS"
            ),

        "train_rows":
            len(train_rows),

        "train_pairs":
            len(train_pairs),

        "train_roles":
            len(train_roles),

        "val_rows":
            len(val_rows),

        "val_pairs":
            len(val_pairs),

        "val_unseen_roles":
            len(val_roles),

        "best_epoch":
            best_epoch,

        "best_selection_score":
            best_score,

        "val":
            best_val,

        "query_only_control":
            query_only,

        "shuffled_output_control":
            shuffled_summary,

        "internal_test_used":
            False,

        "inference_features": [
            "target_role",
            "target_family",
            "query",
            "visible final response",
        ],

        "forbidden_features_used":
            [],
    }

    (
        out / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("FINAL GENERAL SENSOR V1")
    print("=" * 100)
    print("best epoch:", best_epoch)
    print(
        "VAL global AUC     :",
        f"{best_val['global_auc']:.4f}",
    )
    print(
        "VAL macro-role AUC :",
        f"{best_val['macro_role_auc']:.4f}",
    )
    print(
        "VAL pair accuracy  :",
        f"{best_val['pair_accuracy']:.4f}",
    )
    print(
        "VAL pair margin    :",
        f"{best_val['mean_pair_margin']:+.4f}",
    )
    print(
        "QUERY-ONLY AUC     :",
        f"{query_only['global_auc']:.4f}",
    )
    print(
        "SHUFFLED mean AUC  :",
        f"{shuffled_summary['mean_global_auc']:.4f}",
    )
    print(
        "INTERNAL TEST USED : FALSE"
    )
    print()
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
