#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from peft import (
    PeftConfig,
    PeftModel,
)

import analysis.train_v13_6b_activation_judge as tr


INIT_ADAPTER = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/"
    "best_adapter"
)

DATA_ROOT = Path(
    "artifacts/v7_benign_final_pool_v2"
)

OUT = Path(
    "artifacts/v7_benign_output_sensor_pilot_v1"
)

SEED = 20260926


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Path):
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def dump(path: Path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def get_answer(row):
    for k in (
        "final_output",
        "output",
        "answer",
        "response",
    ):
        x = row.get(k)

        if isinstance(x, str) and x.strip():
            return x.strip()

    raise RuntimeError(
        "row has no usable final output"
    )


def get_selected(row):
    x = row.get(
        "selected_roles",
        []
    )

    if not isinstance(
        x,
        (list, tuple)
    ):
        return []

    return [
        str(r)
        for r in x
    ]


# ============================================================
# INPUT FORMAT
#
# IMPORTANT:
#
# NO QUERY.
#
# The only variable information is:
#   candidate role
#   role description
#   final visible MAS output
# ============================================================

def sensor_text(
    role,
    role_desc,
    answer,
):
    return (
        "You are judging whether a candidate specialist "
        "actually participated in producing the final response "
        "of a hidden multi-agent system.\n\n"

        "The original user query is intentionally unavailable. "
        "Judge participation only from evidence present in the "
        "final response. Topic overlap alone is not sufficient.\n\n"

        "[CANDIDATE ROLE]\n"
        f"{role}\n\n"

        "[ROLE DESCRIPTION]\n"
        f"{role_desc}\n\n"

        "[FINAL RESPONSE]\n"
        f"{answer}"
    )


# ============================================================
# TRAINING PAIRS
#
# Same output O:
#
#   positive = (selected role, O)
#   negative = (absent role, O)
#
# This directly learns within-output activation discrimination.
# ============================================================

def build_pairs(
    rows,
    roles,
    role_desc,
    pairs_per_role,
    seed,
):
    rng = random.Random(
        seed
    )

    positive_rows = {
        r: []
        for r in roles
    }

    for row in rows:
        selected = set(
            get_selected(row)
        )

        for r in selected:
            if r in positive_rows:
                positive_rows[r].append(
                    row
                )

    pos_counts = {
        r: len(xs)
        for r, xs in positive_rows.items()
    }

    print(
        "===== TRAIN POSITIVE COUNTS =====",
        flush=True,
    )

    print(
        json.dumps(
            pos_counts,
            indent=2,
        ),
        flush=True,
    )

    # Keep negative-role usage roughly balanced.
    neg_usage = Counter()

    pairs = []

    for pos_role in roles:
        pool = positive_rows[
            pos_role
        ]

        if not pool:
            raise RuntimeError(
                f"no positives for {pos_role}"
            )

        n_take = min(
            pairs_per_role,
            len(pool),
        )

        chosen = rng.sample(
            pool,
            n_take,
        )

        for row in chosen:
            selected = set(
                get_selected(row)
            )

            absent = [
                r
                for r in roles
                if r not in selected
            ]

            if not absent:
                continue

            min_used = min(
                neg_usage[r]
                for r in absent
            )

            candidate_neg = [
                r
                for r in absent
                if neg_usage[r]
                == min_used
            ]

            neg_role = rng.choice(
                candidate_neg
            )

            neg_usage[
                neg_role
            ] += 1

            answer = get_answer(
                row
            )

            pairs.append({
                "session_id":
                    row.get("session_id"),

                "positive_role":
                    pos_role,

                "negative_role":
                    neg_role,

                "positive_text":
                    sensor_text(
                        pos_role,
                        role_desc[pos_role],
                        answer,
                    ),

                "negative_text":
                    sensor_text(
                        neg_role,
                        role_desc[neg_role],
                        answer,
                    ),
            })

    rng.shuffle(
        pairs
    )

    audit = {
        "pairs":
            len(pairs),

        "pairs_per_role_requested":
            pairs_per_role,

        "positive_role_counts":
            dict(
                sorted(
                    Counter(
                        x["positive_role"]
                        for x in pairs
                    ).items()
                )
            ),

        "negative_role_counts":
            dict(
                sorted(
                    Counter(
                        x["negative_role"]
                        for x in pairs
                    ).items()
                )
            ),

        "source_positive_counts":
            pos_counts,
    }

    return pairs, audit


class PairDataset(Dataset):
    def __init__(
        self,
        pairs,
    ):
        self.pairs = pairs

    def __len__(self):
        return len(
            self.pairs
        )

    def __getitem__(
        self,
        idx,
    ):
        return self.pairs[
            idx
        ]


def make_collate(
    tok,
    max_length,
):
    def collate(batch):
        pos = [
            x["positive_text"]
            for x in batch
        ]

        neg = [
            x["negative_text"]
            for x in batch
        ]

        texts = (
            pos + neg
        )

        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        return {
            "enc":
                enc,

            "n":
                len(batch),
        }

    return collate


# ============================================================
# MODEL
# ============================================================

def load_model():
    cfg = PeftConfig.from_pretrained(
        str(INIT_ADAPTER)
    )

    base_name = (
        cfg.base_model_name_or_path
    )

    print(
        "[model] base:",
        base_name,
        flush=True,
    )

    print(
        "[model] adapter:",
        INIT_ADAPTER,
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(
        str(INIT_ADAPTER)
    )

    if tok.pad_token_id is None:
        tok.pad_token = (
            tok.eos_token
        )

    tok.padding_side = "right"
    tok.truncation_side = "right"

    base = (
        AutoModelForSequenceClassification
        .from_pretrained(
            base_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        )
    )

    base.config.pad_token_id = (
        tok.pad_token_id
    )

    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        str(INIT_ADAPTER),
        is_trainable=True,
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
        "[model] trainable:",
        trainable,
        "/",
        total,
        f"({100.0*trainable/total:.4f}%)",
        flush=True,
    )

    return (
        tok,
        model,
        base_name,
    )


# ============================================================
# AUC
# ============================================================

def binary_auc(
    labels,
    scores,
):
    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    s = np.asarray(
        scores,
        dtype=np.float64,
    )

    pos = int(
        y.sum()
    )

    neg = int(
        len(y) - pos
    )

    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(
        s,
        kind="mergesort",
    )

    sorted_s = s[
        order
    ]

    ranks = np.empty(
        len(s),
        dtype=np.float64,
    )

    i = 0

    while i < len(s):
        j = i + 1

        while (
            j < len(s)
            and sorted_s[j]
            == sorted_s[i]
        ):
            j += 1

        # 1-indexed average rank.
        avg_rank = (
            (i + 1) + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = avg_rank

        i = j

    pos_rank_sum = float(
        ranks[
            y == 1
        ].sum()
    )

    auc = (
        pos_rank_sum
        - pos * (
            pos + 1
        ) / 2.0
    ) / (
        pos * neg
    )

    return float(
        auc
    )


# ============================================================
# SCORING
# ============================================================

@torch.inference_mode()
def score_texts(
    model,
    tok,
    texts,
    device,
    max_length,
    batch_size,
):
    scores = []

    model.eval()

    for start in range(
        0,
        len(texts),
        batch_size,
    ):
        batch = texts[
            start:
            start + batch_size
        ]

        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        enc = {
            k: v.to(device)
            for k, v in enc.items()
        }

        logits = model(
            **enc
        ).logits

        logits = (
            logits
            .float()
            .view(-1)
            .cpu()
            .numpy()
            .tolist()
        )

        scores.extend(
            logits
        )

    return np.asarray(
        scores,
        dtype=np.float64,
    )


def evaluate(
    model,
    tok,
    rows,
    roles,
    role_desc,
    device,
    max_length,
    batch_size,
    limit,
    seed,
):
    rng = random.Random(
        seed
    )

    rows = list(
        rows
    )

    if (
        limit is not None
        and len(rows) > limit
    ):
        rows = rng.sample(
            rows,
            limit,
        )

    texts = []
    labels = []
    row_index = []
    role_index = []

    for i, row in enumerate(
        rows
    ):
        selected = set(
            get_selected(row)
        )

        answer = get_answer(
            row
        )

        for j, role in enumerate(
            roles
        ):
            texts.append(
                sensor_text(
                    role,
                    role_desc[role],
                    answer,
                )
            )

            labels.append(
                1
                if role in selected
                else 0
            )

            row_index.append(
                i
            )

            role_index.append(
                j
            )

    scores = score_texts(
        model,
        tok,
        texts,
        device,
        max_length,
        batch_size,
    )

    labels_np = np.asarray(
        labels,
        dtype=np.int64,
    )

    micro_auc = binary_auc(
        labels_np,
        scores,
    )

    # --------------------------------------------------------
    # Per-role AUC
    # --------------------------------------------------------

    role_aucs = {}

    for j, role in enumerate(
        roles
    ):
        mask = (
            np.asarray(
                role_index
            )
            == j
        )

        role_aucs[
            role
        ] = binary_auc(
            labels_np[mask],
            scores[mask],
        )

    valid_role_auc = [
        x
        for x in role_aucs.values()
        if math.isfinite(x)
    ]

    macro_role_auc = (
        float(
            np.mean(
                valid_role_auc
            )
        )
        if valid_role_auc
        else float("nan")
    )

    # --------------------------------------------------------
    # Within-output metrics
    # --------------------------------------------------------

    matrix = scores.reshape(
        len(rows),
        len(roles),
    )

    within_aucs = []
    topk_recalls = []
    exact_sets = []
    margins = []

    for i, row in enumerate(
        rows
    ):
        selected = set(
            get_selected(row)
        )

        y = np.asarray([
            1 if r in selected else 0
            for r in roles
        ])

        s = matrix[
            i
        ]

        auc = binary_auc(
            y,
            s,
        )

        if math.isfinite(
            auc
        ):
            within_aucs.append(
                auc
            )

        k = len(
            selected
        )

        if k > 0:
            top_idx = np.argsort(
                -s
            )[:k]

            predicted = {
                roles[j]
                for j in top_idx
            }

            topk_recalls.append(
                len(
                    predicted
                    & selected
                )
                / k
            )

            exact_sets.append(
                float(
                    predicted
                    == selected
                )
            )

        pos_scores = s[
            y == 1
        ]

        neg_scores = s[
            y == 0
        ]

        if (
            len(pos_scores)
            and len(neg_scores)
        ):
            margins.append(
                float(
                    pos_scores.mean()
                    - neg_scores.mean()
                )
            )

    metrics = {
        "rows":
            len(rows),

        "candidate_scores":
            len(scores),

        "micro_auc":
            float(
                micro_auc
            ),

        "macro_role_auc":
            float(
                macro_role_auc
            ),

        "within_output_auc":
            float(
                np.mean(
                    within_aucs
                )
            ),

        "topk_activation_recall":
            float(
                np.mean(
                    topk_recalls
                )
            ),

        "exact_activation_set":
            float(
                np.mean(
                    exact_sets
                )
            ),

        "mean_selected_minus_absent_logit":
            float(
                np.mean(
                    margins
                )
            ),

        "role_auc":
            role_aucs,
    }

    return metrics


# ============================================================
# TRAIN
# ============================================================

def train(args):
    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_rows = read_jsonl(
        DATA_ROOT / "train.jsonl"
    )

    dev_rows = read_jsonl(
        DATA_ROOT / "dev.jsonl"
    )

    test_rows = read_jsonl(
        DATA_ROOT / "test.jsonl"
    )

    role_desc = dict(
        tr.ROLE_DESCRIPTIONS
    )

    roles = list(
        role_desc.keys()
    )

    print(
        "[data] roles:",
        len(roles),
        flush=True,
    )

    print(
        "[data] train/dev/test:",
        len(train_rows),
        len(dev_rows),
        len(test_rows),
        flush=True,
    )

    pairs, audit = build_pairs(
        train_rows,
        roles,
        role_desc,
        args.pairs_per_role,
        args.seed,
    )

    dump(
        OUT / "train_audit.json",
        audit,
    )

    print(
        "===== TRAIN AUDIT =====",
        flush=True,
    )

    print(
        json.dumps(
            audit,
            indent=2,
        ),
        flush=True,
    )

    tok, model, base_name = (
        load_model()
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(
        device
    )

    ds = PairDataset(
        pairs
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate(
            tok,
            args.max_length,
        ),
    )

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    opt = AdamW(
        params,
        lr=args.lr,
        weight_decay=0.01,
    )

    model.train()

    opt.zero_grad(
        set_to_none=True
    )

    running = 0.0
    optimizer_steps = 0

    print(
        "\n===== TRAINING =====",
        flush=True,
    )

    for step, batch in enumerate(
        loader,
        1,
    ):
        enc = {
            k: v.to(device)
            for k, v in batch[
                "enc"
            ].items()
        }

        n = int(
            batch["n"]
        )

        logits = model(
            **enc
        ).logits

        logits = (
            logits
            .float()
            .view(-1)
        )

        pos = logits[
            :n
        ]

        neg = logits[
            n:
        ]

        rank_loss = F.softplus(
            args.margin
            - (
                pos - neg
            )
        ).mean()

        bce_logits = torch.cat(
            [
                pos,
                neg,
            ],
            dim=0,
        )

        bce_targets = torch.cat(
            [
                torch.ones_like(
                    pos
                ),
                torch.zeros_like(
                    neg
                ),
            ],
            dim=0,
        )

        bce_loss = (
            F.binary_cross_entropy_with_logits(
                bce_logits,
                bce_targets,
            )
        )

        loss = (
            rank_loss
            + args.bce_weight
            * bce_loss
        )

        (
            loss
            / args.grad_accum
        ).backward()

        running += float(
            loss.detach().cpu()
        )

        do_step = (
            step % args.grad_accum
            == 0
            or step == len(loader)
        )

        if do_step:
            torch.nn.utils.clip_grad_norm_(
                params,
                1.0,
            )

            opt.step()

            opt.zero_grad(
                set_to_none=True
            )

            optimizer_steps += 1

        if (
            step % args.log_every == 0
            or step == len(loader)
        ):
            print(
                f"[train] batch "
                f"{step}/{len(loader)} "
                f"loss={running/step:.5f} "
                f"opt_steps={optimizer_steps}",
                flush=True,
            )

    last_dir = (
        OUT / "last_adapter"
    )

    shutil.rmtree(
        last_dir,
        ignore_errors=True,
    )

    model.save_pretrained(
        last_dir
    )

    tok.save_pretrained(
        last_dir
    )

    print(
        "\n===== DEV EVAL =====",
        flush=True,
    )

    dev_metrics = evaluate(
        model,
        tok,
        dev_rows,
        roles,
        role_desc,
        device,
        args.max_length,
        args.eval_batch_size,
        args.dev_limit,
        args.seed + 101,
    )

    print(
        json.dumps(
            dev_metrics,
            indent=2,
        ),
        flush=True,
    )

    best_dir = (
        OUT / "best_adapter"
    )

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
        "\n===== HELD-OUT TEST EVAL =====",
        flush=True,
    )

    test_metrics = evaluate(
        model,
        tok,
        test_rows,
        roles,
        role_desc,
        device,
        args.max_length,
        args.eval_batch_size,
        args.test_limit,
        args.seed + 202,
    )

    print(
        json.dumps(
            test_metrics,
            indent=2,
        ),
        flush=True,
    )

    summary = {
        "experiment":
            "v7_benign_output_sensor_pilot_v1",

        "input_variables": [
            "candidate_role",
            "role_description",
            "final_output",
        ],

        "query_used":
            False,

        "initial_adapter":
            str(
                INIT_ADAPTER
            ),

        "base_model":
            base_name,

        "train_rows_available":
            len(train_rows),

        "train_pairs":
            len(pairs),

        "pairs_per_role":
            args.pairs_per_role,

        "dev_rows_available":
            len(dev_rows),

        "test_rows_available":
            len(test_rows),

        "dev_metrics":
            dev_metrics,

        "test_metrics":
            test_metrics,

        "best_adapter":
            str(
                best_dir
            ),
    }

    dump(
        OUT / "metrics.json",
        summary,
    )

    print(
        "\n===== FINAL SUMMARY =====",
        flush=True,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        ),
        flush=True,
    )


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    p.add_argument(
        "--pairs-per-role",
        type=int,
        default=256,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    p.add_argument(
        "--grad-accum",
        type=int,
        default=8,
    )

    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
    )

    p.add_argument(
        "--max-length",
        type=int,
        default=768,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=5e-6,
    )

    p.add_argument(
        "--margin",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--bce-weight",
        type=float,
        default=0.15,
    )

    p.add_argument(
        "--dev-limit",
        type=int,
        default=512,
    )

    p.add_argument(
        "--test-limit",
        type=int,
        default=512,
    )

    p.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    return p.parse_args()


if __name__ == "__main__":
    train(
        parse_args()
    )
