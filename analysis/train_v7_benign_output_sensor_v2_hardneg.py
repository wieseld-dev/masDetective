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
import analysis.train_v7_benign_output_sensor_pilot_v1 as bos


INIT_ADAPTER = Path(
    "artifacts/v7_benign_output_sensor_pilot_v1/"
    "best_adapter"
)

DATA_ROOT = Path(
    "artifacts/v7_benign_final_pool_v2"
)

CALIBRATION = Path(
    "artifacts/v7_rolewise_calibration_v1/"
    "calibration.json"
)

OUT = Path(
    "artifacts/v7_benign_output_sensor_v2_hardneg"
)

SEED = 20260926


# ============================================================
# IO
# ============================================================

def read_jsonl(path):
    rows = []

    with Path(path).open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def dump(path, obj):
    path = Path(path)

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
    return bos.get_answer(row)


def get_selected(row):
    return bos.get_selected(row)


def sensor_text(
    role,
    role_desc,
    answer,
):
    return bos.sensor_text(
        role,
        role_desc,
        answer,
    )


# ============================================================
# MODEL
# ============================================================

def load_adapter(
    adapter,
    trainable,
):
    cfg = PeftConfig.from_pretrained(
        str(adapter)
    )

    base_name = (
        cfg.base_model_name_or_path
    )

    tok = AutoTokenizer.from_pretrained(
        str(adapter)
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

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
        str(adapter),
        is_trainable=trainable,
    )

    model.to("cuda")

    if trainable:
        model.train()
    else:
        model.eval()

    return tok, model, base_name


# ============================================================
# ROLE-WISE DEV CALIBRATION
# Used ONLY for hard-negative selection.
# ============================================================

def load_role_calibration(
    roles,
):
    obj = json.loads(
        CALIBRATION.read_text(
            encoding="utf-8"
        )
    )

    by_role = obj["roles"]

    scale = np.asarray(
        [
            float(
                by_role[r]["scale"]
            )
            for r in roles
        ],
        dtype=np.float64,
    )

    bias = np.asarray(
        [
            float(
                by_role[r]["bias"]
            )
            for r in roles
        ],
        dtype=np.float64,
    )

    return scale, bias


# ============================================================
# SAMPLE TRAIN OUTPUTS
#
# We want broad role coverage, not just random high-frequency
# planning/research outputs.
# ============================================================

def sample_train_outputs(
    rows,
    roles,
    limit,
    seed,
):
    rng = random.Random(
        seed
    )

    by_role = defaultdict(
        list
    )

    for idx, row in enumerate(
        rows
    ):
        for role in set(
            get_selected(row)
        ):
            if role in roles:
                by_role[role].append(
                    idx
                )

    selected_indices = set()

    target_per_role = max(
        1,
        math.ceil(
            limit / len(roles)
        ),
    )

    # First guarantee roughly balanced role coverage.
    for role in roles:
        pool = list(
            by_role[role]
        )

        rng.shuffle(
            pool
        )

        for idx in pool:
            selected_indices.add(
                idx
            )

            if (
                sum(
                    1
                    for j in selected_indices
                    if role in set(
                        get_selected(
                            rows[j]
                        )
                    )
                )
                >= target_per_role
            ):
                break

            if len(
                selected_indices
            ) >= limit:
                break

        if len(
            selected_indices
        ) >= limit:
            break

    # Fill remaining slots randomly.
    remaining = [
        i
        for i in range(
            len(rows)
        )
        if i not in selected_indices
    ]

    rng.shuffle(
        remaining
    )

    for idx in remaining:
        if len(
            selected_indices
        ) >= limit:
            break

        selected_indices.add(
            idx
        )

    chosen = [
        rows[i]
        for i in sorted(
            selected_indices
        )
    ]

    rng.shuffle(
        chosen
    )

    return chosen[:limit]


# ============================================================
# FROZEN V1 SCORING FOR HARD-NEGATIVE MINING
# ============================================================

@torch.inference_mode()
def score_all_roles(
    model,
    tok,
    rows,
    roles,
    role_desc,
    max_length,
    batch_size,
):
    texts = []

    for row in rows:
        answer = get_answer(
            row
        )

        for role in roles:
            texts.append(
                sensor_text(
                    role,
                    role_desc[role],
                    answer,
                )
            )

    scores = bos.score_texts(
        model,
        tok,
        texts,
        torch.device("cuda"),
        max_length,
        batch_size,
    )

    return scores.reshape(
        len(rows),
        len(roles),
    )


def mine_hard_examples(
    rows,
    raw_scores,
    roles,
    role_desc,
    scale,
    bias,
    hard_k,
):
    calibrated = (
        raw_scores
        * scale.reshape(1, -1)
        + bias.reshape(1, -1)
    )

    examples = []

    positive_counts = Counter()
    negative_counts = Counter()

    for i, row in enumerate(
        rows
    ):
        selected = set(
            get_selected(row)
        )

        selected = {
            r
            for r in selected
            if r in roles
        }

        if not selected:
            continue

        absent = [
            r
            for r in roles
            if r not in selected
        ]

        answer = get_answer(
            row
        )

        absent_sorted = sorted(
            absent,
            key=lambda r: calibrated[
                i,
                roles.index(r)
            ],
            reverse=True,
        )

        hard = absent_sorted[
            :hard_k
        ]

        for pos_role in sorted(
            selected
        ):
            positive_counts[
                pos_role
            ] += 1

            for neg_role in hard:
                negative_counts[
                    neg_role
                ] += 1

                examples.append({
                    "session_id":
                        row.get(
                            "session_id"
                        ),

                    "positive_role":
                        pos_role,

                    "negative_role":
                        neg_role,

                    "positive_text":
                        sensor_text(
                            pos_role,
                            role_desc[
                                pos_role
                            ],
                            answer,
                        ),

                    "negative_text":
                        sensor_text(
                            neg_role,
                            role_desc[
                                neg_role
                            ],
                            answer,
                        ),

                    "negative_mined_score":
                        float(
                            calibrated[
                                i,
                                roles.index(
                                    neg_role
                                )
                            ]
                        ),
                })

    audit = {
        "source_outputs":
            len(rows),

        "hard_k":
            hard_k,

        "training_pairs":
            len(examples),

        "positive_role_counts":
            dict(
                sorted(
                    positive_counts.items()
                )
            ),

        "negative_role_counts":
            dict(
                sorted(
                    negative_counts.items()
                )
            ),
    }

    return examples, audit


# ============================================================
# DATASET
# ============================================================

class PairDataset(Dataset):
    def __init__(
        self,
        examples,
    ):
        self.examples = examples

    def __len__(self):
        return len(
            self.examples
        )

    def __getitem__(
        self,
        idx,
    ):
        return self.examples[
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

        enc = tok(
            pos + neg,
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
# EVAL
# ============================================================

def binary_auc(
    labels,
    scores,
):
    return bos.binary_auc(
        labels,
        scores,
    )


def evaluate(
    model,
    tok,
    rows,
    roles,
    role_desc,
    max_length,
    batch_size,
):
    texts = []
    labels = []
    role_index = []

    for row in rows:
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
                int(
                    role in selected
                )
            )

            role_index.append(
                j
            )

    scores = bos.score_texts(
        model,
        tok,
        texts,
        torch.device("cuda"),
        max_length,
        batch_size,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    matrix = scores.reshape(
        len(rows),
        len(roles),
    )

    label_matrix = labels.reshape(
        len(rows),
        len(roles),
    )

    micro_auc = binary_auc(
        labels,
        scores,
    )

    role_aucs = {}

    for j, role in enumerate(
        roles
    ):
        role_aucs[
            role
        ] = binary_auc(
            label_matrix[:, j],
            matrix[:, j],
        )

    valid_role = [
        x
        for x in role_aucs.values()
        if math.isfinite(x)
    ]

    within = []
    topk = []
    exact = []

    for i in range(
        len(rows)
    ):
        y = label_matrix[i]
        s = matrix[i]

        auc = binary_auc(
            y,
            s,
        )

        if math.isfinite(
            auc
        ):
            within.append(
                auc
            )

        k = int(
            y.sum()
        )

        if k > 0:
            pred_idx = np.argsort(
                -s
            )[:k]

            truth_idx = set(
                np.where(
                    y == 1
                )[0].tolist()
            )

            pred_idx = set(
                pred_idx.tolist()
            )

            topk.append(
                len(
                    truth_idx
                    & pred_idx
                )
                / k
            )

            exact.append(
                float(
                    truth_idx
                    == pred_idx
                )
            )

    return {
        "rows":
            len(rows),

        "micro_auc":
            float(
                micro_auc
            ),

        "macro_role_auc":
            float(
                np.mean(
                    valid_role
                )
            ),

        "within_output_auc":
            float(
                np.mean(
                    within
                )
            ),

        "topk_activation_recall":
            float(
                np.mean(
                    topk
                )
            ),

        "exact_activation_set":
            float(
                np.mean(
                    exact
                )
            ),

        "role_auc":
            role_aucs,
    }


# ============================================================
# TRAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train-outputs",
        type=int,
        default=2048,
    )

    ap.add_argument(
        "--hard-k",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--grad-accum",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--max-length",
        type=int,
        default=768,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=3e-6,
    )

    ap.add_argument(
        "--margin",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--bce-weight",
        type=float,
        default=0.15,
    )

    ap.add_argument(
        "--dev-limit",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--test-limit",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    ap.add_argument(
        "--log-every",
        type=int,
        default=250,
    )

    args = ap.parse_args()

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

    role_desc = dict(
        tr.ROLE_DESCRIPTIONS
    )

    roles = list(
        role_desc.keys()
    )

    train_all = read_jsonl(
        DATA_ROOT / "train.jsonl"
    )

    dev_all = read_jsonl(
        DATA_ROOT / "dev.jsonl"
    )

    test_all = read_jsonl(
        DATA_ROOT / "test.jsonl"
    )

    train_rows = sample_train_outputs(
        train_all,
        roles,
        args.train_outputs,
        args.seed,
    )

    rng = random.Random(
        args.seed + 1
    )

    dev_rows = (
        rng.sample(
            dev_all,
            min(
                args.dev_limit,
                len(dev_all),
            ),
        )
    )

    rng2 = random.Random(
        args.seed + 2
    )

    test_rows = (
        rng2.sample(
            test_all,
            min(
                args.test_limit,
                len(test_all),
            ),
        )
    )

    print(
        "===== DATA =====",
        flush=True,
    )

    print(
        "train outputs:",
        len(train_rows),
        flush=True,
    )

    print(
        "dev/test:",
        len(dev_rows),
        len(test_rows),
        flush=True,
    )

    # --------------------------------------------------------
    # STEP 1: frozen V1 hard-negative mining
    # --------------------------------------------------------

    print(
        "\n===== LOAD FROZEN V1 FOR MINING =====",
        flush=True,
    )

    tok_mine, model_mine, _ = (
        load_adapter(
            INIT_ADAPTER,
            trainable=False,
        )
    )

    print(
        "\n===== SCORE TRAIN OUTPUTS FOR HARD NEGATIVES =====",
        flush=True,
    )

    train_scores = score_all_roles(
        model_mine,
        tok_mine,
        train_rows,
        roles,
        role_desc,
        args.max_length,
        args.eval_batch_size,
    )

    scale, bias = (
        load_role_calibration(
            roles
        )
    )

    examples, audit = (
        mine_hard_examples(
            train_rows,
            train_scores,
            roles,
            role_desc,
            scale,
            bias,
            args.hard_k,
        )
    )

    dump(
        OUT / "hard_negative_audit.json",
        audit,
    )

    print(
        "\n===== HARD-NEGATIVE AUDIT =====",
        flush=True,
    )

    print(
        json.dumps(
            audit,
            indent=2,
        ),
        flush=True,
    )

    del model_mine
    del tok_mine
    del train_scores

    torch.cuda.empty_cache()

    # --------------------------------------------------------
    # STEP 2: train from V1 adapter
    # --------------------------------------------------------

    print(
        "\n===== LOAD TRAINABLE V1 =====",
        flush=True,
    )

    tok, model, base_name = (
        load_adapter(
            INIT_ADAPTER,
            trainable=True,
        )
    )

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    print(
        "trainable params:",
        sum(
            p.numel()
            for p in params
        ),
        flush=True,
    )

    ds = PairDataset(
        examples
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

    opt = AdamW(
        params,
        lr=args.lr,
        weight_decay=0.01,
    )

    # Baseline dev before training.
    print(
        "\n===== BASELINE DEV =====",
        flush=True,
    )

    baseline_dev = evaluate(
        model,
        tok,
        dev_rows,
        roles,
        role_desc,
        args.max_length,
        args.eval_batch_size,
    )

    print(
        json.dumps(
            baseline_dev,
            indent=2,
        ),
        flush=True,
    )

    best_auc = float(
        baseline_dev[
            "within_output_auc"
        ]
    )

    history = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        opt.zero_grad(
            set_to_none=True
        )

        running = 0.0
        optimizer_steps = 0

        print(
            f"\n===== EPOCH {epoch} =====",
            flush=True,
        )

        for step, batch in enumerate(
            loader,
            1,
        ):
            enc = {
                k: v.to("cuda")
                for k, v
                in batch["enc"].items()
            }

            n = int(
                batch["n"]
            )

            logits = model(
                **enc
            ).logits.float().view(-1)

            pos = logits[:n]
            neg = logits[n:]

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
                loss.detach()
                .cpu()
            )

            do_step = (
                step
                % args.grad_accum
                == 0
                or step
                == len(loader)
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
                step
                % args.log_every
                == 0
                or step
                == len(loader)
            ):
                print(
                    f"[train] "
                    f"epoch={epoch} "
                    f"batch={step}/{len(loader)} "
                    f"loss={running/step:.5f} "
                    f"opt_steps={optimizer_steps}",
                    flush=True,
                )

        model.eval()

        print(
            f"\n===== DEV AFTER EPOCH {epoch} =====",
            flush=True,
        )

        dev_metrics = evaluate(
            model,
            tok,
            dev_rows,
            roles,
            role_desc,
            args.max_length,
            args.eval_batch_size,
        )

        print(
            json.dumps(
                dev_metrics,
                indent=2,
            ),
            flush=True,
        )

        history.append({
            "epoch":
                epoch,

            "train_loss":
                running
                / max(
                    len(loader),
                    1,
                ),

            "dev":
                dev_metrics,
        })

        if (
            dev_metrics[
                "within_output_auc"
            ]
            > best_auc
        ):
            best_auc = float(
                dev_metrics[
                    "within_output_auc"
                ]
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
                "[checkpoint] new best "
                f"within_auc={best_auc:.6f}",
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

    # If training never beat baseline, keep last separately but
    # don't pretend it is better.
    best_dir = (
        OUT / "best_adapter"
    )

    if not best_dir.exists():
        print(
            "[checkpoint] no epoch beat V1 baseline",
            flush=True,
        )

        selected_adapter = (
            INIT_ADAPTER
        )

        final_dev = (
            baseline_dev
        )

        test_model = model
        test_tok = tok

    else:
        selected_adapter = (
            best_dir
        )

        # Reload best explicitly.
        del model
        del tok

        torch.cuda.empty_cache()

        test_tok, test_model, _ = (
            load_adapter(
                best_dir,
                trainable=False,
            )
        )

        final_dev = evaluate(
            test_model,
            test_tok,
            dev_rows,
            roles,
            role_desc,
            args.max_length,
            args.eval_batch_size,
        )

    print(
        "\n===== TEST =====",
        flush=True,
    )

    test_metrics = evaluate(
        test_model,
        test_tok,
        test_rows,
        roles,
        role_desc,
        args.max_length,
        args.eval_batch_size,
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
            "v7_benign_output_sensor_v2_hardneg",

        "query_used":
            False,

        "initial_adapter":
            str(
                INIT_ADAPTER
            ),

        "selected_adapter":
            str(
                selected_adapter
            ),

        "base_model":
            base_name,

        "train_outputs":
            len(
                train_rows
            ),

        "training_pairs":
            len(
                examples
            ),

        "hard_k":
            args.hard_k,

        "baseline_dev":
            baseline_dev,

        "final_dev":
            final_dev,

        "test":
            test_metrics,

        "history":
            history,
    }

    dump(
        OUT / "summary.json",
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


if __name__ == "__main__":
    main()
