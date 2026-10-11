#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import PeftConfig, PeftModel

import analysis.train_v13_6b_activation_judge as tr
import analysis.train_v7_benign_output_sensor_pilot_v1 as bos


ADAPTER = Path(
    "artifacts/v7_benign_output_sensor_pilot_v1/best_adapter"
)

RERANKER_ARTIFACT = Path(
    "artifacts/v7_structured_reranker_v1/reranker.pt"
)

RERANKER_SUMMARY = Path(
    "artifacts/v7_structured_reranker_v1/summary.json"
)

DATA_ROOT = Path(
    "artifacts/v7_benign_final_pool_v2"
)

AUDIT = Path(
    "artifacts/v7_actual_query_manual_audit_v1/sample_300.jsonl"
)

OUT = Path(
    "artifacts/v7_count_decoder_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926

TRAIN_LIMIT = 2048
DEV_LIMIT = 512
TEST_LIMIT = 512

MAX_LENGTH = 768
SCORE_BATCH = 8

TARGET_DEV_RECALL = 0.80


CAPABILITY_IDS = {
    28, 45, 69, 70, 82, 93, 110, 133, 147, 156,
    166, 179, 186, 233, 236, 238, 255, 267, 300,
}

DIRECTION_IDS = {
    8, 27, 39, 41, 55, 66, 80, 81, 85, 89,
    91, 100, 108, 112, 172, 182, 184, 189, 205,
    218, 220, 224, 247, 256, 259, 273, 278, 297,
}


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def dump(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def norm_q(x):
    return " ".join(
        str(x or "").strip().split()
    ).lower()


def sample_fixed(rows, n, seed):
    rows = list(rows)

    if len(rows) <= n:
        return rows

    rng = random.Random(seed)
    return rng.sample(rows, n)


def load_sensor():
    cfg = PeftConfig.from_pretrained(
        str(ADAPTER)
    )

    base_name = cfg.base_model_name_or_path

    tok = AutoTokenizer.from_pretrained(
        str(ADAPTER)
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

    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER),
        is_trainable=False,
    )

    model.eval()
    model.to("cuda")

    return tok, model


@torch.inference_mode()
def score_rows(
    model,
    tok,
    rows,
    roles,
    role_desc,
):
    texts = []

    for row in rows:
        answer = bos.get_answer(row)

        for role in roles:
            texts.append(
                bos.sensor_text(
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
        MAX_LENGTH,
        SCORE_BATCH,
    )

    return scores.reshape(
        len(rows),
        len(roles),
    )


def labels_matrix(rows, roles):
    y = np.zeros(
        (len(rows), len(roles)),
        dtype=np.int64,
    )

    for i, row in enumerate(rows):
        selected = set(
            bos.get_selected(row)
        )

        for j, role in enumerate(roles):
            if role in selected:
                y[i, j] = 1

    return y


# ============================================================
# Reconstruct structured reranker from previous artifact.
# ============================================================

class StructuredReranker(nn.Module):
    def __init__(
        self,
        n_roles,
        mean,
        std,
    ):
        super().__init__()

        self.delta = nn.Parameter(
            torch.zeros(
                n_roles,
                n_roles,
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(
                n_roles,
            )
        )

        self.register_buffer(
            "mean",
            torch.tensor(
                mean,
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "std",
            torch.tensor(
                std,
                dtype=torch.float32,
            ),
        )

    def forward(self, scores):
        x = (
            scores - self.mean
        ) / self.std

        return (
            scores
            + x @ self.delta.t()
            + self.bias
        )


def load_reranker(roles):
    art = torch.load(
        RERANKER_ARTIFACT,
        map_location="cpu",
        weights_only=False,
    )

    saved_roles = list(
        art["roles"]
    )

    if saved_roles != roles:
        raise RuntimeError(
            "role ordering mismatch"
        )

    rr = StructuredReranker(
        len(roles),
        art["mean"],
        art["std"],
    )

    rr.load_state_dict(
        art["state_dict"]
    )

    rr.eval()

    return rr


@torch.inference_mode()
def rerank(rr, scores):
    return rr(
        torch.tensor(
            scores,
            dtype=torch.float32,
        )
    ).numpy()


def load_post_calibration(roles):
    obj = json.loads(
        RERANKER_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    cal = obj[
        "role_calibration"
    ]

    scale = np.asarray([
        float(cal[r]["scale"])
        for r in roles
    ])

    bias = np.asarray([
        float(cal[r]["bias"])
        for r in roles
    ])

    return scale, bias


def apply_calibration(
    scores,
    scale,
    bias,
):
    return (
        scores
        * scale.reshape(1, -1)
        + bias.reshape(1, -1)
    )


# ============================================================
# Count features.
#
# We use:
#  - role-specific 20 scores
#  - sorted 20 scores
#  - 19 adjacent score gaps
#  - summary statistics
#
# Thus the decoder can learn both role priors and "elbow"
# geometry without seeing query or output text.
# ============================================================

def make_features(scores):
    sorted_s = np.sort(
        scores,
        axis=1,
    )[:, ::-1]

    gaps = (
        sorted_s[:, :-1]
        - sorted_s[:, 1:]
    )

    mean = scores.mean(
        axis=1,
        keepdims=True,
    )

    std = scores.std(
        axis=1,
        keepdims=True,
    )

    mx = scores.max(
        axis=1,
        keepdims=True,
    )

    mn = scores.min(
        axis=1,
        keepdims=True,
    )

    median = np.median(
        scores,
        axis=1,
        keepdims=True,
    )

    return np.concatenate(
        [
            scores,
            sorted_s,
            gaps,
            mean,
            std,
            mx,
            mn,
            median,
        ],
        axis=1,
    ).astype(np.float32)


class CountNet(nn.Module):
    def __init__(
        self,
        input_dim,
        n_classes=21,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(
                64,
                32,
            ),
            nn.ReLU(),
            nn.Linear(
                32,
                n_classes,
            ),
        )

    def forward(self, x):
        return self.net(x)


def count_metrics(
    true_k,
    pred_k,
):
    true_k = np.asarray(
        true_k
    )

    pred_k = np.asarray(
        pred_k
    )

    return {
        "count_accuracy":
            float(
                np.mean(
                    true_k == pred_k
                )
            ),

        "count_mae":
            float(
                np.mean(
                    np.abs(
                        true_k - pred_k
                    )
                )
            ),

        "mean_true_count":
            float(
                np.mean(true_k)
            ),

        "mean_pred_count":
            float(
                np.mean(pred_k)
            ),

        "count_bias_pred_minus_true":
            float(
                np.mean(
                    pred_k - true_k
                )
            ),
    }


def decode_topk(
    scores,
    counts,
):
    pred = np.zeros_like(
        scores,
        dtype=np.int64,
    )

    for i in range(
        len(scores)
    ):
        k = int(
            np.clip(
                counts[i],
                0,
                scores.shape[1],
            )
        )

        if k <= 0:
            continue

        idx = np.argsort(
            -scores[i]
        )[:k]

        pred[i, idx] = 1

    return pred


def set_metrics(
    pred,
    labels,
):
    pred = pred.astype(
        np.int64
    )

    y = labels.astype(
        np.int64
    )

    tp = int(
        (
            (pred == 1)
            & (y == 1)
        ).sum()
    )

    fp = int(
        (
            (pred == 1)
            & (y == 0)
        ).sum()
    )

    fn = int(
        (
            (pred == 0)
            & (y == 1)
        ).sum()
    )

    tn = int(
        (
            (pred == 0)
            & (y == 0)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    exact = np.all(
        pred == y,
        axis=1,
    )

    return {
        "precision":
            float(precision),

        "recall":
            float(recall),

        "fpr":
            float(fpr),

        "false_positives":
            fp,

        "false_negatives":
            fn,

        "avg_fp_per_output":
            float(
                fp / len(y)
            ),

        "avg_fn_per_output":
            float(
                fn / len(y)
            ),

        "exact_activation_set":
            float(
                exact.mean()
            ),
    }


def train_count_ensemble(
    train_x,
    train_k,
    dev_x,
    dev_k,
    seeds,
):
    mu = train_x.mean(
        axis=0
    )

    sd = train_x.std(
        axis=0
    )

    sd = np.maximum(
        sd,
        1e-5,
    )

    tx = (
        train_x - mu
    ) / sd

    dx = (
        dev_x - mu
    ) / sd

    tx_t = torch.tensor(
        tx,
        dtype=torch.float32,
    )

    ty_t = torch.tensor(
        train_k,
        dtype=torch.long,
    )

    dx_t = torch.tensor(
        dx,
        dtype=torch.float32,
    )

    models = []
    reports = []

    for seed in seeds:
        torch.manual_seed(
            seed
        )

        model = CountNet(
            tx.shape[1],
            21,
        )

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=2e-3,
            weight_decay=1e-3,
        )

        best_state = None
        best_dev_nll = float(
            "inf"
        )

        best_epoch = 0
        patience = 40
        stale = 0

        for epoch in range(
            1,
            501,
        ):
            model.train()

            order = torch.randperm(
                len(tx_t)
            )

            for start in range(
                0,
                len(order),
                128,
            ):
                idx = order[
                    start:start + 128
                ]

                logits = model(
                    tx_t[idx]
                )

                loss = (
                    F.cross_entropy(
                        logits,
                        ty_t[idx],
                    )
                )

                opt.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    2.0,
                )

                opt.step()

            model.eval()

            with torch.no_grad():
                dev_logits = model(
                    dx_t
                )

                dev_target = torch.tensor(
                    dev_k,
                    dtype=torch.long,
                )

                nll = float(
                    F.cross_entropy(
                        dev_logits,
                        dev_target,
                    )
                )

            if (
                nll
                < best_dev_nll
                - 1e-5
            ):
                best_dev_nll = nll
                best_epoch = epoch

                best_state = {
                    k:
                        v.detach()
                        .clone()

                    for k, v
                    in model
                    .state_dict()
                    .items()
                }

                stale = 0

            else:
                stale += 1

            if stale >= patience:
                break

        model.load_state_dict(
            best_state
        )

        model.eval()

        models.append(
            model
        )

        reports.append({
            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "best_dev_nll":
                best_dev_nll,
        })

    return (
        models,
        mu,
        sd,
        reports,
    )


@torch.inference_mode()
def ensemble_probs(
    models,
    x,
    mu,
    sd,
):
    xx = (
        x - mu
    ) / sd

    xx = torch.tensor(
        xx,
        dtype=torch.float32,
    )

    probs = []

    for model in models:
        p = torch.softmax(
            model(xx),
            dim=1,
        )

        probs.append(
            p.numpy()
        )

    return np.mean(
        probs,
        axis=0,
    )


def penalized_counts(
    probs,
    lam,
):
    counts = np.arange(
        probs.shape[1],
        dtype=np.float64,
    )

    logp = np.log(
        np.clip(
            probs,
            1e-12,
            1.0,
        )
    )

    utility = (
        logp
        - lam
        * counts.reshape(
            1,
            -1,
        )
    )

    return np.argmax(
        utility,
        axis=1,
    )


def choose_lambda(
    dev_probs,
    dev_scores,
    dev_y,
    target_recall,
):
    candidates = np.linspace(
        -0.50,
        1.75,
        181,
    )

    feasible = []

    fallback = []

    for lam in candidates:
        k = penalized_counts(
            dev_probs,
            lam,
        )

        pred = decode_topk(
            dev_scores,
            k,
        )

        m = set_metrics(
            pred,
            dev_y,
        )

        row = {
            "lambda":
                float(lam),

            **m,
        }

        fallback.append(
            row
        )

        if (
            m["recall"]
            >= target_recall
        ):
            feasible.append(
                row
            )

    if feasible:
        # Primary objective:
        # minimum false positives subject to recall constraint.
        #
        # Tie-break:
        # greater exact set, then precision.
        best = sorted(
            feasible,
            key=lambda x: (
                x[
                    "avg_fp_per_output"
                ],
                -x[
                    "exact_activation_set"
                ],
                -x[
                    "precision"
                ],
            ),
        )[0]

    else:
        # If 0.80 is impossible, maximize F1.
        def f1(x):
            p = x["precision"]
            r = x["recall"]

            return (
                2 * p * r / (p + r)
                if p + r
                else 0.0
            )

        best = max(
            fallback,
            key=f1,
        )

    return best


def build_audited_test(
    full_test,
):
    audit = read_jsonl(
        AUDIT
    )

    audit_test = [
        x
        for x in audit
        if x["split"] == "test"
    ]

    lookup = {}

    for row in full_test:
        key = (
            str(
                row.get(
                    "session_id"
                )
            ),
            norm_q(
                row.get(
                    "query"
                )
            ),
        )

        lookup.setdefault(
            key,
            []
        ).append(row)

    rows = []
    ids = []

    for a in audit_test:
        key = (
            str(
                a.get(
                    "session_id"
                )
            ),
            norm_q(
                a.get(
                    "query"
                )
            ),
        )

        matches = lookup.get(
            key,
            []
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"audit join failure "
                f"id={a['audit_id']} "
                f"matches={len(matches)}"
            )

        rows.append(
            matches[0]
        )

        ids.append(
            int(
                a["audit_id"]
            )
        )

    strict_idx = [
        i
        for i, aid in enumerate(
            ids
        )
        if (
            aid not in CAPABILITY_IDS
            and aid not in DIRECTION_IDS
        )
    ]

    return (
        rows,
        ids,
        strict_idx,
    )


def evaluate_decoder(
    name,
    scores,
    labels,
    probs,
    lam,
):
    true_k = labels.sum(
        axis=1
    ).astype(
        np.int64
    )

    plain_k = np.argmax(
        probs,
        axis=1,
    )

    tuned_k = penalized_counts(
        probs,
        lam,
    )

    plain_pred = decode_topk(
        scores,
        plain_k,
    )

    tuned_pred = decode_topk(
        scores,
        tuned_k,
    )

    oracle_pred = decode_topk(
        scores,
        true_k,
    )

    return {
        "name":
            name,

        "plain_count": {
            **count_metrics(
                true_k,
                plain_k,
            ),

            **set_metrics(
                plain_pred,
                labels,
            ),
        },

        "fp_tuned_count": {
            "lambda":
                float(lam),

            **count_metrics(
                true_k,
                tuned_k,
            ),

            **set_metrics(
                tuned_pred,
                labels,
            ),
        },

        "oracle_true_count": {
            **set_metrics(
                oracle_pred,
                labels,
            ),
        },
    }


def main():
    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    role_desc = dict(
        tr.ROLE_DESCRIPTIONS
    )

    roles = list(
        role_desc.keys()
    )

    train_all = read_jsonl(
        DATA_ROOT
        / "train.jsonl"
    )

    dev_all = read_jsonl(
        DATA_ROOT
        / "dev.jsonl"
    )

    test_all = read_jsonl(
        DATA_ROOT
        / "test.jsonl"
    )

    train = sample_fixed(
        train_all,
        TRAIN_LIMIT,
        SEED + 11,
    )

    dev = sample_fixed(
        dev_all,
        DEV_LIMIT,
        SEED + 101,
    )

    test = sample_fixed(
        test_all,
        TEST_LIMIT,
        SEED + 202,
    )

    (
        audited,
        audited_ids,
        strict_idx,
    ) = build_audited_test(
        test_all
    )

    print(
        "===== DATA =====",
        flush=True,
    )

    print(
        "train/dev/test/audit/strict:",
        len(train),
        len(dev),
        len(test),
        len(audited),
        len(strict_idx),
        flush=True,
    )

    print(
        "\n===== LOAD FROZEN SENSOR =====",
        flush=True,
    )

    tok, sensor = load_sensor()

    print(
        "\n===== SCORE TRAIN =====",
        flush=True,
    )

    train_raw = score_rows(
        sensor,
        tok,
        train,
        roles,
        role_desc,
    )

    print(
        "\n===== SCORE DEV =====",
        flush=True,
    )

    dev_raw = score_rows(
        sensor,
        tok,
        dev,
        roles,
        role_desc,
    )

    print(
        "\n===== SCORE TEST =====",
        flush=True,
    )

    test_raw = score_rows(
        sensor,
        tok,
        test,
        roles,
        role_desc,
    )

    print(
        "\n===== SCORE AUDIT =====",
        flush=True,
    )

    audit_raw = score_rows(
        sensor,
        tok,
        audited,
        roles,
        role_desc,
    )

    train_y = labels_matrix(
        train,
        roles,
    )

    dev_y = labels_matrix(
        dev,
        roles,
    )

    test_y = labels_matrix(
        test,
        roles,
    )

    audit_y = labels_matrix(
        audited,
        roles,
    )

    strict_y = audit_y[
        strict_idx
    ]

    del sensor
    del tok

    torch.cuda.empty_cache()

    print(
        "\n===== APPLY STRUCTURED RERANKER =====",
        flush=True,
    )

    rr = load_reranker(
        roles
    )

    train_struct = rerank(
        rr,
        train_raw,
    )

    dev_struct = rerank(
        rr,
        dev_raw,
    )

    test_struct = rerank(
        rr,
        test_raw,
    )

    audit_struct = rerank(
        rr,
        audit_raw,
    )

    strict_struct = audit_struct[
        strict_idx
    ]

    scale, bias = (
        load_post_calibration(
            roles
        )
    )

    train_cal = apply_calibration(
        train_struct,
        scale,
        bias,
    )

    dev_cal = apply_calibration(
        dev_struct,
        scale,
        bias,
    )

    test_cal = apply_calibration(
        test_struct,
        scale,
        bias,
    )

    audit_cal = apply_calibration(
        audit_struct,
        scale,
        bias,
    )

    strict_cal = audit_cal[
        strict_idx
    ]

    # ========================================================
    # Train count head on structured score geometry.
    # ========================================================

    train_x = make_features(
        train_struct
    )

    dev_x = make_features(
        dev_struct
    )

    test_x = make_features(
        test_struct
    )

    audit_x = make_features(
        audit_struct
    )

    train_k = train_y.sum(
        axis=1
    ).astype(
        np.int64
    )

    dev_k = dev_y.sum(
        axis=1
    ).astype(
        np.int64
    )

    print(
        "\n===== COUNT DISTRIBUTION TRAIN =====",
        flush=True,
    )

    vals, counts = np.unique(
        train_k,
        return_counts=True,
    )

    print(
        dict(
            zip(
                vals.tolist(),
                counts.tolist(),
            )
        ),
        flush=True,
    )

    print(
        "\n===== TRAIN COUNT ENSEMBLE =====",
        flush=True,
    )

    (
        models,
        mu,
        sd,
        model_reports,
    ) = train_count_ensemble(
        train_x,
        train_k,
        dev_x,
        dev_k,
        seeds=[
            SEED + 1,
            SEED + 2,
            SEED + 3,
            SEED + 4,
            SEED + 5,
        ],
    )

    print(
        "count model reports:",
        json.dumps(
            model_reports,
            indent=2,
        ),
        flush=True,
    )

    train_probs = ensemble_probs(
        models,
        train_x,
        mu,
        sd,
    )

    dev_probs = ensemble_probs(
        models,
        dev_x,
        mu,
        sd,
    )

    test_probs = ensemble_probs(
        models,
        test_x,
        mu,
        sd,
    )

    audit_probs = ensemble_probs(
        models,
        audit_x,
        mu,
        sd,
    )

    strict_probs = audit_probs[
        strict_idx
    ]

    # ========================================================
    # Tune count penalty on DEV ONLY.
    # ========================================================

    print(
        "\n===== TUNE FP PENALTY ON DEV =====",
        flush=True,
    )

    best_operating_point = (
        choose_lambda(
            dev_probs,
            dev_struct,
            dev_y,
            TARGET_DEV_RECALL,
        )
    )

    lam = float(
        best_operating_point[
            "lambda"
        ]
    )

    print(
        json.dumps(
            best_operating_point,
            indent=2,
        ),
        flush=True,
    )

    # ========================================================
    # Evaluate structured scores.
    # ========================================================

    struct_test = evaluate_decoder(
        "structured_test",
        test_struct,
        test_y,
        test_probs,
        lam,
    )

    struct_strict = evaluate_decoder(
        "structured_strict_82",
        strict_struct,
        strict_y,
        strict_probs,
        lam,
    )

    # Calibration changes ranking across roles, so also test it.
    cal_test = evaluate_decoder(
        "structured_calibrated_test",
        test_cal,
        test_y,
        test_probs,
        lam,
    )

    cal_strict = evaluate_decoder(
        "structured_calibrated_strict_82",
        strict_cal,
        strict_y,
        strict_probs,
        lam,
    )

    # ========================================================
    # Also expose oracle count ceilings.
    # ========================================================

    result = {
        "experiment":
            "v7_count_decoder_v1",

        "query_used":
            False,

        "sensor_frozen":
            True,

        "active_count_not_installed_count":
            True,

        "count_training_rows":
            len(train),

        "target_dev_recall":
            TARGET_DEV_RECALL,

        "selected_lambda":
            lam,

        "dev_operating_point":
            best_operating_point,

        "count_models":
            model_reports,

        "structured": {
            "test":
                struct_test,

            "strict_82":
                struct_strict,
        },

        "structured_calibrated": {
            "test":
                cal_test,

            "strict_82":
                cal_strict,
        },
    }

    dump(
        OUT / "summary.json",
        result,
    )

    # Save matrices too, so NEXT analysis will not require
    # another 30-minute Qwen scoring pass.
    np.savez_compressed(
        OUT / "score_cache.npz",

        train_raw=train_raw,
        dev_raw=dev_raw,
        test_raw=test_raw,
        audit_raw=audit_raw,

        train_struct=train_struct,
        dev_struct=dev_struct,
        test_struct=test_struct,
        audit_struct=audit_struct,

        train_cal=train_cal,
        dev_cal=dev_cal,
        test_cal=test_cal,
        audit_cal=audit_cal,

        train_y=train_y,
        dev_y=dev_y,
        test_y=test_y,
        audit_y=audit_y,

        strict_idx=np.asarray(
            strict_idx,
            dtype=np.int64,
        ),
    )

    print(
        "\n===== FINAL COUNT DECODER =====",
        flush=True,
    )

    for variant_name, variant in [
        (
            "STRUCTURED",
            result[
                "structured"
            ],
        ),
        (
            "STRUCTURED+CAL",
            result[
                "structured_calibrated"
            ],
        ),
    ]:
        print(
            f"\n### {variant_name}",
            flush=True,
        )

        for subset in [
            "test",
            "strict_82",
        ]:
            r = variant[
                subset
            ]

            print(
                f"\n{subset}",
                flush=True,
            )

            p = r[
                "plain_count"
            ]

            print(
                "  PRED-K:"
                f" count_acc={p['count_accuracy']:.4f}"
                f" MAE={p['count_mae']:.4f}"
                f" precision={p['precision']:.4f}"
                f" recall={p['recall']:.4f}"
                f" FPR={p['fpr']:.4f}"
                f" FP/output={p['avg_fp_per_output']:.4f}"
                f" exact={p['exact_activation_set']:.4f}",
                flush=True,
            )

            q = r[
                "fp_tuned_count"
            ]

            print(
                "  FP-TUNED:"
                f" lambda={q['lambda']:.4f}"
                f" count_acc={q['count_accuracy']:.4f}"
                f" MAE={q['count_mae']:.4f}"
                f" precision={q['precision']:.4f}"
                f" recall={q['recall']:.4f}"
                f" FPR={q['fpr']:.4f}"
                f" FP/output={q['avg_fp_per_output']:.4f}"
                f" exact={q['exact_activation_set']:.4f}",
                flush=True,
            )

            o = r[
                "oracle_true_count"
            ]

            print(
                "  ORACLE-K:"
                f" precision={o['precision']:.4f}"
                f" recall={o['recall']:.4f}"
                f" FPR={o['fpr']:.4f}"
                f" FP/output={o['avg_fp_per_output']:.4f}"
                f" exact={o['exact_activation_set']:.4f}",
                flush=True,
            )

    print(
        "\nSaved:",
        OUT / "summary.json",
        OUT / "score_cache.npz",
        flush=True,
    )


if __name__ == "__main__":
    main()
