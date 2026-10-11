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


CACHE = Path(
    "artifacts/v7_count_decoder_v1/score_cache.npz"
)

OUT = Path(
    "artifacts/v7_joint_multilabel_decoder_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
TARGET_RECALL = 0.80

FP_LAMBDAS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.40,
]

SEEDS = [
    SEED + 1,
    SEED + 2,
    SEED + 3,
]


def dump(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def make_features(
    raw,
    struct,
    cal,
):
    sorted_s = np.sort(
        struct,
        axis=1,
    )[:, ::-1]

    gaps = (
        sorted_s[:, :-1]
        - sorted_s[:, 1:]
    )

    summary = np.stack(
        [
            struct.mean(axis=1),
            struct.std(axis=1),
            struct.max(axis=1),
            struct.min(axis=1),
            np.median(
                struct,
                axis=1,
            ),
        ],
        axis=1,
    )

    return np.concatenate(
        [
            raw,
            struct,
            cal,
            sorted_s,
            gaps,
            summary,
        ],
        axis=1,
    ).astype(
        np.float32
    )


class Decoder(nn.Module):
    def __init__(
        self,
        d_in,
        n_roles,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                d_in,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),

            nn.Linear(
                64,
                n_roles,
            ),
        )

    def forward(self, x):
        return self.net(x)


def pairwise_loss(
    logits,
    y,
):
    losses = []

    for i in range(
        logits.shape[0]
    ):
        pos = torch.where(
            y[i] > 0.5
        )[0]

        neg = torch.where(
            y[i] < 0.5
        )[0]

        if (
            len(pos) == 0
            or len(neg) == 0
        ):
            continue

        d = (
            logits[i, pos]
            .unsqueeze(1)
            -
            logits[i, neg]
            .unsqueeze(0)
        )

        losses.append(
            F.softplus(
                -d
            ).mean()
        )

    return torch.stack(
        losses
    ).mean()


def set_metrics(
    pred,
    y,
):
    pred = pred.astype(
        np.int64
    )

    y = y.astype(
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

        "fp":
            fp,

        "fn":
            fn,

        "avg_fp_per_output":
            float(
                fp / len(y)
            ),

        "avg_fn_per_output":
            float(
                fn / len(y)
            ),

        "exact_set":
            float(
                exact.mean()
            ),
    }


def within_auc(
    probs,
    y,
):
    vals = []

    for i in range(
        len(probs)
    ):
        pos = probs[i][
            y[i] == 1
        ]

        neg = probs[i][
            y[i] == 0
        ]

        if (
            len(pos) == 0
            or len(neg) == 0
        ):
            continue

        wins = (
            pos[:, None]
            > neg[None, :]
        ).mean()

        ties = (
            pos[:, None]
            == neg[None, :]
        ).mean()

        vals.append(
            float(
                wins
                + 0.5 * ties
            )
        )

    return float(
        np.mean(vals)
    )


def oracle_k_metrics(
    probs,
    y,
):
    pred = np.zeros_like(
        y,
        dtype=np.int64,
    )

    for i in range(
        len(y)
    ):
        k = int(
            y[i].sum()
        )

        if k <= 0:
            continue

        idx = np.argsort(
            -probs[i]
        )[:k]

        pred[i, idx] = 1

    return set_metrics(
        pred,
        y,
    )


def tune_threshold(
    probs,
    y,
    target_recall,
):
    # Include extreme low thresholds so target recall
    # is always measurable rather than silently falling back.
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.001,
                    0.999,
                    999,
                ),
                probs.reshape(-1),
            ]
        )
    )

    feasible = []

    all_rows = []

    for t in thresholds:
        pred = (
            probs >= t
        ).astype(
            np.int64
        )

        m = set_metrics(
            pred,
            y,
        )

        row = {
            "threshold":
                float(t),

            **m,
        }

        all_rows.append(
            row
        )

        if (
            m["recall"]
            >= target_recall
        ):
            feasible.append(
                row
            )

    if not feasible:
        raise RuntimeError(
            "No threshold reached target recall; "
            "this should be impossible with threshold near zero."
        )

    best = min(
        feasible,
        key=lambda x: (
            x["fpr"],
            -x["precision"],
            -x["exact_set"],
        ),
    )

    return best


def train_one(
    train_x,
    train_y,
    dev_x,
    dev_y,
    fp_lambda,
    seed,
):
    torch.manual_seed(
        seed
    )

    model = Decoder(
        train_x.shape[1],
        train_y.shape[1],
    )

    tx = torch.tensor(
        train_x,
        dtype=torch.float32,
    )

    ty = torch.tensor(
        train_y,
        dtype=torch.float32,
    )

    dx = torch.tensor(
        dev_x,
        dtype=torch.float32,
    )

    # Moderate positive weighting:
    # preserve recall without giving positives 9x weight.
    positives = train_y.sum(
        axis=0
    )

    negatives = (
        len(train_y)
        - positives
    )

    pos_weight = (
        negatives
        / np.maximum(
            positives,
            1,
        )
    )

    pos_weight = np.clip(
        pos_weight,
        1.0,
        4.0,
    )

    pos_weight_t = torch.tensor(
        pos_weight,
        dtype=torch.float32,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )

    best_state = None
    best_dev_loss = float(
        "inf"
    )

    best_epoch = 0
    stale = 0
    patience = 35

    for epoch in range(
        1,
        401,
    ):
        model.train()

        order = torch.randperm(
            len(tx)
        )

        for start in range(
            0,
            len(tx),
            128,
        ):
            idx = order[
                start:start + 128
            ]

            xb = tx[idx]
            yb = ty[idx]

            logits = model(
                xb
            )

            bce = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    yb,
                    pos_weight=pos_weight_t,
                )
            )

            rank = pairwise_loss(
                logits,
                yb,
            )

            probs = torch.sigmoid(
                logits
            )

            true_count = yb.sum(
                dim=1
            )

            pred_count = probs.sum(
                dim=1
            )

            count_loss = (
                (
                    pred_count
                    - true_count
                ) ** 2
            ).mean() / 16.0

            neg_mask = (
                1.0 - yb
            )

            fp_soft = (
                (
                    probs
                    * neg_mask
                ).sum()
                /
                torch.clamp(
                    neg_mask.sum(),
                    min=1.0,
                )
            )

            loss = (
                bce
                + 0.20 * rank
                + 0.05 * count_loss
                + fp_lambda * fp_soft
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
            dlogits = model(
                dx
            )

            dbce = (
                F.binary_cross_entropy_with_logits(
                    dlogits,
                    torch.tensor(
                        dev_y,
                        dtype=torch.float32,
                    ),
                    pos_weight=pos_weight_t,
                )
            )

            dprob = torch.sigmoid(
                dlogits
            )

            dy = torch.tensor(
                dev_y,
                dtype=torch.float32,
            )

            drank = pairwise_loss(
                dlogits,
                dy,
            )

            dcount = (
                (
                    dprob.sum(dim=1)
                    - dy.sum(dim=1)
                ) ** 2
            ).mean() / 16.0

            dneg = (
                1.0 - dy
            )

            dfp = (
                (
                    dprob
                    * dneg
                ).sum()
                /
                torch.clamp(
                    dneg.sum(),
                    min=1.0,
                )
            )

            dloss = (
                dbce
                + 0.20 * drank
                + 0.05 * dcount
                + fp_lambda * dfp
            )

            dval = float(
                dloss
            )

        if (
            dval
            < best_dev_loss
            - 1e-5
        ):
            best_dev_loss = dval
            best_epoch = epoch

            best_state = {
                k:
                    v.detach()
                    .clone()

                for k, v
                in model.state_dict()
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

    return (
        model,
        {
            "seed":
                seed,

            "fp_lambda":
                fp_lambda,

            "best_epoch":
                best_epoch,

            "best_dev_loss":
                best_dev_loss,
        },
    )


@torch.inference_mode()
def predict_ensemble(
    models,
    x,
):
    xt = torch.tensor(
        x,
        dtype=torch.float32,
    )

    probs = []

    for m in models:
        probs.append(
            torch.sigmoid(
                m(xt)
            ).numpy()
        )

    return np.mean(
        probs,
        axis=0,
    )


def normalize_features(
    train_x,
    *others,
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

    train_z = (
        train_x - mu
    ) / sd

    out = [
        train_z
    ]

    for x in others:
        out.append(
            (x - mu) / sd
        )

    return (
        *out,
        mu,
        sd,
    )


def evaluate_split(
    probs,
    y,
    threshold,
):
    pred = (
        probs
        >= threshold
    ).astype(
        np.int64
    )

    return {
        "within_output_auc":
            within_auc(
                probs,
                y,
            ),

        "threshold_metrics":
            set_metrics(
                pred,
                y,
            ),

        "oracle_k":
            oracle_k_metrics(
                probs,
                y,
            ),
    }


def main():
    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    cache = np.load(
        CACHE
    )

    train_raw = cache[
        "train_raw"
    ]

    dev_raw = cache[
        "dev_raw"
    ]

    test_raw = cache[
        "test_raw"
    ]

    audit_raw = cache[
        "audit_raw"
    ]

    train_struct = cache[
        "train_struct"
    ]

    dev_struct = cache[
        "dev_struct"
    ]

    test_struct = cache[
        "test_struct"
    ]

    audit_struct = cache[
        "audit_struct"
    ]

    train_cal = cache[
        "train_cal"
    ]

    dev_cal = cache[
        "dev_cal"
    ]

    test_cal = cache[
        "test_cal"
    ]

    audit_cal = cache[
        "audit_cal"
    ]

    train_y = cache[
        "train_y"
    ].astype(
        np.int64
    )

    dev_y = cache[
        "dev_y"
    ].astype(
        np.int64
    )

    test_y = cache[
        "test_y"
    ].astype(
        np.int64
    )

    audit_y = cache[
        "audit_y"
    ].astype(
        np.int64
    )

    strict_idx = cache[
        "strict_idx"
    ].astype(
        np.int64
    )

    strict_y = audit_y[
        strict_idx
    ]

    train_x = make_features(
        train_raw,
        train_struct,
        train_cal,
    )

    dev_x = make_features(
        dev_raw,
        dev_struct,
        dev_cal,
    )

    test_x = make_features(
        test_raw,
        test_struct,
        test_cal,
    )

    audit_x = make_features(
        audit_raw,
        audit_struct,
        audit_cal,
    )

    (
        train_x,
        dev_x,
        test_x,
        audit_x,
        mu,
        sd,
    ) = normalize_features(
        train_x,
        dev_x,
        test_x,
        audit_x,
    )

    print(
        "===== DATA ====="
    )

    print(
        "feature_dim:",
        train_x.shape[1]
    )

    print(
        "train/dev/test/audit/strict:",
        len(train_x),
        len(dev_x),
        len(test_x),
        len(audit_x),
        len(strict_idx),
    )

    variants = []

    for fp_lambda in FP_LAMBDAS:
        print(
            f"\n===== FP LAMBDA {fp_lambda} =====",
            flush=True,
        )

        models = []
        train_reports = []

        for seed in SEEDS:
            model, report = train_one(
                train_x,
                train_y,
                dev_x,
                dev_y,
                fp_lambda,
                seed,
            )

            models.append(
                model
            )

            train_reports.append(
                report
            )

            print(
                report,
                flush=True,
            )

        dev_probs = predict_ensemble(
            models,
            dev_x,
        )

        op = tune_threshold(
            dev_probs,
            dev_y,
            TARGET_RECALL,
        )

        print(
            "DEV OPERATING POINT:",
            json.dumps(
                op,
                indent=2,
            ),
            flush=True,
        )

        variants.append({
            "fp_lambda":
                fp_lambda,

            "models":
                models,

            "reports":
                train_reports,

            "dev_probs":
                dev_probs,

            "dev_operating_point":
                op,
        })

    # Select ONLY on dev:
    # min FPR subject to recall >= .80.
    best = min(
        variants,
        key=lambda v: (
            v[
                "dev_operating_point"
            ]["fpr"],

            -v[
                "dev_operating_point"
            ]["precision"],

            -v[
                "dev_operating_point"
            ]["exact_set"],
        ),
    )

    best_lambda = best[
        "fp_lambda"
    ]

    threshold = best[
        "dev_operating_point"
    ][
        "threshold"
    ]

    models = best[
        "models"
    ]

    print(
        "\n===== SELECTED ON DEV =====",
        flush=True,
    )

    print(
        "fp_lambda:",
        best_lambda,
        flush=True,
    )

    print(
        "threshold:",
        threshold,
        flush=True,
    )

    print(
        "dev:",
        json.dumps(
            best[
                "dev_operating_point"
            ],
            indent=2,
        ),
        flush=True,
    )

    test_probs = predict_ensemble(
        models,
        test_x,
    )

    audit_probs = predict_ensemble(
        models,
        audit_x,
    )

    strict_probs = audit_probs[
        strict_idx
    ]

    test_result = evaluate_split(
        test_probs,
        test_y,
        threshold,
    )

    strict_result = evaluate_split(
        strict_probs,
        strict_y,
        threshold,
    )

    result = {
        "experiment":
            "v7_joint_multilabel_decoder_v1",

        "query_used":
            False,

        "qwen_calls":
            0,

        "target_dev_recall":
            TARGET_RECALL,

        "selected_fp_lambda":
            best_lambda,

        "selected_threshold":
            threshold,

        "selected_dev_operating_point":
            best[
                "dev_operating_point"
            ],

        "test":
            test_result,

        "strict_82":
            strict_result,

        "all_variant_dev_results": [
            {
                "fp_lambda":
                    v["fp_lambda"],

                "dev_operating_point":
                    v[
                        "dev_operating_point"
                    ],

                "training":
                    v["reports"],
            }
            for v in variants
        ],
    }

    dump(
        OUT / "summary.json",
        result,
    )

    print(
        "\n===== FINAL MATCHED-RECALL RESULTS =====",
        flush=True,
    )

    for name, r in [
        (
            "TEST",
            test_result,
        ),
        (
            "STRICT_82",
            strict_result,
        ),
    ]:
        m = r[
            "threshold_metrics"
        ]

        o = r[
            "oracle_k"
        ]

        print(
            f"\n{name}",
            flush=True,
        )

        print(
            "  within_auc="
            f"{r['within_output_auc']:.4f}",
            flush=True,
        )

        print(
            "  THRESHOLD:"
            f" precision={m['precision']:.4f}"
            f" recall={m['recall']:.4f}"
            f" FPR={m['fpr']:.4f}"
            f" FP/output={m['avg_fp_per_output']:.4f}"
            f" FN/output={m['avg_fn_per_output']:.4f}"
            f" exact={m['exact_set']:.4f}",
            flush=True,
        )

        print(
            "  ORACLE-K:"
            f" precision={o['precision']:.4f}"
            f" recall={o['recall']:.4f}"
            f" FPR={o['fpr']:.4f}"
            f" FP/output={o['avg_fp_per_output']:.4f}"
            f" exact={o['exact_set']:.4f}",
            flush=True,
        )

    print(
        "\nSaved:",
        OUT / "summary.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
