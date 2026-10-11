#!/usr/bin/env python3

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CACHE = Path(
    "artifacts/v7_count_decoder_v1/score_cache.npz"
)

OUT = Path(
    "artifacts/v7_targeted_toprank_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926


# ============================================================
# Metrics
# ============================================================

def dump(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def ranks_from_scores(scores):
    order = np.argsort(
        -scores,
        axis=1,
    )

    ranks = np.empty_like(
        order,
        dtype=np.int64,
    )

    rows = np.arange(
        len(scores)
    )[:, None]

    ranks[
        rows,
        order,
    ] = (
        np.arange(
            scores.shape[1]
        )[None, :]
        + 1
    )

    return ranks


def within_auc(scores, y):
    values = []

    for i in range(
        len(scores)
    ):
        pos = scores[i][
            y[i] == 1
        ]

        neg = scores[i][
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

        values.append(
            float(
                wins
                + 0.5 * ties
            )
        )

    return float(
        np.mean(values)
    )


def recall_at_k(
    scores,
    y,
    k,
):
    ranks = ranks_from_scores(
        scores
    )

    active = (
        y == 1
    )

    hits = (
        active
        & (ranks <= k)
    ).sum()

    total = active.sum()

    return float(
        hits / total
    )


def oracle_k_metrics(
    scores,
    y,
):
    pred = np.zeros_like(
        y,
        dtype=np.int64,
    )

    for i in range(
        len(scores)
    ):
        k = int(
            y[i].sum()
        )

        if k <= 0:
            continue

        idx = np.argsort(
            -scores[i]
        )[:k]

        pred[
            i,
            idx,
        ] = 1

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

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    exact = np.all(
        pred == y,
        axis=1,
    )

    return {
        "recall":
            float(recall),

        "exact":
            float(
                exact.mean()
            ),

        "fp_per_output":
            float(
                fp / len(y)
            ),
    }


def metrics(scores, y):
    o = oracle_k_metrics(
        scores,
        y,
    )

    return {
        "within_auc":
            within_auc(
                scores,
                y,
            ),

        "recall_at_3":
            recall_at_k(
                scores,
                y,
                3,
            ),

        "recall_at_5":
            recall_at_k(
                scores,
                y,
                5,
            ),

        "recall_at_7":
            recall_at_k(
                scores,
                y,
                7,
            ),

        "recall_at_8":
            recall_at_k(
                scores,
                y,
                8,
            ),

        "oracle_k_recall":
            o["recall"],

        "oracle_k_exact":
            o["exact"],

        "oracle_k_fp_per_output":
            o[
                "fp_per_output"
            ],
    }


# ============================================================
# Features
#
# Conservative contextual residual:
#   current structured scores
#   normalized score vector
#   current ranks
#
# No text/query/output access.
# ============================================================

def build_features(scores):
    mean = scores.mean(
        axis=1,
        keepdims=True,
    )

    std = scores.std(
        axis=1,
        keepdims=True,
    )

    z = (
        scores - mean
    ) / np.maximum(
        std,
        1e-5,
    )

    ranks = ranks_from_scores(
        scores
    ).astype(
        np.float32
    )

    ranks = (
        ranks - 1.0
    ) / (
        scores.shape[1]
        - 1.0
    )

    return np.concatenate(
        [
            scores,
            z,
            ranks,
        ],
        axis=1,
    ).astype(
        np.float32
    )


class TargetedRanker(nn.Module):
    def __init__(
        self,
        d_in,
        n_roles,
    ):
        super().__init__()

        # Intentionally small.
        self.hidden = nn.Linear(
            d_in,
            32,
        )

        self.out = nn.Linear(
            32,
            n_roles,
        )

        # Identity at initialization.
        nn.init.zeros_(
            self.out.weight
        )

        nn.init.zeros_(
            self.out.bias
        )

    def forward(
        self,
        x,
        base,
    ):
        h = torch.tanh(
            self.hidden(x)
        )

        correction = self.out(
            h
        )

        return (
            base + correction,
            correction,
        )


# ============================================================
# Targeted loss
#
# 1. ordinary selected > absent ranking
#
# 2. EXTRA weight if:
#      true role is currently buried below top-5
#      false role is currently inside top-5
#
# 3. worst-positive vs strongest-negative loss
#    attacks exact/top-K recovery directly.
# ============================================================

def targeted_pair_loss(
    scores,
    y,
    base_ranks,
    hard_weight,
    buried_weight,
):
    diff = (
        scores.unsqueeze(2)
        - scores.unsqueeze(1)
    )

    pos_mask = (
        y.unsqueeze(2)
        > 0.5
    )

    neg_mask = (
        y.unsqueeze(1)
        < 0.5
    )

    mask = (
        pos_mask
        & neg_mask
    )

    # rank of positive role
    pos_rank = (
        base_ranks
        .unsqueeze(2)
    )

    # rank of negative role
    neg_rank = (
        base_ranks
        .unsqueeze(1)
    )

    weight = torch.ones_like(
        diff
    )

    # False role already contaminating top-5.
    weight = (
        weight
        + hard_weight
        * (
            neg_rank <= 5
        ).float()
    )

    # True role buried too deep.
    weight = (
        weight
        + buried_weight
        * (
            pos_rank > 5
        ).float()
    )

    # Both simultaneously = most important error.
    weight = (
        weight
        + (
            hard_weight
            * buried_weight
        )
        * (
            (
                neg_rank <= 5
            )
            &
            (
                pos_rank > 5
            )
        ).float()
    )

    losses = F.softplus(
        -diff
    )

    return (
        (
            losses[mask]
            * weight[mask]
        ).sum()
        /
        weight[mask].sum()
    )


def worst_true_loss(
    scores,
    y,
    margin=0.5,
):
    losses = []

    for i in range(
        scores.shape[0]
    ):
        pos = scores[i][
            y[i] > 0.5
        ]

        neg = scores[i][
            y[i] < 0.5
        ]

        if (
            len(pos) == 0
            or len(neg) == 0
        ):
            continue

        # Smooth approximation:
        # weakest positive vs strongest negative.
        tau = 0.5

        weakest_pos = (
            -tau
            * torch.logsumexp(
                -pos / tau,
                dim=0,
            )
        )

        strongest_neg = (
            tau
            * torch.logsumexp(
                neg / tau,
                dim=0,
            )
        )

        losses.append(
            F.softplus(
                strongest_neg
                - weakest_pos
                + margin
            )
        )

    return torch.stack(
        losses
    ).mean()


# ============================================================
# Training
# ============================================================

def train_variant(
    train_x,
    train_base,
    train_y,
    dev_x,
    dev_base,
    dev_y,
    *,
    seed,
    lr,
    hard_weight,
    buried_weight,
    worst_weight,
):
    torch.manual_seed(
        seed
    )

    model = TargetedRanker(
        train_x.shape[1],
        train_y.shape[1],
    )

    tx = torch.tensor(
        train_x,
        dtype=torch.float32,
    )

    tb = torch.tensor(
        train_base,
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

    db = torch.tensor(
        dev_base,
        dtype=torch.float32,
    )

    train_ranks = torch.tensor(
        ranks_from_scores(
            train_base
        ),
        dtype=torch.float32,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=2e-3,
    )

    baseline = metrics(
        dev_base,
        dev_y,
    )

    # Step zero is always legal.
    best_state = {
        k:
            v.detach().clone()

        for k, v
        in model.state_dict()
        .items()
    }

    best_metric = baseline
    best_epoch = 0

    stale = 0
    patience = 35

    for epoch in range(
        1,
        251,
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

            scores, correction = model(
                tx[idx],
                tb[idx],
            )

            pair = targeted_pair_loss(
                scores,
                ty[idx],
                train_ranks[idx],
                hard_weight,
                buried_weight,
            )

            worst = worst_true_loss(
                scores,
                ty[idx],
            )

            reg = (
                correction
                .pow(2)
                .mean()
            )

            loss = (
                pair
                + worst_weight
                * worst
                + 0.01
                * reg
            )

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

        if (
            epoch % 5 != 0
            and epoch != 1
        ):
            continue

        model.eval()

        with torch.no_grad():
            dev_scores, _ = model(
                dx,
                db,
            )

        dev_scores = (
            dev_scores.numpy()
        )

        dm = metrics(
            dev_scores,
            dev_y,
        )

        # Primary objective is now TOP-RANK recovery.
        #
        # Constraint: don't destroy AUC.
        valid_auc = (
            dm["within_auc"]
            >= baseline["within_auc"]
            - 0.005
        )

        best_valid_auc = (
            best_metric["within_auc"]
            >= baseline["within_auc"]
            - 0.005
        )

        def score_key(m):
            return (
                m[
                    "oracle_k_recall"
                ],
                m[
                    "recall_at_5"
                ],
                m[
                    "oracle_k_exact"
                ],
                m[
                    "within_auc"
                ],
            )

        improved = False

        if valid_auc:
            if not best_valid_auc:
                improved = True

            elif (
                score_key(dm)
                > score_key(
                    best_metric
                )
            ):
                improved = True

        if improved:
            best_state = {
                k:
                    v.detach().clone()

                for k, v
                in model
                .state_dict()
                .items()
            }

            best_metric = dm
            best_epoch = epoch
            stale = 0

        else:
            stale += 5

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

            "lr":
                lr,

            "hard_weight":
                hard_weight,

            "buried_weight":
                buried_weight,

            "worst_weight":
                worst_weight,

            "best_epoch":
                best_epoch,

            "dev":
                best_metric,
        },
    )


@torch.inference_mode()
def predict(
    model,
    x,
    base,
):
    x = torch.tensor(
        x,
        dtype=torch.float32,
    )

    base = torch.tensor(
        base,
        dtype=torch.float32,
    )

    scores, _ = model(
        x,
        base,
    )

    return scores.numpy()


def normalize(
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

    out = [
        (
            train_x - mu
        ) / sd
    ]

    for x in others:
        out.append(
            (
                x - mu
            ) / sd
        )

    return (
        *out,
        mu,
        sd,
    )


# ============================================================
# Blend targeted ranking with existing calibrated ranking.
#
# This is selected ONLY on dev.
# ============================================================

def search_blend(
    dev_target,
    dev_cal,
    dev_y,
):
    baseline = metrics(
        dev_cal,
        dev_y,
    )

    best = {
        "alpha":
            0.0,

        "metrics":
            baseline,
    }

    for alpha in np.linspace(
        0.0,
        1.5,
        151,
    ):
        score = (
            dev_cal
            + alpha
            * (
                dev_target
                - dev_cal
            )
        )

        m = metrics(
            score,
            dev_y,
        )

        # Preserve essentially all AUC quality.
        if (
            m["within_auc"]
            < baseline["within_auc"]
            - 0.005
        ):
            continue

        key = (
            m[
                "oracle_k_recall"
            ],
            m[
                "recall_at_5"
            ],
            m[
                "oracle_k_exact"
            ],
            m[
                "within_auc"
            ],
        )

        best_key = (
            best["metrics"][
                "oracle_k_recall"
            ],
            best["metrics"][
                "recall_at_5"
            ],
            best["metrics"][
                "oracle_k_exact"
            ],
            best["metrics"][
                "within_auc"
            ],
        )

        if key > best_key:
            best = {
                "alpha":
                    float(alpha),

                "metrics":
                    m,
            }

    return best


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

    train_base = cache[
        "train_struct"
    ]

    dev_base = cache[
        "dev_struct"
    ]

    test_base = cache[
        "test_struct"
    ]

    audit_base = cache[
        "audit_struct"
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

    train_x = build_features(
        train_base
    )

    dev_x = build_features(
        dev_base
    )

    test_x = build_features(
        test_base
    )

    audit_x = build_features(
        audit_base
    )

    (
        train_x,
        dev_x,
        test_x,
        audit_x,
        mu,
        sd,
    ) = normalize(
        train_x,
        dev_x,
        test_x,
        audit_x,
    )

    print(
        "===== CURRENT BASELINES =====",
        flush=True,
    )

    print(
        "DEV structured:",
        json.dumps(
            metrics(
                dev_base,
                dev_y,
            )
        ),
        flush=True,
    )

    print(
        "DEV calibrated:",
        json.dumps(
            metrics(
                dev_cal,
                dev_y,
            )
        ),
        flush=True,
    )

    # Small hyperparameter sweep.
    configs = []

    for (
        lr,
        hard_w,
        buried_w,
        worst_w,
    ) in itertools.product(
        [
            0.001,
            0.003,
        ],
        [
            1.0,
            2.0,
        ],
        [
            1.0,
            2.0,
        ],
        [
            0.10,
            0.25,
            0.50,
        ],
    ):
        configs.append(
            (
                lr,
                hard_w,
                buried_w,
                worst_w,
            )
        )

    runs = []

    for i, (
        lr,
        hard_w,
        buried_w,
        worst_w,
    ) in enumerate(
        configs
    ):
        seed = (
            SEED
            + 100
            + i
        )

        print(
            "\n===== CONFIG "
            f"{i+1}/{len(configs)} "
            f"lr={lr} "
            f"hard={hard_w} "
            f"buried={buried_w} "
            f"worst={worst_w} "
            "=====",
            flush=True,
        )

        model, report = (
            train_variant(
                train_x,
                train_base,
                train_y,
                dev_x,
                dev_base,
                dev_y,
                seed=seed,
                lr=lr,
                hard_weight=hard_w,
                buried_weight=buried_w,
                worst_weight=worst_w,
            )
        )

        runs.append(
            (
                model,
                report,
            )
        )

        print(
            json.dumps(
                report,
                indent=2,
            ),
            flush=True,
        )

    # ========================================================
    # Pick targeted model ONLY on dev.
    # ========================================================

    baseline_auc = metrics(
        dev_cal,
        dev_y,
    )[
        "within_auc"
    ]

    valid = []

    for model, report in runs:
        dm = report[
            "dev"
        ]

        if (
            dm["within_auc"]
            >= baseline_auc - 0.01
        ):
            valid.append(
                (
                    model,
                    report,
                )
            )

    if not valid:
        valid = runs

    model, report = max(
        valid,
        key=lambda mr: (
            mr[1]["dev"][
                "oracle_k_recall"
            ],
            mr[1]["dev"][
                "recall_at_5"
            ],
            mr[1]["dev"][
                "oracle_k_exact"
            ],
            mr[1]["dev"][
                "within_auc"
            ],
        ),
    )

    dev_target = predict(
        model,
        dev_x,
        dev_base,
    )

    test_target = predict(
        model,
        test_x,
        test_base,
    )

    audit_target = predict(
        model,
        audit_x,
        audit_base,
    )

    # ========================================================
    # Blend with current high-AUC calibrated system.
    # ========================================================

    blend = search_blend(
        dev_target,
        dev_cal,
        dev_y,
    )

    alpha = blend[
        "alpha"
    ]

    dev_final = (
        dev_cal
        + alpha
        * (
            dev_target
            - dev_cal
        )
    )

    test_final = (
        test_cal
        + alpha
        * (
            test_target
            - test_cal
        )
    )

    audit_final = (
        audit_cal
        + alpha
        * (
            audit_target
            - audit_cal
        )
    )

    strict_final = audit_final[
        strict_idx
    ]

    result = {
        "experiment":
            "v7_targeted_toprank_v1",

        "qwen_calls":
            0,

        "selected_only_on_dev":
            True,

        "selected_training_config":
            report,

        "selected_blend_alpha":
            alpha,

        "dev_blend_metrics":
            blend["metrics"],

        "baseline_calibrated": {
            "dev":
                metrics(
                    dev_cal,
                    dev_y,
                ),

            "test":
                metrics(
                    test_cal,
                    test_y,
                ),

            "strict_82":
                metrics(
                    audit_cal[
                        strict_idx
                    ],
                    strict_y,
                ),
        },

        "targeted_final": {
            "dev":
                metrics(
                    dev_final,
                    dev_y,
                ),

            "test":
                metrics(
                    test_final,
                    test_y,
                ),

            "strict_82":
                metrics(
                    strict_final,
                    strict_y,
                ),
        },

        "all_runs": [
            r
            for _, r
            in runs
        ],
    }

    dump(
        OUT / "summary.json",
        result,
    )

    torch.save(
        {
            "model_state":
                model.state_dict(),

            "feature_mean":
                mu,

            "feature_std":
                sd,

            "blend_alpha":
                alpha,

            "selected_config":
                report,
        },
        OUT / "targeted_ranker.pt",
    )

    print(
        "\n===== FINAL TARGETED RESULTS =====",
        flush=True,
    )

    print(
        "SELECTED CONFIG:",
        json.dumps(
            report,
            ensure_ascii=False,
        ),
        flush=True,
    )

    print(
        "BLEND ALPHA:",
        alpha,
        flush=True,
    )

    for split in [
        "dev",
        "test",
        "strict_82",
    ]:
        b = result[
            "baseline_calibrated"
        ][split]

        n = result[
            "targeted_final"
        ][split]

        print(
            f"\n{split.upper()}",
            flush=True,
        )

        print(
            "  BASE:"
            f" AUC={b['within_auc']:.4f}"
            f" R@3={b['recall_at_3']:.4f}"
            f" R@5={b['recall_at_5']:.4f}"
            f" R@7={b['recall_at_7']:.4f}"
            f" oracleR={b['oracle_k_recall']:.4f}"
            f" oracleExact={b['oracle_k_exact']:.4f}",
            flush=True,
        )

        print(
            "  TARGET:"
            f" AUC={n['within_auc']:.4f}"
            f" R@3={n['recall_at_3']:.4f}"
            f" R@5={n['recall_at_5']:.4f}"
            f" R@7={n['recall_at_7']:.4f}"
            f" oracleR={n['oracle_k_recall']:.4f}"
            f" oracleExact={n['oracle_k_exact']:.4f}",
            flush=True,
        )

    print(
        "\nSaved:",
        OUT / "summary.json",
        OUT / "targeted_ranker.pt",
        flush=True,
    )


if __name__ == "__main__":
    main()
