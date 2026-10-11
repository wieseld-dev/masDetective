#!/usr/bin/env python3

from __future__ import annotations

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
    "artifacts/v7_relative_threshold_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
TARGET_RECALL = 0.80


def dump(path, x):
    Path(path).write_text(
        json.dumps(
            x,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def metrics(pred, y):
    pred = pred.astype(np.int64)
    y = y.astype(np.int64)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    exact = np.all(
        pred == y,
        axis=1,
    ).mean()

    return {
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "fp_per_output": float(fp / len(y)),
        "fn_per_output": float(fn / len(y)),
        "mean_predicted_roles": float(pred.sum(axis=1).mean()),
        "exact_set": float(exact),
    }


def within_auc(scores, y):
    out = []

    for i in range(len(scores)):
        p = scores[i][y[i] == 1]
        n = scores[i][y[i] == 0]

        if not len(p) or not len(n):
            continue

        wins = (p[:, None] > n[None, :]).mean()
        ties = (p[:, None] == n[None, :]).mean()

        out.append(
            float(wins + 0.5 * ties)
        )

    return float(np.mean(out))


# ============================================================
# Relative score representations.
# All are monotonic WITHIN one output, so ranking is preserved.
# ============================================================

def transforms(scores):
    eps = 1e-6

    mean = scores.mean(
        axis=1,
        keepdims=True,
    )

    std = scores.std(
        axis=1,
        keepdims=True,
    )

    median = np.median(
        scores,
        axis=1,
        keepdims=True,
    )

    mad = np.median(
        np.abs(
            scores - median
        ),
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

    result = {
        "identity":
            scores,

        "center_mean":
            scores - mean,

        "center_median":
            scores - median,

        "zscore":
            (scores - mean)
            / np.maximum(
                std,
                eps,
            ),

        "robust_z":
            (scores - median)
            / np.maximum(
                1.4826 * mad,
                eps,
            ),

        "relative_to_max":
            scores - mx,

        "minmax":
            (scores - mn)
            / np.maximum(
                mx - mn,
                eps,
            ),
    }

    for temp in [
        0.25,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
    ]:
        z = scores / temp

        z = z - z.max(
            axis=1,
            keepdims=True,
        )

        ez = np.exp(
            np.clip(
                z,
                -50,
                50,
            )
        )

        p = ez / ez.sum(
            axis=1,
            keepdims=True,
        )

        result[
            f"softmax_T{temp}"
        ] = p

    return result


def threshold_search(
    scores,
    y,
    target_recall,
):
    flat = scores.reshape(-1)

    # Quantile candidates are enough and much faster
    # than evaluating every distinct float.
    qs = np.linspace(
        0.0,
        1.0,
        2001,
    )

    thresholds = np.unique(
        np.quantile(
            flat,
            qs,
        )
    )

    best = None

    for threshold in thresholds:
        pred = (
            scores >= threshold
        ).astype(np.int64)

        m = metrics(
            pred,
            y,
        )

        if (
            m["recall"]
            < target_recall
        ):
            continue

        row = {
            "threshold":
                float(threshold),

            **m,
        }

        if best is None:
            best = row
            continue

        key = (
            row["fp_per_output"],
            row["fpr"],
            -row["precision"],
            -row["exact_set"],
        )

        best_key = (
            best["fp_per_output"],
            best["fpr"],
            -best["precision"],
            -best["exact_set"],
        )

        if key < best_key:
            best = row

    if best is None:
        raise RuntimeError(
            "Could not reach target recall."
        )

    return best


# ============================================================
# Dynamic threshold:
#
# predict ONE threshold per output from distribution statistics.
# This cannot reorder roles.
# ============================================================

def threshold_features(scores):
    sorted_s = np.sort(
        scores,
        axis=1,
    )[:, ::-1]

    q25 = np.quantile(
        scores,
        0.25,
        axis=1,
    )

    q75 = np.quantile(
        scores,
        0.75,
        axis=1,
    )

    return np.stack(
        [
            scores.mean(axis=1),
            scores.std(axis=1),
            np.median(
                scores,
                axis=1,
            ),
            scores.min(axis=1),
            scores.max(axis=1),
            q25,
            q75,
            q75 - q25,

            sorted_s[:, 0],
            sorted_s[:, 1],
            sorted_s[:, 2],

            sorted_s[:, 0]
            - sorted_s[:, 1],

            sorted_s[:, 1]
            - sorted_s[:, 2],

            sorted_s[:, 2]
            - sorted_s[:, 3],
        ],
        axis=1,
    ).astype(np.float32)


class ThresholdNet(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                d,
                32,
            ),
            nn.GELU(),

            nn.Linear(
                32,
                16,
            ),
            nn.GELU(),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_threshold_net(
    train_scores,
    train_y,
    dev_scores,
    dev_y,
    seed,
):
    torch.manual_seed(seed)

    train_x = threshold_features(
        train_scores
    )

    dev_x = threshold_features(
        dev_scores
    )

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

    train_x = (
        train_x - mu
    ) / sd

    dev_x = (
        dev_x - mu
    ) / sd

    tx = torch.tensor(
        train_x,
        dtype=torch.float32,
    )

    ts = torch.tensor(
        train_scores,
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

    ds = torch.tensor(
        dev_scores,
        dtype=torch.float32,
    )

    model = ThresholdNet(
        tx.shape[1]
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )

    # Mild positive weighting because positives are sparse.
    neg = float(
        (train_y == 0).sum()
    )

    pos = float(
        (train_y == 1).sum()
    )

    pos_weight = min(
        neg / max(pos, 1.0),
        4.0,
    )

    pos_weight_t = torch.tensor(
        pos_weight,
        dtype=torch.float32,
    )

    best_state = None
    best_dev = float("inf")
    stale = 0

    for epoch in range(
        1,
        401,
    ):
        model.train()

        t = model(tx)

        logits = (
            ts
            - t.unsqueeze(1)
        )

        loss = (
            F.binary_cross_entropy_with_logits(
                logits,
                ty,
                pos_weight=pos_weight_t,
            )
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()

        with torch.no_grad():
            dt = model(dx)

            dlogits = (
                ds
                - dt.unsqueeze(1)
            )

            dloss = float(
                F.binary_cross_entropy_with_logits(
                    dlogits,
                    torch.tensor(
                        dev_y,
                        dtype=torch.float32,
                    ),
                    pos_weight=pos_weight_t,
                )
            )

        if (
            dloss
            < best_dev - 1e-6
        ):
            best_dev = dloss

            best_state = {
                k:
                    v.detach().clone()

                for k, v
                in model.state_dict().items()
            }

            stale = 0

        else:
            stale += 1

        if stale >= 40:
            break

    model.load_state_dict(
        best_state
    )

    return (
        model,
        mu,
        sd,
    )


@torch.inference_mode()
def dynamic_relative_scores(
    model,
    mu,
    sd,
    scores,
):
    x = threshold_features(
        scores
    )

    x = (
        x - mu
    ) / sd

    x = torch.tensor(
        x,
        dtype=torch.float32,
    )

    threshold = model(
        x
    ).numpy()

    # Relative to predicted per-output threshold.
    # Ranking is exactly unchanged.
    return (
        scores
        - threshold.reshape(
            -1,
            1,
        )
    )


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    cache = np.load(CACHE)

    data = {
        "raw": {
            "train": cache["train_raw"],
            "dev": cache["dev_raw"],
            "test": cache["test_raw"],
            "audit": cache["audit_raw"],
        },

        "structured": {
            "train": cache["train_struct"],
            "dev": cache["dev_struct"],
            "test": cache["test_struct"],
            "audit": cache["audit_struct"],
        },

        "structured_calibrated": {
            "train": cache["train_cal"],
            "dev": cache["dev_cal"],
            "test": cache["test_cal"],
            "audit": cache["audit_cal"],
        },
    }

    train_y = cache[
        "train_y"
    ].astype(np.int64)

    dev_y = cache[
        "dev_y"
    ].astype(np.int64)

    test_y = cache[
        "test_y"
    ].astype(np.int64)

    audit_y = cache[
        "audit_y"
    ].astype(np.int64)

    strict_idx = cache[
        "strict_idx"
    ].astype(np.int64)

    strict_y = audit_y[
        strict_idx
    ]

    candidates = []

    print(
        "===== STATIC RELATIVE NORMALIZATION =====",
        flush=True,
    )

    for base_name, splits in data.items():
        dev_transforms = transforms(
            splits["dev"]
        )

        test_transforms = transforms(
            splits["test"]
        )

        audit_transforms = transforms(
            splits["audit"]
        )

        for transform_name, dev_scores in (
            dev_transforms.items()
        ):
            op = threshold_search(
                dev_scores,
                dev_y,
                TARGET_RECALL,
            )

            test_scores = (
                test_transforms[
                    transform_name
                ]
            )

            audit_scores = (
                audit_transforms[
                    transform_name
                ]
            )

            strict_scores = (
                audit_scores[
                    strict_idx
                ]
            )

            test_m = metrics(
                (
                    test_scores
                    >= op["threshold"]
                ).astype(np.int64),
                test_y,
            )

            strict_m = metrics(
                (
                    strict_scores
                    >= op["threshold"]
                ).astype(np.int64),
                strict_y,
            )

            row = {
                "kind":
                    "static",

                "base":
                    base_name,

                "transform":
                    transform_name,

                "dev":
                    op,

                "test":
                    test_m,

                "strict_82":
                    strict_m,

                "within_auc_test":
                    within_auc(
                        test_scores,
                        test_y,
                    ),

                "within_auc_strict":
                    within_auc(
                        strict_scores,
                        strict_y,
                    ),
            }

            candidates.append(
                row
            )

            print(
                f"{base_name:24s} "
                f"{transform_name:18s} "
                f"dev FP/out={op['fp_per_output']:.3f} "
                f"recall={op['recall']:.3f}",
                flush=True,
            )

    print(
        "\n===== DYNAMIC OUTPUT-LEVEL THRESHOLD =====",
        flush=True,
    )

    for base_name, splits in data.items():
        dynamic_runs = []

        for seed in [
            SEED + 1,
            SEED + 2,
            SEED + 3,
            SEED + 4,
            SEED + 5,
        ]:
            model, mu, sd = (
                train_threshold_net(
                    splits["train"],
                    train_y,
                    splits["dev"],
                    dev_y,
                    seed,
                )
            )

            dev_rel = (
                dynamic_relative_scores(
                    model,
                    mu,
                    sd,
                    splits["dev"],
                )
            )

            test_rel = (
                dynamic_relative_scores(
                    model,
                    mu,
                    sd,
                    splits["test"],
                )
            )

            audit_rel = (
                dynamic_relative_scores(
                    model,
                    mu,
                    sd,
                    splits["audit"],
                )
            )

            dynamic_runs.append(
                (
                    dev_rel,
                    test_rel,
                    audit_rel,
                )
            )

        # Ensemble relative margins.
        dev_rel = np.mean(
            [
                x[0]
                for x in dynamic_runs
            ],
            axis=0,
        )

        test_rel = np.mean(
            [
                x[1]
                for x in dynamic_runs
            ],
            axis=0,
        )

        audit_rel = np.mean(
            [
                x[2]
                for x in dynamic_runs
            ],
            axis=0,
        )

        strict_rel = audit_rel[
            strict_idx
        ]

        # Global OFFSET selected only on DEV.
        op = threshold_search(
            dev_rel,
            dev_y,
            TARGET_RECALL,
        )

        test_m = metrics(
            (
                test_rel
                >= op["threshold"]
            ).astype(np.int64),
            test_y,
        )

        strict_m = metrics(
            (
                strict_rel
                >= op["threshold"]
            ).astype(np.int64),
            strict_y,
        )

        row = {
            "kind":
                "dynamic",

            "base":
                base_name,

            "transform":
                "threshold_net",

            "dev":
                op,

            "test":
                test_m,

            "strict_82":
                strict_m,

            "within_auc_test":
                within_auc(
                    test_rel,
                    test_y,
                ),

            "within_auc_strict":
                within_auc(
                    strict_rel,
                    strict_y,
                ),
        }

        candidates.append(
            row
        )

        print(
            f"DYNAMIC {base_name}: "
            f"dev FP/out={op['fp_per_output']:.3f} "
            f"recall={op['recall']:.3f}",
            flush=True,
        )

    # ========================================================
    # Selection ONLY on DEV.
    # ========================================================

    feasible = [
        x
        for x in candidates
        if (
            x["dev"]["recall"]
            >= TARGET_RECALL
        )
    ]

    best = min(
        feasible,
        key=lambda x: (
            x["dev"][
                "fp_per_output"
            ],
            x["dev"]["fpr"],
            -x["dev"]["precision"],
        ),
    )

    result = {
        "experiment":
            "v7_relative_threshold_v1",

        "target_dev_recall":
            TARGET_RECALL,

        "qwen_calls":
            0,

        "selected_on":
            "dev only",

        "best":
            best,

        "all_candidates":
            candidates,
    }

    dump(
        OUT / "summary.json",
        result,
    )

    print(
        "\n===== WINNER SELECTED ON DEV =====",
        flush=True,
    )

    print(
        "kind:",
        best["kind"],
        flush=True,
    )

    print(
        "base:",
        best["base"],
        flush=True,
    )

    print(
        "transform:",
        best["transform"],
        flush=True,
    )

    for split in [
        "dev",
        "test",
        "strict_82",
    ]:
        m = best[split]

        print(
            f"\n{split.upper()}:"
            f" precision={m['precision']:.4f}"
            f" recall={m['recall']:.4f}"
            f" FPR={m['fpr']:.4f}"
            f" FP/out={m['fp_per_output']:.4f}"
            f" FN/out={m['fn_per_output']:.4f}"
            f" meanPred={m['mean_predicted_roles']:.4f}"
            f" exact={m['exact_set']:.4f}",
            flush=True,
        )

    print(
        "\nAUC test:",
        best["within_auc_test"],
        flush=True,
    )

    print(
        "AUC strict:",
        best["within_auc_strict"],
        flush=True,
    )

    print(
        "\nSaved:",
        OUT / "summary.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
