#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CACHE = Path(
    "artifacts/v7_count_decoder_v1/score_cache.npz"
)

OUT = Path(
    "artifacts/v7_nonlinear_residual_ranker_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926

TRAIN_SEEDS = [
    SEED + 11,
    SEED + 22,
    SEED + 33,
    SEED + 44,
    SEED + 55,
]

TARGET_RECALL = 0.80


def dump(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def within_auc(scores, y):
    vals = []

    for i in range(len(scores)):
        pos = scores[i][y[i] == 1]
        neg = scores[i][y[i] == 0]

        if len(pos) == 0 or len(neg) == 0:
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
            float(wins + 0.5 * ties)
        )

    return float(np.mean(vals))


def oracle_k_metrics(scores, y):
    pred = np.zeros_like(
        y,
        dtype=np.int64,
    )

    for i in range(len(y)):
        k = int(y[i].sum())

        if k <= 0:
            continue

        idx = np.argsort(
            -scores[i]
        )[:k]

        pred[i, idx] = 1

    return set_metrics(
        pred,
        y,
    )


def set_metrics(pred, y):
    pred = pred.astype(np.int64)
    y = y.astype(np.int64)

    tp = int(
        ((pred == 1) & (y == 1)).sum()
    )

    fp = int(
        ((pred == 1) & (y == 0)).sum()
    )

    fn = int(
        ((pred == 0) & (y == 1)).sum()
    )

    tn = int(
        ((pred == 0) & (y == 0)).sum()
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

        "avg_fp_per_output":
            float(fp / len(y)),

        "avg_fn_per_output":
            float(fn / len(y)),

        "exact_set":
            float(exact.mean()),
    }


def make_features(
    raw,
    struct,
):
    # Role-specific relative score.
    centered = (
        struct
        - struct.mean(
            axis=1,
            keepdims=True,
        )
    )

    std = struct.std(
        axis=1,
        keepdims=True,
    )

    zscore = centered / np.maximum(
        std,
        1e-5,
    )

    # Rank of each role within the same output.
    order = np.argsort(
        -struct,
        axis=1,
    )

    ranks = np.empty_like(
        order,
        dtype=np.float32,
    )

    rows = np.arange(
        len(struct)
    )[:, None]

    ranks[
        rows,
        order,
    ] = np.arange(
        struct.shape[1]
    )[None, :]

    ranks = (
        ranks
        / max(
            struct.shape[1] - 1,
            1,
        )
    )

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
            sorted_s[:, 0]
            - sorted_s[:, 1],
            sorted_s[:, 1]
            - sorted_s[:, 2],
        ],
        axis=1,
    )

    return np.concatenate(
        [
            raw,
            struct,
            centered,
            zscore,
            ranks,
            sorted_s,
            gaps,
            summary,
        ],
        axis=1,
    ).astype(np.float32)


class ResidualRanker(nn.Module):
    def __init__(
        self,
        d_in,
        n_roles,
    ):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(
                d_in,
                128,
            ),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(
                128,
                64,
            ),
            nn.GELU(),
        )

        self.head = nn.Linear(
            64,
            n_roles,
        )

        # EXACT identity at initialization:
        # correction = 0.
        nn.init.zeros_(
            self.head.weight
        )
        nn.init.zeros_(
            self.head.bias
        )

    def forward(
        self,
        x,
        base,
    ):
        correction = self.head(
            self.body(x)
        )

        return (
            base
            + correction
        ), correction


def all_pair_loss(
    scores,
    y,
):
    diff = (
        scores.unsqueeze(2)
        - scores.unsqueeze(1)
    )

    mask = (
        (y.unsqueeze(2) > 0.5)
        &
        (y.unsqueeze(1) < 0.5)
    )

    return F.softplus(
        -diff[mask]
    ).mean()


def hard_pair_loss(
    scores,
    base,
    y,
    hard_k=5,
):
    losses = []

    for i in range(
        scores.shape[0]
    ):
        pos = torch.where(
            y[i] > 0.5
        )[0]

        neg = torch.where(
            y[i] < 0.5
        )[0]

        if len(pos) == 0 or len(neg) == 0:
            continue

        # Fixed hard negatives according to the stable
        # structured baseline, not dynamic model noise.
        neg_base = base[
            i,
            neg,
        ]

        k = min(
            hard_k,
            len(neg),
        )

        hard_local = torch.topk(
            neg_base,
            k=k,
        ).indices

        hard_neg = neg[
            hard_local
        ]

        d = (
            scores[i, pos]
            .unsqueeze(1)
            -
            scores[i, hard_neg]
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


def train_one(
    train_x,
    train_base,
    train_y,
    dev_x,
    dev_base,
    dev_y,
    seed,
):
    torch.manual_seed(seed)

    model = ResidualRanker(
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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=2e-3,
    )

    baseline_auc = within_auc(
        dev_base,
        dev_y,
    )

    baseline_oracle = (
        oracle_k_metrics(
            dev_base,
            dev_y,
        )["recall"]
    )

    best_auc = baseline_auc
    best_oracle = baseline_oracle
    best_epoch = 0

    best_state = {
        k: v.detach().clone()
        for k, v
        in model.state_dict().items()
    }

    stale = 0
    patience = 35

    for epoch in range(
        1,
        301,
    ):
        model.train()

        order = torch.randperm(
            len(tx)
        )

        epoch_loss = 0.0
        nb = 0

        for start in range(
            0,
            len(tx),
            128,
        ):
            idx = order[
                start:start + 128
            ]

            xb = tx[idx]
            bb = tb[idx]
            yb = ty[idx]

            scores, correction = model(
                xb,
                bb,
            )

            pair = all_pair_loss(
                scores,
                yb,
            )

            hard = hard_pair_loss(
                scores,
                bb,
                yb,
                hard_k=5,
            )

            # Keep correction conservative.
            reg = (
                correction.pow(2)
                .mean()
            )

            loss = (
                pair
                + 0.35 * hard
                + 0.01 * reg
            )

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            epoch_loss += float(
                loss.detach()
            )

            nb += 1

        if (
            epoch % 5 != 0
            and epoch != 1
        ):
            continue

        model.eval()

        with torch.no_grad():
            ds, _ = model(
                dx,
                db,
            )

        ds = ds.numpy()

        auc = within_auc(
            ds,
            dev_y,
        )

        oracle = oracle_k_metrics(
            ds,
            dev_y,
        )

        ore = oracle["recall"]

        print(
            f"[seed={seed}] "
            f"epoch={epoch:03d} "
            f"loss={epoch_loss/max(nb,1):.5f} "
            f"dev_auc={auc:.6f} "
            f"oracleK_recall={ore:.6f}",
            flush=True,
        )

        improved = (
            auc > best_auc + 1e-5
            or (
                abs(
                    auc - best_auc
                ) <= 1e-5
                and ore > best_oracle
            )
        )

        if improved:
            best_auc = auc
            best_oracle = ore
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
            stale += 5

        if stale >= patience:
            break

    model.load_state_dict(
        best_state
    )

    model.eval()

    return model, {
        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "best_dev_auc":
            float(best_auc),

        "best_dev_oracle_k_recall":
            float(best_oracle),
    }


@torch.inference_mode()
def predict_model(
    model,
    x,
    base,
):
    x = torch.tensor(
        x,
        dtype=torch.float32,
    )

    base_t = torch.tensor(
        base,
        dtype=torch.float32,
    )

    scores, _ = model(
        x,
        base_t,
    )

    return scores.numpy()


def normalize(
    train,
    *others,
):
    mu = train.mean(
        axis=0
    )

    sd = train.std(
        axis=0
    )

    sd = np.maximum(
        sd,
        1e-5,
    )

    out = [
        (train - mu) / sd
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


def blend_search(
    dev_base,
    dev_new,
    dev_y,
):
    best = None

    # Includes alpha=0 => exact structured baseline.
    for alpha in np.linspace(
        0.0,
        1.5,
        151,
    ):
        scores = (
            dev_base
            + alpha
            * (
                dev_new
                - dev_base
            )
        )

        auc = within_auc(
            scores,
            dev_y,
        )

        oracle = oracle_k_metrics(
            scores,
            dev_y,
        )

        row = {
            "alpha":
                float(alpha),

            "within_auc":
                float(auc),

            "oracle_k_recall":
                float(
                    oracle["recall"]
                ),

            "oracle_k_exact":
                float(
                    oracle["exact_set"]
                ),
        }

        if best is None:
            best = row
            continue

        key = (
            row["within_auc"],
            row["oracle_k_recall"],
            row["oracle_k_exact"],
        )

        best_key = (
            best["within_auc"],
            best["oracle_k_recall"],
            best["oracle_k_exact"],
        )

        if key > best_key:
            best = row

    return best


def tune_global_threshold(
    dev_scores,
    dev_y,
):
    vals = np.unique(
        np.concatenate(
            [
                dev_scores.reshape(-1),
                np.asarray(
                    [
                        dev_scores.min()
                        - 1.0
                    ]
                ),
            ]
        )
    )

    feasible = []

    for threshold in vals:
        pred = (
            dev_scores
            >= threshold
        ).astype(
            np.int64
        )

        m = set_metrics(
            pred,
            dev_y,
        )

        if m["recall"] >= TARGET_RECALL:
            feasible.append({
                "threshold":
                    float(threshold),

                **m,
            })

    if not feasible:
        raise RuntimeError(
            "No threshold reached target recall"
        )

    return min(
        feasible,
        key=lambda x: (
            x["fpr"],
            x["avg_fp_per_output"],
            -x["precision"],
        ),
    )


def evaluate(
    scores,
    y,
    threshold=None,
):
    out = {
        "within_auc":
            within_auc(
                scores,
                y,
            ),

        "oracle_k":
            oracle_k_metrics(
                scores,
                y,
            ),
    }

    if threshold is not None:
        pred = (
            scores
            >= threshold
        ).astype(
            np.int64
        )

        out["matched_recall_threshold"] = (
            set_metrics(
                pred,
                y,
            )
        )

    return out


def main():
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

    train_x = make_features(
        train_raw,
        train_struct,
    )

    dev_x = make_features(
        dev_raw,
        dev_struct,
    )

    test_x = make_features(
        test_raw,
        test_struct,
    )

    audit_x = make_features(
        audit_raw,
        audit_struct,
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
        "===== BASELINES =====",
        flush=True,
    )

    print(
        "dev structured AUC:",
        within_auc(
            dev_struct,
            dev_y,
        ),
        flush=True,
    )

    print(
        "test structured AUC:",
        within_auc(
            test_struct,
            test_y,
        ),
        flush=True,
    )

    print(
        "strict structured AUC:",
        within_auc(
            audit_struct[
                strict_idx
            ],
            strict_y,
        ),
        flush=True,
    )

    models = []
    reports = []

    for seed in TRAIN_SEEDS:
        print(
            f"\n===== TRAIN SEED {seed} =====",
            flush=True,
        )

        model, report = train_one(
            train_x,
            train_struct,
            train_y,
            dev_x,
            dev_struct,
            dev_y,
            seed,
        )

        models.append(
            model
        )

        reports.append(
            report
        )

    # Robust ensemble: average all residual rankers.
    dev_preds = []
    test_preds = []
    audit_preds = []

    for model in models:
        dev_preds.append(
            predict_model(
                model,
                dev_x,
                dev_struct,
            )
        )

        test_preds.append(
            predict_model(
                model,
                test_x,
                test_struct,
            )
        )

        audit_preds.append(
            predict_model(
                model,
                audit_x,
                audit_struct,
            )
        )

    dev_new = np.mean(
        dev_preds,
        axis=0,
    )

    test_new = np.mean(
        test_preds,
        axis=0,
    )

    audit_new = np.mean(
        audit_preds,
        axis=0,
    )

    # Conservative blend chosen only on DEV.
    blend = blend_search(
        dev_struct,
        dev_new,
        dev_y,
    )

    alpha = blend[
        "alpha"
    ]

    print(
        "\n===== DEV BLEND =====",
        flush=True,
    )

    print(
        json.dumps(
            blend,
            indent=2,
        ),
        flush=True,
    )

    dev_final = (
        dev_struct
        + alpha
        * (
            dev_new
            - dev_struct
        )
    )

    test_final = (
        test_struct
        + alpha
        * (
            test_new
            - test_struct
        )
    )

    audit_final = (
        audit_struct
        + alpha
        * (
            audit_new
            - audit_struct
        )
    )

    strict_struct = audit_struct[
        strict_idx
    ]

    strict_final = audit_final[
        strict_idx
    ]

    # Matched-recall threshold fitted ONLY on dev.
    threshold_op = tune_global_threshold(
        dev_final,
        dev_y,
    )

    threshold = threshold_op[
        "threshold"
    ]

    result = {
        "experiment":
            "v7_nonlinear_residual_ranker_v1",

        "query_used":
            False,

        "qwen_calls":
            0,

        "training_rows":
            int(
                len(train_x)
            ),

        "seed_reports":
            reports,

        "selected_alpha_on_dev":
            alpha,

        "dev_blend_selection":
            blend,

        "dev_threshold_operating_point":
            threshold_op,

        "structured_baseline": {
            "dev":
                evaluate(
                    dev_struct,
                    dev_y,
                ),

            "test":
                evaluate(
                    test_struct,
                    test_y,
                ),

            "strict_82":
                evaluate(
                    strict_struct,
                    strict_y,
                ),
        },

        "nonlinear_residual": {
            "dev":
                evaluate(
                    dev_final,
                    dev_y,
                    threshold,
                ),

            "test":
                evaluate(
                    test_final,
                    test_y,
                    threshold,
                ),

            "strict_82":
                evaluate(
                    strict_final,
                    strict_y,
                    threshold,
                ),
        },
    }

    dump(
        OUT / "summary.json",
        result,
    )

    torch.save(
        {
            "models": [
                m.state_dict()
                for m in models
            ],

            "feature_mean":
                mu,

            "feature_std":
                sd,

            "alpha":
                alpha,

            "seed_reports":
                reports,
        },
        OUT / "ranker_ensemble.pt",
    )

    print(
        "\n===== FINAL RESULTS =====",
        flush=True,
    )

    for subset in [
        "dev",
        "test",
        "strict_82",
    ]:
        b = result[
            "structured_baseline"
        ][subset]

        n = result[
            "nonlinear_residual"
        ][subset]

        print(
            f"\n{subset}",
            flush=True,
        )

        print(
            "  STRUCT:"
            f" AUC={b['within_auc']:.4f}"
            f" oracleK_recall={b['oracle_k']['recall']:.4f}"
            f" oracleK_exact={b['oracle_k']['exact_set']:.4f}"
            f" oracleK_FP/out={b['oracle_k']['avg_fp_per_output']:.4f}",
            flush=True,
        )

        print(
            "  NONLIN:"
            f" AUC={n['within_auc']:.4f}"
            f" oracleK_recall={n['oracle_k']['recall']:.4f}"
            f" oracleK_exact={n['oracle_k']['exact_set']:.4f}"
            f" oracleK_FP/out={n['oracle_k']['avg_fp_per_output']:.4f}",
            flush=True,
        )

        if (
            "matched_recall_threshold"
            in n
        ):
            t = n[
                "matched_recall_threshold"
            ]

            print(
                "  @DEV-80%-RECALL-THRESH:"
                f" precision={t['precision']:.4f}"
                f" recall={t['recall']:.4f}"
                f" FPR={t['fpr']:.4f}"
                f" FP/out={t['avg_fp_per_output']:.4f}"
                f" exact={t['exact_set']:.4f}",
                flush=True,
            )

    print(
        "\nselected alpha:",
        alpha,
        flush=True,
    )

    print(
        "saved:",
        OUT / "summary.json",
        OUT / "ranker_ensemble.pt",
        flush=True,
    )


if __name__ == "__main__":
    main()
