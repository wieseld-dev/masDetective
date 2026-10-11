#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import analysis.v7_count_decoder_v1 as cd
import analysis.v7_nonlinear_residual_ranker_v1 as nr


CACHE = Path(
    "artifacts/v7_count_decoder_v1/score_cache.npz"
)

NONLIN_ART = Path(
    "artifacts/v7_nonlinear_residual_ranker_v1/"
    "ranker_ensemble.pt"
)

OUT = Path(
    "artifacts/v7_hybrid_topk_decoder_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
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


def reconstruct_nonlin(
    raw,
    struct,
    artifact,
):
    mean = np.asarray(
        artifact["feature_mean"]
    )

    std = np.asarray(
        artifact["feature_std"]
    )

    alpha = float(
        artifact["alpha"]
    )

    x = nr.make_features(
        raw,
        struct,
    )

    x = (
        x - mean
    ) / std

    states = artifact[
        "models"
    ]

    preds = []

    for state in states:
        model = nr.ResidualRanker(
            x.shape[1],
            struct.shape[1],
        )

        model.load_state_dict(
            state
        )

        model.eval()

        pred = nr.predict_model(
            model,
            x,
            struct,
        )

        preds.append(
            pred
        )

    new = np.mean(
        preds,
        axis=0,
    )

    final = (
        struct
        + alpha
        * (
            new - struct
        )
    )

    return final


def oracle_metrics(
    scores,
    y,
):
    return nr.oracle_k_metrics(
        scores,
        y,
    )


def within_auc(
    scores,
    y,
):
    return nr.within_auc(
        scores,
        y,
    )


def search_hybrid(
    struct,
    cal,
    nonlin,
    y,
):
    baseline_auc = within_auc(
        struct,
        y,
    )

    candidates = []

    # Convex combinations:
    # ws + wc + wn = 1.
    for wc_i in range(
        0,
        21,
    ):
        wc = wc_i / 20.0

        for wn_i in range(
            0,
            21 - wc_i,
        ):
            wn = wn_i / 20.0

            ws = (
                1.0 - wc - wn
            )

            score = (
                ws * struct
                + wc * cal
                + wn * nonlin
            )

            auc = within_auc(
                score,
                y,
            )

            om = oracle_metrics(
                score,
                y,
            )

            candidates.append({
                "w_struct":
                    float(ws),

                "w_cal":
                    float(wc),

                "w_nonlin":
                    float(wn),

                "within_auc":
                    float(auc),

                "oracle_k_recall":
                    float(
                        om["recall"]
                    ),

                "oracle_k_exact":
                    float(
                        om["exact_set"]
                    ),

                "oracle_k_fp_per_output":
                    float(
                        om[
                            "avg_fp_per_output"
                        ]
                    ),
            })

    # Primary target: top-K recovery.
    # Do not allow major destruction of overall ranking.
    allowed = [
        x
        for x in candidates
        if (
            x["within_auc"]
            >= baseline_auc - 0.005
        )
    ]

    best = max(
        allowed,
        key=lambda x: (
            x[
                "oracle_k_recall"
            ],
            x[
                "oracle_k_exact"
            ],
            x[
                "within_auc"
            ],
        ),
    )

    return best


def apply_hybrid(
    struct,
    cal,
    nonlin,
    weights,
):
    return (
        weights["w_struct"]
        * struct
        +
        weights["w_cal"]
        * cal
        +
        weights["w_nonlin"]
        * nonlin
    )


def count_base(
    probs,
    method,
):
    classes = np.arange(
        probs.shape[1],
        dtype=np.float64,
    )

    if method == "argmax":
        return np.argmax(
            probs,
            axis=1,
        )

    expected = (
        probs
        * classes.reshape(
            1,
            -1,
        )
    ).sum(
        axis=1
    )

    if method == "round":
        return np.rint(
            expected
        ).astype(
            np.int64
        )

    if method == "ceil":
        return np.ceil(
            expected
        ).astype(
            np.int64
        )

    if method == "floor":
        return np.floor(
            expected
        ).astype(
            np.int64
        )

    raise ValueError(
        method
    )


def decode_with_extra(
    scores,
    base_k,
    extra,
):
    k = np.clip(
        base_k + extra,
        0,
        scores.shape[1],
    )

    return cd.decode_topk(
        scores,
        k,
    )


def search_operating_point(
    probs,
    scores,
    y,
):
    rows = []

    for method in [
        "argmax",
        "round",
        "ceil",
        "floor",
    ]:
        base = count_base(
            probs,
            method,
        )

        for extra in range(
            0,
            11,
        ):
            pred = decode_with_extra(
                scores,
                base,
                extra,
            )

            m = cd.set_metrics(
                pred,
                y,
            )

            row = {
                "count_method":
                    method,

                "extra_roles":
                    int(extra),

                "mean_predicted_count":
                    float(
                        np.mean(
                            np.clip(
                                base
                                + extra,
                                0,
                                scores.shape[1],
                            )
                        )
                    ),

                **m,
            }

            rows.append(
                row
            )

    feasible = [
        x
        for x in rows
        if (
            x["recall"]
            >= TARGET_RECALL
        )
    ]

    if not feasible:
        best = max(
            rows,
            key=lambda x: (
                x["recall"],
                -x[
                    "avg_fp_per_output"
                ],
            ),
        )

        best[
            "target_recall_reached"
        ] = False

        return best

    best = min(
        feasible,
        key=lambda x: (
            x[
                "avg_fp_per_output"
            ],
            x["fpr"],
            -x["precision"],
            -x[
                "exact_activation_set"
            ],
        ),
    )

    best[
        "target_recall_reached"
    ] = True

    return best


def evaluate(
    scores,
    y,
    probs,
    op,
):
    base = count_base(
        probs,
        op[
            "count_method"
        ],
    )

    pred = decode_with_extra(
        scores,
        base,
        op[
            "extra_roles"
        ],
    )

    setm = cd.set_metrics(
        pred,
        y,
    )

    true_k = y.sum(
        axis=1
    ).astype(
        np.int64
    )

    pred_k = np.clip(
        base
        + op[
            "extra_roles"
        ],
        0,
        scores.shape[1],
    )

    countm = cd.count_metrics(
        true_k,
        pred_k,
    )

    oracle = oracle_metrics(
        scores,
        y,
    )

    return {
        "within_auc":
            within_auc(
                scores,
                y,
            ),

        "decoder":
            {
                **setm,
                **countm,
            },

        "oracle_k":
            oracle,
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

    artifact = torch.load(
        NONLIN_ART,
        map_location="cpu",
        weights_only=False,
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

    print(
        "===== RECONSTRUCT NONLINEAR SCORES =====",
        flush=True,
    )

    train_nonlin = reconstruct_nonlin(
        train_raw,
        train_struct,
        artifact,
    )

    dev_nonlin = reconstruct_nonlin(
        dev_raw,
        dev_struct,
        artifact,
    )

    test_nonlin = reconstruct_nonlin(
        test_raw,
        test_struct,
        artifact,
    )

    audit_nonlin = reconstruct_nonlin(
        audit_raw,
        audit_struct,
        artifact,
    )

    print(
        "\n===== SEARCH TOP-K HYBRID ON DEV =====",
        flush=True,
    )

    weights = search_hybrid(
        dev_struct,
        dev_cal,
        dev_nonlin,
        dev_y,
    )

    print(
        json.dumps(
            weights,
            indent=2,
        ),
        flush=True,
    )

    train_hybrid = apply_hybrid(
        train_struct,
        train_cal,
        train_nonlin,
        weights,
    )

    dev_hybrid = apply_hybrid(
        dev_struct,
        dev_cal,
        dev_nonlin,
        weights,
    )

    test_hybrid = apply_hybrid(
        test_struct,
        test_cal,
        test_nonlin,
        weights,
    )

    audit_hybrid = apply_hybrid(
        audit_struct,
        audit_cal,
        audit_nonlin,
        weights,
    )

    strict_hybrid = audit_hybrid[
        strict_idx
    ]

    # ========================================================
    # Retrain tiny count ensemble using hybrid geometry.
    # No LLM work here.
    # ========================================================

    print(
        "\n===== TRAIN COUNT ENSEMBLE =====",
        flush=True,
    )

    train_x = cd.make_features(
        train_hybrid
    )

    dev_x = cd.make_features(
        dev_hybrid
    )

    test_x = cd.make_features(
        test_hybrid
    )

    audit_x = cd.make_features(
        audit_hybrid
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

    (
        models,
        mu,
        sd,
        reports,
    ) = cd.train_count_ensemble(
        train_x,
        train_k,
        dev_x,
        dev_k,
        seeds=[
            SEED + 101,
            SEED + 102,
            SEED + 103,
            SEED + 104,
            SEED + 105,
        ],
    )

    dev_probs = cd.ensemble_probs(
        models,
        dev_x,
        mu,
        sd,
    )

    test_probs = cd.ensemble_probs(
        models,
        test_x,
        mu,
        sd,
    )

    audit_probs = cd.ensemble_probs(
        models,
        audit_x,
        mu,
        sd,
    )

    strict_probs = audit_probs[
        strict_idx
    ]

    print(
        "\n===== SEARCH K+DELTA OPERATING POINT ON DEV =====",
        flush=True,
    )

    op = search_operating_point(
        dev_probs,
        dev_hybrid,
        dev_y,
    )

    print(
        json.dumps(
            op,
            indent=2,
        ),
        flush=True,
    )

    test_result = evaluate(
        test_hybrid,
        test_y,
        test_probs,
        op,
    )

    strict_result = evaluate(
        strict_hybrid,
        strict_y,
        strict_probs,
        op,
    )

    dev_result = evaluate(
        dev_hybrid,
        dev_y,
        dev_probs,
        op,
    )

    result = {
        "experiment":
            "v7_hybrid_topk_decoder_v1",

        "query_used":
            False,

        "qwen_calls":
            0,

        "target_dev_recall":
            TARGET_RECALL,

        "hybrid_weights":
            weights,

        "count_training":
            reports,

        "selected_operating_point":
            op,

        "dev":
            dev_result,

        "test":
            test_result,

        "strict_82":
            strict_result,
    }

    dump(
        OUT / "summary.json",
        result,
    )

    print(
        "\n===== FINAL HYBRID TOP-K RESULTS =====",
        flush=True,
    )

    print(
        "HYBRID WEIGHTS:",
        json.dumps(
            weights,
            ensure_ascii=False,
        ),
        flush=True,
    )

    print(
        "OPERATING POINT:",
        json.dumps(
            op,
            ensure_ascii=False,
        ),
        flush=True,
    )

    for name, r in [
        (
            "DEV",
            dev_result,
        ),
        (
            "TEST",
            test_result,
        ),
        (
            "STRICT_82",
            strict_result,
        ),
    ]:
        d = r["decoder"]
        o = r["oracle_k"]

        print(
            f"\n{name}",
            flush=True,
        )

        print(
            "  AUC="
            f"{r['within_auc']:.4f}",
            flush=True,
        )

        print(
            "  K+DELTA:"
            f" precision={d['precision']:.4f}"
            f" recall={d['recall']:.4f}"
            f" FPR={d['fpr']:.4f}"
            f" FP/out={d['avg_fp_per_output']:.4f}"
            f" FN/out={d['avg_fn_per_output']:.4f}"
            f" exact={d['exact_activation_set']:.4f}"
            f" meanK={d['mean_pred_count']:.4f}",
            flush=True,
        )

        print(
            "  ORACLE-K:"
            f" recall={o['recall']:.4f}"
            f" FP/out={o['avg_fp_per_output']:.4f}"
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
