#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import analysis.train_v13_6b_activation_judge as tr


CACHE = Path(
    "artifacts/v7_count_decoder_v1/score_cache.npz"
)

OUT = Path(
    "artifacts/v7_rank_error_audit_v1"
)
OUT.mkdir(parents=True, exist_ok=True)


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


def recall_at_k(
    ranks,
    y,
    k,
):
    active = (
        y == 1
    )

    hit = (
        active
        & (ranks <= k)
    ).sum()

    total = active.sum()

    return float(
        hit / total
    )


def fp_per_output_at_k(
    ranks,
    y,
    k,
):
    pred = (
        ranks <= k
    )

    fp = (
        pred
        & (y == 0)
    ).sum()

    return float(
        fp / len(y)
    )


def exact_at_k(
    ranks,
    y,
    k,
):
    pred = (
        ranks <= k
    ).astype(
        np.int64
    )

    return float(
        np.all(
            pred == y,
            axis=1,
        ).mean()
    )


def per_role_stats(
    ranks,
    y,
    roles,
):
    out = {}

    for j, role in enumerate(
        roles
    ):
        pos_mask = (
            y[:, j] == 1
        )

        neg_mask = (
            y[:, j] == 0
        )

        pos_ranks = ranks[
            pos_mask,
            j,
        ]

        neg_ranks = ranks[
            neg_mask,
            j,
        ]

        if len(pos_ranks):
            mean_rank = float(
                pos_ranks.mean()
            )

            median_rank = float(
                np.median(
                    pos_ranks
                )
            )

            r3 = float(
                np.mean(
                    pos_ranks <= 3
                )
            )

            r5 = float(
                np.mean(
                    pos_ranks <= 5
                )
            )

            r7 = float(
                np.mean(
                    pos_ranks <= 7
                )
            )

            r10 = float(
                np.mean(
                    pos_ranks <= 10
                )
            )

        else:
            mean_rank = None
            median_rank = None
            r3 = None
            r5 = None
            r7 = None
            r10 = None

        if len(neg_ranks):
            fp3 = float(
                np.mean(
                    neg_ranks <= 3
                )
            )

            fp5 = float(
                np.mean(
                    neg_ranks <= 5
                )
            )

            fp7 = float(
                np.mean(
                    neg_ranks <= 7
                )
            )

        else:
            fp3 = None
            fp5 = None
            fp7 = None

        out[role] = {
            "positive_count":
                int(
                    pos_mask.sum()
                ),

            "mean_active_rank":
                mean_rank,

            "median_active_rank":
                median_rank,

            "recall_at_3":
                r3,

            "recall_at_5":
                r5,

            "recall_at_7":
                r7,

            "recall_at_10":
                r10,

            "inactive_top3_rate":
                fp3,

            "inactive_top5_rate":
                fp5,

            "inactive_top7_rate":
                fp7,
        }

    return out


def confusion_pairs(
    ranks,
    y,
    roles,
    k=5,
):
    counts = Counter()

    for i in range(
        len(y)
    ):
        active = [
            j
            for j in range(
                len(roles)
            )
            if y[i, j] == 1
        ]

        false_top = [
            j
            for j in range(
                len(roles)
            )
            if (
                y[i, j] == 0
                and ranks[i, j] <= k
            )
        ]

        for true_j in active:
            for false_j in false_top:
                counts[
                    (
                        roles[true_j],
                        roles[false_j],
                    )
                ] += 1

    return [
        {
            "true_role":
                a,

            "false_top_role":
                b,

            "count":
                int(c),
        }

        for (a, b), c
        in counts.most_common(40)
    ]


def worst_true_rank_stats(
    ranks,
    y,
):
    vals = []

    for i in range(
        len(y)
    ):
        active = np.where(
            y[i] == 1
        )[0]

        if not len(active):
            continue

        vals.append(
            int(
                ranks[
                    i,
                    active,
                ].max()
            )
        )

    vals = np.asarray(
        vals,
        dtype=np.int64,
    )

    return {
        "mean":
            float(
                vals.mean()
            ),

        "median":
            float(
                np.median(
                    vals
                )
            ),

        "p75":
            float(
                np.quantile(
                    vals,
                    0.75,
                )
            ),

        "p90":
            float(
                np.quantile(
                    vals,
                    0.90,
                )
            ),

        "pct_all_true_within_top3":
            float(
                np.mean(
                    vals <= 3
                )
            ),

        "pct_all_true_within_top5":
            float(
                np.mean(
                    vals <= 5
                )
            ),

        "pct_all_true_within_top7":
            float(
                np.mean(
                    vals <= 7
                )
            ),

        "pct_all_true_within_top10":
            float(
                np.mean(
                    vals <= 10
                )
            ),
    }


def audit(
    name,
    scores,
    y,
    roles,
):
    ranks = ranks_from_scores(
        scores
    )

    curve = []

    for k in range(
        1,
        len(roles) + 1,
    ):
        curve.append({
            "k":
                k,

            "recall":
                recall_at_k(
                    ranks,
                    y,
                    k,
                ),

            "fp_per_output":
                fp_per_output_at_k(
                    ranks,
                    y,
                    k,
                ),

            "exact_set":
                exact_at_k(
                    ranks,
                    y,
                    k,
                ),
        })

    return {
        "name":
            name,

        "rows":
            len(y),

        "recall_fp_curve":
            curve,

        "per_role":
            per_role_stats(
                ranks,
                y,
                roles,
            ),

        "worst_true_rank":
            worst_true_rank_stats(
                ranks,
                y,
            ),

        "top5_confusions":
            confusion_pairs(
                ranks,
                y,
                roles,
                k=5,
            ),

        "top7_confusions":
            confusion_pairs(
                ranks,
                y,
                roles,
                k=7,
            ),
    }


def main():
    cache = np.load(
        CACHE
    )

    roles = list(
        tr.ROLE_DESCRIPTIONS.keys()
    )

    test_scores = cache[
        "test_cal"
    ]

    audit_scores = cache[
        "audit_cal"
    ]

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

    strict_scores = audit_scores[
        strict_idx
    ]

    strict_y = audit_y[
        strict_idx
    ]

    test = audit(
        "test_structured_calibrated",
        test_scores,
        test_y,
        roles,
    )

    strict = audit(
        "strict82_structured_calibrated",
        strict_scores,
        strict_y,
        roles,
    )

    result = {
        "test":
            test,

        "strict_82":
            strict,
    }

    dump(
        OUT / "rank_audit.json",
        result,
    )

    print(
        "===== RECALL@K / FP@K ====="
    )

    print(
        "\nTEST"
    )

    for row in test[
        "recall_fp_curve"
    ]:
        if row["k"] <= 12:
            print(
                f"K={row['k']:2d} "
                f"recall={row['recall']:.4f} "
                f"FP/out={row['fp_per_output']:.4f} "
                f"exact={row['exact_set']:.4f}"
            )

    print(
        "\nSTRICT_82"
    )

    for row in strict[
        "recall_fp_curve"
    ]:
        if row["k"] <= 12:
            print(
                f"K={row['k']:2d} "
                f"recall={row['recall']:.4f} "
                f"FP/out={row['fp_per_output']:.4f} "
                f"exact={row['exact_set']:.4f}"
            )

    print(
        "\n===== WORST TRUE ROLE RANK ====="
    )

    print(
        "TEST:",
        json.dumps(
            test[
                "worst_true_rank"
            ],
            indent=2,
        )
    )

    print(
        "STRICT:",
        json.dumps(
            strict[
                "worst_true_rank"
            ],
            indent=2,
        )
    )

    print(
        "\n===== WORST ROLES BY ACTIVE RANK: STRICT ====="
    )

    rows = []

    for role, m in strict[
        "per_role"
    ].items():
        if (
            m[
                "mean_active_rank"
            ]
            is not None
        ):
            rows.append(
                (
                    m[
                        "mean_active_rank"
                    ],
                    role,
                    m,
                )
            )

    for _, role, m in sorted(
        rows,
        reverse=True,
    ):
        print(
            f"{role:28s} "
            f"N+={m['positive_count']:3d} "
            f"meanRank={m['mean_active_rank']:.2f} "
            f"R@3={m['recall_at_3']:.3f} "
            f"R@5={m['recall_at_5']:.3f} "
            f"R@7={m['recall_at_7']:.3f} "
            f"inactiveTop5={m['inactive_top5_rate']:.3f}"
        )

    print(
        "\n===== TOP STRICT CONFUSIONS @5 ====="
    )

    for row in strict[
        "top5_confusions"
    ][:25]:
        print(
            f"{row['true_role']:28s} -> "
            f"{row['false_top_role']:28s} "
            f"{row['count']}"
        )

    print(
        "\nSaved:",
        OUT / "rank_audit.json"
    )


if __name__ == "__main__":
    main()
