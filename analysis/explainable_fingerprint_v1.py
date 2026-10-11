from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


SRC = Path(
    "artifacts/frozen_probe_auc_v2/"
    "responses.jsonl"
)

OUT = Path(
    "artifacts/explainable_fingerprint_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


ROLES = [
    "code",
    "writing",
    "math",
    "planning",
    "research",
    "data",
    "design",
    "operations",
]


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


rows = [
    x
    for x in read_jsonl(SRC)
    if (
        x.get(
            "execution_status",
            x.get(
                "status",
                "ok",
            ),
        )
        == "ok"
    )
]


assert len(rows) == 320, len(rows)


for x in rows:

    x["comp_id"] = str(
        x.get("comp_id")
        or str(
            x["system_id"]
        ).split("_")[0]
    )


print(
    "rows:",
    len(rows),
    flush=True,
)


# ============================================================
# EXACT CURRENT FEATURE REPRESENTATION
# ============================================================

from v8.embedding import (
    FrozenOutputEmbedderV8,
)

from v11.role_universe import (
    ROLE_DESCRIPTIONS,
)

from analysis.adaptive_fanout_count_v12 import (
    SharedPresenceDetector,
)


print(
    "loading embedder...",
    flush=True,
)

embedder = FrozenOutputEmbedderV8(
    device="cpu"
)

builder = SharedPresenceDetector(
    embedder,
    ROLE_DESCRIPTIONS,
    c=0.05,
)


print(
    "building detector features...",
    flush=True,
)


X = np.stack(
    [
        builder.features(
            str(
                x["target_role"]
            ),
            str(
                x["query"]
            ),
            str(
                x["final_output"]
            ),
        )
        for x in rows
    ]
).astype(
    np.float32
)


y = np.asarray(
    [
        int(
            x[
                "target_present_eval_only"
            ]
        )
        for x in rows
    ],
    dtype=int,
)


role_arr = np.asarray(
    [
        str(
            x["target_role"]
        )
        for x in rows
    ],
    dtype=object,
)


comp_arr = np.asarray(
    [
        str(
            x["comp_id"]
        )
        for x in rows
    ],
    dtype=object,
)


system_arr = np.asarray(
    [
        str(
            x["system_id"]
        )
        for x in rows
    ],
    dtype=object,
)


query_arr = np.asarray(
    [
        str(
            x["query"]
        )
        for x in rows
    ],
    dtype=object,
)


feature_dim = X.shape[1]

assert (
    feature_dim - 15
) % 4 == 0

D = (
    feature_dim - 15
) // 4


print(
    "feature_dim:",
    feature_dim,
    flush=True,
)

print(
    "embedding_dim:",
    D,
    flush=True,
)


# ============================================================
# FEATURE BLOCKS
# ============================================================

BLOCKS = {
    "semantic_scalars":
        np.arange(
            0,
            6,
        ),

    "style":
        np.arange(
            6,
            15,
        ),

    "output_x_role":
        np.arange(
            15,
            15 + D,
        ),

    "output_x_query":
        np.arange(
            15 + D,
            15 + 2 * D,
        ),

    "abs_output_role":
        np.arange(
            15 + 2 * D,
            15 + 3 * D,
        ),

    "abs_output_query":
        np.arange(
            15 + 3 * D,
            15 + 4 * D,
        ),
}


ALL_IDX = np.arange(
    feature_dim
)


# More interpretable super-groups.
SUPERBLOCKS = {
    "explicit_semantic":
        np.concatenate(
            [
                BLOCKS[
                    "semantic_scalars"
                ],
                BLOCKS[
                    "style"
                ],
            ]
        ),

    "role_latent_relation":
        np.concatenate(
            [
                BLOCKS[
                    "output_x_role"
                ],
                BLOCKS[
                    "abs_output_role"
                ],
            ]
        ),

    "query_latent_relation":
        np.concatenate(
            [
                BLOCKS[
                    "output_x_query"
                ],
                BLOCKS[
                    "abs_output_query"
                ],
            ]
        ),

    "all":
        ALL_IDX,
}


# ============================================================
# HELPERS
# ============================================================

def fit_model(
    Xtr,
    ytr,
):

    w = np.ones(
        len(ytr),
        dtype=float,
    )

    return (
        SharedPresenceDetector
        ._fit_pipe(
            Xtr,
            ytr,
            w,
            0.05,
        )
    )


def auc(
    yy,
    ss,
):

    yy = np.asarray(
        yy,
        dtype=int,
    )

    ss = np.asarray(
        ss,
        dtype=float,
    )

    if len(
        np.unique(yy)
    ) != 2:
        return None

    return float(
        roc_auc_score(
            yy,
            ss,
        )
    )


unique_comps = np.asarray(
    sorted(
        set(
            comp_arr.tolist()
        )
    ),
    dtype=object,
)


# Fixed composition folds.
comp_cv = list(
    GroupKFold(
        n_splits=5
    ).split(
        unique_comps,
        np.zeros(
            len(
                unique_comps
            )
        ),
        unique_comps,
    )
)


# ============================================================
# PART A:
# TRUE LEAVE-ROLE-OUT + COMPOSITION-HOLDOUT
#
# Test example is scored by a model that saw:
#   - NO example of that target role
#   - NO example from the held-out compositions
#
# This isolates generic detector transfer.
# ============================================================

print()
print(
    "=" * 120
)

print(
    "A. UNSEEN-ROLE + UNSEEN-COMPOSITION FINGERPRINT"
)

print(
    "=" * 120
)


loo_scores = np.full(
    len(rows),
    np.nan,
    dtype=float,
)


loo_role_auc = {}


for held_role in ROLES:

    target_mask = (
        role_arr
        == held_role
    )


    for fold, (
        trc,
        tec,
    ) in enumerate(
        comp_cv,
        1,
    ):

        test_comps = set(
            unique_comps[
                tec
            ].tolist()
        )


        train_mask = np.asarray(
            [
                (
                    role_arr[i]
                    != held_role
                    and
                    comp_arr[i]
                    not in test_comps
                )
                for i in range(
                    len(rows)
                )
            ],
            dtype=bool,
        )


        test_mask = np.asarray(
            [
                (
                    role_arr[i]
                    == held_role
                    and
                    comp_arr[i]
                    in test_comps
                )
                for i in range(
                    len(rows)
                )
            ],
            dtype=bool,
        )


        assert not np.any(
            role_arr[
                train_mask
            ]
            == held_role
        )


        assert not (
            set(
                comp_arr[
                    train_mask
                ].tolist()
            )
            & test_comps
        )


        model = fit_model(
            X[
                train_mask
            ],
            y[
                train_mask
            ],
        )


        loo_scores[
            test_mask
        ] = (
            model.decision_function(
                X[
                    test_mask
                ]
            )
        )


    rm = target_mask

    assert np.isfinite(
        loo_scores[
            rm
        ]
    ).all()


    loo_role_auc[
        held_role
    ] = auc(
        y[
            rm
        ],
        loo_scores[
            rm
        ],
    )


    print(
        f"{held_role:<12} "
        f"unseen-role AUC="
        f"{loo_role_auc[held_role]:.4f}"
    )


assert np.isfinite(
    loo_scores
).all()


global_loo_auc = auc(
    y,
    loo_scores,
)


print()
print(
    "GLOBAL unseen-role OOF AUC:",
    f"{global_loo_auc:.4f}",
)

print(
    "MEAN unseen-role AUC:",
    f"{np.mean(list(loo_role_auc.values())):.4f}",
)

print(
    "MEDIAN unseen-role AUC:",
    f"{np.median(list(loo_role_auc.values())):.4f}",
)


# ============================================================
# PART B:
# WHICH REPRESENTATION BLOCKS TRANSFER TO UNSEEN ROLES?
# ============================================================

print()
print(
    "=" * 120
)

print(
    "B. UNSEEN-ROLE FEATURE-BLOCK TRANSFER"
)

print(
    "=" * 120
)


block_results = {}


for block_name, idx in (
    SUPERBLOCKS.items()
):

    scores = np.full(
        len(rows),
        np.nan,
        dtype=float,
    )


    for held_role in ROLES:

        for trc, tec in comp_cv:

            test_comps = set(
                unique_comps[
                    tec
                ].tolist()
            )


            train_mask = np.asarray(
                [
                    (
                        role_arr[i]
                        != held_role
                        and
                        comp_arr[i]
                        not in test_comps
                    )
                    for i in range(
                        len(rows)
                    )
                ],
                dtype=bool,
            )


            test_mask = np.asarray(
                [
                    (
                        role_arr[i]
                        == held_role
                        and
                        comp_arr[i]
                        in test_comps
                    )
                    for i in range(
                        len(rows)
                    )
                ],
                dtype=bool,
            )


            model = fit_model(
                X[
                    train_mask
                ][:, idx],
                y[
                    train_mask
                ],
            )


            scores[
                test_mask
            ] = (
                model.decision_function(
                    X[
                        test_mask
                    ][:, idx]
                )
            )


    assert np.isfinite(
        scores
    ).all()


    per_role = {}


    for role in ROLES:

        rm = (
            role_arr
            == role
        )

        per_role[
            role
        ] = auc(
            y[
                rm
            ],
            scores[
                rm
            ],
        )


    block_results[
        block_name
    ] = {
        "global_auc":
            auc(
                y,
                scores,
            ),

        "mean_role_auc":
            float(
                np.mean(
                    list(
                        per_role.values()
                    )
                )
            ),

        "median_role_auc":
            float(
                np.median(
                    list(
                        per_role.values()
                    )
                )
            ),

        "per_role_auc":
            per_role,
    }


    print(
        f"{block_name:<24} "
        f"global="
        f"{block_results[block_name]['global_auc']:.4f} "
        f"meanRole="
        f"{block_results[block_name]['mean_role_auc']:.4f} "
        f"medianRole="
        f"{block_results[block_name]['median_role_auc']:.4f}"
    )


# ============================================================
# PART C:
# ABLATION UNDER THE ORIGINAL COMPOSITION-OOF SETTING.
#
# Which block is actually necessary for current performance?
# ============================================================

print()
print(
    "=" * 120
)

print(
    "C. CURRENT-FINGERPRINT BLOCK ABLATION"
)

print(
    "=" * 120
)


ablation_results = {}


for removed_name, removed_idx in (
    BLOCKS.items()
):

    keep = np.setdiff1d(
        ALL_IDX,
        removed_idx,
    )


    scores = np.full(
        len(rows),
        np.nan,
        dtype=float,
    )


    for trc, tec in comp_cv:

        test_comps = set(
            unique_comps[
                tec
            ].tolist()
        )


        train_mask = np.asarray(
            [
                comp_arr[i]
                not in test_comps
                for i in range(
                    len(rows)
                )
            ],
            dtype=bool,
        )


        test_mask = ~train_mask


        model = fit_model(
            X[
                train_mask
            ][:, keep],
            y[
                train_mask
            ],
        )


        scores[
            test_mask
        ] = (
            model.decision_function(
                X[
                    test_mask
                ][:, keep]
            )
        )


    result_auc = auc(
        y,
        scores,
    )


    ablation_results[
        removed_name
    ] = result_auc


    print(
        f"remove "
        f"{removed_name:<20} "
        f"AUC="
        f"{result_auc:.4f}"
    )


# ============================================================
# PART D:
# LINEAR CONTRIBUTION AUDIT
#
# For every OOF prediction, calculate:
#
#   standardized_feature * LR_weight
#
# Then sum within interpretable feature blocks.
#
# This tells us what pushed a prediction toward PRESENT.
# ============================================================

print()
print(
    "=" * 120
)

print(
    "D. OOF LINEAR FINGERPRINT CONTRIBUTIONS"
)

print(
    "=" * 120
)


block_contrib = {
    name:
        np.full(
            len(rows),
            np.nan,
            dtype=float,
        )
    for name in BLOCKS
}


full_scores = np.full(
    len(rows),
    np.nan,
    dtype=float,
)


for trc, tec in comp_cv:

    test_comps = set(
        unique_comps[
            tec
        ].tolist()
    )


    train_mask = np.asarray(
        [
            comp_arr[i]
            not in test_comps
            for i in range(
                len(rows)
            )
        ],
        dtype=bool,
    )

    test_mask = ~train_mask


    model = fit_model(
        X[
            train_mask
        ],
        y[
            train_mask
        ],
    )


    scaler = model.named_steps[
        "standardscaler"
    ]

    clf = model.named_steps[
        "logisticregression"
    ]


    z = scaler.transform(
        X[
            test_mask
        ]
    )


    coef = clf.coef_[
        0
    ]


    contributions = (
        z
        * coef[
            None,
            :
        ]
    )


    full_scores[
        test_mask
    ] = (
        model.decision_function(
            X[
                test_mask
            ]
        )
    )


    for name, idx in (
        BLOCKS.items()
    ):

        block_contrib[
            name
        ][
            test_mask
        ] = (
            contributions[
                :,
                idx
            ].sum(
                axis=1
            )
        )


assert np.isfinite(
    full_scores
).all()


print(
    "full composition-OOF AUC:",
    f"{auc(y, full_scores):.4f}",
)


contribution_summary = {}


for name, vals in (
    block_contrib.items()
):

    pos_mean = float(
        vals[
            y == 1
        ].mean()
    )

    neg_mean = float(
        vals[
            y == 0
        ].mean()
    )

    delta = (
        pos_mean
        - neg_mean
    )


    contribution_summary[
        name
    ] = {
        "present_mean":
            pos_mean,

        "absent_mean":
            neg_mean,

        "present_minus_absent":
            delta,
    }


print()
print(
    f"{'block':<22} "
    f"{'PRESENT':>10} "
    f"{'ABSENT':>10} "
    f"{'DELTA':>10}"
)


for name, s in sorted(
    contribution_summary.items(),
    key=lambda kv:
        -abs(
            kv[
                1
            ][
                "present_minus_absent"
            ]
        ),
):

    print(
        f"{name:<22} "
        f"{s['present_mean']:>+10.3f} "
        f"{s['absent_mean']:>+10.3f} "
        f"{s['present_minus_absent']:>+10.3f}"
    )


# ============================================================
# PART E:
# PER-ROLE EXPLANATION PROFILE
# ============================================================

print()
print(
    "=" * 120
)

print(
    "E. PER-ROLE FINGERPRINT CONTRIBUTION PROFILE"
)

print(
    "=" * 120
)


role_profiles = {}


for role in ROLES:

    rm = (
        role_arr
        == role
    )


    profile = []


    for name, vals in (
        block_contrib.items()
    ):

        pos = vals[
            rm
            & (
                y == 1
            )
        ]

        neg = vals[
            rm
            & (
                y == 0
            )
        ]


        delta = float(
            pos.mean()
            - neg.mean()
        )


        profile.append(
            (
                name,
                delta,
            )
        )


    profile.sort(
        key=lambda z:
            -abs(
                z[1]
            )
    )


    role_profiles[
        role
    ] = {
        k:
            float(v)
        for k, v
        in profile
    }


    print(
        f"\n### {role}"
    )


    for name, delta in profile:

        print(
            f"  "
            f"{name:<22} "
            f"Δpresent-absent="
            f"{delta:+.3f}"
        )


# ============================================================
# PART F:
# SAVE PER-EXAMPLE ATTRIBUTIONS
# ============================================================

attribution_rows = []


for i, x in enumerate(
    rows
):

    attribution_rows.append({
        "system_id":
            x[
                "system_id"
            ],

        "comp_id":
            x[
                "comp_id"
            ],

        "role":
            x[
                "target_role"
            ],

        "query":
            x[
                "query"
            ],

        "final_output":
            x[
                "final_output"
            ],

        "present_eval_only":
            int(
                x[
                    "target_present_eval_only"
                ]
            ),

        "current_oof_score":
            float(
                full_scores[
                    i
                ]
            ),

        "unseen_role_oof_score":
            float(
                loo_scores[
                    i
                ]
            ),

        "block_contributions":
            {
                name:
                    float(
                        vals[
                            i
                        ]
                    )
                for name, vals
                in block_contrib.items()
            },
    })


summary = {
    "experiment":
        "explainable_fingerprint_v1",

    "n_rows":
        len(
            rows
        ),

    "feature_dim":
        int(
            feature_dim
        ),

    "embedding_dim":
        int(
            D
        ),

    "unseen_role_plus_composition":
        {
            "global_auc":
                global_loo_auc,

            "mean_role_auc":
                float(
                    np.mean(
                        list(
                            loo_role_auc.values()
                        )
                    )
                ),

            "median_role_auc":
                float(
                    np.median(
                        list(
                            loo_role_auc.values()
                        )
                    )
                ),

            "per_role_auc":
                loo_role_auc,
        },

    "block_transfer":
        block_results,

    "ablations":
        ablation_results,

    "contribution_summary":
        contribution_summary,

    "role_profiles":
        role_profiles,
}


(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


with (
    OUT
    / "attributions.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for x in attribution_rows:

        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print(
    "wrote:",
    OUT
    / "summary.json"
)

print(
    "wrote:",
    OUT
    / "attributions.jsonl"
)
