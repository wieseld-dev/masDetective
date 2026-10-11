from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


SRC = Path(
    "artifacts/v11_probe_discovery.jsonl"
)

OUT = Path(
    "artifacts/generic_fingerprint_v2"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


from v11.role_universe import (
    TRAIN_ROLES,
    DEV_UNSEEN_ROLES,
    ROLE_DESCRIPTIONS,
)

from v8.embedding import (
    FrozenOutputEmbedderV8,
)

from analysis.adaptive_fanout_count_v12 import (
    SharedPresenceDetector,
)


TRAIN_ROLES = tuple(
    map(
        str,
        TRAIN_ROLES,
    )
)

DEV_ROLES = tuple(
    map(
        str,
        DEV_UNSEEN_ROLES,
    )
)

assert not (
    set(TRAIN_ROLES)
    & set(DEV_ROLES)
)


def read_jsonl(path):

    return [
        json.loads(x)
        for x in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


raw = read_jsonl(
    SRC
)


# ============================================================
# 1. BUILD MATCHED COUNTERFACTUAL BRANCHES
#
# Each row contains:
# role, query, plus_output, sham_output, minus_output
#
# plus/sham = role-present branches
# minus     = role-absent branch
# ============================================================

branches = []

seen = set()


for rec_idx, rec in enumerate(
    raw
):

    role = str(
        rec.get(
            "target_role",
            "",
        )
    )

    phase = str(
        rec.get(
            "phase",
            "",
        )
    )

    query = str(
        rec.get(
            "query",
            "",
        )
    ).strip()


    if (
        role
        not in set(
            TRAIN_ROLES
        )
        | set(
            DEV_ROLES
        )
    ):
        continue

    if phase not in {
        "train",
        "holdout",
    }:
        continue

    if not query:
        continue


    bb = list(
        rec.get(
            "branches_training_only"
        )
        or []
    )

    bb += list(
        rec.get(
            "confirmation_branches_training_only"
        )
        or []
    )


    for branch_idx, b in enumerate(
        bb
    ):

        plus = str(
            b.get(
                "plus_output",
                "",
            )
            or ""
        ).strip()

        sham = str(
            b.get(
                "sham_output",
                "",
            )
            or ""
        ).strip()

        minus = str(
            b.get(
                "minus_output",
                "",
            )
            or ""
        ).strip()


        if not (
            plus
            and sham
            and minus
        ):
            continue


        key = (
            role,
            query,
            plus,
            sham,
            minus,
        )

        if key in seen:
            continue

        seen.add(
            key
        )


        split = (
            "train"
            if role
            in TRAIN_ROLES
            else "dev_unseen"
        )


        branches.append({
            "branch_id":
                (
                    f"{phase}:"
                    f"{rec_idx}:"
                    f"{branch_idx}"
                ),

            "split":
                split,

            "phase":
                phase,

            "role":
                role,

            "query":
                query,

            "plus_output":
                plus,

            "sham_output":
                sham,

            "minus_output":
                minus,
        })


train_branches = [
    x
    for x in branches
    if x[
        "split"
    ] == "train"
]

dev_branches = [
    x
    for x in branches
    if x[
        "split"
    ] == "dev_unseen"
]


print(
    "===== DATASET ====="
)

print(
    "train roles:",
    len(
        TRAIN_ROLES
    ),
    list(
        TRAIN_ROLES
    ),
)

print(
    "dev unseen roles:",
    len(
        DEV_ROLES
    ),
    list(
        DEV_ROLES
    ),
)

print(
    "train branches:",
    len(
        train_branches
    ),
)

print(
    "dev branches:",
    len(
        dev_branches
    ),
)


for split_name, rr in [
    (
        "TRAIN",
        train_branches,
    ),
    (
        "DEV_UNSEEN",
        dev_branches,
    ),
]:

    print()
    print(
        split_name
    )

    for role in (
        TRAIN_ROLES
        if split_name
        == "TRAIN"
        else DEV_ROLES
    ):

        n = sum(
            x[
                "role"
            ]
            == role
            for x in rr
        )

        nq = len({
            x[
                "query"
            ]
            for x in rr
            if x[
                "role"
            ]
            == role
        })

        print(
            f"{role:<28} "
            f"branches={n:<4} "
            f"queries={nq}"
        )


if not train_branches:
    raise RuntimeError(
        "No train branches"
    )

if not dev_branches:
    raise RuntimeError(
        "No unseen DEV branches"
    )


# ============================================================
# 2. CONVERT BRANCHES INTO INDIVIDUAL OUTPUT EXAMPLES
# ============================================================

samples = []


for b in branches:

    for kind, y, field in [
        (
            "plus",
            1,
            "plus_output",
        ),
        (
            "sham",
            1,
            "sham_output",
        ),
        (
            "minus",
            0,
            "minus_output",
        ),
    ]:

        samples.append({
            "branch_id":
                b[
                    "branch_id"
                ],

            "split":
                b[
                    "split"
                ],

            "role":
                b[
                    "role"
                ],

            "query":
                b[
                    "query"
                ],

            "kind":
                kind,

            "y":
                y,

            "output":
                b[
                    field
                ],
        })


train_idx = np.asarray(
    [
        x[
            "split"
        ]
        == "train"
        for x in samples
    ],
    dtype=bool,
)

dev_idx = np.asarray(
    [
        x[
            "split"
        ]
        == "dev_unseen"
        for x in samples
    ],
    dtype=bool,
)


y = np.asarray(
    [
        x[
            "y"
        ]
        for x in samples
    ],
    dtype=int,
)


print()
print(
    "train output samples:",
    int(
        train_idx.sum()
    ),
)

print(
    "dev output samples:",
    int(
        dev_idx.sum()
    ),
)


# ============================================================
# 3. EMBEDDINGS
# ============================================================

print()
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


embedding_cache = {}


def emb(text):

    text = str(
        text
    )

    if text not in embedding_cache:

        embedding_cache[
            text
        ] = (
            embedder
            .encode(
                text
            )
            .astype(
                np.float32
            )
        )

    return embedding_cache[
        text
    ]


def cosine(a, b):

    na = float(
        np.linalg.norm(
            a
        )
    )

    nb = float(
        np.linalg.norm(
            b
        )
    )

    if (
        na <= 1e-12
        or nb <= 1e-12
    ):
        return 0.0

    return float(
        np.dot(
            a,
            b,
        )
        / (
            na
            * nb
        )
    )


# ============================================================
# 4. FULL CURRENT REPRESENTATION
# ============================================================

print(
    "building FULL features...",
    flush=True,
)


X_full = np.stack(
    [
        builder.features(
            x[
                "role"
            ],
            x[
                "query"
            ],
            x[
                "output"
            ],
        )
        for x in samples
    ]
).astype(
    np.float32
)


# ============================================================
# 5. GENERIC GEOMETRIC REPRESENTATION
#
# No raw embedding coordinate is exposed.
# Only relationships among role/query/output.
# ============================================================

GEOM_NAMES = [
    "cos_output_role",
    "cos_output_query",
    "cos_query_role",

    "dist_output_role",
    "dist_output_query",
    "dist_query_role",

    "role_alignment_gain",
    "query_alignment_gain",

    "cos_response_shift_role",
    "cos_response_shift_role_minus_query",

    "cos_output_minus_role_query",
    "cos_output_minus_role_query_minus_role",

    "norm_response_shift",
    "norm_output_minus_role",
    "norm_role_minus_query",

    "projection_shift_on_role",
    "projection_shift_on_role_minus_query",

    "triangle_slack_role_via_query",
    "triangle_slack_query_via_role",

    "response_shift_over_role_query_distance",
]


def geom_features(
    role,
    query,
    output,
):

    desc = ROLE_DESCRIPTIONS.get(
        role,
        role,
    )

    r = emb(
        desc
    )

    q = emb(
        query
    )

    o = emb(
        output
    )


    oq = (
        o
        - q
    )

    or_ = (
        o
        - r
    )

    rq = (
        r
        - q
    )

    qr = (
        q
        - r
    )


    d_or = float(
        np.linalg.norm(
            or_
        )
    )

    d_oq = float(
        np.linalg.norm(
            oq
        )
    )

    d_qr = float(
        np.linalg.norm(
            q
            - r
        )
    )


    cos_or = cosine(
        o,
        r,
    )

    cos_oq = cosine(
        o,
        q,
    )

    cos_qr = cosine(
        q,
        r,
    )


    return np.asarray(
        [
            cos_or,
            cos_oq,
            cos_qr,

            d_or,
            d_oq,
            d_qr,

            cos_or
            - cos_qr,

            cos_oq
            - cos_qr,

            cosine(
                oq,
                r,
            ),

            cosine(
                oq,
                rq,
            ),

            cosine(
                or_,
                q,
            ),

            cosine(
                or_,
                qr,
            ),

            d_oq,
            d_or,
            d_qr,

            float(
                np.dot(
                    oq,
                    r,
                )
            ),

            float(
                np.dot(
                    oq,
                    rq,
                )
            ),

            d_oq
            + d_qr
            - d_or,

            d_or
            + d_qr
            - d_oq,

            d_oq
            / max(
                d_qr,
                1e-6,
            ),
        ],
        dtype=np.float32,
    )


print(
    "building GEOM features...",
    flush=True,
)


X_geom = np.stack(
    [
        geom_features(
            x[
                "role"
            ],
            x[
                "query"
            ],
            x[
                "output"
            ],
        )
        for x in samples
    ]
)


print()
print(
    "FULL dim:",
    X_full.shape[
        1
    ],
)

print(
    "GEOM dim:",
    X_geom.shape[
        1
    ],
)


# ============================================================
# 6. TRAINING
# ============================================================

C = 0.05


def fit_pointwise(
    X,
):

    scaler = StandardScaler()

    ztr = scaler.fit_transform(
        X[
            train_idx
        ]
    )


    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        random_state=0,
    )


    clf.fit(
        ztr,
        y[
            train_idx
        ],
    )


    return (
        scaler,
        clf,
    )


sample_lookup = {}


for i, x in enumerate(
    samples
):

    sample_lookup[
        (
            x[
                "branch_id"
            ],
            x[
                "kind"
            ],
        )
    ] = i


def make_pairwise_diffs(
    Z,
):

    diffs = []


    for b in train_branches:

        p = sample_lookup[
            (
                b[
                    "branch_id"
                ],
                "plus",
            )
        ]

        s = sample_lookup[
            (
                b[
                    "branch_id"
                ],
                "sham",
            )
        ]

        m = sample_lookup[
            (
                b[
                    "branch_id"
                ],
                "minus",
            )
        ]


        diffs.append(
            Z[p]
            - Z[m]
        )

        diffs.append(
            Z[s]
            - Z[m]
        )


    return np.stack(
        diffs
    )


def fit_pairwise(
    X,
):

    scaler = StandardScaler()

    scaler.fit(
        X[
            train_idx
        ]
    )


    Z = scaler.transform(
        X
    )


    Dpos = make_pairwise_diffs(
        Z
    )


    D = np.vstack(
        [
            Dpos,
            -Dpos,
        ]
    )


    yy = np.concatenate(
        [
            np.ones(
                len(
                    Dpos
                ),
                dtype=int,
            ),

            np.zeros(
                len(
                    Dpos
                ),
                dtype=int,
            ),
        ]
    )


    clf = LogisticRegression(
        C=C,
        fit_intercept=False,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        random_state=0,
    )


    clf.fit(
        D,
        yy,
    )


    return (
        scaler,
        clf,
    )


def decision(
    scaler,
    clf,
    X,
):

    Z = scaler.transform(
        X
    )


    if clf.fit_intercept:

        return clf.decision_function(
            Z
        )


    return (
        Z
        @ clf.coef_[
            0
        ]
    )


MODELS = {}


for rep_name, X in [
    (
        "full",
        X_full,
    ),
    (
        "geom",
        X_geom,
    ),
]:

    print()
    print(
        f"training POINTWISE_{rep_name.upper()}...",
        flush=True,
    )

    MODELS[
        f"pointwise_{rep_name}"
    ] = (
        *fit_pointwise(
            X
        ),
        X,
    )


    print(
        f"training PAIRWISE_{rep_name.upper()}...",
        flush=True,
    )

    MODELS[
        f"pairwise_{rep_name}"
    ] = (
        *fit_pairwise(
            X
        ),
        X,
    )


# ============================================================
# 7. METRICS
# ============================================================

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
        np.unique(
            yy
        )
    ) != 2:
        return None

    return float(
        roc_auc_score(
            yy,
            ss,
        )
    )


results = {}


for model_name, (
    scaler,
    clf,
    X,
) in MODELS.items():

    score = decision(
        scaler,
        clf,
        X,
    )


    dev_score = score[
        dev_idx
    ]

    dev_y = y[
        dev_idx
    ]


    per_role_auc = {}

    per_role_pair = {}

    per_role_strict = {}

    per_role_query_pair = {}


    for role in DEV_ROLES:

        rm = np.asarray(
            [
                (
                    x[
                        "split"
                    ]
                    == "dev_unseen"
                    and
                    x[
                        "role"
                    ]
                    == role
                )
                for x in samples
            ],
            dtype=bool,
        )


        per_role_auc[
            role
        ] = auc(
            y[
                rm
            ],
            score[
                rm
            ],
        )


        role_branches = [
            b
            for b in dev_branches
            if b[
                "role"
            ]
            == role
        ]


        wins = 0

        total = 0

        strict_wins = 0


        query_stats = defaultdict(
            lambda: [
                0,
                0,
            ]
        )


        for b in role_branches:

            p = sample_lookup[
                (
                    b[
                        "branch_id"
                    ],
                    "plus",
                )
            ]

            s = sample_lookup[
                (
                    b[
                        "branch_id"
                    ],
                    "sham",
                )
            ]

            m = sample_lookup[
                (
                    b[
                        "branch_id"
                    ],
                    "minus",
                )
            ]


            wp = int(
                score[p]
                > score[m]
            )

            ws = int(
                score[s]
                > score[m]
            )


            wins += (
                wp
                + ws
            )

            total += 2


            strict_wins += int(
                wp
                and ws
            )


            qkey = b[
                "query"
            ]

            query_stats[
                qkey
            ][0] += (
                wp
                + ws
            )

            query_stats[
                qkey
            ][1] += 2


        per_role_pair[
            role
        ] = (
            wins
            / total
        )


        per_role_strict[
            role
        ] = (
            strict_wins
            / len(
                role_branches
            )
        )


        q_acc = [
            w
            / n
            for w, n
            in query_stats.values()
        ]


        per_role_query_pair[
            role
        ] = float(
            np.mean(
                q_acc
            )
        )


    pair_values = list(
        per_role_pair.values()
    )

    strict_values = list(
        per_role_strict.values()
    )

    query_values = list(
        per_role_query_pair.values()
    )


    result = {
        "global_dev_auc":
            auc(
                dev_y,
                dev_score,
            ),

        "mean_role_auc":
            float(
                np.mean(
                    list(
                        per_role_auc.values()
                    )
                )
            ),

        "median_role_auc":
            float(
                np.median(
                    list(
                        per_role_auc.values()
                    )
                )
            ),

        "mean_role_pair_accuracy":
            float(
                np.mean(
                    pair_values
                )
            ),

        "mean_role_strict_branch_win":
            float(
                np.mean(
                    strict_values
                )
            ),

        "mean_role_query_macro_pair_accuracy":
            float(
                np.mean(
                    query_values
                )
            ),

        "per_role_auc":
            per_role_auc,

        "per_role_pair_accuracy":
            per_role_pair,

        "per_role_strict_branch_win":
            per_role_strict,

        "per_role_query_macro_pair_accuracy":
            per_role_query_pair,
    }


    results[
        model_name
    ] = result


    print()
    print(
        "=" * 110
    )

    print(
        model_name.upper()
    )

    print(
        "=" * 110
    )


    for role in DEV_ROLES:

        print(
            f"{role:<28} "
            f"AUC="
            f"{per_role_auc[role]:.4f} "
            f"pairAcc="
            f"{per_role_pair[role]:.4f} "
            f"strict="
            f"{per_role_strict[role]:.4f} "
            f"queryMacro="
            f"{per_role_query_pair[role]:.4f}"
        )


    print()
    print(
        "GLOBAL DEV AUC:",
        f"{result['global_dev_auc']:.4f}",
    )

    print(
        "MEAN ROLE AUC:",
        f"{result['mean_role_auc']:.4f}",
    )

    print(
        "MEDIAN ROLE AUC:",
        f"{result['median_role_auc']:.4f}",
    )

    print(
        "MEAN ROLE PAIR ACC:",
        f"{result['mean_role_pair_accuracy']:.4f}",
    )

    print(
        "MEAN STRICT BRANCH WIN:",
        f"{result['mean_role_strict_branch_win']:.4f}",
    )

    print(
        "MEAN QUERY-MACRO PAIR ACC:",
        f"{result['mean_role_query_macro_pair_accuracy']:.4f}",
    )


# ============================================================
# 8. EXPLAINABLE GEOM COEFFICIENTS
# ============================================================

print()
print(
    "=" * 110
)

print(
    "PAIRWISE_GEOM — EXPLAINABLE FINGERPRINT WEIGHTS"
)

print(
    "=" * 110
)


geom_scaler, geom_clf, _ = MODELS[
    "pairwise_geom"
]


coef = geom_clf.coef_[
    0
]


weight_rows = sorted(
    zip(
        GEOM_NAMES,
        coef.tolist(),
    ),
    key=lambda z:
        -abs(
            z[1]
        ),
)


for name, w in weight_rows:

    direction = (
        "PRESENT"
        if w > 0
        else "ABSENT"
    )

    print(
        f"{name:<48} "
        f"weight="
        f"{w:+.4f} "
        f"toward={direction}"
    )


# ============================================================
# 9. FINAL COMPARISON
# ============================================================

print()
print(
    "=" * 110
)

print(
    "FINAL 20-TRAIN -> 6-UNSEEN-ROLE DEV COMPARISON"
)

print(
    "=" * 110
)


for name, r in results.items():

    print(
        f"{name:<20} "
        f"globalAUC="
        f"{r['global_dev_auc']:.4f} "
        f"meanRoleAUC="
        f"{r['mean_role_auc']:.4f} "
        f"medianRoleAUC="
        f"{r['median_role_auc']:.4f} "
        f"pairAcc="
        f"{r['mean_role_pair_accuracy']:.4f} "
        f"strict="
        f"{r['mean_role_strict_branch_win']:.4f} "
        f"queryMacro="
        f"{r['mean_role_query_macro_pair_accuracy']:.4f}"
    )


summary = {
    "experiment":
        "generic_fingerprint_v2",

    "train_roles":
        list(
            TRAIN_ROLES
        ),

    "dev_unseen_roles":
        list(
            DEV_ROLES
        ),

    "train_branches":
        len(
            train_branches
        ),

    "dev_branches":
        len(
            dev_branches
        ),

    "C":
        C,

    "results":
        results,

    "pairwise_geom_weights":
        [
            {
                "feature":
                    name,

                "weight":
                    float(
                        w
                    ),
            }
            for name, w
            in weight_rows
        ],
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


print()
print(
    "wrote:",
    OUT
    / "summary.json"
)
