from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


SRC = Path(
    "artifacts/frozen_probe_auc_v2/"
    "responses.jsonl"
)

OUT = Path(
    "artifacts/generic_pairwise_fingerprint_v1"
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

    x["role"] = str(
        x["target_role"]
    )

    x["comp_id"] = str(
        x.get("comp_id")
        or str(
            x["system_id"]
        ).split("_")[0]
    )

    x["y"] = int(
        x["target_present_eval_only"]
    )


print(
    "rows:",
    len(rows),
    flush=True,
)


# ============================================================
# EMBEDDINGS + CURRENT FULL FEATURES
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


# Cache embeddings because role/query repeat heavily.
emb_cache = {}


def emb(text):

    text = str(text)

    if text not in emb_cache:

        emb_cache[text] = (
            embedder
            .encode(text)
            .astype(
                np.float32
            )
        )

    return emb_cache[text]


def cosine(a, b):

    na = float(
        np.linalg.norm(a)
    )

    nb = float(
        np.linalg.norm(b)
    )

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b)
        / (
            na
            * nb
        )
    )


print(
    "building features...",
    flush=True,
)


X_full = np.stack(
    [
        builder.features(
            x["role"],
            x["query"],
            x["final_output"],
        )
        for x in rows
    ]
).astype(
    np.float32
)


# ============================================================
# GENERIC GEOMETRIC FEATURES
#
# No raw embedding coordinate enters directly.
# Everything is a relation between role/query/output.
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

    "cos_output_minus_query__role",
    "cos_output_minus_query__role_minus_query",
    "cos_output_minus_role__query",
    "cos_output_minus_role__query_minus_role",

    "norm_output_minus_query",
    "norm_output_minus_role",
    "norm_role_minus_query",

    "projection_response_shift_on_role",
    "projection_response_shift_on_role_minus_query",

    "triangle_slack_role_via_query",
    "triangle_slack_query_via_role",

    "response_shift_over_role_query_distance",
]


def geom_features(row):

    role_desc = (
        ROLE_DESCRIPTIONS.get(
            row["role"],
            row["role"],
        )
    )

    r = emb(
        role_desc
    )

    q = emb(
        row["query"]
    )

    o = emb(
        row["final_output"]
    )


    oq = (
        o - q
    )

    or_ = (
        o - r
    )

    rq = (
        r - q
    )

    qr = (
        q - r
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
            q - r
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


    vals = [
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
    ]


    return np.asarray(
        vals,
        dtype=np.float32,
    )


X_geom = np.stack(
    [
        geom_features(x)
        for x in rows
    ]
)


print(
    "full dim:",
    X_full.shape[1],
    flush=True,
)

print(
    "geom dim:",
    X_geom.shape[1],
    flush=True,
)


y = np.asarray(
    [
        x["y"]
        for x in rows
    ],
    dtype=int,
)

role_arr = np.asarray(
    [
        x["role"]
        for x in rows
    ],
    dtype=object,
)

comp_arr = np.asarray(
    [
        x["comp_id"]
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


# ============================================================
# PAIRWISE DATA
#
# Pair only examples with SAME:
#   role + exact query
#
# Therefore the only thing that differs is
# response generated by present vs absent MAS.
# ============================================================

def make_pairs(
    X,
    mask,
):

    groups = defaultdict(
        lambda: {
            0: [],
            1: [],
        }
    )


    idxs = np.flatnonzero(
        mask
    )


    for i in idxs:

        key = (
            role_arr[i],
            query_arr[i],
        )

        groups[
            key
        ][
            int(
                y[i]
            )
        ].append(
            i
        )


    diffs = []


    for key, g in groups.items():

        pos = g[1]

        neg = g[0]


        for ip in pos:

            for ineg in neg:

                diffs.append(
                    X[ip]
                    - X[ineg]
                )


    if not diffs:

        raise RuntimeError(
            "No counterfactual pairs"
        )


    return np.stack(
        diffs
    ).astype(
        np.float32
    )


# ============================================================
# MODEL
#
# Pairwise logistic ranking:
#
#   wᵀ(x+ - x-) > 0
#
# We train both forward and reverse differences.
# No intercept: score must come from evidence, not pair prior.
# ============================================================

def fit_pairwise_ranker(
    X,
    train_mask,
    C,
):

    scaler = StandardScaler()

    scaler.fit(
        X[
            train_mask
        ]
    )


    Xz = scaler.transform(
        X
    )


    Dpos = make_pairs(
        Xz,
        train_mask,
    )


    D = np.vstack(
        [
            Dpos,
            -Dpos,
        ]
    )


    yp = np.concatenate(
        [
            np.ones(
                len(Dpos),
                dtype=int,
            ),
            np.zeros(
                len(Dpos),
                dtype=int,
            ),
        ]
    )


    clf = LogisticRegression(
        C=float(C),
        fit_intercept=False,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        random_state=0,
    )


    clf.fit(
        D,
        yp,
    )


    return (
        scaler,
        clf,
        len(
            Dpos
        ),
    )


def score_ranker(
    scaler,
    clf,
    X,
    mask,
):

    z = scaler.transform(
        X[
            mask
        ]
    )

    return (
        z
        @ clf.coef_[
            0
        ]
    )


# ============================================================
# POINTWISE BASELINE WITH IDENTICAL SPLITS
# ============================================================

def fit_pointwise(
    X,
    train_mask,
    C,
):

    scaler = StandardScaler()

    Xtr = scaler.fit_transform(
        X[
            train_mask
        ]
    )


    clf = LogisticRegression(
        C=float(C),
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        random_state=0,
    )


    clf.fit(
        Xtr,
        y[
            train_mask
        ],
    )


    return (
        scaler,
        clf,
    )


# ============================================================
# OUTER EVALUATION
#
# Hold out:
#   1. entire target role
#   2. test compositions
#
# Hyperparameter selection is done only from training roles.
# ============================================================

unique_comps = np.asarray(
    sorted(
        set(
            comp_arr.tolist()
        )
    ),
    dtype=object,
)


outer_comp_cv = list(
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


C_GRID = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
]


REPRESENTATIONS = {
    "full":
        X_full,

    "geom":
        X_geom,
}


METHODS = [
    "pointwise",
    "pairwise",
]


def safe_auc(
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


def inner_select_c(
    X,
    base_train_mask,
    method,
):

    available_roles = [
        r
        for r in ROLES
        if np.any(
            base_train_mask
            & (
                role_arr
                == r
            )
        )
    ]


    results = []


    for C in C_GRID:

        aucs = []


        # Inner leave-one-role-out.
        for vr in available_roles:

            tr = (
                base_train_mask
                & (
                    role_arr
                    != vr
                )
            )

            va = (
                base_train_mask
                & (
                    role_arr
                    == vr
                )
            )


            if (
                len(
                    np.unique(
                        y[
                            va
                        ]
                    )
                )
                != 2
            ):
                continue


            try:

                if method == "pairwise":

                    scaler, clf, _ = (
                        fit_pairwise_ranker(
                            X,
                            tr,
                            C,
                        )
                    )

                    s = score_ranker(
                        scaler,
                        clf,
                        X,
                        va,
                    )

                else:

                    scaler, clf = (
                        fit_pointwise(
                            X,
                            tr,
                            C,
                        )
                    )

                    s = (
                        clf.decision_function(
                            scaler.transform(
                                X[
                                    va
                                ]
                            )
                        )
                    )


                a = safe_auc(
                    y[
                        va
                    ],
                    s,
                )


                if a is not None:

                    aucs.append(
                        a
                    )


            except Exception:

                continue


        mean_auc = (
            float(
                np.mean(
                    aucs
                )
            )
            if aucs
            else -1.0
        )


        results.append(
            (
                C,
                mean_auc,
            )
        )


    results.sort(
        key=lambda z:
            (
                -z[1],
                z[0],
            )
    )


    return (
        results[0][0],
        results,
    )


all_results = {}


for rep_name, Xrep in (
    REPRESENTATIONS.items()
):

    for method in METHODS:

        label = (
            f"{method}_{rep_name}"
        )


        print()
        print(
            "=" * 120
        )

        print(
            label.upper()
        )

        print(
            "=" * 120
        )


        oof = np.full(
            len(rows),
            np.nan,
            dtype=float,
        )


        chosen_cs = []


        for held_role in ROLES:

            for fold, (
                trc,
                tec,
            ) in enumerate(
                outer_comp_cv,
                1,
            ):

                test_comps = set(
                    unique_comps[
                        tec
                    ].tolist()
                )


                base_train = np.asarray(
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
                        base_train
                    ]
                    == held_role
                )


                assert not (
                    set(
                        comp_arr[
                            base_train
                        ].tolist()
                    )
                    & test_comps
                )


                best_C, inner = (
                    inner_select_c(
                        Xrep,
                        base_train,
                        method,
                    )
                )


                chosen_cs.append(
                    {
                        "held_role":
                            held_role,

                        "fold":
                            fold,

                        "C":
                            best_C,

                        "inner":
                            inner,
                    }
                )


                if method == "pairwise":

                    scaler, clf, npairs = (
                        fit_pairwise_ranker(
                            Xrep,
                            base_train,
                            best_C,
                        )
                    )

                    scores = score_ranker(
                        scaler,
                        clf,
                        Xrep,
                        test_mask,
                    )

                else:

                    scaler, clf = (
                        fit_pointwise(
                            Xrep,
                            base_train,
                            best_C,
                        )
                    )

                    scores = (
                        clf.decision_function(
                            scaler.transform(
                                Xrep[
                                    test_mask
                                ]
                            )
                        )
                    )

                    npairs = None


                oof[
                    test_mask
                ] = scores


        assert np.isfinite(
            oof
        ).all()


        per_role = {}


        for role in ROLES:

            rm = (
                role_arr
                == role
            )

            per_role[
                role
            ] = safe_auc(
                y[
                    rm
                ],
                oof[
                    rm
                ],
            )


            print(
                f"{role:<12} "
                f"AUC="
                f"{per_role[role]:.4f}"
            )


        result = {
            "global_auc":
                safe_auc(
                    y,
                    oof,
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

            "chosen_cs":
                chosen_cs,

            "oof_scores":
                oof.tolist(),
        }


        all_results[
            label
        ] = result


        print()

        print(
            "GLOBAL AUC:",
            f"{result['global_auc']:.4f}",
        )

        print(
            "MEAN ROLE AUC:",
            f"{result['mean_role_auc']:.4f}",
        )

        print(
            "MEDIAN ROLE AUC:",
            f"{result['median_role_auc']:.4f}",
        )


# ============================================================
# DIRECT COUNTERFACTUAL PAIR ACCURACY
#
# For same role+query:
# how often does held-out model rank PRESENT > ABSENT?
# ============================================================

print()
print(
    "=" * 120
)

print(
    "COUNTERFACTUAL PAIRWISE ORDERING"
)

print(
    "=" * 120
)


for label, result in (
    all_results.items()
):

    scores = np.asarray(
        result[
            "oof_scores"
        ],
        dtype=float,
    )


    wins = 0

    total = 0


    for role in ROLES:

        for query in sorted(
            set(
                query_arr[
                    role_arr
                    == role
                ].tolist()
            )
        ):

            mask = (
                (
                    role_arr
                    == role
                )
                &
                (
                    query_arr
                    == query
                )
            )


            pos = scores[
                mask
                & (
                    y == 1
                )
            ]

            neg = scores[
                mask
                & (
                    y == 0
                )
            ]


            for p in pos:

                for n in neg:

                    total += 1

                    if p > n:

                        wins += 1


    acc = (
        wins
        / total
    )


    all_results[
        label
    ][
        "counterfactual_pair_accuracy"
    ] = float(
        acc
    )


    print(
        f"{label:<22} "
        f"{wins}/{total} "
        f"= {acc:.4f}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "=" * 120
)

print(
    "FINAL UNSEEN-ROLE COMPARISON"
)

print(
    "=" * 120
)


for label, r in (
    all_results.items()
):

    print(
        f"{label:<22} "
        f"global="
        f"{r['global_auc']:.4f} "
        f"meanRole="
        f"{r['mean_role_auc']:.4f} "
        f"medianRole="
        f"{r['median_role_auc']:.4f} "
        f"pairAcc="
        f"{r['counterfactual_pair_accuracy']:.4f}"
    )


# ============================================================
# SAVE
# ============================================================

save = {}


for label, r in (
    all_results.items()
):

    save[
        label
    ] = {
        k: v
        for k, v
        in r.items()
        if k
        != "oof_scores"
    }


(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        save,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


with (
    OUT
    / "oof_scores.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for i, row in enumerate(
        rows
    ):

        rec = {
            "system_id":
                row[
                    "system_id"
                ],

            "comp_id":
                row[
                    "comp_id"
                ],

            "role":
                row[
                    "role"
                ],

            "query":
                row[
                    "query"
                ],

            "y":
                row[
                    "y"
                ],
        }


        for label, result in (
            all_results.items()
        ):

            rec[
                label
            ] = float(
                result[
                    "oof_scores"
                ][
                    i
                ]
            )


        f.write(
            json.dumps(
                rec,
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
    / "oof_scores.jsonl"
)
