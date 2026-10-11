from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from v11.role_universe import (
    TRAIN_ROLES,
    DEV_UNSEEN_ROLES,
    ROLE_DESCRIPTIONS,
)

from v8.embedding import FrozenOutputEmbedderV8


SRC = Path("artifacts/v11_probe_discovery.jsonl")
OUT = Path("artifacts/lowrank_fingerprint_v4")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_ROLES = tuple(map(str, TRAIN_ROLES))
DEV_ROLES = tuple(map(str, DEV_UNSEEN_ROLES))
ALL_ROLES = TRAIN_ROLES + DEV_ROLES

C = 0.05
PCA_DIMS = (4, 8, 16, 32)


# ============================================================
# DATA
# ============================================================

def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


raw = read_jsonl(SRC)

branches = []
seen = set()

allowed = set(ALL_ROLES)

for rec_i, rec in enumerate(raw):
    role = str(rec.get("target_role", ""))
    phase = str(rec.get("phase", ""))
    query = str(rec.get("query", "")).strip()

    if role not in allowed:
        continue

    if phase not in {"train", "holdout"}:
        continue

    if not query:
        continue

    bb = list(
        rec.get("branches_training_only")
        or []
    )

    bb += list(
        rec.get(
            "confirmation_branches_training_only"
        )
        or []
    )

    for branch_i, b in enumerate(bb):
        plus = str(
            b.get("plus_output", "")
            or ""
        ).strip()

        sham = str(
            b.get("sham_output", "")
            or ""
        ).strip()

        minus = str(
            b.get("minus_output", "")
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

        seen.add(key)

        branches.append({
            "branch_id":
                f"{phase}:{rec_i}:{branch_i}",

            "split":
                (
                    "train"
                    if role in TRAIN_ROLES
                    else "dev"
                ),

            "role":
                role,

            "query":
                query,

            "plus":
                plus,

            "sham":
                sham,

            "minus":
                minus,
        })


samples = []

for b in branches:
    for kind, y in (
        ("plus", 1),
        ("sham", 1),
        ("minus", 0),
    ):
        samples.append({
            "branch_id":
                b["branch_id"],

            "split":
                b["split"],

            "role":
                b["role"],

            "query":
                b["query"],

            "kind":
                kind,

            "y":
                y,

            "output":
                b[kind],
        })


lookup = {
    (
        x["branch_id"],
        x["kind"],
    ): i
    for i, x in enumerate(samples)
}


y = np.asarray(
    [
        x["y"]
        for x in samples
    ],
    dtype=int,
)


train_branches = [
    b
    for b in branches
    if b["split"] == "train"
]

dev_branches = [
    b
    for b in branches
    if b["split"] == "dev"
]


print("===== DATA =====")
print("train roles:", len(TRAIN_ROLES))
print("dev unseen roles:", len(DEV_ROLES))
print("train branches:", len(train_branches))
print("dev branches:", len(dev_branches))
print("samples:", len(samples))


# ============================================================
# EMBEDDINGS
# ============================================================

print()
print("loading embedder...", flush=True)

embedder = FrozenOutputEmbedderV8(
    device="cpu"
)

cache = {}


def emb(text):
    text = str(text)

    if text not in cache:
        cache[text] = (
            embedder
            .encode(text)
            .astype(np.float32)
        )

    return cache[text]


print("building raw role/query/output embeddings...", flush=True)

R = np.stack(
    [
        emb(
            ROLE_DESCRIPTIONS.get(
                x["role"],
                x["role"],
            )
        )
        for x in samples
    ]
)

Q = np.stack(
    [
        emb(
            x["query"]
        )
        for x in samples
    ]
)

O = np.stack(
    [
        emb(
            x["output"]
        )
        for x in samples
    ]
)


print(
    "embedding dim:",
    R.shape[1],
)


# ============================================================
# 20-D GEOMETRY BASELINE
# ============================================================

def row_norm(a):
    return np.sqrt(
        np.sum(
            a * a,
            axis=1,
        )
    )


def row_cos(a, b):
    na = row_norm(a)
    nb = row_norm(b)

    den = np.maximum(
        na * nb,
        1e-12,
    )

    return np.sum(
        a * b,
        axis=1,
    ) / den


def geom_features(RR, QQ, OO):
    oq = OO - QQ
    or_ = OO - RR
    rq = RR - QQ
    qr = QQ - RR

    d_or = row_norm(or_)
    d_oq = row_norm(oq)
    d_qr = row_norm(QQ - RR)

    cor = row_cos(OO, RR)
    coq = row_cos(OO, QQ)
    cqr = row_cos(QQ, RR)

    return np.column_stack(
        [
            cor,
            coq,
            cqr,

            d_or,
            d_oq,
            d_qr,

            cor - cqr,
            coq - cqr,

            row_cos(
                oq,
                RR,
            ),

            row_cos(
                oq,
                rq,
            ),

            row_cos(
                or_,
                QQ,
            ),

            row_cos(
                or_,
                qr,
            ),

            d_oq,
            d_or,
            d_qr,

            np.sum(
                oq * RR,
                axis=1,
            ),

            np.sum(
                oq * rq,
                axis=1,
            ),

            d_oq
            + d_qr
            - d_or,

            d_or
            + d_qr
            - d_oq,

            d_oq
            / np.maximum(
                d_qr,
                1e-6,
            ),
        ]
    ).astype(np.float32)


X_GEOM = geom_features(
    R,
    Q,
    O,
)

print(
    "GEOM dim:",
    X_GEOM.shape[1],
)


# ============================================================
# LOW-RANK RELATIONAL REPRESENTATION
#
# PCA is fitted ONLY on allowed training-role text.
#
# Features are not raw projected output coordinates.
# They describe compatibility between:
#
#   response shift = P(output) - P(query)
#
# and
#
#   role / role-minus-query
#
# ============================================================

def fit_pca_for_roles(
    allowed_roles,
):
    mask = np.asarray(
        [
            (
                x["split"] == "train"
                and
                x["role"] in allowed_roles
            )
            for x in samples
        ],
        dtype=bool,
    )

    # Training-only semantic manifold.
    corpus = np.vstack(
        [
            R[mask],
            Q[mask],
            O[mask],
        ]
    )

    pca = PCA(
        n_components=max(PCA_DIMS),
        svd_solver="randomized",
        random_state=0,
    )

    pca.fit(
        corpus
    )

    return pca


def lowrank_features(
    pca,
    dim,
):
    RP = pca.transform(R)[:, :dim]
    QP = pca.transform(Q)[:, :dim]
    OP = pca.transform(O)[:, :dim]

    shift = OP - QP
    role_from_query = RP - QP

    # Every block is relational.
    return np.concatenate(
        [
            shift * RP,
            shift * role_from_query,

            np.abs(
                shift - RP
            ),

            np.abs(
                shift - role_from_query
            ),
        ],
        axis=1,
    ).astype(np.float32)


# ============================================================
# PAIRWISE RANKER
# ============================================================

def fit_pairwise(
    X,
    allowed_roles,
):
    sample_mask = np.asarray(
        [
            (
                x["split"] == "train"
                and
                x["role"] in allowed_roles
            )
            for x in samples
        ],
        dtype=bool,
    )

    scaler = StandardScaler()

    scaler.fit(
        X[sample_mask]
    )

    Z = scaler.transform(X)

    diffs = []

    for b in train_branches:
        if b["role"] not in allowed_roles:
            continue

        p = lookup[
            (
                b["branch_id"],
                "plus",
            )
        ]

        s = lookup[
            (
                b["branch_id"],
                "sham",
            )
        ]

        m = lookup[
            (
                b["branch_id"],
                "minus",
            )
        ]

        diffs.append(
            Z[p] - Z[m]
        )

        diffs.append(
            Z[s] - Z[m]
        )

    Dpos = np.stack(
        diffs
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

    score = (
        Z
        @ clf.coef_[0]
    )

    return score


# ============================================================
# METRICS
# ============================================================

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
        return np.nan

    return float(
        roc_auc_score(
            yy,
            ss,
        )
    )


def evaluate_role(
    score,
    role,
    split,
):
    idx = np.asarray(
        [
            (
                x["split"] == split
                and
                x["role"] == role
            )
            for x in samples
        ],
        dtype=bool,
    )

    auc = safe_auc(
        y[idx],
        score[idx],
    )

    bb = (
        train_branches
        if split == "train"
        else dev_branches
    )

    rr = [
        b
        for b in bb
        if b["role"] == role
    ]

    wins = 0
    total = 0
    strict = 0

    for b in rr:
        p = lookup[
            (
                b["branch_id"],
                "plus",
            )
        ]

        s = lookup[
            (
                b["branch_id"],
                "sham",
            )
        ]

        m = lookup[
            (
                b["branch_id"],
                "minus",
            )
        ]

        wp = int(
            score[p] > score[m]
        )

        ws = int(
            score[s] > score[m]
        )

        wins += wp + ws
        total += 2

        strict += int(
            wp and ws
        )

    return {
        "auc":
            float(auc),

        "pair_acc":
            wins / total,

        "strict":
            strict / len(rr),
    }


# ============================================================
# LEAVE-ONE-TRAIN-ROLE-OUT MODEL SELECTION
#
# DEV roles are NOT used here.
# ============================================================

candidate_names = [
    "geom",
]

for d in PCA_DIMS:
    candidate_names.append(
        f"pca_rel_{d}"
    )

    candidate_names.append(
        f"geom_plus_pca_rel_{d}"
    )


cv = {
    name: []
    for name in candidate_names
}


print()
print("=" * 120)
print(
    "TRAIN-ROLE LEAVE-ONE-ROLE-OUT MODEL SELECTION"
)
print(
    "DEV ROLES ARE NOT USED"
)
print("=" * 120)


for fold_i, held_role in enumerate(
    TRAIN_ROLES,
    1,
):
    allowed_roles = tuple(
        r
        for r in TRAIN_ROLES
        if r != held_role
    )

    print(
        f"[fold {fold_i:02d}/20] "
        f"held_role={held_role}",
        flush=True,
    )

    # Baseline GEOM.
    score = fit_pairwise(
        X_GEOM,
        allowed_roles,
    )

    metric = evaluate_role(
        score,
        held_role,
        "train",
    )

    cv["geom"].append(
        metric
    )

    # PCA fit never sees held role.
    pca = fit_pca_for_roles(
        set(
            allowed_roles
        )
    )

    for d in PCA_DIMS:
        X_PCA = lowrank_features(
            pca,
            d,
        )

        score = fit_pairwise(
            X_PCA,
            allowed_roles,
        )

        metric = evaluate_role(
            score,
            held_role,
            "train",
        )

        cv[
            f"pca_rel_{d}"
        ].append(
            metric
        )

        X_COMBO = np.concatenate(
            [
                X_GEOM,
                X_PCA,
            ],
            axis=1,
        )

        score = fit_pairwise(
            X_COMBO,
            allowed_roles,
        )

        metric = evaluate_role(
            score,
            held_role,
            "train",
        )

        cv[
            f"geom_plus_pca_rel_{d}"
        ].append(
            metric
        )


print()
print("=" * 120)
print(
    "TRAIN-ONLY CV SUMMARY"
)
print("=" * 120)


cv_summary = {}


for name in candidate_names:
    rr = cv[name]

    aucs = [
        x["auc"]
        for x in rr
    ]

    pairs = [
        x["pair_acc"]
        for x in rr
    ]

    strict = [
        x["strict"]
        for x in rr
    ]

    cv_summary[name] = {
        "mean_auc":
            float(
                np.mean(
                    aucs
                )
            ),

        "median_auc":
            float(
                np.median(
                    aucs
                )
            ),

        "mean_pair_acc":
            float(
                np.mean(
                    pairs
                )
            ),

        "mean_strict":
            float(
                np.mean(
                    strict
                )
            ),
    }

    print(
        f"{name:<28} "
        f"AUC="
        f"{np.mean(aucs):.4f} "
        f"median="
        f"{np.median(aucs):.4f} "
        f"pairAcc="
        f"{np.mean(pairs):.4f} "
        f"strict="
        f"{np.mean(strict):.4f}"
    )


# Primary selection criterion:
# unseen-role mean AUC on TRAIN-role CV.
# PairAcc is tie-breaker.

selected_name = max(
    candidate_names,
    key=lambda name: (
        cv_summary[
            name
        ][
            "mean_auc"
        ],

        cv_summary[
            name
        ][
            "mean_pair_acc"
        ],
    ),
)


print()
print(
    "SELECTED USING TRAIN-ROLE CV ONLY:",
    selected_name,
)


# ============================================================
# FINAL FIT ON ALL 20 TRAIN ROLES
# THEN ONE DEV EVALUATION
# ============================================================

all_train = set(
    TRAIN_ROLES
)

pca_final = fit_pca_for_roles(
    all_train
)


def make_final_feature(
    name,
):
    if name == "geom":
        return X_GEOM

    if name.startswith(
        "pca_rel_"
    ):
        d = int(
            name.rsplit(
                "_",
                1,
            )[1]
        )

        return lowrank_features(
            pca_final,
            d,
        )

    if name.startswith(
        "geom_plus_pca_rel_"
    ):
        d = int(
            name.rsplit(
                "_",
                1,
            )[1]
        )

        XP = lowrank_features(
            pca_final,
            d,
        )

        return np.concatenate(
            [
                X_GEOM,
                XP,
            ],
            axis=1,
        )

    raise KeyError(name)


def evaluate_dev(
    score,
):
    per_role = {}

    for role in DEV_ROLES:
        per_role[
            role
        ] = evaluate_role(
            score,
            role,
            "dev",
        )

    return {
        "mean_role_auc":
            float(
                np.mean(
                    [
                        v["auc"]
                        for v in per_role.values()
                    ]
                )
            ),

        "median_role_auc":
            float(
                np.median(
                    [
                        v["auc"]
                        for v in per_role.values()
                    ]
                )
            ),

        "mean_pair_acc":
            float(
                np.mean(
                    [
                        v["pair_acc"]
                        for v in per_role.values()
                    ]
                )
            ),

        "mean_strict":
            float(
                np.mean(
                    [
                        v["strict"]
                        for v in per_role.values()
                    ]
                )
            ),

        "per_role":
            per_role,
    }


# Existing baseline.
baseline_score = fit_pairwise(
    X_GEOM,
    all_train,
)

baseline_dev = evaluate_dev(
    baseline_score
)


# TRAIN-CV-selected low-rank model.
X_SELECTED = make_final_feature(
    selected_name
)

selected_score = fit_pairwise(
    X_SELECTED,
    all_train,
)

selected_dev = evaluate_dev(
    selected_score
)


print()
print("=" * 120)
print(
    "FINAL 20-TRAIN -> 6-UNSEEN DEV"
)
print(
    "MODEL WAS SELECTED WITHOUT DEV"
)
print("=" * 120)


print()
print("BASELINE: GEOM")

for role in DEV_ROLES:
    m = baseline_dev[
        "per_role"
    ][role]

    print(
        f"{role:<28} "
        f"AUC={m['auc']:.4f} "
        f"pairAcc={m['pair_acc']:.4f} "
        f"strict={m['strict']:.4f}"
    )

print(
    f"MEAN AUC={baseline_dev['mean_role_auc']:.4f} "
    f"PAIR={baseline_dev['mean_pair_acc']:.4f} "
    f"STRICT={baseline_dev['mean_strict']:.4f}"
)


print()
print(
    f"SELECTED: {selected_name}"
)

for role in DEV_ROLES:
    m = selected_dev[
        "per_role"
    ][role]

    print(
        f"{role:<28} "
        f"AUC={m['auc']:.4f} "
        f"pairAcc={m['pair_acc']:.4f} "
        f"strict={m['strict']:.4f}"
    )

print(
    f"MEAN AUC={selected_dev['mean_role_auc']:.4f} "
    f"PAIR={selected_dev['mean_pair_acc']:.4f} "
    f"STRICT={selected_dev['mean_strict']:.4f}"
)


print()
print("=" * 120)
print(
    "FINAL LOW-RANK FINGERPRINT V4 COMPARISON"
)
print("=" * 120)

print(
    f"pairwise_geom              "
    f"AUC={baseline_dev['mean_role_auc']:.4f} "
    f"pairAcc={baseline_dev['mean_pair_acc']:.4f} "
    f"strict={baseline_dev['mean_strict']:.4f}"
)

print(
    f"{selected_name:<26} "
    f"AUC={selected_dev['mean_role_auc']:.4f} "
    f"pairAcc={selected_dev['mean_pair_acc']:.4f} "
    f"strict={selected_dev['mean_strict']:.4f}"
)


summary = {
    "experiment":
        "lowrank_fingerprint_v4",

    "selection_policy":
        (
            "leave-one-training-role-out only; "
            "DEV roles excluded from representation/model selection"
        ),

    "C":
        C,

    "pca_dims":
        list(PCA_DIMS),

    "selected_model":
        selected_name,

    "train_cv":
        cv_summary,

    "baseline_dev":
        baseline_dev,

    "selected_dev":
        selected_dev,
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
    OUT / "summary.json"
)
