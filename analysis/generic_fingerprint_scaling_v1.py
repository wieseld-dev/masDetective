from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
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
OUT = Path("artifacts/generic_fingerprint_scaling_v1")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_ROLES = tuple(map(str, TRAIN_ROLES))
DEV_ROLES = tuple(map(str, DEV_UNSEEN_ROLES))

C = 0.05

ROLE_COUNTS = (4, 8, 12, 16, 20)
REPEATS = 30

FIXED_PAIR_BUDGET = 100
BASE_SEED = 20260927


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


# ============================================================
# DATA
# ============================================================

raw = read_jsonl(SRC)

branches = []
seen = set()

allowed = set(TRAIN_ROLES) | set(DEV_ROLES)

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


print("===== DATA =====")
print("train roles:", len(TRAIN_ROLES))
print("dev roles:", len(DEV_ROLES))
print(
    "train branches:",
    sum(
        b["split"] == "train"
        for b in branches
    ),
)
print(
    "dev branches:",
    sum(
        b["split"] == "dev"
        for b in branches
    ),
)


# ============================================================
# 20-D RELATIONAL GEOMETRY
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


def cosine(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na <= 1e-12 or nb <= 1e-12:
        return 0.0

    return float(
        np.dot(a, b)
        / (na * nb)
    )


def geom(role, query, output):
    r = emb(
        ROLE_DESCRIPTIONS.get(
            role,
            role,
        )
    )

    q = emb(query)
    o = emb(output)

    oq = o - q
    or_ = o - r
    rq = r - q
    qr = q - r

    d_or = float(np.linalg.norm(or_))
    d_oq = float(np.linalg.norm(oq))
    d_qr = float(np.linalg.norm(q - r))

    cor = cosine(o, r)
    coq = cosine(o, q)
    cqr = cosine(q, r)

    return np.asarray(
        [
            cor,
            coq,
            cqr,

            d_or,
            d_oq,
            d_qr,

            cor - cqr,
            coq - cqr,

            cosine(oq, r),
            cosine(oq, rq),

            cosine(or_, q),
            cosine(or_, qr),

            d_oq,
            d_or,
            d_qr,

            float(np.dot(oq, r)),
            float(np.dot(oq, rq)),

            d_oq + d_qr - d_or,
            d_or + d_qr - d_oq,

            d_oq / max(d_qr, 1e-6),
        ],
        dtype=np.float32,
    )


print("building geometry...", flush=True)

X = np.stack(
    [
        geom(
            x["role"],
            x["query"],
            x["output"],
        )
        for x in samples
    ]
)

print("X:", X.shape)


# ============================================================
# TRAIN PAIR POOLS
# ============================================================

pairs_by_role = defaultdict(list)

dev_branches = []

for b in branches:
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

    if b["split"] == "train":
        pairs_by_role[
            b["role"]
        ].append(
            (p, m)
        )

        pairs_by_role[
            b["role"]
        ].append(
            (s, m)
        )

    else:
        dev_branches.append(
            (
                b["role"],
                p,
                s,
                m,
            )
        )


print()
print("===== TRAIN PAIRS PER ROLE =====")

for role in TRAIN_ROLES:
    print(
        f"{role:<28} "
        f"{len(pairs_by_role[role])}"
    )


# ============================================================
# MODEL
# ============================================================

def fit_pairwise(pair_indices):
    used = sorted({
        i
        for pair in pair_indices
        for i in pair
    })

    scaler = StandardScaler()

    scaler.fit(
        X[used]
    )

    Z = scaler.transform(X)

    Dpos = np.stack(
        [
            Z[p] - Z[m]
            for p, m
            in pair_indices
        ]
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


def safe_auc(yy, ss):
    if len(np.unique(yy)) != 2:
        return np.nan

    return float(
        roc_auc_score(
            yy,
            ss,
        )
    )


def evaluate(score):
    per_auc = {}
    per_pair = {}
    per_strict = {}

    for role in DEV_ROLES:
        idx = np.asarray(
            [
                (
                    x["split"] == "dev"
                    and
                    x["role"] == role
                )
                for x in samples
            ],
            dtype=bool,
        )

        per_auc[
            role
        ] = safe_auc(
            y[idx],
            score[idx],
        )

        rr = [
            x
            for x in dev_branches
            if x[0] == role
        ]

        wins = 0
        total = 0
        strict = 0

        for _, p, s, m in rr:
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

        per_pair[
            role
        ] = wins / total

        per_strict[
            role
        ] = strict / len(rr)

    return {
        "mean_role_auc":
            float(
                np.mean(
                    list(
                        per_auc.values()
                    )
                )
            ),

        "median_role_auc":
            float(
                np.median(
                    list(
                        per_auc.values()
                    )
                )
            ),

        "mean_pair_acc":
            float(
                np.mean(
                    list(
                        per_pair.values()
                    )
                )
            ),

        "mean_strict":
            float(
                np.mean(
                    list(
                        per_strict.values()
                    )
                )
            ),
    }


# ============================================================
# FIXED-BUDGET ROLE-DIVERSITY CURVE
#
# Exactly 100 positive training comparisons at every K.
# Therefore label budget is constant; only number/diversity
# of training role types changes.
# ============================================================

def balanced_sample_pairs(
    selected_roles,
    rng,
    budget,
):
    roles = list(
        selected_roles
    )

    roles = list(
        rng.permutation(
            roles
        )
    )

    base = budget // len(roles)
    rem = budget % len(roles)

    out = []

    for i, role in enumerate(roles):
        n = base + int(
            i < rem
        )

        pool = pairs_by_role[
            role
        ]

        if n > len(pool):
            raise RuntimeError(
                f"{role}: need {n}, have {len(pool)}"
            )

        take = rng.choice(
            len(pool),
            size=n,
            replace=False,
        )

        out.extend(
            pool[j]
            for j in take
        )

    assert len(out) == budget

    return out


all_results = {
    "fixed_budget_100": [],
    "all_available_data": [],
}


print()
print("=" * 120)
print("FIXED BUDGET = 100 POSITIVE PAIRWISE COMPARISONS")
print("=" * 120)


for k in ROLE_COUNTS:
    runs = []

    for rep in range(REPEATS):
        rng = np.random.default_rng(
            BASE_SEED
            + 1000 * k
            + rep
        )

        selected = tuple(
            rng.choice(
                TRAIN_ROLES,
                size=k,
                replace=False,
            )
        )

        pairs = balanced_sample_pairs(
            selected,
            rng,
            FIXED_PAIR_BUDGET,
        )

        score = fit_pairwise(
            pairs
        )

        metric = evaluate(
            score
        )

        runs.append(
            metric
        )

        all_results[
            "fixed_budget_100"
        ].append({
            "k_roles":
                k,

            "repeat":
                rep,

            "selected_roles":
                list(selected),

            "pair_budget":
                len(pairs),

            **metric,
        })

    print(
        f"K={k:<2} "
        f"AUC="
        f"{np.mean([r['mean_role_auc'] for r in runs]):.4f}"
        f" ± "
        f"{np.std([r['mean_role_auc'] for r in runs]):.4f}   "
        f"pairAcc="
        f"{np.mean([r['mean_pair_acc'] for r in runs]):.4f}"
        f" ± "
        f"{np.std([r['mean_pair_acc'] for r in runs]):.4f}   "
        f"strict="
        f"{np.mean([r['mean_strict'] for r in runs]):.4f}"
    )


# ============================================================
# NATURAL CURVE: USE ALL AVAILABLE PAIRS
# ============================================================

print()
print("=" * 120)
print("ALL AVAILABLE TRAINING PAIRS")
print("=" * 120)


for k in ROLE_COUNTS:
    runs = []

    nreps = (
        1
        if k == 20
        else REPEATS
    )

    for rep in range(nreps):
        rng = np.random.default_rng(
            BASE_SEED
            + 50000
            + 1000 * k
            + rep
        )

        if k == 20:
            selected = tuple(
                TRAIN_ROLES
            )
        else:
            selected = tuple(
                rng.choice(
                    TRAIN_ROLES,
                    size=k,
                    replace=False,
                )
            )

        pairs = [
            pair
            for role in selected
            for pair in pairs_by_role[
                role
            ]
        ]

        score = fit_pairwise(
            pairs
        )

        metric = evaluate(
            score
        )

        runs.append(
            metric
        )

        all_results[
            "all_available_data"
        ].append({
            "k_roles":
                k,

            "repeat":
                rep,

            "selected_roles":
                list(selected),

            "pair_budget":
                len(pairs),

            **metric,
        })

    print(
        f"K={k:<2} "
        f"pairs="
        f"{np.mean([x['pair_budget'] for x in all_results['all_available_data'] if x['k_roles'] == k]):.1f}   "
        f"AUC="
        f"{np.mean([r['mean_role_auc'] for r in runs]):.4f}"
        f" ± "
        f"{np.std([r['mean_role_auc'] for r in runs]):.4f}   "
        f"pairAcc="
        f"{np.mean([r['mean_pair_acc'] for r in runs]):.4f}"
        f" ± "
        f"{np.std([r['mean_pair_acc'] for r in runs]):.4f}   "
        f"strict="
        f"{np.mean([r['mean_strict'] for r in runs]):.4f}"
    )


# ============================================================
# COMPACT SUMMARY
# ============================================================

summary = {
    "experiment":
        "generic_fingerprint_scaling_v1",

    "representation":
        "20-dimensional coordinate-free relational geometry",

    "objective":
        "pairwise counterfactual ranking",

    "C":
        C,

    "dev_roles":
        list(DEV_ROLES),

    "role_counts":
        list(ROLE_COUNTS),

    "repeats":
        REPEATS,

    "fixed_pair_budget":
        FIXED_PAIR_BUDGET,

    "runs":
        all_results,
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
