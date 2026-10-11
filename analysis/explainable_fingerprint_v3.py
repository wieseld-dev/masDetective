from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


SRC = Path("artifacts/v11_probe_discovery.jsonl")
OUT = Path("artifacts/explainable_fingerprint_v3")
OUT.mkdir(parents=True, exist_ok=True)

PROFILE_PATH = OUT / "concept_profiles.json"

PROFILE_MODEL = os.environ.get(
    "PROFILE_MODEL",
    "gpt-4o-mini",
)

PROFILE_WORKERS = int(
    os.environ.get(
        "PROFILE_WORKERS",
        "6",
    )
)

C = 0.05


from v11.role_universe import (
    TRAIN_ROLES,
    DEV_UNSEEN_ROLES,
    ROLE_DESCRIPTIONS,
)

from v8.embedding import FrozenOutputEmbedderV8


TRAIN_ROLES = tuple(map(str, TRAIN_ROLES))
DEV_ROLES = tuple(map(str, DEV_UNSEEN_ROLES))
ALL_ROLES = TRAIN_ROLES + DEV_ROLES

assert not (set(TRAIN_ROLES) & set(DEV_ROLES))


CONCEPT_KEYS = (
    "evidence_artifacts",
    "decision_criteria",
    "verification_behavior",
    "failure_modes",
    "constraints_dependencies",
    "reasoning_actions",
)


# =====================================================================
# 1. ROLE -> EXPLAINABLE CONCEPT PROFILE
#
# IMPORTANT:
# Generator sees ONLY public role name + public capability description.
# It never sees MAS responses, labels, routes, or evaluation results.
# =====================================================================

def load_profiles():
    if not PROFILE_PATH.exists():
        return {}

    return json.loads(
        PROFILE_PATH.read_text(
            encoding="utf-8"
        )
    )


def valid_profile(p):
    return (
        isinstance(p, dict)
        and all(
            isinstance(p.get(k), str)
            and p[k].strip()
            for k in CONCEPT_KEYS
        )
    )


def generate_profile(role):
    from openai import OpenAI

    client = OpenAI()

    desc = ROLE_DESCRIPTIONS.get(
        role,
        role,
    )

    system = """
You convert a PUBLIC specialist-role description into an interpretable
behavioral fingerprint specification.

The specification will later be used to inspect whether a response exhibits
behavior consistent with that specialist.

Use ONLY the supplied role name and capability description.
Do not assume any particular hidden system, agent architecture, training set,
probe, response, label, or evaluation result.

Return JSON with EXACTLY these six keys:
evidence_artifacts
decision_criteria
verification_behavior
failure_modes
constraints_dependencies
reasoning_actions

For each key write one concise, concrete phrase (roughly 6-20 words).
Describe distinctive professional behavior, not generic competence.
Avoid phrases such as "be accurate", "be helpful", or "analyze carefully".
Do not mention the role name merely to make the phrase role-specific.
""".strip()

    user = f"""
ROLE NAME:
{role}

PUBLIC CAPABILITY DESCRIPTION:
{desc}
""".strip()

    last = None

    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=PROFILE_MODEL,
                temperature=0.0,
                response_format={
                    "type": "json_object"
                },
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
            )

            text = (
                r.choices[0]
                .message
                .content
            )

            p = json.loads(text)

            # Keep exact stable schema.
            p = {
                k: str(p.get(k, "")).strip()
                for k in CONCEPT_KEYS
            }

            if not valid_profile(p):
                raise ValueError(
                    f"invalid concept profile: {p}"
                )

            return role, p

        except Exception as e:
            last = e

            if attempt < 3:
                time.sleep(
                    2 ** attempt
                )

    raise RuntimeError(
        f"profile generation failed for {role}: {last}"
    )


profiles = load_profiles()

missing_roles = [
    r
    for r in ALL_ROLES
    if not valid_profile(
        profiles.get(r)
    )
]


print("===== CONCEPT PROFILE CACHE =====")
print("cached:", len(ALL_ROLES) - len(missing_roles))
print("missing:", len(missing_roles))

if missing_roles:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY required for missing concept profiles"
        )

    with ThreadPoolExecutor(
        max_workers=PROFILE_WORKERS
    ) as ex:
        futures = {
            ex.submit(
                generate_profile,
                role,
            ): role
            for role in missing_roles
        }

        for i, fut in enumerate(
            as_completed(futures),
            1,
        ):
            role, profile = fut.result()
            profiles[role] = profile

            print(
                f"[profile {i}/{len(missing_roles)}] "
                f"{role}",
                flush=True,
            )

    PROFILE_PATH.write_text(
        json.dumps(
            profiles,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


assert all(
    valid_profile(
        profiles[r]
    )
    for r in ALL_ROLES
)


print()
print("===== CONCEPT PROFILES =====")

for role in ALL_ROLES:
    print(f"\n### {role}")
    print(
        "description:",
        ROLE_DESCRIPTIONS.get(
            role,
            role,
        ),
    )

    for k in CONCEPT_KEYS:
        print(
            f"  {k:<26}: "
            f"{profiles[role][k]}"
        )


# =====================================================================
# 2. LOAD MATCHED V11 COUNTERFACTUALS
# =====================================================================

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


for rec_idx, rec in enumerate(raw):
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

    if role not in set(ALL_ROLES):
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

    for branch_idx, b in enumerate(bb):
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

        seen.add(key)

        branches.append({
            "branch_id":
                f"{phase}:{rec_idx}:{branch_idx}",

            "split":
                (
                    "train"
                    if role in TRAIN_ROLES
                    else "dev_unseen"
                ),

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
    b
    for b in branches
    if b["split"] == "train"
]

dev_branches = [
    b
    for b in branches
    if b["split"] == "dev_unseen"
]


print()
print("===== COUNTERFACTUAL DATA =====")
print("train roles:", len(TRAIN_ROLES))
print("dev unseen roles:", len(DEV_ROLES))
print("train branches:", len(train_branches))
print("dev branches:", len(dev_branches))


# =====================================================================
# 3. FLATTEN TO OUTPUT SAMPLES
# =====================================================================

samples = []

for b in branches:
    for kind, y, field in (
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
                b[field],
        })


train_mask = np.asarray(
    [
        x["split"] == "train"
        for x in samples
    ],
    dtype=bool,
)

dev_mask = np.asarray(
    [
        x["split"] == "dev_unseen"
        for x in samples
    ],
    dtype=bool,
)

y = np.asarray(
    [
        int(x["y"])
        for x in samples
    ],
    dtype=int,
)


sample_lookup = {
    (
        x["branch_id"],
        x["kind"],
    ): i
    for i, x in enumerate(samples)
}


print(
    "train samples:",
    int(train_mask.sum()),
)

print(
    "dev samples:",
    int(dev_mask.sum()),
)


# =====================================================================
# 4. EMBEDDINGS
# =====================================================================

print()
print("loading embedding model...", flush=True)

embedder = FrozenOutputEmbedderV8(
    device="cpu"
)

emb_cache = {}


def emb(text):
    text = str(text)

    if text not in emb_cache:
        emb_cache[text] = (
            embedder
            .encode(text)
            .astype(np.float32)
        )

    return emb_cache[text]


def cosine(a, b):
    na = float(
        np.linalg.norm(a)
    )

    nb = float(
        np.linalg.norm(b)
    )

    if (
        na <= 1e-12
        or nb <= 1e-12
    ):
        return 0.0

    return float(
        np.dot(a, b)
        / (
            na
            * nb
        )
    )


# =====================================================================
# 5. 20-D GENERIC GEOMETRY
# =====================================================================

GEOM_NAMES = (
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
)


def geom_features(
    role,
    query,
    output,
):
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

    d_or = float(
        np.linalg.norm(or_)
    )

    d_oq = float(
        np.linalg.norm(oq)
    )

    d_qr = float(
        np.linalg.norm(
            q - r
        )
    )

    cos_or = cosine(o, r)
    cos_oq = cosine(o, q)
    cos_qr = cosine(q, r)

    return np.asarray(
        [
            cos_or,
            cos_oq,
            cos_qr,

            d_or,
            d_oq,
            d_qr,

            cos_or - cos_qr,
            cos_oq - cos_qr,

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


print()
print("building GEOM features...", flush=True)

X_geom = np.stack(
    [
        geom_features(
            x["role"],
            x["query"],
            x["output"],
        )
        for x in samples
    ]
)


# =====================================================================
# 6. EXPLAINABLE CONCEPT FEATURES
#
# For each concept:
#
#   output_alignment
#   query_alignment
#   added_alignment = output_alignment - query_alignment
#   response_shift_projection
#   response_shift_cosine
#
# The last three specifically ask:
# "what did the RESPONSE add beyond the QUERY?"
# =====================================================================

CONCEPT_CHANNELS = (
    "output_alignment",
    "query_alignment",
    "added_alignment",
    "response_shift_projection",
    "response_shift_cosine",
)


CONCEPT_FEATURE_NAMES = []

for ck in CONCEPT_KEYS:
    for channel in CONCEPT_CHANNELS:
        CONCEPT_FEATURE_NAMES.append(
            f"{ck}::{channel}"
        )


def concept_features(
    role,
    query,
    output,
):
    q = emb(query)
    o = emb(output)

    shift = o - q

    vals = []

    for ck in CONCEPT_KEYS:
        concept_text = profiles[
            role
        ][
            ck
        ]

        c = emb(concept_text)

        out_sim = cosine(
            o,
            c,
        )

        query_sim = cosine(
            q,
            c,
        )

        vals.extend(
            [
                out_sim,

                query_sim,

                out_sim
                - query_sim,

                float(
                    np.dot(
                        shift,
                        c,
                    )
                ),

                cosine(
                    shift,
                    c,
                ),
            ]
        )

    return np.asarray(
        vals,
        dtype=np.float32,
    )


print(
    "building CONCEPT features...",
    flush=True,
)

X_concept = np.stack(
    [
        concept_features(
            x["role"],
            x["query"],
            x["output"],
        )
        for x in samples
    ]
)


X_gc = np.concatenate(
    [
        X_geom,
        X_concept,
    ],
    axis=1,
)


print()
print(
    "GEOM dim:",
    X_geom.shape[1],
)

print(
    "CONCEPT dim:",
    X_concept.shape[1],
)

print(
    "GEOM+CONCEPT dim:",
    X_gc.shape[1],
)


# =====================================================================
# 7. TRAINING
# =====================================================================

def fit_pointwise(X):
    scaler = StandardScaler()

    Z = scaler.fit_transform(
        X[train_mask]
    )

    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        random_state=0,
    )

    clf.fit(
        Z,
        y[train_mask],
    )

    return scaler, clf


def make_pairwise_diffs(Z):
    diffs = []

    for b in train_branches:
        p = sample_lookup[
            (
                b["branch_id"],
                "plus",
            )
        ]

        s = sample_lookup[
            (
                b["branch_id"],
                "sham",
            )
        ]

        m = sample_lookup[
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

    return np.stack(
        diffs
    )


def fit_pairwise(X):
    scaler = StandardScaler()
    scaler.fit(
        X[train_mask]
    )

    Z = scaler.transform(X)

    Dpos = make_pairwise_diffs(Z)

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

    return scaler, clf


def score_model(
    scaler,
    clf,
    X,
):
    Z = scaler.transform(X)

    if clf.fit_intercept:
        return clf.decision_function(Z)

    return (
        Z
        @ clf.coef_[0]
    )


MODELS = {}


print()
print("training PAIRWISE_GEOM...", flush=True)

MODELS[
    "pairwise_geom"
] = (
    *fit_pairwise(
        X_geom
    ),
    X_geom,
)


print(
    "training PAIRWISE_CONCEPT...",
    flush=True,
)

MODELS[
    "pairwise_concept"
] = (
    *fit_pairwise(
        X_concept
    ),
    X_concept,
)


print(
    "training POINTWISE_GEOM_CONCEPT...",
    flush=True,
)

MODELS[
    "pointwise_geom_concept"
] = (
    *fit_pointwise(
        X_gc
    ),
    X_gc,
)


print(
    "training PAIRWISE_GEOM_CONCEPT...",
    flush=True,
)

MODELS[
    "pairwise_geom_concept"
] = (
    *fit_pairwise(
        X_gc
    ),
    X_gc,
)


# =====================================================================
# 8. METRICS
# =====================================================================

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


def evaluate_scores(score):
    per_role_auc = {}
    per_role_pair = {}
    per_role_strict = {}

    for role in DEV_ROLES:
        rm = np.asarray(
            [
                (
                    x["split"]
                    == "dev_unseen"
                    and
                    x["role"]
                    == role
                )
                for x in samples
            ],
            dtype=bool,
        )

        per_role_auc[
            role
        ] = safe_auc(
            y[rm],
            score[rm],
        )

        rr = [
            b
            for b in dev_branches
            if b["role"] == role
        ]

        wins = 0
        total = 0
        strict = 0

        for b in rr:
            p = sample_lookup[
                (
                    b["branch_id"],
                    "plus",
                )
            ]

            s = sample_lookup[
                (
                    b["branch_id"],
                    "sham",
                )
            ]

            m = sample_lookup[
                (
                    b["branch_id"],
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

            wins += wp + ws
            total += 2

            strict += int(
                wp
                and ws
            )

        per_role_pair[
            role
        ] = wins / total

        per_role_strict[
            role
        ] = strict / len(rr)

    return {
        "global_dev_auc":
            safe_auc(
                y[dev_mask],
                score[dev_mask],
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
                    list(
                        per_role_pair.values()
                    )
                )
            ),

        "mean_role_strict_branch_win":
            float(
                np.mean(
                    list(
                        per_role_strict.values()
                    )
                )
            ),

        "per_role_auc":
            per_role_auc,

        "per_role_pair_accuracy":
            per_role_pair,

        "per_role_strict_branch_win":
            per_role_strict,
    }


results = {}
scores_by_model = {}


for name, (
    scaler,
    clf,
    X,
) in MODELS.items():
    score = score_model(
        scaler,
        clf,
        X,
    )

    scores_by_model[
        name
    ] = score

    result = evaluate_scores(
        score
    )

    results[
        name
    ] = result

    print()
    print("=" * 115)
    print(name.upper())
    print("=" * 115)

    for role in DEV_ROLES:
        print(
            f"{role:<28} "
            f"AUC="
            f"{result['per_role_auc'][role]:.4f} "
            f"pairAcc="
            f"{result['per_role_pair_accuracy'][role]:.4f} "
            f"strict="
            f"{result['per_role_strict_branch_win'][role]:.4f}"
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


# =====================================================================
# 9. GROUNDED EXPLANATIONS FOR PAIRWISE GEOM+CONCEPT
#
# Explain POSITIVE-vs-MINUS MARGIN using actual linear contributions.
# No free-form post-hoc LLM explanation is used.
# =====================================================================

scaler, clf, _ = MODELS[
    "pairwise_geom_concept"
]

Zgc = scaler.transform(
    X_gc
)

coef = clf.coef_[0]


concept_start = X_geom.shape[1]

category_indices = {}

for ci, ck in enumerate(
    CONCEPT_KEYS
):
    start = (
        concept_start
        + ci
        * len(
            CONCEPT_CHANNELS
        )
    )

    category_indices[
        ck
    ] = np.arange(
        start,
        start
        + len(
            CONCEPT_CHANNELS
        ),
    )


explanations = []


for b in dev_branches:
    p = sample_lookup[
        (
            b["branch_id"],
            "plus",
        )
    ]

    s = sample_lookup[
        (
            b["branch_id"],
            "sham",
        )
    ]

    m = sample_lookup[
        (
            b["branch_id"],
            "minus",
        )
    ]

    # Average the two role-present branches.
    zpos = (
        Zgc[p]
        + Zgc[s]
    ) / 2.0

    delta_contrib = (
        zpos
        - Zgc[m]
    ) * coef

    geom_contribution = float(
        delta_contrib[
            :concept_start
        ].sum()
    )

    concept_contrib = {
        ck:
            float(
                delta_contrib[
                    idx
                ].sum()
            )
        for ck, idx
        in category_indices.items()
    }

    concept_rank = sorted(
        concept_contrib.items(),
        key=lambda z:
            -z[1],
    )

    total_margin = float(
        delta_contrib.sum()
    )

    explanations.append({
        "branch_id":
            b["branch_id"],

        "role":
            b["role"],

        "query":
            b["query"],

        "total_pairwise_margin":
            total_margin,

        "geom_contribution":
            geom_contribution,

        "concept_contributions":
            concept_contrib,

        "top_positive_concepts":
            [
                {
                    "category":
                        ck,

                    "concept":
                        profiles[
                            b["role"]
                        ][ck],

                    "contribution":
                        float(v),
                }
                for ck, v
                in concept_rank[:3]
            ],

        "top_negative_concepts":
            [
                {
                    "category":
                        ck,

                    "concept":
                        profiles[
                            b["role"]
                        ][ck],

                    "contribution":
                        float(v),
                }
                for ck, v
                in sorted(
                    concept_contrib.items(),
                    key=lambda z:
                        z[1],
                )[:3]
            ],
    })


# Aggregate explanation by unseen role.
print()
print("=" * 115)
print(
    "GROUNDED CONCEPT FINGERPRINT EXPLANATIONS"
)
print("=" * 115)


role_explanation_summary = {}


for role in DEV_ROLES:
    rr = [
        x
        for x in explanations
        if x["role"] == role
    ]

    mean_contrib = {
        ck:
            float(
                np.mean(
                    [
                        x[
                            "concept_contributions"
                        ][ck]
                        for x in rr
                    ]
                )
            )
        for ck in CONCEPT_KEYS
    }

    ranked = sorted(
        mean_contrib.items(),
        key=lambda z:
            -z[1],
    )

    role_explanation_summary[
        role
    ] = mean_contrib

    print(f"\n### {role}")

    for ck, v in ranked:
        print(
            f"{ck:<26} "
            f"mean_margin_contribution="
            f"{v:+.4f}"
        )

        print(
            "   concept:",
            profiles[
                role
            ][ck],
        )


# =====================================================================
# 10. FINAL COMPARISON
# =====================================================================

print()
print("=" * 115)
print(
    "FINAL EXPLAINABLE FINGERPRINT V3 COMPARISON"
)
print("=" * 115)


for name, r in results.items():
    print(
        f"{name:<28} "
        f"globalAUC="
        f"{r['global_dev_auc']:.4f} "
        f"meanRoleAUC="
        f"{r['mean_role_auc']:.4f} "
        f"medianRoleAUC="
        f"{r['median_role_auc']:.4f} "
        f"pairAcc="
        f"{r['mean_role_pair_accuracy']:.4f} "
        f"strict="
        f"{r['mean_role_strict_branch_win']:.4f}"
    )


summary = {
    "experiment":
        "explainable_fingerprint_v3",

    "development_only":
        True,

    "concept_profile_input":
        "public role name + public role description only",

    "concept_profile_model":
        PROFILE_MODEL,

    "C":
        C,

    "train_roles":
        list(TRAIN_ROLES),

    "dev_unseen_roles":
        list(DEV_ROLES),

    "train_branches":
        len(train_branches),

    "dev_branches":
        len(dev_branches),

    "concept_keys":
        list(CONCEPT_KEYS),

    "results":
        results,

    "role_explanation_summary":
        role_explanation_summary,
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
    / "dev_explanations.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:
    for x in explanations:
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
    OUT / "summary.json",
)

print(
    "wrote:",
    PROFILE_PATH,
)

print(
    "wrote:",
    OUT / "dev_explanations.jsonl",
)
