#!/usr/bin/env python3

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score


EXP = Path("artifacts/eqs_v3_2_closed_loop/pilot")
MANIFEST = Path("artifacts/v14_adaptive_role_v1/manifest_privileged.json")
V31 = Path("scripts/v3_1_mas_auc_semantic.py")
OUT = Path("artifacts/eqs_v3_3_forensic")

POLICIES = [
    "random",
    "query_only",
    "history_belief_query",
]

ROUNDS = range(1, 7)


def read_jsonl(path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted(
        set().union(
            *(r.keys() for r in rows)
        )
    )
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    xs = [
        float(x)
        for x in xs
        if x is not None
        and np.isfinite(float(x))
    ]
    return float(np.mean(xs)) if xs else None


def corr(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None
        and y is not None
        and np.isfinite(float(x))
        and np.isfinite(float(y))
    ]

    if len(pairs) < 3:
        return None

    a = np.asarray([x for x, _ in pairs])
    b = np.asarray([y for _, y in pairs])

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None

    return float(np.corrcoef(a, b)[0, 1])


def binary_entropy(probs):
    vals = []
    for p in probs:
        p = min(
            max(float(p), 1e-8),
            1 - 1e-8,
        )
        vals.append(
            -(
                p * math.log2(p)
                + (1 - p) * math.log2(1 - p)
            )
        )
    return mean(vals)


def load_v31():
    spec = importlib.util.spec_from_file_location(
        "v31_forensic",
        V31,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import V3.1")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_torch(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def find_tensor_dict(obj):
    if isinstance(obj, dict):
        tensors = {
            k: v
            for k, v in obj.items()
            if isinstance(k, str)
            and torch.is_tensor(v)
            and v.ndim == 1
        }

        if len(tensors) >= 10:
            return tensors

        for key in [
            "cache",
            "embeddings",
            "vectors",
            "data",
        ]:
            if key in obj:
                found = find_tensor_dict(obj[key])
                if found:
                    return found

    return None


# ------------------------------------------------------------
# Load manifest / truth
# ------------------------------------------------------------

manifest = json.loads(
    MANIFEST.read_text()
)

systems = [
    x
    for x in manifest["systems"]
    if x.get("split") == "dev"
]

truth = {
    str(x["session_id"]):
    set(map(str, x["roles"]))
    for x in systems
}

sids = sorted(truth)

print(
    f"[LOAD] DEV systems={len(sids)}"
)


# ------------------------------------------------------------
# Candidate selections are canonical.
# Do not rely on selection_log because resume may have overwritten
# round-1 diagnostics.
# ------------------------------------------------------------

selected = {}

all_queries = []

for policy in POLICIES:
    for r in ROUNDS:
        path = (
            EXP
            / policy
            / f"candidates_dev_round_{r:02d}.jsonl"
        )

        rows = read_jsonl(path)

        if len(rows) != len(sids):
            raise RuntimeError(
                f"{path}: expected {len(sids)} rows, "
                f"got {len(rows)}"
            )

        for x in rows:
            sid = str(x["session_id"])
            q = str(x["query"])

            selected[
                (policy, r, sid)
            ] = q

            all_queries.append(q)


# ------------------------------------------------------------
# Load selector diagnostics when available.
# ------------------------------------------------------------

selector_diag = {}

for x in read_jsonl(
    EXP / "selection_log.jsonl"
):
    key = (
        str(x["policy"]),
        int(x["round"]),
        str(x["session_id"]),
    )

    selector_diag[key] = (
        x.get("diagnostics")
        or {}
    )


# ------------------------------------------------------------
# Load all belief states.
# ------------------------------------------------------------

states = {}

role_names = None

for policy in POLICIES:
    for r in range(0, 7):
        path = (
            EXP
            / policy
            / f"states_dev_round_{r:02d}.jsonl"
        )

        rows = read_jsonl(path)

        if len(rows) != len(sids):
            raise RuntimeError(
                f"{path}: expected {len(sids)} rows, "
                f"got {len(rows)}"
            )

        for x in rows:
            sid = str(x["session_id"])

            role_probs = {
                str(k): float(v)
                for k, v
                in x["role_probs"].items()
            }

            states[
                (policy, r, sid)
            ] = role_probs

            if role_names is None:
                role_names = sorted(
                    role_probs
                )

print(
    f"[LOAD] roles={len(role_names)}"
)


# ------------------------------------------------------------
# Calculate AUC directly from state + truth.
# This avoids depending on summary aggregation.
# ------------------------------------------------------------

def auc_of(policy, r, sid):
    probs = states[
        (policy, r, sid)
    ]

    y = np.asarray(
        [
            int(role in truth[sid])
            for role in role_names
        ],
        dtype=int,
    )

    p = np.asarray(
        [
            probs[role]
            for role in role_names
        ],
        dtype=float,
    )

    return float(
        roc_auc_score(y, p)
    )


auc = {}

for policy in POLICIES:
    for r in range(0, 7):
        for sid in sids:
            auc[
                (policy, r, sid)
            ] = auc_of(
                policy,
                r,
                sid,
            )


# ------------------------------------------------------------
# Try to recover exact Qwen semantic embeddings used by V3.2.
# If unavailable, automatically fall back to TF-IDF.
# ------------------------------------------------------------

semantic_mode = None
query_vec = {}

embed_file = (
    EXP
    / "embedding_runtime"
    / "semantic_embeddings.pt"
)

try:
    if embed_file.exists():
        v31 = load_v31()

        obj = load_torch(
            embed_file
        )

        cache = find_tensor_dict(
            obj
        )

        if cache:
            missing = 0

            for q in set(all_queries):
                candidates = [
                    v31.query_text(q),
                    q,
                ]

                v = None

                for k in candidates:
                    if k in cache:
                        v = cache[k]
                        break

                if v is None:
                    missing += 1
                    continue

                v = v.float()
                v = v / v.norm().clamp_min(
                    1e-12
                )

                query_vec[q] = (
                    v.numpy()
                )

            if missing == 0:
                semantic_mode = (
                    "Qwen2.5 semantic embeddings"
                )

except Exception as e:
    print(
        "[WARN] exact semantic embeddings "
        f"unavailable: {e}"
    )


if semantic_mode is None:
    unique_queries = sorted(
        set(all_queries)
    )

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        norm="l2",
    )

    X = tfidf.fit_transform(
        unique_queries
    ).toarray()

    for q, v in zip(
        unique_queries,
        X,
    ):
        query_vec[q] = v

    semantic_mode = (
        "TF-IDF fallback"
    )


print(
    f"[SIMILARITY] mode={semantic_mode}"
)


def cosine(q1, q2):
    a = query_vec[q1]
    b = query_vec[q2]

    den = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if den < 1e-12:
        return 0.0

    return float(
        np.dot(a, b) / den
    )


def mean_pairwise_similarity(queries):
    if len(queries) < 2:
        return None

    vals = [
        cosine(a, b)
        for a, b
        in itertools.combinations(
            queries,
            2,
        )
    ]

    return mean(vals)


# ------------------------------------------------------------
# Build per-session/per-round forensic table.
# ------------------------------------------------------------

rows = []

for policy in POLICIES:
    for sid in sids:
        previous_queries = []

        for r in ROUNDS:
            q = selected[
                (policy, r, sid)
            ]

            before_auc = auc[
                (policy, r - 1, sid)
            ]

            after_auc = auc[
                (policy, r, sid)
            ]

            delta_auc = (
                after_auc
                - before_auc
            )

            if previous_queries:
                max_prior_similarity = max(
                    cosine(q, old)
                    for old
                    in previous_queries
                )

                novelty = (
                    1.0
                    - max_prior_similarity
                )
            else:
                max_prior_similarity = None
                novelty = 1.0

            prev_belief = states[
                (policy, r - 1, sid)
            ]

            new_belief = states[
                (policy, r, sid)
            ]

            belief_shift = mean(
                [
                    abs(
                        new_belief[x]
                        - prev_belief[x]
                    )
                    for x in role_names
                ]
            )

            confidence_before = mean(
                [
                    2.0
                    * abs(
                        prev_belief[x] - 0.5
                    )
                    for x in role_names
                ]
            )

            confidence_after = mean(
                [
                    2.0
                    * abs(
                        new_belief[x] - 0.5
                    )
                    for x in role_names
                ]
            )

            entropy_before = binary_entropy(
                [
                    prev_belief[x]
                    for x in role_names
                ]
            )

            entropy_after = binary_entropy(
                [
                    new_belief[x]
                    for x in role_names
                ]
            )

            diag = selector_diag.get(
                (policy, r, sid),
                {},
            )

            top5 = (
                diag.get("top5")
                or []
            )

            selector_score = (
                diag.get("score")
            )

            selector_margin = None
            selector_top5_spread = None

            if len(top5) >= 2:
                # supports both {"q","s"} and {"query","score"}
                def score_of(x):
                    if "s" in x:
                        return float(x["s"])
                    if "score" in x:
                        return float(x["score"])
                    return None

                s1 = score_of(top5[0])
                s2 = score_of(top5[1])

                if (
                    s1 is not None
                    and s2 is not None
                ):
                    selector_margin = (
                        s1 - s2
                    )

                slast = score_of(
                    top5[-1]
                )

                if (
                    s1 is not None
                    and slast is not None
                ):
                    selector_top5_spread = (
                        s1 - slast
                    )

            qonly_q = selected[
                (
                    "query_only",
                    r,
                    sid,
                )
            ]

            similarity_to_query_only = cosine(
                q,
                qonly_q,
            )

            rows.append(
                {
                    "policy": policy,
                    "session_id": sid,
                    "round": r,
                    "query": q,
                    "auc_before": before_auc,
                    "auc_after": after_auc,
                    "delta_auc": delta_auc,
                    "max_similarity_to_prior_query":
                    max_prior_similarity,
                    "query_novelty": novelty,
                    "belief_mean_abs_shift":
                    belief_shift,
                    "belief_confidence_before":
                    confidence_before,
                    "belief_confidence_after":
                    confidence_after,
                    "belief_entropy_before":
                    entropy_before,
                    "belief_entropy_after":
                    entropy_after,
                    "selector_score":
                    selector_score,
                    "selector_margin":
                    selector_margin,
                    "selector_top5_spread":
                    selector_top5_spread,
                    "similarity_to_query_only":
                    similarity_to_query_only,
                }
            )

            previous_queries.append(
                q
            )


# ------------------------------------------------------------
# Per-policy/per-round aggregation.
# ------------------------------------------------------------

round_summary = []

for policy in POLICIES:
    for r in ROUNDS:
        rr = [
            x
            for x in rows
            if x["policy"] == policy
            and x["round"] == r
        ]

        queries = [
            x["query"]
            for x in rr
        ]

        round_summary.append(
            {
                "policy": policy,
                "round": r,
                "mean_auc_after":
                mean(
                    [
                        x["auc_after"]
                        for x in rr
                    ]
                ),
                "mean_delta_auc":
                mean(
                    [
                        x["delta_auc"]
                        for x in rr
                    ]
                ),
                "positive_delta_fraction":
                mean(
                    [
                        float(
                            x["delta_auc"] > 0
                        )
                        for x in rr
                    ]
                ),
                "unique_exact_queries":
                len(set(queries)),
                "mean_pairwise_query_similarity":
                mean_pairwise_similarity(
                    queries
                ),
                "mean_query_novelty":
                mean(
                    [
                        x["query_novelty"]
                        for x in rr
                    ]
                ),
                "mean_selector_margin":
                mean(
                    [
                        x["selector_margin"]
                        for x in rr
                    ]
                ),
                "mean_selector_top5_spread":
                mean(
                    [
                        x["selector_top5_spread"]
                        for x in rr
                    ]
                ),
                "mean_belief_shift":
                mean(
                    [
                        x[
                            "belief_mean_abs_shift"
                        ]
                        for x in rr
                    ]
                ),
                "mean_entropy_after":
                mean(
                    [
                        x["belief_entropy_after"]
                        for x in rr
                    ]
                ),
            }
        )


# ------------------------------------------------------------
# Correlation diagnostics.
# For adaptivity, rounds >=2 matter most.
# ------------------------------------------------------------

hbq = [
    x
    for x in rows
    if (
        x["policy"]
        == "history_belief_query"
        and x["round"] >= 2
    )
]

correlations = {
    "HBQ_novelty_vs_delta_auc":
    corr(
        [
            x["query_novelty"]
            for x in hbq
        ],
        [
            x["delta_auc"]
            for x in hbq
        ],
    ),

    "HBQ_selector_margin_vs_delta_auc":
    corr(
        [
            x["selector_margin"]
            for x in hbq
        ],
        [
            x["delta_auc"]
            for x in hbq
        ],
    ),

    "HBQ_belief_shift_vs_delta_auc":
    corr(
        [
            x["belief_mean_abs_shift"]
            for x in hbq
        ],
        [
            x["delta_auc"]
            for x in hbq
        ],
    ),

    "HBQ_similarity_to_QOnly_vs_delta_auc":
    corr(
        [
            x["similarity_to_query_only"]
            for x in hbq
        ],
        [
            x["delta_auc"]
            for x in hbq
        ],
    ),
}


# ------------------------------------------------------------
# Identify best/worst HBQ transitions.
# ------------------------------------------------------------

hbq_sorted = sorted(
    [
        x
        for x in rows
        if x["policy"]
        == "history_belief_query"
    ],
    key=lambda x: x["delta_auc"],
)

worst = hbq_sorted[:10]
best = list(
    reversed(
        hbq_sorted[-10:]
    )
)


# ------------------------------------------------------------
# Direct round-level HBQ vs QOnly / Random.
# ------------------------------------------------------------

round_comparison = []

for r in range(0, 7):
    for A, B in [
        (
            "history_belief_query",
            "query_only",
        ),
        (
            "history_belief_query",
            "random",
        ),
    ]:
        effects = [
            auc[(A, r, sid)]
            - auc[(B, r, sid)]
            for sid in sids
        ]

        round_comparison.append(
            {
                "round": r,
                "comparison":
                f"{A}_minus_{B}",
                "mean_auc_difference":
                mean(effects),
                "wins":
                sum(
                    x > 0
                    for x in effects
                ),
                "losses":
                sum(
                    x < 0
                    for x in effects
                ),
                "ties":
                sum(
                    abs(x) < 1e-12
                    for x in effects
                ),
            }
        )


# ------------------------------------------------------------
# Query divergence:
# how often does HBQ actually choose a different probe from QOnly?
# ------------------------------------------------------------

divergence = {}

for r in ROUNDS:
    exact_diff = []

    semantic_sim = []

    for sid in sids:
        a = selected[
            (
                "history_belief_query",
                r,
                sid,
            )
        ]

        b = selected[
            (
                "query_only",
                r,
                sid,
            )
        ]

        exact_diff.append(
            float(a != b)
        )

        semantic_sim.append(
            cosine(a, b)
        )

    divergence[r] = {
        "fraction_HBQ_exactly_different_from_QOnly":
        mean(exact_diff),

        "mean_HBQ_QOnly_semantic_similarity":
        mean(semantic_sim),
    }


# ------------------------------------------------------------
# Heuristic diagnosis flags.
# These are descriptive, not statistical conclusions.
# ------------------------------------------------------------

rs = {
    (
        x["policy"],
        x["round"],
    ): x
    for x in round_summary
}

diagnostic_flags = []

hbq_late_delta = mean(
    [
        rs[
            (
                "history_belief_query",
                r,
            )
        ]["mean_delta_auc"]
        for r in [4, 5, 6]
    ]
)

hbq_early_delta = mean(
    [
        rs[
            (
                "history_belief_query",
                r,
            )
        ]["mean_delta_auc"]
        for r in [1, 2, 3]
    ]
)

random_div = mean(
    [
        rs[
            ("random", r)
        ][
            "mean_pairwise_query_similarity"
        ]
        for r in ROUNDS
    ]
)

hbq_div = mean(
    [
        rs[
            (
                "history_belief_query",
                r,
            )
        ][
            "mean_pairwise_query_similarity"
        ]
        for r in ROUNDS
    ]
)

if (
    hbq_late_delta is not None
    and hbq_early_delta is not None
    and hbq_late_delta
    < hbq_early_delta
):
    diagnostic_flags.append(
        "HBQ information gain degrades in later rounds."
    )

if (
    random_div is not None
    and hbq_div is not None
    and hbq_div > random_div + 0.10
):
    diagnostic_flags.append(
        "HBQ probes are substantially more semantically clustered "
        "than Random probes, consistent with insufficient exploration."
    )

nov_corr = correlations[
    "HBQ_novelty_vs_delta_auc"
]

if (
    nov_corr is not None
    and nov_corr > 0.20
):
    diagnostic_flags.append(
        "More novel HBQ probes correlate positively with AUC gain, "
        "suggesting diversity/exploration may matter."
    )

margin_corr = correlations[
    "HBQ_selector_margin_vs_delta_auc"
]

if (
    margin_corr is not None
    and margin_corr < -0.20
):
    diagnostic_flags.append(
        "Higher selector confidence correlates with worse AUC gain, "
        "consistent with selector overconfidence/miscalibration."
    )

shift_corr = correlations[
    "HBQ_belief_shift_vs_delta_auc"
]

if (
    shift_corr is not None
    and shift_corr < -0.20
):
    diagnostic_flags.append(
        "Large belief updates correlate with poorer AUC gain, "
        "consistent with noisy evidence destabilizing the belief state."
    )


# ------------------------------------------------------------
# Save machine-readable artifacts.
# ------------------------------------------------------------

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

write_csv(
    OUT / "session_round_forensics.csv",
    rows,
)

write_csv(
    OUT / "round_summary.csv",
    round_summary,
)

write_csv(
    OUT / "round_comparison.csv",
    round_comparison,
)

summary = {
    "semantic_similarity_mode":
    semantic_mode,

    "n_systems":
    len(sids),

    "rounds":
    6,

    "correlations":
    correlations,

    "query_divergence_HBQ_vs_QOnly":
    divergence,

    "diagnostic_flags":
    diagnostic_flags,

    "worst_HBQ_transitions":
    worst,

    "best_HBQ_transitions":
    best,
}

(
    OUT / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


# ------------------------------------------------------------
# Human-readable report
# ------------------------------------------------------------

report = []

report.append(
    "=" * 100
)

report.append(
    "V3.3 FORENSIC DIAGNOSTIC — CLOSED-LOOP SELECTOR"
)

report.append(
    "=" * 100
)

report.append(
    f"Systems: {len(sids)}"
)

report.append(
    f"Semantic similarity: {semantic_mode}"
)

report.append(
    "No new MAS/API executions were performed."
)

report.append("")

report.append(
    "ROUND SUMMARY"
)

report.append(
    "-" * 100
)

report.append(
    "policy                    r   AUC      dAUC     uniqueQ   pairSim   novelty   selMargin   beliefShift"
)

for x in round_summary:
    report.append(
        f"{x['policy']:25s} "
        f"{x['round']:1d} "
        f"{x['mean_auc_after']:8.4f} "
        f"{x['mean_delta_auc']:+8.4f} "
        f"{x['unique_exact_queries']:7d} "
        f"{(x['mean_pairwise_query_similarity'] if x['mean_pairwise_query_similarity'] is not None else float('nan')):9.4f} "
        f"{(x['mean_query_novelty'] if x['mean_query_novelty'] is not None else float('nan')):9.4f} "
        f"{(x['mean_selector_margin'] if x['mean_selector_margin'] is not None else float('nan')):10.4f} "
        f"{(x['mean_belief_shift'] if x['mean_belief_shift'] is not None else float('nan')):11.4f}"
    )

report.append("")

report.append(
    "HBQ VS QUERY-ONLY DIVERGENCE"
)

report.append(
    "-" * 100
)

for r in ROUNDS:
    x = divergence[r]

    report.append(
        f"round={r} "
        f"exactly_different="
        f"{x['fraction_HBQ_exactly_different_from_QOnly']:.3f} "
        f"semantic_similarity="
        f"{x['mean_HBQ_QOnly_semantic_similarity']:.3f}"
    )

report.append("")

report.append(
    "CORRELATIONS WITH HBQ ROUND-TO-ROUND ΔAUC (ROUNDS 2–6)"
)

report.append(
    "-" * 100
)

for k, v in correlations.items():
    report.append(
        f"{k:48s}: "
        f"{v if v is not None else 'NA'}"
    )

report.append("")

report.append(
    "HBQ VS BASELINES BY ROUND"
)

report.append(
    "-" * 100
)

for x in round_comparison:
    report.append(
        f"r={x['round']} "
        f"{x['comparison']:45s} "
        f"mean_diff={x['mean_auc_difference']:+.4f} "
        f"W/L/T="
        f"{x['wins']}/"
        f"{x['losses']}/"
        f"{x['ties']}"
    )

report.append("")

report.append(
    "10 WORST HBQ TRANSITIONS"
)

report.append(
    "-" * 100
)

for x in worst:
    report.append(
        f"{x['session_id']} "
        f"r={x['round']} "
        f"AUC {x['auc_before']:.3f}"
        f"->{x['auc_after']:.3f} "
        f"d={x['delta_auc']:+.3f} "
        f"novelty={x['query_novelty']:.3f} "
        f"margin={x['selector_margin']} "
        f"shift={x['belief_mean_abs_shift']:.3f}"
    )

    report.append(
        f"    Q: {x['query']}"
    )

report.append("")

report.append(
    "10 BEST HBQ TRANSITIONS"
)

report.append(
    "-" * 100
)

for x in best:
    report.append(
        f"{x['session_id']} "
        f"r={x['round']} "
        f"AUC {x['auc_before']:.3f}"
        f"->{x['auc_after']:.3f} "
        f"d={x['delta_auc']:+.3f} "
        f"novelty={x['query_novelty']:.3f} "
        f"margin={x['selector_margin']} "
        f"shift={x['belief_mean_abs_shift']:.3f}"
    )

    report.append(
        f"    Q: {x['query']}"
    )

report.append("")

report.append(
    "AUTOMATIC DIAGNOSTIC FLAGS"
)

report.append(
    "-" * 100
)

if diagnostic_flags:
    for x in diagnostic_flags:
        report.append(
            "* " + x
        )
else:
    report.append(
        "* No single simple failure mode dominates; inspect transition table."
    )

report.append("")

report.append(
    "FILES"
)

report.append(
    "-" * 100
)

report.append(
    str(
        OUT / "session_round_forensics.csv"
    )
)

report.append(
    str(
        OUT / "round_summary.csv"
    )
)

report.append(
    str(
        OUT / "round_comparison.csv"
    )
)

report.append(
    str(
        OUT / "summary.json"
    )
)

text = "\n".join(
    report
)

(
    OUT / "report.txt"
).write_text(
    text
)

print(text)
