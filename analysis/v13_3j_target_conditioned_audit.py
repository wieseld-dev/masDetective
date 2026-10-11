#!/usr/bin/env python3

import json
import math
from pathlib import Path
import numpy as np

INPUTS = [
    Path("artifacts/v13_3c_cross_role/cross_role_matrix.json"),
    Path("artifacts/v13_3c_cross_role_v12/cross_role_matrix.json"),
]

SIGNALS = [
    ("prob", "probability"),
    ("evid", "evidence_delta"),
    ("attr", "answer_attribution"),
]

AGGS = ["mean", "median", "top2", "top3"]


def agg(xs, mode):
    x = np.asarray(xs, dtype=float)

    if not len(x):
        return float("-inf")

    if mode == "mean":
        return float(np.mean(x))

    if mode == "median":
        return float(np.median(x))

    if mode == "top2":
        n = min(2, len(x))
        return float(np.mean(np.sort(x)[-n:]))

    if mode == "top3":
        n = min(3, len(x))
        return float(np.mean(np.sort(x)[-n:]))

    raise ValueError(mode)


def ranks(scores):
    order = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0])
    )
    return order, {
        r: i + 1
        for i, (r, _) in enumerate(order)
    }


systems = []
roles = None

for path in INPUTS:
    with open(path) as f:
        d = json.load(f)

    if roles is None:
        roles = d["candidate_roles"]

    systems.extend(d["systems"])


print("=" * 105)
print("V13.3j TARGET-CONDITIONED OBSERVATION AUDIT")
print("=" * 105)
print("systems:", len(systems))
print("API calls: 0")
print("model calls: 0")


for s in systems:
    sid = s["system_id"]
    truth = set(s["true_roles_eval_only"])
    K = len(truth)
    events = s["events"]

    print("\n" + "=" * 105)
    print("SYSTEM:", sid)
    print("K:", K)
    print("truth:", sorted(truth))
    print("=" * 105)

    # ----------------------------------------------------------
    # How many probes were explicitly targeted to each role?
    # ----------------------------------------------------------
    target_counts = {
        role: sum(e["target_role"] == role for e in events)
        for role in roles
    }

    print("\nTARGET COUNTS")
    for role in roles:
        mark = "TRUE" if role in truth else "false"
        print(
            f"{role:25s} "
            f"n={target_counts[role]:2d} "
            f"[{mark}]"
        )

    # ----------------------------------------------------------
    # Target-only ranking
    #
    # For candidate role R, ONLY use events whose target_role=R.
    # No truth is used in scoring.
    # ----------------------------------------------------------
    print("\nTARGET-ONLY FINAL RANKINGS")

    for short, field in SIGNALS:
        for mode in AGGS:

            score = {}

            for role in roles:
                vals = [
                    float(e["all_role_scores"][role][field])
                    for e in events
                    if e["target_role"] == role
                ]

                score[role] = agg(vals, mode)

            order, rank = ranks(score)

            topk = [r for r, _ in order[:K]]
            hits = len(set(topk) & truth)
            exact = set(topk) == truth

            tr = [rank[r] for r in truth]

            print(
                f"{short}_{mode:6s} "
                f"hits={hits}/{K} "
                f"recall@K={hits/K:.3f} "
                f"exact={int(exact)} "
                f"meanTR={np.mean(tr):.2f} "
                f"worstTR={max(tr):2d} "
                f"topK={topk}"
            )

    # ----------------------------------------------------------
    # Rank-support version.
    #
    # Raw probability scales may differ by role.
    # So for each targeted event, ask:
    # how highly did the Judge rank the TARGET role among all 20?
    # ----------------------------------------------------------
    print("\nTARGETED WITHIN-EVENT RANK SUPPORT")

    support = {}

    for role in roles:
        rr = []

        for e in events:
            if e["target_role"] != role:
                continue

            event_scores = {
                r: float(
                    e["all_role_scores"][r]["probability"]
                )
                for r in roles
            }

            _, event_rank = ranks(event_scores)
            rr.append(event_rank[role])

        if not rr:
            support[role] = float("-inf")
            continue

        top3_rate = np.mean([r <= 3 for r in rr])
        top5_rate = np.mean([r <= 5 for r in rr])
        mrr = np.mean([1.0 / r for r in rr])

        # Diagnostic support statistic only.
        support[role] = (
            0.50 * top3_rate
            + 0.20 * top5_rate
            + 0.30 * mrr
        )

    order, rank = ranks(support)
    topk = [r for r, _ in order[:K]]
    hits = len(set(topk) & truth)

    print(
        f"rank_support "
        f"hits={hits}/{K} "
        f"recall@K={hits/K:.3f} "
        f"exact={int(set(topk) == truth)} "
        f"topK={topk}"
    )

    print("\nTOP 10 TARGETED SUPPORT")
    for i, (role, sc) in enumerate(order[:10], 1):
        mark = "TRUE" if role in truth else "false"
        print(
            f"{i:02d}. {role:25s} "
            f"{sc:.4f} [{mark}]"
        )

    # ----------------------------------------------------------
    # Detailed true-role targeted vs cross behavior
    # ----------------------------------------------------------
    print("\nTRUE ROLE: TARGETED VS CROSS")

    for role in sorted(truth):

        target_events = [
            e for e in events
            if e["target_role"] == role
        ]

        cross_events = [
            e for e in events
            if e["target_role"] != role
        ]

        def summarize(group):
            if not group:
                return None

            role_ranks = []

            for e in group:
                scores = {
                    r: float(
                        e["all_role_scores"][r]["probability"]
                    )
                    for r in roles
                }

                _, rk = ranks(scores)
                role_ranks.append(rk[role])

            return {
                "n": len(group),
                "top1": np.mean(
                    [r == 1 for r in role_ranks]
                ),
                "top3": np.mean(
                    [r <= 3 for r in role_ranks]
                ),
                "top5": np.mean(
                    [r <= 5 for r in role_ranks]
                ),
                "mean_rank": np.mean(role_ranks),
                "best_rank": min(role_ranks),
            }

        t = summarize(target_events)
        c = summarize(cross_events)

        print(
            f"{role:25s} "
            f"TARGET "
            f"n={t['n']:2d} "
            f"top1={t['top1']:.3f} "
            f"top3={t['top3']:.3f} "
            f"top5={t['top5']:.3f} "
            f"meanR={t['mean_rank']:.2f} "
            f"best={t['best_rank']:2d} "
            f"| CROSS "
            f"n={c['n']:3d} "
            f"top1={c['top1']:.3f} "
            f"top3={c['top3']:.3f} "
            f"top5={c['top5']:.3f} "
            f"meanR={c['mean_rank']:.2f} "
            f"best={c['best_rank']:2d}"
        )


print("\nDONE — API calls: 0, model calls: 0")
