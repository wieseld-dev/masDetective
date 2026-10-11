#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score


INPUTS = [
    Path("artifacts/v13_3c_cross_role/cross_role_matrix.json"),
    Path("artifacts/v13_3c_cross_role_v12/cross_role_matrix.json"),
]

OUT = Path("artifacts/v13_3k_semantic_bias")
OUT.mkdir(parents=True, exist_ok=True)

SIGNALS = {
    "prob": "probability",
    "evid": "evidence_delta",
    "attr": "answer_attribution",
}


def rank_dict(scores):
    ordered = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0])
    )
    ranks = {
        role: i + 1
        for i, (role, _) in enumerate(ordered)
    }
    return ordered, ranks


def robust_scale(xs):
    x = np.asarray(xs, dtype=float)

    if len(x) == 0:
        return 1.0

    med = np.median(x)
    mad = np.median(np.abs(x - med))

    # robust sigma estimate
    scale = 1.4826 * mad

    # avoid unstable division
    return max(float(scale), 0.25)


# ------------------------------------------------------------------
# Load systems
# ------------------------------------------------------------------

systems = []
roles = None

for p in INPUTS:
    with p.open() as f:
        d = json.load(f)

    if roles is None:
        roles = list(d["candidate_roles"])

    systems.extend(d["systems"])


print("=" * 110)
print("V13.3k — TARGET SEMANTIC SELF-BIAS AUDIT")
print("=" * 110)

print("systems:", len(systems))
print("roles:", len(roles))
print("API calls: 0")
print("model calls: 0")

for s in systems:
    print(
        s["system_id"],
        "K=", len(s["true_roles_eval_only"]),
        "truth=", s["true_roles_eval_only"]
    )


# ------------------------------------------------------------------
# Collect SELF-TARGET samples
#
# Sample:
# query explicitly targeted role R
# score evaluated for the SAME role R
#
# label = whether R really exists in that system
# ------------------------------------------------------------------

samples = []

for s in systems:
    sid = s["system_id"]
    truth = set(s["true_roles_eval_only"])

    for e in s["events"]:
        target = e["target_role"]

        if target not in roles:
            continue

        score_node = e["all_role_scores"][target]

        # target rank among all roles for combined probability
        event_prob = {
            r: float(e["all_role_scores"][r]["probability"])
            for r in roles
        }
        _, ranks = rank_dict(event_prob)

        best_other_prob = max(
            float(e["all_role_scores"][r]["probability"])
            for r in roles
            if r != target
        )

        samples.append({
            "system_id": sid,
            "role": target,
            "present": int(target in truth),

            "prob": float(score_node["probability"]),
            "evid": float(score_node["evidence_delta"]),
            "attr": float(score_node["answer_attribution"]),

            "rank": ranks[target],
            "prob_margin":
                float(score_node["probability"])
                - best_other_prob,
        })


print("\n" + "=" * 110)
print("A. GLOBAL SELF-TARGET PRESENT vs ABSENT")
print("=" * 110)

y = np.array([x["present"] for x in samples])

for sig in ["prob", "evid", "attr", "prob_margin"]:
    x = np.array([x[sig] for x in samples])

    pos = x[y == 1]
    neg = x[y == 0]

    auc = roc_auc_score(y, x)

    print(
        f"{sig:12s} "
        f"present_n={len(pos):3d} "
        f"absent_n={len(neg):3d} "
        f"present_mean={np.mean(pos):+.4f} "
        f"absent_mean={np.mean(neg):+.4f} "
        f"delta={np.mean(pos)-np.mean(neg):+.4f} "
        f"AUC={auc:.4f}"
    )

# rank: lower is better, hence minus rank
rank_signal = np.array([-x["rank"] for x in samples])
auc_rank = roc_auc_score(y, rank_signal)

print(
    f"{'target_rank':12s} "
    f"present_mean={np.mean([x['rank'] for x in samples if x['present']]):.3f} "
    f"absent_mean={np.mean([x['rank'] for x in samples if not x['present']]):.3f} "
    f"AUC={auc_rank:.4f}"
)


# ------------------------------------------------------------------
# Per-role separation
# ------------------------------------------------------------------

print("\n" + "=" * 110)
print("B. PER-ROLE SELF-TARGET SEPARATION")
print("=" * 110)

per_role = {}

for role in roles:
    rr = [x for x in samples if x["role"] == role]

    pos = [x for x in rr if x["present"]]
    neg = [x for x in rr if not x["present"]]

    per_role[role] = {
        "pos": pos,
        "neg": neg,
    }

    print(f"\nROLE: {role}")
    print(
        f"present_samples={len(pos)} "
        f"absent_samples={len(neg)}"
    )

    for sig in ["prob", "evid", "attr", "prob_margin"]:
        if pos:
            p = np.array([x[sig] for x in pos])
            pm = float(np.mean(p))
            pmed = float(np.median(p))
        else:
            pm = pmed = None

        if neg:
            n = np.array([x[sig] for x in neg])
            nm = float(np.mean(n))
            nmed = float(np.median(n))
        else:
            nm = nmed = None

        if pos and neg:
            yy = np.array(
                [1] * len(pos) + [0] * len(neg)
            )
            xx = np.array(
                [x[sig] for x in pos]
                + [x[sig] for x in neg]
            )
            auc = roc_auc_score(yy, xx)
        else:
            auc = None

        print(
            f"  {sig:11s} "
            f"present_mean={pm if pm is not None else 'NA'} "
            f"absent_mean={nm if nm is not None else 'NA'} "
            f"present_med={pmed if pmed is not None else 'NA'} "
            f"absent_med={nmed if nmed is not None else 'NA'} "
            f"AUC={auc if auc is not None else 'NA'}"
        )


# ------------------------------------------------------------------
# Cross-system null calibration
#
# Diagnostic only:
#
# For system S and candidate role R:
#   1. take R-targeted probes in S
#   2. summarize their score
#   3. subtract baseline from OTHER systems where R is absent
#
# This uses labels from OTHER dev systems to estimate a null.
# It is NOT a final deployable estimator.
# ------------------------------------------------------------------

print("\n" + "=" * 110)
print("C. LEAVE-ONE-SYSTEM-OUT ABSENT-NULL CORRECTION")
print("DIAGNOSTIC ONLY — NULL BASELINE LEARNED FROM OTHER LABELED SYSTEMS")
print("K USED ONLY FOR EVALUATION")
print("=" * 110)


def system_role_target_values(system, role, signal):
    field = SIGNALS[signal]

    return [
        float(e["all_role_scores"][role][field])
        for e in system["events"]
        if e["target_role"] == role
    ]


for signal in ["prob", "evid", "attr"]:

    print("\n" + "-" * 110)
    print("SIGNAL:", signal)
    print("-" * 110)

    for s in systems:
        sid = s["system_id"]
        truth = set(s["true_roles_eval_only"])
        K = len(truth)

        corrected = {}
        raw = {}

        for role in roles:

            current_vals = system_role_target_values(
                s, role, signal
            )

            if not current_vals:
                corrected[role] = float("-inf")
                raw[role] = float("-inf")
                continue

            # Median current self-target score
            current_stat = float(np.median(current_vals))
            raw[role] = current_stat

            # NULL:
            # targeted probes for role R from OTHER systems
            # where R is known absent.
            null_vals = []

            for other in systems:
                if other["system_id"] == sid:
                    continue

                other_truth = set(
                    other["true_roles_eval_only"]
                )

                if role in other_truth:
                    continue

                null_vals.extend(
                    system_role_target_values(
                        other, role, signal
                    )
                )

            if not null_vals:
                corrected[role] = current_stat
                continue

            null_med = float(np.median(null_vals))
            null_scale = robust_scale(null_vals)

            corrected[role] = (
                current_stat - null_med
            ) / null_scale

        for label, score_map in [
            ("RAW_TARGET_MEDIAN", raw),
            ("NULL_CORRECTED", corrected),
        ]:
            ordered, ranks = rank_dict(score_map)

            topk = [
                r for r, _ in ordered[:K]
            ]

            hits = len(set(topk) & truth)
            true_ranks = [
                ranks[r] for r in truth
            ]

            print(
                f"{sid:30s} "
                f"{label:17s} "
                f"hits={hits}/{K} "
                f"recall@K={hits/K:.3f} "
                f"exact={int(set(topk)==truth)} "
                f"meanTR={np.mean(true_ranks):.2f} "
                f"worstTR={max(true_ranks):2d} "
                f"topK={topk}"
            )


# ------------------------------------------------------------------
# Inspect the strongest self-confirming FALSE roles
# ------------------------------------------------------------------

print("\n" + "=" * 110)
print("D. STRONGEST FALSE SELF-CONFIRMATION EVENTS")
print("=" * 110)

false_samples = [
    x for x in samples
    if x["present"] == 0
]

false_samples.sort(
    key=lambda x: -x["prob"]
)

for x in false_samples[:40]:
    print(
        f'{x["system_id"]:30s} '
        f'role={x["role"]:25s} '
        f'prob={x["prob"]:.4f} '
        f'evid={x["evid"]:+.3f} '
        f'attr={x["attr"]:+.3f} '
        f'rank={x["rank"]:2d} '
        f'margin={x["prob_margin"]:+.4f}'
    )


summary = {
    "protocol":
        "v13.3k_semantic_self_bias_audit",
    "systems": [
        s["system_id"]
        for s in systems
    ],
    "n_samples": len(samples),
    "api_calls": 0,
    "model_calls": 0,
    "note":
        "Null correction is diagnostic only and uses "
        "ground-truth absence labels from other development systems."
}

with (OUT / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)


print("\nDONE")
print("saved:", OUT / "summary.json")
print("API calls: 0")
print("model calls: 0")
