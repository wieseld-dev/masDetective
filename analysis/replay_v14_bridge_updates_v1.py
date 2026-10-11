from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(
    "artifacts/ten_system_belief_trace_control_v14_dropin"
)

DIAG = ROOT / "bridge_diagnostic_v1.json"


# ============================================================
# Metrics
# ============================================================

def pair_auc(scores, true_roles):
    roles = list(scores)

    pos = [
        float(scores[r])
        for r in roles
        if r in true_roles
    ]

    neg = [
        float(scores[r])
        for r in roles
        if r not in true_roles
    ]

    wins = 0.0
    total = 0

    for p in pos:
        for n in neg:
            total += 1

            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5

    return wins / total


def top3(scores):
    return sorted(
        scores,
        key=scores.get,
        reverse=True,
    )[:3]


def exact_top3(scores, true_roles):
    return (
        set(top3(scores))
        == set(true_roles)
    )


def mean(xs):
    return (
        sum(xs) / len(xs)
        if xs
        else 0.0
    )


# ============================================================
# Load target identity reconstructed by previous diagnostic
# ============================================================

if not DIAG.exists():
    raise RuntimeError(
        f"Missing diagnostic file: {DIAG}"
    )

diag = json.loads(
    DIAG.read_text()
)

chosen_target = {
    (
        str(r["system_id"]),
        int(r["round"]),
    ):
        str(r["chosen_target"])
    for r in diag["rounds"]
}


# ============================================================
# Load systems
# ============================================================

summaries = {}

for p in sorted(
    ROOT.glob("mas*_summary.json")
):
    d = json.loads(
        p.read_text()
    )

    summaries[
        d["system_id"]
    ] = d


if len(summaries) != 10:
    print(
        f"WARNING: expected 10 summaries, "
        f"found {len(summaries)}"
    )


results = {}


# ============================================================
# Replay each system
# ============================================================

for sid, summary in summaries.items():

    true_roles = set(
        summary["true_roles"]
    )

    tp = (
        ROOT
        / f"{sid}_trajectory.json"
    )

    traj = json.loads(
        tp.read_text()
    )

    if not traj:
        continue

    roles = list(
        traj[0][
            "fingerprint_scores"
        ].keys()
    )

    # --------------------------------------------------------
    # A. Historical Bayes
    # --------------------------------------------------------

    A = {
        str(r): float(v)
        for r, v
        in summary[
            "final_role_probs"
        ].items()
    }


    # --------------------------------------------------------
    # B. Mean raw sensor score for EVERY role across
    #    all 8 observed responses.
    #
    #    This ignores the Bayes bridge entirely.
    # --------------------------------------------------------

    all_role_scores = defaultdict(
        list
    )

    for rec in traj:
        fp = rec[
            "fingerprint_scores"
        ]

        for r in roles:
            all_role_scores[r].append(
                float(fp[r])
            )

    B = {
        r: mean(
            all_role_scores[r]
        )
        for r in roles
    }


    # --------------------------------------------------------
    # C/D/E use only the role that the selected V14 query
    # was actually intended to test.
    # --------------------------------------------------------

    targeted_scores = defaultdict(
        list
    )

    for rec in traj:
        rnd = int(
            rec["round"]
        )

        target = chosen_target[
            (sid, rnd)
        ]

        fp = rec[
            "fingerprint_scores"
        ]

        targeted_scores[
            target
        ].append(
            float(fp[target])
        )


    # --------------------------------------------------------
    # C. Target-only mean raw score.
    #
    # Never-targeted roles get neutral score 0.5.
    # --------------------------------------------------------

    C = {}

    for r in roles:
        xs = targeted_scores.get(
            r,
            [],
        )

        C[r] = (
            mean(xs)
            if xs
            else 0.5
        )


    # --------------------------------------------------------
    # D. Target-only asymmetric evidence.
    #
    # Fixed rule, NO truth tuning:
    #
    #   e = sensor_score - 0.5
    #
    # Positive evidence: full weight.
    # Negative evidence: 0.25 weight.
    #
    # Aggregate by mean rather than sum so repeatedly-targeted
    # roles do not win merely because they were queried more.
    # --------------------------------------------------------

    D = {}

    for r in roles:
        xs = targeted_scores.get(
            r,
            [],
        )

        if not xs:
            D[r] = 0.0
            continue

        ev = []

        for s in xs:
            e = float(s) - 0.5

            if e < 0:
                e *= 0.25

            ev.append(
                e
            )

        D[r] = mean(
            ev
        )


    # --------------------------------------------------------
    # E. Target-only positive evidence.
    #
    # Negative observations are ignored completely.
    # Again mean over target opportunities.
    # --------------------------------------------------------

    E = {}

    for r in roles:
        xs = targeted_scores.get(
            r,
            [],
        )

        if not xs:
            E[r] = 0.0
            continue

        ev = [
            max(
                float(s) - 0.5,
                0.0,
            )
            for s in xs
        ]

        E[r] = mean(
            ev
        )


    methods = {
        "A_historical_bayes":
            A,

        "B_allrole_mean_sensor":
            B,

        "C_target_mean_sensor":
            C,

        "D_target_asym_neg025":
            D,

        "E_target_positive_only":
            E,
    }


    per_method = {}

    for name, scores in methods.items():
        per_method[name] = {
            "auc":
                pair_auc(
                    scores,
                    true_roles,
                ),

            "top3":
                top3(
                    scores
                ),

            "top3_exact":
                exact_top3(
                    scores,
                    true_roles,
                ),

            "scores":
                scores,
        }


    results[sid] = {
        "true_roles":
            sorted(
                true_roles
            ),

        "target_counts":
            {
                r: len(xs)
                for r, xs
                in targeted_scores.items()
            },

        "methods":
            per_method,
    }


# ============================================================
# Aggregate
# ============================================================

method_names = [
    "A_historical_bayes",
    "B_allrole_mean_sensor",
    "C_target_mean_sensor",
    "D_target_asym_neg025",
    "E_target_positive_only",
]

aggregate = {}

for name in method_names:
    aucs = [
        results[sid][
            "methods"
        ][name]["auc"]
        for sid in results
    ]

    exact = sum(
        int(
            results[sid][
                "methods"
            ][name][
                "top3_exact"
            ]
        )
        for sid in results
    )

    aggregate[name] = {
        "mean_auc":
            statistics.mean(
                aucs
            ),

        "median_auc":
            statistics.median(
                aucs
            ),

        "min_auc":
            min(
                aucs
            ),

        "max_auc":
            max(
                aucs
            ),

        "top3_exact":
            exact,

        "n_systems":
            len(
                aucs
            ),
    }


# ============================================================
# Print
# ============================================================

print()
print("=" * 118)
print("NO-TRAINING REPLAY: V14 OBSERVATIONS → ROLE EVIDENCE")
print("=" * 118)

print()
print("AGGREGATE")
print("-" * 118)

for name in method_names:
    x = aggregate[name]

    print(
        f"{name:30s}  "
        f"meanAUC={x['mean_auc']:.4f}  "
        f"median={x['median_auc']:.4f}  "
        f"range={x['min_auc']:.4f}-{x['max_auc']:.4f}  "
        f"Top3Exact={x['top3_exact']}/{x['n_systems']}"
    )


print()
print("=" * 118)
print("PER SYSTEM")
print("=" * 118)

for sid, x in results.items():

    print()
    print(
        f"{sid}"
    )

    print(
        f"TRUE: "
        f"{x['true_roles']}"
    )

    print(
        "TARGET COUNTS: "
        + json.dumps(
            x[
                "target_counts"
            ],
            sort_keys=True,
        )
    )

    for name in method_names:
        m = x[
            "methods"
        ][name]

        print(
            f"  {name:28s} "
            f"AUC={m['auc']:.4f}  "
            f"TOP3={m['top3']}  "
            f"exact={m['top3_exact']}"
        )


# ============================================================
# Save full result
# ============================================================

out = (
    ROOT
    / "bridge_replay_no_training_v1.json"
)

out.write_text(
    json.dumps(
        {
            "aggregate":
                aggregate,

            "systems":
                results,

            "definitions": {
                "A":
                    "Historical final Bayes role probabilities.",

                "B":
                    "Mean raw fingerprint score per role across all observed responses.",

                "C":
                    "Mean raw fingerprint score only when that role was the intended target; never-targeted roles = 0.5.",

                "D":
                    "Target-only evidence e=s-0.5; positive full weight; negative multiplied by 0.25; mean per targeted role.",

                "E":
                    "Target-only positive evidence max(s-0.5,0); mean per targeted role; negative ignored.",
            },
        },
        indent=2,
    )
)

print()
print("=" * 118)
print(
    "Saved:",
    out,
)
print("=" * 118)
