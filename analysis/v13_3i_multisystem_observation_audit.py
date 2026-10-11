#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np


INPUTS = [
    Path("artifacts/v13_3c_cross_role/cross_role_matrix.json"),
    Path("artifacts/v13_3c_cross_role_v12/cross_role_matrix.json"),
]

OUT = Path("artifacts/v13_3i_multisystem_audit")
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Frozen, pre-existing aggregation rules only.
# This script is DIAGNOSTIC; it does not fit/tune anything.
# ---------------------------------------------------------------------
METHODS = [
    ("prob_top3",   "probability",        "top3"),
    ("prob_mean",   "probability",        "mean"),
    ("prob_median", "probability",        "median"),
    ("prob_p90",    "probability",        "p90"),

    ("evid_top3",   "evidence_delta",     "top3"),
    ("evid_mean",   "evidence_delta",     "mean"),
    ("evid_median", "evidence_delta",     "median"),
    ("evid_trim10", "evidence_delta",     "trim10"),
    ("evid_clip3",  "evidence_delta",     "clip3"),
    ("evid_clip5",  "evidence_delta",     "clip5"),

    ("attr_top3",   "answer_attribution", "top3"),
    ("attr_mean",   "answer_attribution", "mean"),
    ("attr_median", "answer_attribution", "median"),
    ("attr_trim10", "answer_attribution", "trim10"),
]

EVENT_SIGNALS = [
    ("prob", "probability"),
    ("evid", "evidence_delta"),
    ("attr", "answer_attribution"),
]


def agg(xs, mode):
    x = np.asarray(xs, dtype=float)
    if len(x) == 0:
        return float("nan")

    if mode == "top3":
        n = min(3, len(x))
        return float(np.mean(np.sort(x)[-n:]))

    if mode == "mean":
        return float(np.mean(x))

    if mode == "median":
        return float(np.median(x))

    if mode == "p90":
        return float(np.percentile(x, 90))

    if mode == "trim10":
        y = np.sort(x)
        k = int(math.floor(0.10 * len(y)))
        if k == 0 or 2 * k >= len(y):
            return float(np.mean(y))
        return float(np.mean(y[k:-k]))

    if mode == "clip3":
        return float(np.mean(np.clip(x, -3.0, 3.0)))

    if mode == "clip5":
        return float(np.mean(np.clip(x, -5.0, 5.0)))

    raise ValueError(mode)


def rank_scores(scores):
    ordered = sorted(scores.items(), key=lambda z: (-z[1], z[0]))
    return (
        ordered,
        {r: i + 1 for i, (r, _) in enumerate(ordered)}
    )


def fmt(x, nd=3):
    if x is None:
        return "NA"
    if isinstance(x, float) and math.isnan(x):
        return "NA"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# ---------------------------------------------------------------------
# Load all four systems
# ---------------------------------------------------------------------
systems = []
candidate_roles = None

for path in INPUTS:
    with path.open() as f:
        d = json.load(f)

    roles = list(d["candidate_roles"])

    if candidate_roles is None:
        candidate_roles = roles
    elif candidate_roles != roles:
        raise RuntimeError(
            f"candidate role mismatch in {path}"
        )

    systems.extend(d["systems"])

print("=" * 100)
print("V13.3i MULTI-SYSTEM OBSERVATION AUDIT")
print("=" * 100)
print("systems:", len(systems))
print("candidate roles:", len(candidate_roles))
print("API calls: 0")
print("GPU/model calls: 0")
print()

for s in systems:
    print(
        s["system_id"],
        "events=", len(s["events"]),
        "K=", len(s["true_roles_eval_only"]),
        "truth=", s["true_roles_eval_only"],
    )


# ---------------------------------------------------------------------
# 1. Final ranking quality for each frozen method
# ---------------------------------------------------------------------
method_rows = []
system_method_cache = {}

for s in systems:
    sid = s["system_id"]
    truth = set(s["true_roles_eval_only"])
    K = len(truth)
    events = s["events"]

    for method, field, mode in METHODS:
        scores = {}

        for role in candidate_roles:
            xs = [
                float(e["all_role_scores"][role][field])
                for e in events
            ]
            scores[role] = agg(xs, mode)

        ordered, ranks = rank_scores(scores)
        topk = [r for r, _ in ordered[:K]]

        hits = len(truth.intersection(topk))
        recall_k = hits / K
        exact = set(topk) == truth

        true_ranks = [ranks[r] for r in truth]
        false_roles = [r for r in candidate_roles if r not in truth]

        weakest_true = min(
            ((r, scores[r]) for r in truth),
            key=lambda x: x[1]
        )
        strongest_false = max(
            ((r, scores[r]) for r in false_roles),
            key=lambda x: x[1]
        )

        margin = weakest_true[1] - strongest_false[1]

        row = {
            "system_id": sid,
            "K": K,
            "method": method,
            "signal": field,
            "aggregator": mode,
            "hits_at_K": hits,
            "recall_at_K": recall_k,
            "exact_set": int(exact),
            "mean_true_rank": float(np.mean(true_ranks)),
            "worst_true_rank": max(true_ranks),
            "weakest_true_role": weakest_true[0],
            "weakest_true_score": weakest_true[1],
            "strongest_false_role": strongest_false[0],
            "strongest_false_score": strongest_false[1],
            "true_false_boundary_margin": margin,
            "topK": "|".join(topk),
        }

        method_rows.append(row)

        system_method_cache[(sid, method)] = {
            "scores": scores,
            "ranks": ranks,
            "ordered": ordered,
            "topk": topk,
        }


# ---------------------------------------------------------------------
# 2. Per-role event observability
#
# Key diagnostic:
# - best rank ever
# - repeated top1/top3/top5 support
# - direct/targeted probes
# - cross-role/non-targeted probes
# ---------------------------------------------------------------------
role_event_rows = []

for s in systems:
    sid = s["system_id"]
    truth = set(s["true_roles_eval_only"])
    events = s["events"]

    # event-level rank maps for each raw signal
    event_rank_maps = {
        short: []
        for short, _ in EVENT_SIGNALS
    }

    for e in events:
        for short, field in EVENT_SIGNALS:
            ss = {
                role: float(e["all_role_scores"][role][field])
                for role in candidate_roles
            }
            _, ranks = rank_scores(ss)
            event_rank_maps[short].append(ranks)

    for role in candidate_roles:
        is_true = role in truth

        for short, field in EVENT_SIGNALS:
            ranks_all = [
                event_rank_maps[short][i][role]
                for i in range(len(events))
            ]

            vals_all = [
                float(e["all_role_scores"][role][field])
                for e in events
            ]

            idx_target = [
                i for i, e in enumerate(events)
                if e["target_role"] == role
            ]
            idx_other = [
                i for i, e in enumerate(events)
                if e["target_role"] != role
            ]

            def group_metrics(idxs):
                if not idxs:
                    return {
                        "n": 0,
                        "best_rank": None,
                        "top1_rate": None,
                        "top3_rate": None,
                        "top5_rate": None,
                        "mean_rank": None,
                        "max_score": None,
                        "mean_score": None,
                    }

                rr = [ranks_all[i] for i in idxs]
                vv = [vals_all[i] for i in idxs]

                return {
                    "n": len(idxs),
                    "best_rank": min(rr),
                    "top1_rate": sum(r <= 1 for r in rr) / len(rr),
                    "top3_rate": sum(r <= 3 for r in rr) / len(rr),
                    "top5_rate": sum(r <= 5 for r in rr) / len(rr),
                    "mean_rank": float(np.mean(rr)),
                    "max_score": max(vv),
                    "mean_score": float(np.mean(vv)),
                }

            ga = group_metrics(list(range(len(events))))
            gt = group_metrics(idx_target)
            gn = group_metrics(idx_other)

            role_event_rows.append({
                "system_id": sid,
                "role": role,
                "is_true": int(is_true),
                "signal": short,

                "n_events": ga["n"],
                "best_rank_any": ga["best_rank"],
                "top1_rate_any": ga["top1_rate"],
                "top3_rate_any": ga["top3_rate"],
                "top5_rate_any": ga["top5_rate"],
                "mean_rank_any": ga["mean_rank"],
                "max_score_any": ga["max_score"],
                "mean_score_any": ga["mean_score"],

                "n_targeted": gt["n"],
                "best_rank_targeted": gt["best_rank"],
                "top1_rate_targeted": gt["top1_rate"],
                "top3_rate_targeted": gt["top3_rate"],
                "mean_rank_targeted": gt["mean_rank"],
                "max_score_targeted": gt["max_score"],
                "mean_score_targeted": gt["mean_score"],

                "n_nontargeted": gn["n"],
                "best_rank_nontargeted": gn["best_rank"],
                "top1_rate_nontargeted": gn["top1_rate"],
                "top3_rate_nontargeted": gn["top3_rate"],
                "mean_rank_nontargeted": gn["mean_rank"],
                "max_score_nontargeted": gn["max_score"],
                "mean_score_nontargeted": gn["mean_score"],
            })


# ---------------------------------------------------------------------
# 3. Present-vs-absent cross-system role audit
#
# Does a role rise when actually present?
# Also exposes roles that rank highly even when ALWAYS absent.
# ---------------------------------------------------------------------
bias_rows = []

for method, field, mode in METHODS:
    for role in candidate_roles:
        present_ranks = []
        absent_ranks = []
        present_scores = []
        absent_scores = []
        false_topk_count = 0
        absent_system_count = 0

        for s in systems:
            sid = s["system_id"]
            truth = set(s["true_roles_eval_only"])
            K = len(truth)

            obj = system_method_cache[(sid, method)]
            r = obj["ranks"][role]
            sc = obj["scores"][role]

            if role in truth:
                present_ranks.append(r)
                present_scores.append(sc)
            else:
                absent_system_count += 1
                absent_ranks.append(r)
                absent_scores.append(sc)
                if r <= K:
                    false_topk_count += 1

        pr = float(np.mean(present_ranks)) if present_ranks else None
        ar = float(np.mean(absent_ranks)) if absent_ranks else None
        ps = float(np.mean(present_scores)) if present_scores else None
        ass = float(np.mean(absent_scores)) if absent_scores else None

        bias_rows.append({
            "method": method,
            "role": role,
            "n_present_systems": len(present_ranks),
            "n_absent_systems": len(absent_ranks),
            "present_mean_rank": pr,
            "absent_mean_rank": ar,
            "rank_lift_absent_minus_present":
                (ar - pr)
                if ar is not None and pr is not None
                else None,
            "present_mean_score": ps,
            "absent_mean_score": ass,
            "score_lift_present_minus_absent":
                (ps - ass)
                if ps is not None and ass is not None
                else None,
            "false_topK_count_when_absent": false_topk_count,
            "false_topK_rate_when_absent":
                false_topk_count / absent_system_count
                if absent_system_count
                else None,
        })


# ---------------------------------------------------------------------
# 4. Prefix stability.
#
# IMPORTANT:
# True K is used ONLY for evaluation.
# It is NOT available to the inference method.
# ---------------------------------------------------------------------
prefix_rows = []

for s in systems:
    sid = s["system_id"]
    truth = set(s["true_roles_eval_only"])
    K = len(truth)
    events = s["events"]

    for method, field, mode in METHODS:
        exacts = []
        recalls = []

        for n in range(1, len(events) + 1):
            prefix = events[:n]

            scores = {}
            for role in candidate_roles:
                xs = [
                    float(e["all_role_scores"][role][field])
                    for e in prefix
                ]
                scores[role] = agg(xs, mode)

            ordered, _ = rank_scores(scores)
            topk = [r for r, _ in ordered[:K]]

            exact = set(topk) == truth
            recall = len(set(topk) & truth) / K

            exacts.append(exact)
            recalls.append(recall)

        exact_idxs = [
            i + 1 for i, x in enumerate(exacts)
            if x
        ]
        first_exact = exact_idxs[0] if exact_idxs else None

        permanent_exact = None
        for i in range(len(exacts)):
            if all(exacts[i:]):
                permanent_exact = i + 1
                break

        transitions = sum(
            exacts[i] != exacts[i - 1]
            for i in range(1, len(exacts))
        )

        post20 = exacts[19:] if len(exacts) >= 20 else exacts
        rec_post20 = recalls[19:] if len(recalls) >= 20 else recalls

        prefix_rows.append({
            "system_id": sid,
            "K_eval_only": K,
            "method": method,
            "first_exact": first_exact,
            "permanent_exact": permanent_exact,
            "final_exact": int(exacts[-1]),
            "exact_fraction_all": float(np.mean(exacts)),
            "exact_fraction_post20": float(np.mean(post20)),
            "mean_recall_post20": float(np.mean(rec_post20)),
            "transitions": transitions,
        })


# ---------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------
def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


write_csv(OUT / "method_summary.csv", method_rows)
write_csv(OUT / "role_event_observability.csv", role_event_rows)
write_csv(OUT / "role_present_absent_bias.csv", bias_rows)
write_csv(OUT / "prefix_stability.csv", prefix_rows)


# ---------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------
print("\n" + "=" * 100)
print("A. FINAL RECOVERY — FROZEN METHODS")
print("K IS USED ONLY FOR EVALUATION")
print("=" * 100)

focus_methods = [
    "prob_top3",
    "prob_mean",
    "evid_top3",
    "evid_mean",
    "evid_trim10",
    "evid_clip3",
    "evid_clip5",
    "attr_top3",
    "attr_mean",
]

for sid in [s["system_id"] for s in systems]:
    print("\nSYSTEM:", sid)

    for r in method_rows:
        if r["system_id"] != sid:
            continue
        if r["method"] not in focus_methods:
            continue

        print(
            f'{r["method"]:14s} '
            f'hits={r["hits_at_K"]}/{r["K"]} '
            f'recall@K={r["recall_at_K"]:.3f} '
            f'exact={r["exact_set"]} '
            f'meanTR={r["mean_true_rank"]:.2f} '
            f'worstTR={r["worst_true_rank"]:2d} '
            f'gap={r["true_false_boundary_margin"]:+.4f} '
            f'weak={r["weakest_true_role"]} '
            f'false={r["strongest_false_role"]}'
        )


print("\n" + "=" * 100)
print("B. TRUE-ROLE OBSERVABILITY — EVENT LEVEL")
print("combined probability only")
print("=" * 100)

true_prob_rows = [
    r for r in role_event_rows
    if r["is_true"] == 1 and r["signal"] == "prob"
]

for r in sorted(
    true_prob_rows,
    key=lambda x: (x["system_id"], x["best_rank_any"], x["role"])
):
    print(
        f'{r["system_id"]:30s} '
        f'{r["role"]:25s} '
        f'best={r["best_rank_any"]:2d} '
        f'top1={r["top1_rate_any"]:.3f} '
        f'top3={r["top3_rate_any"]:.3f} '
        f'| TARGET n={r["n_targeted"]:2d} '
        f'best={fmt(r["best_rank_targeted"],0):>3s} '
        f'top3={fmt(r["top3_rate_targeted"]):>5s} '
        f'| CROSS n={r["n_nontargeted"]:3d} '
        f'best={fmt(r["best_rank_nontargeted"],0):>3s} '
        f'top3={fmt(r["top3_rate_nontargeted"]):>5s}'
    )


print("\n" + "=" * 100)
print("C. FALSE-POSITIVE ROLE BIAS")
print("prob_top3; lower absent rank = more suspicious")
print("=" * 100)

b = [
    r for r in bias_rows
    if r["method"] == "prob_top3"
    and r["n_absent_systems"] > 0
]

b.sort(
    key=lambda x: (
        x["absent_mean_rank"]
        if x["absent_mean_rank"] is not None
        else 999,
        -x["false_topK_count_when_absent"],
    )
)

for r in b:
    print(
        f'{r["role"]:25s} '
        f'present_n={r["n_present_systems"]} '
        f'absent_n={r["n_absent_systems"]} '
        f'present_rank={fmt(r["present_mean_rank"]):>6s} '
        f'absent_rank={fmt(r["absent_mean_rank"]):>6s} '
        f'rank_lift={fmt(r["rank_lift_absent_minus_present"]):>7s} '
        f'falseTopK={r["false_topK_count_when_absent"]}/'
        f'{r["n_absent_systems"]}'
    )


print("\n" + "=" * 100)
print("D. PREFIX STABILITY")
print("K IS EVALUATION-ONLY")
print("=" * 100)

for sid in [s["system_id"] for s in systems]:
    print("\nSYSTEM:", sid)

    for r in prefix_rows:
        if r["system_id"] != sid:
            continue
        if r["method"] not in focus_methods:
            continue

        print(
            f'{r["method"]:14s} '
            f'first={str(r["first_exact"]):>4s} '
            f'permanent={str(r["permanent_exact"]):>4s} '
            f'final={r["final_exact"]} '
            f'exactPost20={r["exact_fraction_post20"]:.3f} '
            f'recallPost20={r["mean_recall_post20"]:.3f} '
            f'transitions={r["transitions"]}'
        )


# ---------------------------------------------------------------------
# Compact machine-readable summary
# ---------------------------------------------------------------------
summary = {
    "protocol": "v13.3i_multisystem_observation_audit",
    "inputs": [str(p) for p in INPUTS],
    "systems": [
        {
            "system_id": s["system_id"],
            "K_eval_only": len(s["true_roles_eval_only"]),
            "true_roles_eval_only": s["true_roles_eval_only"],
            "events": len(s["events"]),
        }
        for s in systems
    ],
    "api_calls": 0,
    "model_calls": 0,
    "important_note":
        "K and true roles are used only for diagnostic evaluation, "
        "never for scoring or ranking.",
}

with (OUT / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)


print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)
for p in [
    OUT / "method_summary.csv",
    OUT / "role_event_observability.csv",
    OUT / "role_present_absent_bias.csv",
    OUT / "prefix_stability.csv",
    OUT / "summary.json",
]:
    print(p)

print("\nDONE — API calls: 0, model calls: 0")
