#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np


AGGREGATORS = (
    "max",
    "top2",
    "top3",
    "p90",
    "median",
    "mean",
)


def aggregate(xs, method):
    xs = [float(x) for x in xs]
    if not xs:
        return 0.5

    if method == "max":
        return max(xs)

    if method == "top2":
        z = sorted(xs, reverse=True)[:2]
        return sum(z) / len(z)

    if method == "top3":
        z = sorted(xs, reverse=True)[:3]
        return sum(z) / len(z)

    if method == "p90":
        return float(np.quantile(np.asarray(xs, dtype=float), 0.90))

    if method == "median":
        return float(statistics.median(xs))

    if method == "mean":
        return float(np.mean(xs))

    raise ValueError(method)


def set_metrics(true, pred):
    true = set(true)
    pred = set(pred)

    tp = len(true & pred)
    fp = len(pred - true)
    fn = len(true - pred)

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix",
        default="artifacts/v13_3c_cross_role/cross_role_matrix.json",
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/v13_3c2_sequential",
    )
    args = ap.parse_args()

    matrix_path = Path(args.matrix)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    obj = json.loads(matrix_path.read_text())

    all_reports = []

    for system in obj["systems"]:
        sid = system["system_id"]
        truth = set(system["true_roles_eval_only"])
        events = system["events"]
        roles = list(obj["candidate_roles"])

        k_true = len(truth)

        print("=" * 100)
        print("SYSTEM:", sid)
        print("events:", len(events))
        print("truth:", sorted(truth))
        print("k_true eval-only:", k_true)
        print("=" * 100)

        # Evidence seen so far for every candidate role.
        seen = {r: [] for r in roles}

        rows = []

        # Store previous rankings for stability diagnostics.
        previous_topk = {a: None for a in AGGREGATORS}
        stable_streak = {a: 0 for a in AGGREGATORS}

        for t, event in enumerate(events, 1):

            # One visible MAS answer produces evidence for ALL hypotheses.
            for role in roles:
                seen[role].append(
                    float(
                        event["all_role_scores"][role]["probability"]
                    )
                )

            for agg in AGGREGATORS:
                scores = {
                    r: aggregate(seen[r], agg)
                    for r in roles
                }

                ranking = sorted(
                    roles,
                    key=lambda r: scores[r],
                    reverse=True,
                )

                ranks = {
                    r: i + 1
                    for i, r in enumerate(ranking)
                }

                # IMPORTANT:
                # K is used only for evaluation here, never to drive queries.
                pred_topk = set(ranking[:k_true])
                metrics = set_metrics(truth, pred_topk)

                true_ranks = [ranks[r] for r in truth]

                all_true_top3 = all(
                    ranks[r] <= 3 for r in truth
                )
                all_true_top5 = all(
                    ranks[r] <= 5 for r in truth
                )

                mean_true_rank = float(
                    np.mean(true_ranks)
                )
                worst_true_rank = max(true_ranks)

                # Rank separation:
                # weakest true score versus strongest false score.
                min_true_score = min(
                    scores[r] for r in truth
                )
                max_false_score = max(
                    scores[r]
                    for r in roles
                    if r not in truth
                )
                true_false_gap = (
                    min_true_score - max_false_score
                )

                current_topk_tuple = tuple(
                    ranking[:k_true]
                )

                if (
                    previous_topk[agg]
                    == current_topk_tuple
                ):
                    stable_streak[agg] += 1
                else:
                    stable_streak[agg] = 1

                previous_topk[agg] = current_topk_tuple

                rows.append({
                    "system_id": sid,
                    "step": t,
                    "asked_role": event["target_role"],
                    "aggregator": agg,
                    "f1_top_k_eval_only": metrics["f1"],
                    "precision_top_k_eval_only": metrics["precision"],
                    "recall_top_k_eval_only": metrics["recall"],
                    "exact_top_k_eval_only": (
                        pred_topk == truth
                    ),
                    "all_true_in_top3_eval_only": all_true_top3,
                    "all_true_in_top5_eval_only": all_true_top5,
                    "mean_true_rank_eval_only": mean_true_rank,
                    "worst_true_rank_eval_only": worst_true_rank,
                    "true_false_gap_eval_only": true_false_gap,
                    "top_k_stable_streak": stable_streak[agg],
                    "ranking_top10": ranking[:10],
                    "top_k": ranking[:k_true],
                    "true_ranks": {
                        r: ranks[r]
                        for r in sorted(truth)
                    },
                    "true_scores": {
                        r: scores[r]
                        for r in sorted(truth)
                    },
                    "best_false_role": max(
                        (r for r in roles if r not in truth),
                        key=lambda r: scores[r],
                    ),
                    "best_false_score": max_false_score,
                })

        # ------------------------------------------------------------
        # Summarize each aggregation strategy across prefixes.
        # ------------------------------------------------------------

        summary_by_agg = {}

        for agg in AGGREGATORS:
            rr = [
                x for x in rows
                if x["aggregator"] == agg
            ]

            first_f1_1 = next(
                (
                    x["step"] for x in rr
                    if x["f1_top_k_eval_only"] >= 0.999999
                ),
                None,
            )

            first_all_top5 = next(
                (
                    x["step"] for x in rr
                    if x["all_true_in_top5_eval_only"]
                ),
                None,
            )

            first_positive_gap = next(
                (
                    x["step"] for x in rr
                    if x["true_false_gap_eval_only"] > 0
                ),
                None,
            )

            # First point from which exact top-K remains correct
            # through the END of the historical trajectory.
            permanent_exact = None

            for i, x in enumerate(rr):
                if all(
                    z["exact_top_k_eval_only"]
                    for z in rr[i:]
                ):
                    permanent_exact = x["step"]
                    break

            # Same concept for all true roles inside top-5.
            permanent_top5 = None

            for i, x in enumerate(rr):
                if all(
                    z["all_true_in_top5_eval_only"]
                    for z in rr[i:]
                ):
                    permanent_top5 = x["step"]
                    break

            best = max(
                rr,
                key=lambda x: (
                    x["f1_top_k_eval_only"],
                    -x["step"],
                ),
            )

            final = rr[-1]

            # Area under F1-query curve, normalized to [0,1].
            f1s = np.asarray(
                [x["f1_top_k_eval_only"] for x in rr],
                dtype=float,
            )
            auc_f1 = float(np.mean(f1s))

            summary_by_agg[agg] = {
                "first_exact_top_k_step_eval_only": first_f1_1,
                "permanent_exact_top_k_step_eval_only": permanent_exact,
                "first_all_true_top5_step_eval_only": first_all_top5,
                "permanent_all_true_top5_step_eval_only": permanent_top5,
                "first_positive_true_false_gap_step_eval_only": (
                    first_positive_gap
                ),
                "best_f1_eval_only": best["f1_top_k_eval_only"],
                "best_f1_first_step_eval_only": best["step"],
                "final_f1_eval_only": final["f1_top_k_eval_only"],
                "final_top_k_eval_only": final["top_k"],
                "final_true_ranks_eval_only": final["true_ranks"],
                "final_true_false_gap_eval_only": (
                    final["true_false_gap_eval_only"]
                ),
                "mean_prefix_f1_auc_eval_only": auc_f1,
            }

        report = {
            "system_id": sid,
            "true_roles_eval_only": sorted(truth),
            "k_true_eval_only": k_true,
            "n_events": len(events),
            "summary_by_aggregator_eval_only": summary_by_agg,
            "prefix_rows": rows,
        }

        all_reports.append(report)

        print()
        print("=== PREFIX SUMMARY ===")

        for agg in AGGREGATORS:
            s = summary_by_agg[agg]

            print(
                f"{agg:7s} | "
                f"first_exact={str(s['first_exact_top_k_step_eval_only']):>4s} "
                f"permanent_exact={str(s['permanent_exact_top_k_step_eval_only']):>4s} "
                f"first_top5={str(s['first_all_true_top5_step_eval_only']):>4s} "
                f"permanent_top5={str(s['permanent_all_true_top5_step_eval_only']):>4s} "
                f"positive_gap={str(s['first_positive_true_false_gap_step_eval_only']):>4s} "
                f"AUC_F1={s['mean_prefix_f1_auc_eval_only']:.3f} "
                f"final_F1={s['final_f1_eval_only']:.3f}"
            )

        # Detailed checkpoints for the current leading diagnostic:
        # top3 mean. This is NOT declared the final policy.
        rr = [
            x for x in rows
            if x["aggregator"] == "top3"
        ]

        checkpoints = sorted(
            set(
                [
                    1, 2, 3, 5, 10, 15, 20,
                    30, 40, 60, 80, 100,
                    len(events),
                ]
            )
        )

        print()
        print("=== TOP3 PREFIX CHECKPOINTS ===")

        for step in checkpoints:
            if step > len(rr):
                continue

            x = rr[step - 1]

            print(
                f"step={step:3d} "
                f"F1={x['f1_top_k_eval_only']:.3f} "
                f"exact={str(x['exact_top_k_eval_only']):5s} "
                f"mean_true_rank={x['mean_true_rank_eval_only']:.2f} "
                f"gap={x['true_false_gap_eval_only']:+.4f} "
                f"stable={x['top_k_stable_streak']:3d} "
                f"topK={x['top_k']} "
                f"best_false={x['best_false_role']}:{x['best_false_score']:.3f}"
            )

        # Print every transition in Top-K for top3 aggregation.
        print()
        print("=== TOP3 TOP-K TRANSITIONS ===")

        prev = None
        for x in rr:
            now = tuple(x["top_k"])

            if now != prev:
                print(
                    f"step={x['step']:3d} "
                    f"F1={x['f1_top_k_eval_only']:.3f} "
                    f"gap={x['true_false_gap_eval_only']:+.4f} "
                    f"topK={list(now)}"
                )
                prev = now

    output = {
        "protocol": "v13.3c2_sequential_prefix_cross_role_replay",
        "api_calls": 0,
        "gpu_required": False,
        "training": False,
        "truth_used_for_action_selection": False,
        "truth_used_for_evaluation_only": True,
        "important_caveat": (
            "Historical query order is replayed. "
            "This does not evaluate a new active controller."
        ),
        "matrix_source": str(matrix_path),
        "systems": all_reports,
    }

    json_path = outdir / "summary.json"
    json_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
    )

    csv_path = outdir / "prefix_metrics.csv"

    fieldnames = [
        "system_id",
        "step",
        "asked_role",
        "aggregator",
        "f1_top_k_eval_only",
        "precision_top_k_eval_only",
        "recall_top_k_eval_only",
        "exact_top_k_eval_only",
        "all_true_in_top3_eval_only",
        "all_true_in_top5_eval_only",
        "mean_true_rank_eval_only",
        "worst_true_rank_eval_only",
        "true_false_gap_eval_only",
        "top_k_stable_streak",
        "best_false_role",
        "best_false_score",
    ]

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        w.writeheader()

        for rep in all_reports:
            for x in rep["prefix_rows"]:
                w.writerow({
                    k: x[k]
                    for k in fieldnames
                })

    print()
    print("=" * 100)
    print("DONE")
    print("saved:", json_path)
    print("saved:", csv_path)
    print("API CALLS USED: 0")
    print("GPU REQUIRED: NO")
    print("=" * 100)


if __name__ == "__main__":
    main()
