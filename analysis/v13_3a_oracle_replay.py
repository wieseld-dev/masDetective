#!/usr/bin/env python3

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


EPS = 1e-6


def clip_prob(p):
    return min(1.0 - EPS, max(EPS, float(p)))


def logit(p):
    p = clip_prob(p)
    return math.log(p / (1.0 - p))


def sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def percentile(xs, q):
    xs = sorted(xs)
    if not xs:
        raise ValueError("empty percentile")
    if len(xs) == 1:
        return xs[0]

    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))

    if lo == hi:
        return xs[lo]

    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def aggregate_logits(values, method):
    if not values:
        raise ValueError("cannot aggregate empty list")

    if method == "median":
        return statistics.median(values)

    if method == "max":
        return max(values)

    if method == "top2":
        x = sorted(values, reverse=True)[:2]
        return sum(x) / len(x)

    if method == "top3":
        x = sorted(values, reverse=True)[:3]
        return sum(x) / len(x)

    if method == "p75":
        return percentile(values, 0.75)

    raise ValueError(f"unknown aggregator: {method}")


def get_roles(system):
    roles = set(system.get("role_summary", {}).keys())

    for h in system.get("replay_history", []):
        r = h.get("role")
        if r:
            roles.add(r)

    return sorted(roles)


def usable_history(system):
    out = []

    for i, h in enumerate(system.get("replay_history", [])):
        role = h.get("role")
        p = h.get("combined_probability")

        if role is None or p is None:
            continue

        h2 = dict(h)
        h2["_original_index"] = i
        out.append(h2)

    return out


def infer_probabilities(selected, roles, method, prior):
    by_role = {r: [] for r in roles}

    for h in selected:
        role = h["role"]
        p = clip_prob(h["combined_probability"])
        by_role[role].append(logit(p))

    probs = {}

    for role in roles:
        vals = by_role[role]

        if vals:
            z = aggregate_logits(vals, method)
            probs[role] = sigmoid(z)
        else:
            probs[role] = prior

    return probs


def evaluate(probs, true_roles, threshold=0.5):
    roles = sorted(probs)
    truth = set(true_roles)
    pred = {r for r, p in probs.items() if p >= threshold}

    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    ranked = sorted(
        roles,
        key=lambda r: (-probs[r], r),
    )

    rank_map = {r: i + 1 for i, r in enumerate(ranked)}
    true_ranks = [rank_map[r] for r in truth if r in rank_map]

    mean_true_rank = (
        sum(true_ranks) / len(true_ranks)
        if true_ranks
        else float("inf")
    )

    recall_at_5 = (
        sum(1 for r in truth if rank_map.get(r, 10**9) <= 5) / len(truth)
        if truth
        else 0.0
    )

    all_true_top5 = (
        all(rank_map.get(r, 10**9) <= 5 for r in truth)
        if truth
        else False
    )

    bce = 0.0
    brier = 0.0

    for r in roles:
        y = 1.0 if r in truth else 0.0
        p = clip_prob(probs[r])

        bce += -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))
        brier += (p - y) ** 2

    bce /= max(1, len(roles))
    brier /= max(1, len(roles))

    true_ps = [probs[r] for r in roles if r in truth]
    false_ps = [probs[r] for r in roles if r not in truth]

    true_mean = sum(true_ps) / len(true_ps) if true_ps else 0.0
    false_mean = sum(false_ps) / len(false_ps) if false_ps else 0.0

    separation = true_mean - false_mean

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "bce": bce,
        "brier": brier,
        "mean_true_rank": mean_true_rank,
        "recall_at_5": recall_at_5,
        "all_true_top5": all_true_top5,
        "separation": separation,
        "predicted_roles": sorted(pred),
        "ranking": ranked,
        "true_role_ranks": {
            r: rank_map[r]
            for r in sorted(truth)
            if r in rank_map
        },
    }


def state_after(selected, roles, truth, method, prior, threshold):
    probs = infer_probabilities(
        selected=selected,
        roles=roles,
        method=method,
        prior=prior,
    )

    m = evaluate(
        probs=probs,
        true_roles=truth,
        threshold=threshold,
    )

    return probs, m


def replay_curve(history, roles, truth, method, prior, threshold):
    selected = []
    curve = []

    for step, h in enumerate(history, start=1):
        selected.append(h)

        probs, m = state_after(
            selected,
            roles,
            truth,
            method,
            prior,
            threshold,
        )

        curve.append({
            "step": step,
            "chosen_role": h["role"],
            "chosen_query": h.get("query"),
            "chosen_probability": h.get("combined_probability"),
            **m,
        })

    return curve


def oracle_curve(history, roles, truth, method, prior, threshold):
    """
    Clairvoyant oracle.

    It is intentionally allowed to know the labels AND the realized replay
    answer. This is an upper bound, not a deployable policy.

    Primary objective:
        minimize multilabel BCE after the next probe.

    Tie-breaks:
        maximize F1
        maximize recall
        maximize truth-vs-false separation
        minimize false positives
    """

    remaining = list(range(len(history)))
    selected = []
    curve = []

    for step in range(1, len(history) + 1):
        best_idx = None
        best_metrics = None
        best_probs = None
        best_key = None

        for idx in remaining:
            trial = selected + [history[idx]]

            probs, m = state_after(
                trial,
                roles,
                truth,
                method,
                prior,
                threshold,
            )

            key = (
                -m["bce"],
                m["f1"],
                m["recall"],
                m["separation"],
                -m["fp"],
            )

            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx
                best_metrics = m
                best_probs = probs

        h = history[best_idx]
        selected.append(h)
        remaining.remove(best_idx)

        curve.append({
            "step": step,
            "chosen_original_index": h["_original_index"],
            "chosen_role": h["role"],
            "chosen_query": h.get("query"),
            "chosen_probability": h.get("combined_probability"),
            **best_metrics,
        })

    return curve


def random_curves(
    history,
    roles,
    truth,
    method,
    prior,
    threshold,
    trials,
    seed,
):
    rng = random.Random(seed)

    sums = {}
    sums_sq_f1 = {}

    for trial in range(trials):
        perm = list(history)
        rng.shuffle(perm)

        curve = replay_curve(
            perm,
            roles,
            truth,
            method,
            prior,
            threshold,
        )

        for row in curve:
            step = row["step"]

            if step not in sums:
                sums[step] = {
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "bce": 0.0,
                    "brier": 0.0,
                    "mean_true_rank": 0.0,
                    "recall_at_5": 0.0,
                    "separation": 0.0,
                }
                sums_sq_f1[step] = 0.0

            for k in sums[step]:
                sums[step][k] += row[k]

            sums_sq_f1[step] += row["f1"] ** 2

    out = []

    for step in sorted(sums):
        row = {"step": step}

        for k, v in sums[step].items():
            row[k] = v / trials

        mean_f1 = row["f1"]
        variance = max(
            0.0,
            sums_sq_f1[step] / trials - mean_f1**2,
        )

        row["f1_std"] = math.sqrt(variance)
        out.append(row)

    return out


def first_step_reaching(curve, key, target):
    for row in curve:
        if row[key] >= target:
            return row["step"]
    return None


def curve_area(curve, key="f1"):
    if not curve:
        return 0.0
    return sum(row[key] for row in curve) / len(curve)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--replay",
        default="artifacts/v13_2_pilot_replay_resume/replay.json",
    )

    ap.add_argument(
        "--output-dir",
        default="artifacts/v13_3a_oracle",
    )

    ap.add_argument(
        "--prior",
        type=float,
        default=0.10,
        help="Probability assigned to roles not probed yet.",
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.50,
    )

    ap.add_argument(
        "--random-trials",
        type=int,
        default=200,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=1333,
    )

    args = ap.parse_args()

    replay_path = Path(args.replay)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with replay_path.open() as f:
        replay = json.load(f)

    aggregators = [
        "median",
        "max",
        "top2",
        "top3",
        "p75",
    ]

    all_results = {
        "protocol": "v13.3a_offline_clairvoyant_oracle",
        "api_calls": 0,
        "v11_modified": False,
        "v13_2_modified": False,
        "replay_source": str(replay_path),
        "prior": args.prior,
        "threshold": args.threshold,
        "random_trials": args.random_trials,
        "systems": [],
    }

    print()
    print("=" * 90)
    print("V13.3a OFFLINE ORACLE")
    print("API CALLS: 0")
    print("V11: untouched")
    print("V13.2: untouched")
    print("=" * 90)

    for si, system in enumerate(replay.get("systems", [])):
        sid = system.get("system_id", f"system_{si}")
        truth = set(system.get("true_roles_eval_only", []))
        roles = get_roles(system)
        history = usable_history(system)

        print()
        print("=" * 90)
        print(f"SYSTEM: {sid}")
        print(f"candidate roles: {len(roles)}")
        print(f"true roles: {sorted(truth)}")
        print(f"available replay probes: {len(history)}")
        print("=" * 90)

        sys_result = {
            "system_id": sid,
            "true_roles": sorted(truth),
            "candidate_roles": roles,
            "n_available_probes": len(history),
            "aggregators": {},
        }

        for method in aggregators:
            full_probs, full_metrics = state_after(
                history,
                roles,
                truth,
                method,
                args.prior,
                args.threshold,
            )

            replay_order = replay_curve(
                history,
                roles,
                truth,
                method,
                args.prior,
                args.threshold,
            )

            oracle = oracle_curve(
                history,
                roles,
                truth,
                method,
                args.prior,
                args.threshold,
            )

            random_curve = random_curves(
                history,
                roles,
                truth,
                method,
                args.prior,
                args.threshold,
                trials=args.random_trials,
                seed=args.seed,
            )

            sys_result["aggregators"][method] = {
                "full_data": {
                    "probabilities": full_probs,
                    **full_metrics,
                },
                "replay_order_curve": replay_order,
                "oracle_curve": oracle,
                "random_curve_mean": random_curve,
                "oracle_mean_f1_over_budget": curve_area(oracle),
                "replay_mean_f1_over_budget": curve_area(replay_order),
                "oracle_first_f1_0_50": first_step_reaching(
                    oracle, "f1", 0.50
                ),
                "oracle_first_f1_0_67": first_step_reaching(
                    oracle, "f1", 0.67
                ),
                "oracle_first_f1_0_80": first_step_reaching(
                    oracle, "f1", 0.80
                ),
                "oracle_first_f1_1_00": first_step_reaching(
                    oracle, "f1", 0.999999
                ),
            }

            print()
            print(f"--- AGGREGATOR: {method} ---")
            print(
                "FULL DATA:"
                f" F1={full_metrics['f1']:.3f}"
                f" P={full_metrics['precision']:.3f}"
                f" R={full_metrics['recall']:.3f}"
                f" mean_true_rank={full_metrics['mean_true_rank']:.2f}"
            )
            print("predicted:", full_metrics["predicted_roles"])
            print("true ranks:", full_metrics["true_role_ranks"])

            print(
                "ORACLE first budget:"
                f" F1>=.50: {first_step_reaching(oracle, 'f1', .50)}"
                f" | >=.67: {first_step_reaching(oracle, 'f1', .67)}"
                f" | >=.80: {first_step_reaching(oracle, 'f1', .80)}"
                f" | =1.00: {first_step_reaching(oracle, 'f1', .999999)}"
            )

        all_results["systems"].append(sys_result)

        budgets = [
            b for b in [5, 10, 15, 20, 30, 40, 60, 80, 120]
            if b <= len(history)
        ]

        print()
        print("=" * 90)
        print("BUDGET COMPARISON")
        print("F1: oracle / replay-order / random-mean")
        print("=" * 90)

        for method in aggregators:
            d = sys_result["aggregators"][method]

            print()
            print(method.upper())

            for b in budgets:
                o = d["oracle_curve"][b - 1]["f1"]
                r = d["replay_order_curve"][b - 1]["f1"]
                z = d["random_curve_mean"][b - 1]["f1"]
                sd = d["random_curve_mean"][b - 1]["f1_std"]

                print(
                    f"  budget={b:3d}"
                    f"  oracle={o:.3f}"
                    f"  replay={r:.3f}"
                    f"  random={z:.3f}±{sd:.3f}"
                )

        # Print first oracle actions for each aggregator.
        print()
        print("=" * 90)
        print("FIRST 15 ORACLE ACTIONS")
        print("=" * 90)

        for method in aggregators:
            print()
            print(f"[{method}]")

            curve = sys_result["aggregators"][method]["oracle_curve"]

            for row in curve[:15]:
                q = (row.get("chosen_query") or "").replace("\n", " ")

                if len(q) > 100:
                    q = q[:97] + "..."

                print(
                    f"{row['step']:02d}. "
                    f"{row['chosen_role']:26s} "
                    f"obs_p={row['chosen_probability']:.3f} "
                    f"F1={row['f1']:.3f} "
                    f"BCE={row['bce']:.3f} "
                    f"| {q}"
                )

    json_path = outdir / "summary.json"

    with json_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    # Compact CSV with all curves.
    csv_path = outdir / "curves.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "system_id",
            "aggregator",
            "strategy",
            "step",
            "f1",
            "precision",
            "recall",
            "bce",
            "brier",
            "mean_true_rank",
            "recall_at_5",
            "separation",
        ])

        for s in all_results["systems"]:
            sid = s["system_id"]

            for method, d in s["aggregators"].items():
                for strategy, key in [
                    ("oracle", "oracle_curve"),
                    ("replay_order", "replay_order_curve"),
                    ("random_mean", "random_curve_mean"),
                ]:
                    for row in d[key]:
                        writer.writerow([
                            sid,
                            method,
                            strategy,
                            row["step"],
                            row["f1"],
                            row["precision"],
                            row["recall"],
                            row["bce"],
                            row["brier"],
                            row["mean_true_rank"],
                            row["recall_at_5"],
                            row["separation"],
                        ])

    print()
    print("=" * 90)
    print("DONE")
    print(f"summary: {json_path}")
    print(f"curves : {csv_path}")
    print("API CALLS USED: 0")
    print("=" * 90)


if __name__ == "__main__":
    main()
