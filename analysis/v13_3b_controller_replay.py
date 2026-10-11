#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


EPS = 1e-6


def clip(p):
    return min(1.0 - EPS, max(EPS, float(p)))


def logit(p):
    p = clip(p)
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def entropy(p):
    p = clip(p)
    return -(p * math.log(p) + (1-p) * math.log(1-p))


def metrics(probs, truth, threshold=0.5):
    pred = {r for r,p in probs.items() if p >= threshold}
    truth = set(truth)

    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)

    precision = tp/(tp+fp) if tp+fp else 0.0
    recall = tp/(tp+fn) if tp+fn else 0.0
    f1 = (
        2*precision*recall/(precision+recall)
        if precision+recall else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted": sorted(pred),
    }


def build_role_queues(history):
    q = defaultdict(list)

    for h in history:
        role = h.get("role")
        p = h.get("combined_probability")

        if role is None or p is None:
            continue

        q[role].append(h)

    return q


def update_belief(old_p, observation_p, weight=1.0):
    """
    Bayesian-like log-odds accumulation.

    The observation probability is treated as evidence,
    not as an absolute replacement for the current belief.
    """
    return sigmoid(
        logit(old_p)
        + weight * logit(observation_p)
    )


def role_priority(role, beliefs, counts, queues):
    """
    Deployable uncertainty heuristic.

    Does NOT inspect the future answer.
    """
    if not queues[role]:
        return -1e9

    p = beliefs[role]

    # High around 0.5.
    u = entropy(p)

    # Prefer roles we have spent less budget on.
    exploration_penalty = 0.08 * counts[role]

    return u - exploration_penalty


def stop_rule(
    beliefs,
    counts,
    total,
    *,
    min_total,
    positive_threshold,
    negative_threshold,
    max_uncertain,
):
    if total < min_total:
        return False, "minimum_budget"

    # A role that has never been probed is still unresolved.
    # Do NOT interpret the sparse prior itself as negative evidence.
    uncertain = [
        r for r,p in beliefs.items()
        if counts[r] == 0
        or negative_threshold < p < positive_threshold
    ]

    if len(uncertain) <= max_uncertain:
        return True, f"uncertain_roles={len(uncertain)}"

    return False, f"uncertain_roles={len(uncertain)}"


def simulate(
    system,
    *,
    prior,
    threshold,
    min_total,
    positive_threshold,
    negative_threshold,
    max_uncertain,
    max_budget,
):
    truth = set(system["true_roles_eval_only"])
    history = system["replay_history"]

    queues = build_role_queues(history)
    roles = sorted(queues)

    beliefs = {r: prior for r in roles}
    counts = {r: 0 for r in roles}

    trajectory = []

    for step in range(1, max_budget + 1):
        should_stop, reason = stop_rule(
            beliefs,
            counts,
            step-1,
            min_total=min_total,
            positive_threshold=positive_threshold,
            negative_threshold=negative_threshold,
            max_uncertain=max_uncertain,
        )

        if should_stop:
            return {
                "stopped": True,
                "stop_reason": reason,
                "steps": step-1,
                "beliefs": beliefs,
                "metrics": metrics(beliefs, truth, threshold),
                "trajectory": trajectory,
            }

        available = [
            r for r in roles
            if queues[r]
        ]

        if not available:
            break

        # Screening phase: before spending a second query on any role,
        # give every still-unseen role one opportunity.
        unseen = [r for r in available if counts[r] == 0]

        if unseen:
            role = max(
                unseen,
                key=lambda r: role_priority(
                    r, beliefs, counts, queues
                )
            )
        else:
            role = max(
                available,
                key=lambda r: role_priority(
                    r, beliefs, counts, queues
                )
            )

        h = queues[role].pop(0)

        obs_p = clip(h["combined_probability"])

        before = beliefs[role]
        after = update_belief(before, obs_p)

        beliefs[role] = after
        counts[role] += 1

        m = metrics(beliefs, truth, threshold)

        trajectory.append({
            "step": step,
            "role": role,
            "query": h.get("query"),
            "observation_probability": obs_p,
            "belief_before": before,
            "belief_after": after,
            "f1_eval_only": m["f1"],
            "precision_eval_only": m["precision"],
            "recall_eval_only": m["recall"],
        })

    return {
        "stopped": False,
        "stop_reason": "budget_or_exhausted",
        "steps": len(trajectory),
        "beliefs": beliefs,
        "metrics": metrics(beliefs, truth, threshold),
        "trajectory": trajectory,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--replay",
        default="artifacts/v13_2_pilot_replay_resume/replay.json",
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/v13_3b_controller",
    )

    ap.add_argument("--prior", type=float, default=0.15)
    ap.add_argument("--threshold", type=float, default=0.50)

    ap.add_argument("--min-total", type=int, default=5)

    ap.add_argument(
        "--positive-threshold",
        type=float,
        default=0.85,
    )
    ap.add_argument(
        "--negative-threshold",
        type=float,
        default=0.15,
    )

    ap.add_argument(
        "--max-uncertain",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--max-budget",
        type=int,
        default=120,
    )

    args = ap.parse_args()

    with open(args.replay) as f:
        data = json.load(f)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = {
        "protocol": "v13.3b_offline_uncertainty_controller",
        "api_calls": 0,
        "truth_used_for_action_selection": False,
        "future_answers_used_for_action_selection": False,
        "config": vars(args),
        "systems": [],
    }

    print("="*90)
    print("V13.3b OFFLINE UNCERTAINTY CONTROLLER")
    print("API CALLS = 0")
    print("Truth used for controller = NO")
    print("Future answers used for controller = NO")
    print("="*90)

    for s in data["systems"]:
        r = simulate(
            s,
            prior=args.prior,
            threshold=args.threshold,
            min_total=args.min_total,
            positive_threshold=args.positive_threshold,
            negative_threshold=args.negative_threshold,
            max_uncertain=args.max_uncertain,
            max_budget=args.max_budget,
        )

        row = {
            "system_id": s["system_id"],
            "truth_eval_only": s["true_roles_eval_only"],
            **r,
        }

        summary["systems"].append(row)

        print()
        print("SYSTEM:", s["system_id"])
        print("truth:", s["true_roles_eval_only"])
        print("steps:", r["steps"])
        print("stop:", r["stop_reason"])
        print("pred:", r["metrics"]["predicted"])
        print(
            "P={:.3f} R={:.3f} F1={:.3f}".format(
                r["metrics"]["precision"],
                r["metrics"]["recall"],
                r["metrics"]["f1"],
            )
        )

        print()
        print("TRAJECTORY:")

        for x in r["trajectory"]:
            q = (x.get("query") or "").replace("\n"," ")
            if len(q) > 80:
                q = q[:77] + "..."

            print(
                f"{x['step']:02d}. "
                f"{x['role']:26s} "
                f"obs={x['observation_probability']:.3f} "
                f"belief={x['belief_before']:.3f}"
                f"->{x['belief_after']:.3f} "
                f"F1(eval)={x['f1_eval_only']:.3f} "
                f"| {q}"
            )

    p = outdir / "summary.json"

    with p.open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("="*90)
    print("saved:", p)
    print("API CALLS USED: 0")
    print("="*90)


if __name__ == "__main__":
    main()
