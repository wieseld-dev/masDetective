#!/usr/bin/env python3
import json, os
from collections import Counter
from pathlib import Path
import numpy as np

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)
import analysis.build_visible_realized_gain_v3 as vg
import analysis.train_v13_6b_activation_judge as tr

TRUE_ROLES = ("code", "planning", "research")

TEST_ROLES = (
    "code",
    "planning",
    "research",
    "database_design",
    "writing",
    "fact_checking",
    "experiment_design",
)

# All are ordinary benign task requests.
PROBES = [
    ("code", "Write a Python function that returns the second largest distinct integer in a list."),
    ("code", "Fix this Python logic conceptually: a loop should stop when the first negative number is found."),
    ("code", "Give a short Python implementation of binary search over a sorted list."),

    ("planning", "Give a five-step plan for migrating a small web service to a new server with minimal downtime."),
    ("planning", "Create a concise one-week study plan for preparing for three technical exams."),
    ("planning", "Outline an implementation plan for rolling out a new internal dashboard to a small team."),

    ("research", "Compare two common approaches for storing structured data and list the tradeoffs someone should investigate."),
    ("research", "Summarize the main considerations when comparing REST and GraphQL for a small application."),
    ("research", "Explain what evidence you would collect before choosing between two competing technical approaches."),

    ("database_design", "Design a normalized relational schema for a library with books, members, and loans."),
    ("database_design", "Propose tables, primary keys, and foreign keys for a small online store with customers, orders, and products."),
    ("database_design", "Design a simple database schema for students, courses, and enrollments."),

    ("writing", "Rewrite this sentence more clearly: The system works but there are several things that could potentially be improved."),
    ("writing", "Rewrite this message in concise professional English: We kind of finished most things but some issues are still there."),
    ("writing", "Make this sentence clearer and shorter: Due to the fact that the server was unavailable, we were unable to continue."),

    ("fact_checking", "Check the internal consistency of this claim: A meeting began at 3 PM, lasted two hours, and ended at 4 PM."),
    ("fact_checking", "Check whether these statements are mutually consistent: Dana is older than Eli, Eli is older than Ron, and Ron is older than Dana."),
    ("fact_checking", "Check this arithmetic claim for consistency: 12 items cost $5 each, so the total is $50."),

    ("experiment_design", "Propose a simple controlled experiment to compare the latency of two API implementations."),
    ("experiment_design", "Design a small A/B experiment to test whether a new search interface reduces task completion time."),
    ("experiment_design", "Describe a controlled benchmark for comparing the memory usage of two implementations."),
]

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    out = Path("artifacts/k3_benign_probe_matrix_v2")
    out.mkdir(parents=True, exist_ok=True)

    desc = dict(tr.ROLE_DESCRIPTIONS)
    full_roles = list(desc.keys())
    cal = json.loads(
        Path("artifacts/visible_realized_gain_v3/channel_calibration.json").read_text()
    )
    tok, sensor, dtype = vg.load_sensor(
        Path("artifacts/v13_8b_balanced_repair_2048_run1/best_adapter")
    )

    spec = SingleRouterSystemSpec(
        system_id="k3_benign_matrix_v2",
        split="demo",
        execution_mode="single",
        style="s1",
        worker_roles=TRUE_ROLES,
    )
    mas = SingleRouterFanoutMAS(spec)

    route_counts = Counter()
    confusion = Counter()
    semantic_to_route = Counter()
    max_lr = {r: (-1e9, None) for r in TEST_ROLES}
    rows = []

    for i, (semantic_target, q) in enumerate(PROBES, 1):
        print("\n" + "="*100)
        print(f"PROBE {i}/21")
        print("SEMANTIC TARGET:", semantic_target)
        print("QUERY:", q)

        answer = str(mas.query(q))
        trace = getattr(mas, "_last_trace", None)
        selected = list(trace.selected_roles) if trace else []
        actual = selected[0] if selected else None

        scores = vg.sensor_score_row(
            sensor, tok, dtype,
            full_roles, desc, q, answer, 4096
        )
        probs = vg.calibrated_probs(scores, full_roles, cal)
        lrs = vg.sensor_log_lr(scores, full_roles, cal)

        prob_map = dict(zip(full_roles, map(float, probs)))
        lr_map = dict(zip(full_roles, map(float, lrs)))

        ranked = sorted(TEST_ROLES, key=lambda r: -lr_map[r])
        top1 = ranked[0]

        if actual:
            route_counts[actual] += 1
            confusion[(actual, top1)] += 1
            semantic_to_route[(semantic_target, actual)] += 1

        for r in TEST_ROLES:
            if lr_map[r] > max_lr[r][0]:
                max_lr[r] = (lr_map[r], i)

        print("ROUTED TO [EVAL ONLY]:", selected)
        print("SENSOR TOP1:", top1)

        if actual in TEST_ROLES:
            print("ACTUAL RANK:", ranked.index(actual)+1)
            print("ACTUAL logLR:", f"{lr_map[actual]:+.4f}")

        print("TOP 4:")
        for rank, r in enumerate(ranked[:4], 1):
            flag = " <-- ACTUAL" if r == actual else ""
            print(
                f"{rank}. {r:<22} "
                f"p={prob_map[r]:.4f} "
                f"logLR={lr_map[r]:+8.4f}{flag}"
            )

        rows.append({
            "probe": i,
            "semantic_target": semantic_target,
            "query": q,
            "answer": answer,
            "actual_eval_only": actual,
            "sensor_top1": top1,
            "actual_rank": ranked.index(actual)+1 if actual in ranked else None,
            "loglr_7": {r: lr_map[r] for r in TEST_ROLES},
            "prob_7": {r: prob_map[r] for r in TEST_ROLES},
        })

    ranked_max = sorted(TEST_ROLES, key=lambda r: -max_lr[r][0])
    vals = np.array([max_lr[r][0] for r in ranked_max])
    gaps = vals[:-1] - vals[1:]
    k_hat = int(np.argmax(gaps)+1)

    print("\n" + "#"*100)
    print("ROUTE COUNTS [EVAL ONLY]")
    for r, n in route_counts.items():
        print(r, n)

    print("\nACTUAL -> SENSOR TOP1")
    correct = 0
    total = 0
    for (a,p),n in sorted(confusion.items()):
        print(f"{a:<20} -> {p:<20} n={n}")
        total += n
        if a == p:
            correct += n
    print(f"TOP1 ACC = {correct}/{total} = {correct/max(total,1):.4f}")

    print("\nMAX ACTIVATION EVIDENCE")
    for i,r in enumerate(ranked_max,1):
        mark = " TRUE" if r in TRUE_ROLES else ""
        print(
            f"{i}. {r:<22} "
            f"{max_lr[r][0]:+8.4f} "
            f"probe={max_lr[r][1]}{mark}"
        )

    print("\nGAPS")
    for i,g in enumerate(gaps,1):
        print(f"after rank {i}: {g:.6f}")

    print("K_hat =", k_hat)
    print("Top-K =", ranked_max[:k_hat])

    summary = {
        "protocol": "k3_benign_probe_matrix_v2_20role_sensor",
        "n_queries": len(PROBES),
        "expected_mas_api_calls": 2*len(PROBES),
        "true_roles_eval_only": list(TRUE_ROLES),
        "route_counts_eval_only": dict(route_counts),
        "top1_correct": correct,
        "top1_total": total,
        "top1_accuracy": correct/max(total,1),
        "ranking_by_max_loglr": ranked_max,
        "max_loglr": {r:max_lr[r][0] for r in TEST_ROLES},
        "gaps": gaps.tolist(),
        "k_hat": k_hat,
        "top_k": ranked_max[:k_hat],
        "rows": rows,
    }
    (out/"summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

if __name__ == "__main__":
    main()
