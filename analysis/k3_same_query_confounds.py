#!/usr/bin/env python3
import inspect
import json
import os
from pathlib import Path

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)

# IMPORTANT: side-effect import registers V11 EXTRA_ROLE_PROMPTS.
import targets.single_router_zoo_v11
import analysis.build_visible_realized_gain_v3 as vg
import analysis.train_v13_6b_activation_judge as tr

TEST_ROLES = (
    "code",
    "planning",
    "research",
    "database_design",
    "writing",
    "fact_checking",
    "experiment_design",
)

CASES = [
    (
        "database_vs_planning",
        "Design a normalized relational schema for a library with books, members, and loans.",
        "planning",
        "database_design",
    ),
    (
        "factcheck_vs_research",
        "Check the internal consistency of this claim: A meeting began at 3 PM, lasted two hours, and ended at 4 PM.",
        "research",
        "fact_checking",
    ),
    (
        "experiment_vs_research",
        "Propose a simple controlled experiment to compare the latency of two API implementations.",
        "research",
        "experiment_design",
    ),
    (
        "writing_vs_research",
        "Rewrite this sentence more clearly: The system works but there are several things that could potentially be improved.",
        "research",
        "writing",
    ),
    (
        "code_vs_planning",
        "Write a Python function that returns the second largest distinct integer in a list.",
        "code",
        "planning",
    ),
]

def call_worker(mas, role, query):
    matches = [
        wid for wid, r in mas.role_by_worker.items()
        if r == role
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one worker for role={role}, "
            f"got {matches}; role_by_worker={mas.role_by_worker}"
        )

    worker_id = matches[0]
    return str(mas._worker_answer(worker_id, query))

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    out = Path("artifacts/k3_same_query_confounds")
    out.mkdir(parents=True, exist_ok=True)

    desc = dict(tr.ROLE_DESCRIPTIONS)
    full_roles = list(desc.keys())

    cal = json.loads(
        Path("artifacts/visible_realized_gain_v3/channel_calibration.json").read_text()
    )
    tok, sensor, dtype = vg.load_sensor(
        Path("artifacts/v13_8b_balanced_repair_2048_run1/best_adapter")
    )

    # Includes every role we may force during this evaluation-only diagnostic.
    spec = SingleRouterSystemSpec(
        system_id="same_query_confounds",
        split="diagnostic",
        execution_mode="single",
        style="s1",
        worker_roles=TEST_ROLES,
    )
    mas = SingleRouterFanoutMAS(spec)

    rows = []
    pair_lines = []
    source_top1_correct = 0
    total = 0

    for family, query, role_a, role_b in CASES:
        print("\n" + "="*100)
        print("CASE:", family)
        print("QUERY:", query)

        for source_role, competitor in [(role_a, role_b), (role_b, role_a)]:
            print("\nFORCED SOURCE [EVAL ONLY]:", source_role)

            answer = call_worker(mas, source_role, query)

            scores = vg.sensor_score_row(
                sensor, tok, dtype,
                full_roles, desc, query, answer, 4096
            )
            probs = vg.calibrated_probs(scores, full_roles, cal)
            lrs = vg.sensor_log_lr(scores, full_roles, cal)

            prob_map = dict(zip(full_roles, map(float, probs)))
            lr_map = dict(zip(full_roles, map(float, lrs)))

            ranked = sorted(TEST_ROLES, key=lambda r: -lr_map[r])
            top1 = ranked[0]

            total += 1
            source_top1_correct += int(top1 == source_role)

            margin_vs_competitor = (
                lr_map[source_role] - lr_map[competitor]
            )

            print("SENSOR TOP1:", top1)
            print(
                f"SOURCE {source_role}: "
                f"p={prob_map[source_role]:.4f} "
                f"logLR={lr_map[source_role]:+.4f}"
            )
            print(
                f"CONFUSER {competitor}: "
                f"p={prob_map[competitor]:.4f} "
                f"logLR={lr_map[competitor]:+.4f}"
            )
            print(
                "SOURCE-CONFOUNDER MARGIN:",
                f"{margin_vs_competitor:+.4f}"
            )

            rows.append({
                "family": family,
                "query": query,
                "forced_source_eval_only": source_role,
                "competitor": competitor,
                "answer": answer,
                "sensor_top1": top1,
                "source_rank": ranked.index(source_role)+1,
                "source_loglr": lr_map[source_role],
                "competitor_loglr": lr_map[competitor],
                "margin": margin_vs_competitor,
                "loglr_7": {r:lr_map[r] for r in TEST_ROLES},
            })

            pair_lines.append({
                "pair_type": "same_query_different_worker",
                "family": family,
                "query": query,
                "answer": answer,
                "positive_role": source_role,
                "negative_role": competitor,
                "positive_label": 1,
                "negative_label": 0,
                "source_eval_only": source_role,
            })

    acc = source_top1_correct / max(total,1)

    print("\n" + "#"*100)
    print(
        f"FORCED-WORKER SOURCE TOP1 ACC: "
        f"{source_top1_correct}/{total} = {acc:.4f}"
    )

    for x in rows:
        print(
            f"{x['family']:<28} "
            f"source={x['forced_source_eval_only']:<18} "
            f"competitor={x['competitor']:<18} "
            f"margin={x['margin']:+.4f} "
            f"top1={x['sensor_top1']}"
        )

    (out/"summary.json").write_text(
        json.dumps({
            "protocol": "same_query_different_forced_worker",
            "n_worker_calls": total,
            "source_top1_correct": source_top1_correct,
            "source_top1_total": total,
            "source_top1_accuracy": acc,
            "rows": rows,
        }, indent=2, ensure_ascii=False)
    )

    with (out/"hard_pairs.jsonl").open("w") as f:
        for x in pair_lines:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
