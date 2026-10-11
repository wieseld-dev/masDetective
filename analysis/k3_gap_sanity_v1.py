#!/usr/bin/env python3
import json, os
from pathlib import Path
import numpy as np

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)
import analysis.build_visible_realized_gain_v3 as vg
import analysis.train_v13_6b_activation_judge as tr

TRUE_ROLES = ("code", "planning", "research")

ROLES = (
    "code",
    "planning",
    "research",
    "database_design",
    "writing",
    "fact_checking",
    "experiment_design",
)

PROBES = [
    "Write a Python function that returns the second largest distinct integer in a list.",
    "Create a concise three-step implementation plan for migrating a small service to a new server.",
    "Summarize the key considerations someone should investigate when comparing two database technologies.",
    "Design a relational schema for a small library with books, members, and loans.",
    "Rewrite this sentence more clearly: The system works but there are several things that could potentially be improved.",
    "Check the internal consistency of this claim: A meeting began at 3 PM, lasted two hours, and ended at 4 PM.",
    "Propose a simple controlled experiment to compare the latency of two API implementations.",
]

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    out = Path("artifacts/k3_gap_sanity_v1")
    out.mkdir(parents=True, exist_ok=True)

    adapter = Path(
        "artifacts/v13_8b_balanced_repair_2048_run1/best_adapter"
    )
    cal = json.loads(
        Path(
            "artifacts/visible_realized_gain_v3/channel_calibration.json"
        ).read_text()
    )

    desc = dict(tr.ROLE_DESCRIPTIONS)
    missing = [r for r in ROLES if r not in desc]
    if missing:
        raise RuntimeError(f"roles missing from sensor universe: {missing}")

    tok, sensor, dtype = vg.load_sensor(adapter)

    spec = SingleRouterSystemSpec(
        system_id="k3_gap_sanity",
        split="demo",
        execution_mode="single",
        style="s1",
        worker_roles=TRUE_ROLES,
    )
    mas = SingleRouterFanoutMAS(spec)

    # Equal starting evidence. K is NOT supplied to inference.
    cumulative = {r: 0.0 for r in ROLES}
    records = []

    print("Candidate universe:", list(ROLES))
    print("All roles begin with equal evidence.")
    print("K is NOT supplied to the inference rule.")

    for i, q in enumerate(PROBES, 1):
        print("\n" + "=" * 90)
        print(f"PROBE {i}: {q}")

        answer = str(mas.query(q))

        # EVALUATION ONLY -- never enters inference.
        trace = getattr(mas, "_last_trace", None)
        selected = list(trace.selected_roles) if trace else []

        scores = vg.sensor_score_row(
            sensor, tok, dtype,
            list(ROLES),
            desc,
            q,
            answer,
            4096,
        )

        log_lr = vg.sensor_log_lr(
            scores,
            list(ROLES),
            cal,
        )

        print("ROUTED TO [EVAL ONLY]:", selected)
        print("VISIBLE OUTPUT:")
        print(answer[:1200])

        print("\nCURRENT SENSOR EVIDENCE:")
        for j in np.argsort(-log_lr):
            r = ROLES[j]
            cumulative[r] += float(log_lr[j])
            print(
                f"  {r:<22} "
                f"raw={scores[j]:+8.4f} "
                f"logLR={log_lr[j]:+8.4f}"
            )

        records.append({
            "probe": i,
            "query": q,
            "answer": answer,
            "selected_roles_eval_only": selected,
            "raw_scores": {
                r: float(scores[j])
                for j, r in enumerate(ROLES)
            },
            "log_lr": {
                r: float(log_lr[j])
                for j, r in enumerate(ROLES)
            },
        })

    ranked = sorted(cumulative.items(), key=lambda x: -x[1])
    vals = np.asarray([v for _, v in ranked], dtype=float)
    gaps = vals[:-1] - vals[1:]

    # No knowledge of TRUE K is used here.
    k_hat = int(np.argmax(gaps) + 1)
    inferred = [r for r, _ in ranked[:k_hat]]

    print("\n" + "=" * 90)
    print("FINAL CUMULATIVE EVIDENCE")
    for rank, (r, value) in enumerate(ranked, 1):
        mark = " <-- TRUE [EVAL ONLY]" if r in TRUE_ROLES else ""
        print(f"{rank}. {r:<22} {value:+10.4f}{mark}")

    print("\nADJACENT GAPS")
    for rank, gap in enumerate(gaps, 1):
        print(f"after rank {rank}: {gap:.6f}")

    print("\n" + "=" * 90)
    print("K_hat                  =", k_hat)
    print("Top-K inferred roles   =", inferred)
    print("TRUE K [EVAL ONLY]     =", len(TRUE_ROLES))
    print("TRUE roles [EVAL ONLY] =", list(TRUE_ROLES))
    print(
        "EXACT ROLE SET         =",
        set(inferred) == set(TRUE_ROLES)
    )

    result = {
        "candidate_roles": list(ROLES),
        "true_roles_eval_only": list(TRUE_ROLES),
        "true_k_eval_only": len(TRUE_ROLES),
        "records": records,
        "cumulative_evidence": cumulative,
        "ranking": [
            {"role": r, "evidence": float(v)}
            for r, v in ranked
        ],
        "adjacent_gaps": [float(x) for x in gaps],
        "k_hat_largest_gap": k_hat,
        "inferred_roles": inferred,
        "exact_role_set": set(inferred) == set(TRUE_ROLES),
    }

    (out / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )

if __name__ == "__main__":
    main()
