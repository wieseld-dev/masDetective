from __future__ import annotations

from argparse import Namespace
from collections import Counter
from pathlib import Path

import analysis.train_v13_6_activation_judge as m


KEY_PAIRS = [
    ("database_design", "code"),
    ("database_design", "database_design"),
    ("math", "code"),
    ("math", "math"),
    ("incident_management", "code"),
    ("incident_management", "incident_management"),
    ("financial_analysis", "financial_analysis"),
]


def iter_branches(c):
    yield "plus", c.plus_output, c.plus_selected
    yield "minus", c.minus_output, c.minus_selected
    yield "sham", c.sham_output, c.sham_selected


def audit(name, cases):
    pair_counts = Counter()
    target_counts = Counter()
    mismatch_counts = Counter()

    examples = []

    for c in cases:
        target = c.target
        target_counts[target] += 1

        for branch, output, selected in iter_branches(c):
            if not output:
                continue

            selected = list(selected or [])

            for role in selected:
                pair_counts[(target, role)] += 1

                if target != role:
                    mismatch_counts[(target, role)] += 1

            if (
                target == "database_design"
                and "code" in selected
            ):
                examples.append({
                    "source": c.source,
                    "branch": branch,
                    "query": c.query,
                    "target": target,
                    "selected": selected,
                    "output": output,
                })

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)
    print(f"cases={len(cases)}")

    print("\nTARGET COUNTS")
    for t in [
        "database_design",
        "math",
        "incident_management",
        "financial_analysis",
    ]:
        print(f"{t:28s} {target_counts[t]}")

    print("\nIMPORTANT TARGET -> SELECTED")
    for t, s in KEY_PAIRS:
        print(
            f"{t:28s} -> {s:28s} "
            f"{pair_counts[(t, s)]}"
        )

    print("\nTOP MISMATCHES")
    for (t, s), n in mismatch_counts.most_common(30):
        print(
            f"{t:28s} -> {s:28s} {n}"
        )

    print("\nDATABASE_DESIGN -> CODE EXAMPLES")
    print(f"count={len(examples)}")

    for i, e in enumerate(examples[:15], 1):
        print()
        print(f"--- EXAMPLE {i} ---")
        print("SOURCE :", e["source"])
        print("BRANCH :", e["branch"])
        print("QUERY  :", e["query"])
        print("SELECT :", e["selected"])
        print("OUTPUT :", e["output"][:1200])


def main():
    rd = dict(m.ROLE_DESCRIPTIONS)

    r13 = m.read_jsonl(
        "artifacts/v13_probe_disambiguation.jsonl"
    )

    r11 = m.read_jsonl(
        "artifacts/v11_probe_discovery.jsonl"
    )

    c13, p13 = m.normalize_rows(
        r13,
        "v13",
        rd,
        1.0,
    )

    c11, p11 = m.normalize_rows(
        r11,
        "v11",
        rd,
        0.70,
    )

    cases = m.attach_components(c13 + c11)

    args = Namespace(
        seed=1331,
        n_splits=5,
        fold=0,
    )

    train_cases, val_cases, split_report = m.split_cases(
        cases,
        args,
    )

    print("=" * 100)
    print("EXACT V13.6 TRAINING-DATA COVERAGE AUDIT")
    print("=" * 100)

    print(
        f"all={len(cases)} "
        f"v13={len(c13)} "
        f"v11={len(c11)} "
        f"train={len(train_cases)} "
        f"val={len(val_cases)}"
    )

    audit("TRAIN SPLIT", train_cases)
    audit("VAL SPLIT", val_cases)
    audit("ALL CASES", cases)


if __name__ == "__main__":
    main()
