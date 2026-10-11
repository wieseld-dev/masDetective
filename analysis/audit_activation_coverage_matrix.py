from __future__ import annotations

from argparse import Namespace
from collections import Counter
from pathlib import Path
import csv
import json

import analysis.train_v13_6_activation_judge as m


OUT = Path("artifacts/activation_coverage_matrix_v1")
OUT.mkdir(parents=True, exist_ok=True)


def iter_branches(c):
    yield "plus", c.plus_output, c.plus_selected
    yield "minus", c.minus_output, c.minus_selected
    yield "sham", c.sham_output, c.sham_selected


def build_counts(cases, roles):
    branch_counts = Counter()
    case_counts = Counter()
    target_case_counts = Counter()

    for c in cases:
        target = c.target

        if target not in roles:
            continue

        target_case_counts[target] += 1
        seen_pairs = set()

        for branch, output, selected in iter_branches(c):
            if not output:
                continue

            for role in list(selected or []):
                if role not in roles:
                    continue

                branch_counts[(target, role)] += 1
                seen_pairs.add((target, role))

        for pair in seen_pairs:
            case_counts[pair] += 1

    return branch_counts, case_counts, target_case_counts


def write_matrix(path, roles, counts):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["target"] + roles)

        for target in roles:
            w.writerow(
                [target] +
                [counts[(target, role)] for role in roles]
            )


def report_split(name, cases, roles):
    branch, case, target_cases = build_counts(cases, roles)

    write_matrix(
        OUT / f"{name}_branch_matrix.tsv",
        roles,
        branch,
    )

    write_matrix(
        OUT / f"{name}_case_matrix.tsv",
        roles,
        case,
    )

    offdiag = []
    zero_offdiag = []

    for target in roles:
        for selected in roles:
            if target == selected:
                continue

            b = branch[(target, selected)]
            c = case[(target, selected)]

            rec = {
                "target": target,
                "selected": selected,
                "branch_count": b,
                "case_count": c,
            }

            offdiag.append(rec)

            if b == 0:
                zero_offdiag.append(rec)

    nonzero_offdiag = [
        x for x in offdiag
        if x["branch_count"] > 0
    ]

    nonzero_offdiag.sort(
        key=lambda x: (
            x["branch_count"],
            x["case_count"],
            x["target"],
            x["selected"],
        )
    )

    summary = {
        "split": name,
        "n_cases": len(cases),
        "n_roles": len(roles),
        "offdiag_pairs_total": len(offdiag),
        "offdiag_pairs_zero": len(zero_offdiag),
        "offdiag_pairs_nonzero": len(nonzero_offdiag),
        "offdiag_coverage_fraction":
            len(nonzero_offdiag) / max(1, len(offdiag)),
        "target_case_counts": dict(target_cases),
        "zero_offdiag_pairs": zero_offdiag,
        "lowest_nonzero_offdiag_pairs": nonzero_offdiag[:100],
    }

    (OUT / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print(name.upper())
    print("=" * 110)

    print(f"cases={len(cases)}")
    print(
        f"offdiag coverage = "
        f"{len(nonzero_offdiag)}/{len(offdiag)} "
        f"({summary['offdiag_coverage_fraction']:.3f})"
    )
    print(f"zero offdiag pairs = {len(zero_offdiag)}")

    print("\nKEY PAIRS")
    for t, s in [
        ("database_design", "code"),
        ("math", "code"),
        ("incident_management", "code"),
        ("financial_analysis", "code"),
        ("data", "code"),
        ("code", "database_design"),
        ("code", "math"),
        ("code", "incident_management"),
    ]:
        print(
            f"{t:28s} -> {s:28s} "
            f"branches={branch[(t,s)]:3d} "
            f"cases={case[(t,s)]:3d}"
        )

    print("\nLOWEST NONZERO OFF-DIAGONAL PAIRS")
    for x in nonzero_offdiag[:40]:
        print(
            f"{x['target']:28s} -> "
            f"{x['selected']:28s} "
            f"branches={x['branch_count']:3d} "
            f"cases={x['case_count']:3d}"
        )

    print("\nZERO OFF-DIAGONAL PAIRS")
    for x in zero_offdiag[:100]:
        print(
            f"{x['target']:28s} -> "
            f"{x['selected']:28s}"
        )

    return branch, case, summary


def main():
    roles = list(m.ROLE_DESCRIPTIONS.keys())

    print("=" * 110)
    print("EXACT V13.6/V13.6B TARGET -> SELECTED COVERAGE MATRIX")
    print("=" * 110)
    print(f"roles={len(roles)}")

    r13 = m.read_jsonl(
        "artifacts/v13_probe_disambiguation.jsonl"
    )

    r11 = m.read_jsonl(
        "artifacts/v11_probe_discovery.jsonl"
    )

    c13, _ = m.normalize_rows(
        r13,
        "v13",
        dict(m.ROLE_DESCRIPTIONS),
        1.0,
    )

    c11, _ = m.normalize_rows(
        r11,
        "v11",
        dict(m.ROLE_DESCRIPTIONS),
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

    print(
        f"all={len(cases)} "
        f"train={len(train_cases)} "
        f"val={len(val_cases)} "
        f"v13={len(c13)} "
        f"v11={len(c11)}"
    )

    _, _, train_summary = report_split(
        "train",
        train_cases,
        roles,
    )

    _, _, val_summary = report_split(
        "val",
        val_cases,
        roles,
    )

    _, _, all_summary = report_split(
        "all",
        cases,
        roles,
    )

    combined = {
        "train": train_summary,
        "val": val_summary,
        "all": all_summary,
    }

    (OUT / "summary.json").write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("FILES")
    print("=" * 110)

    for p in sorted(OUT.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
