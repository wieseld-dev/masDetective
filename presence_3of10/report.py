from __future__ import annotations

import argparse
import json
from pathlib import Path

from v11.role_universe import TRAIN_ROLES
from presence_3of10.config import POOL, HIDDEN


def read_one(path):
    rows = [
        json.loads(x)
        for x in Path(path).open(encoding="utf-8")
        if x.strip()
    ]

    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one row")

    return rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--horizon", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.root)
    truth = set(HIDDEN)

    lines = []

    lines += [
        "PRESENCE 3-of-10 — TRAINED ADAPTIVE V11",
        "========================================",
        "",
        "Candidate pool:",
        "  " + ", ".join(POOL),
        "",
        "Hidden truth [evaluation only]:",
        "  " + ", ".join(HIDDEN),
        "",
    ]

    exact_rounds = []

    for r in range(args.horizon + 1):
        st = read_one(
            root / f"states_dev_round_{r:02d}.jsonl"
        )

        probs = st["role_probs"]

        ranked = sorted(
            POOL,
            key=lambda x: float(probs[x]),
            reverse=True,
        )

        top3 = ranked[:3]
        exact = set(top3) == truth
        recall = len(set(top3) & truth) / 3.0

        tp = sum(float(probs[x]) for x in HIDDEN) / 3.0

        negatives = [x for x in POOL if x not in truth]
        fp = (
            sum(float(probs[x]) for x in negatives)
            / len(negatives)
        )

        strict20 = sorted(
            TRAIN_ROLES,
            key=lambda x: float(probs[x]),
            reverse=True,
        )[:3]

        if exact:
            exact_rounds.append(r)

        lines.append("=" * 78)
        lines.append(f"ROUND {r}")

        if r > 0:
            c = read_one(
                root / f"candidates_dev_round_{r:02d}.jsonl"
            )
            lines.append(f"PROBE: {c['query']}")

        lines.append(
            f"Top3(pool)={top3} "
            f"recall@3={recall:.3f} "
            f"exact@3={exact}"
        )

        lines.append(
            f"true_mean={tp:.4f} "
            f"false_mean={fp:.4f} "
            f"separation={tp-fp:.4f}"
        )

        lines.append(
            f"strict_top3_all20={strict20}"
        )

        lines.append("")
        lines.append(
            f"{'rank':<5}{'role':<28}{'p':>9}  truth"
        )

        for i, role in enumerate(ranked, 1):
            mark = "*" if role in truth else ""
            lines.append(
                f"{i:<5}{role:<28}"
                f"{float(probs[role]):>9.4f}  {mark}"
            )

        lines.append("")

    lines += [
        "=" * 78,
        "FINAL",
        f"final_top3={sorted(POOL, key=lambda x: read_one(root / f'states_dev_round_{args.horizon:02d}.jsonl')['role_probs'][x], reverse=True)[:3]}",
        (
            "first_exact_round="
            + (
                str(min(exact_rounds))
                if exact_rounds
                else "NEVER"
            )
        ),
    ]

    report = "\n".join(lines) + "\n"

    (root / "REPORT.txt").write_text(
        report,
        encoding="utf-8",
    )

    print(report)


if __name__ == "__main__":
    main()
