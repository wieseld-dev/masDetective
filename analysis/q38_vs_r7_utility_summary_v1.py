from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path("artifacts/q38_vs_r7_utility_v1")
PARTS = ROOT / "parts"


def read_all():
    rows = []

    for p in sorted(PARTS.glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))

    return rows


def summarize(xs):
    margins = np.asarray(
        [x["pair_margin"] for x in xs],
        dtype=float,
    )

    plus = np.asarray(
        [x["plus_target_score"] for x in xs],
        dtype=float,
    )

    minus = np.asarray(
        [x["minus_target_score"] for x in xs],
        dtype=float,
    )

    labels = np.asarray(
        [1] * len(plus) + [0] * len(minus)
    )

    scores = np.concatenate(
        [plus, minus]
    )

    correct = int(np.sum(margins > 0))
    ties = int(np.sum(margins == 0))

    activated = [
        bool(x["target_activated_plus_eval_only"])
        for x in xs
    ]

    return {
        "pairs": len(xs),

        "target_presence_auc":
            float(roc_auc_score(labels, scores)),

        "pair_accuracy":
            float(np.mean(margins > 0)),

        "pair_correct": correct,
        "pair_ties": ties,

        "mean_pair_margin":
            float(np.mean(margins)),

        "median_pair_margin":
            float(np.median(margins)),

        "p25_pair_margin":
            float(np.quantile(margins, 0.25)),

        "p75_pair_margin":
            float(np.quantile(margins, 0.75)),

        "positive_margin_rate":
            float(np.mean(margins > 0)),

        "margin_gt_005_rate":
            float(np.mean(margins > 0.05)),

        "mean_plus_target_score":
            float(np.mean(plus)),

        "mean_minus_target_score":
            float(np.mean(minus)),

        "target_activation_rate_plus_eval_only":
            float(np.mean(activated)),
    }


def main():
    rows = read_all()

    print("ROWS =", len(rows))

    if len(rows) != 200:
        raise RuntimeError(
            f"Expected 200 paired probe rows, got {len(rows)}"
        )

    by_arm = defaultdict(list)
    by_arm_role = defaultdict(list)

    for x in rows:
        by_arm[x["arm"]].append(x)
        by_arm_role[
            (x["arm"], x["role"])
        ].append(x)

    arms = {
        arm: summarize(xs)
        for arm, xs in sorted(by_arm.items())
    }

    role_metrics = {}

    for (arm, role), xs in sorted(
        by_arm_role.items()
    ):
        role_metrics.setdefault(
            role,
            {}
        )[arm] = summarize(xs)

    arm_names = sorted(arms)

    comparison = {}

    if len(arm_names) == 2:
        a, b = arm_names

        # Explicitly detect names rather than relying on alphabetical
        r7 = next(
            x for x in arm_names
            if x.startswith("R7")
        )

        q38 = next(
            x for x in arm_names
            if x.startswith("QWEN38")
        )

        comparison = {
            "q38_minus_r7_presence_auc":
                arms[q38]["target_presence_auc"]
                - arms[r7]["target_presence_auc"],

            "q38_minus_r7_pair_accuracy":
                arms[q38]["pair_accuracy"]
                - arms[r7]["pair_accuracy"],

            "q38_minus_r7_mean_margin":
                arms[q38]["mean_pair_margin"]
                - arms[r7]["mean_pair_margin"],

            "q38_minus_r7_activation_rate":
                arms[q38][
                    "target_activation_rate_plus_eval_only"
                ]
                - arms[r7][
                    "target_activation_rate_plus_eval_only"
                ],

            "roles_q38_higher_mean_margin": sum(
                role_metrics[r][q38][
                    "mean_pair_margin"
                ]
                >
                role_metrics[r][r7][
                    "mean_pair_margin"
                ]
                for r in role_metrics
            ),

            "roles_total":
                len(role_metrics),
        }

    summary = {
        "experiment":
            "q38_vs_r7_generator_utility_v1",

        "definition":
            (
                "Same probe evaluated on matched PLUS/MINUS "
                "hidden MAS pair; visible target score computed "
                "from query + final response only."
            ),

        "truth_used_for":
            "evaluation only",

        "arms":
            arms,

        "comparison":
            comparison,

        "per_role":
            role_metrics,
    }

    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        ROOT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print()
    print("=" * 100)
    print("GENERATOR UTILITY A/B")
    print("=" * 100)

    for arm in sorted(arms):
        s = arms[arm]

        print()
        print(arm)
        print(
            "  presence AUC       =",
            f"{s['target_presence_auc']:.4f}",
        )
        print(
            "  pair accuracy      =",
            f"{s['pair_accuracy']:.4f}",
        )
        print(
            "  mean pair margin   =",
            f"{s['mean_pair_margin']:+.5f}",
        )
        print(
            "  median pair margin =",
            f"{s['median_pair_margin']:+.5f}",
        )
        print(
            "  PLUS activation    =",
            f"{s['target_activation_rate_plus_eval_only']:.4f}",
        )

    print()
    print("COMPARISON")
    print(json.dumps(
        comparison,
        indent=2,
    ))

    print()
    print("UTILITY_AB_OK")


if __name__ == "__main__":
    main()
