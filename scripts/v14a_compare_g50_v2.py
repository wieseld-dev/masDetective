#!/usr/bin/env python3

import json
from pathlib import Path
import statistics


TRAIN = Path("artifacts/v14_adaptive_role_v1")

OLD = Path(
    "artifacts/v14_adaptive_g50_v2_eval_old_r6"
)

NEW = Path(
    "artifacts/v14_adaptive_g50_v2_eval_new_r10"
)


def readj(p):
    if not p.exists():
        return []

    return [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ]


def mean(xs):
    return (
        statistics.mean(xs)
        if xs
        else float("nan")
    )


print("=" * 108)
print("V14 ADAPTIVE G=50 CONTINUATION — FINAL AUDIT")
print("=" * 108)

print()
print("TRAINING POOL / REWARD QUALITY")
print("-" * 108)

for r in range(7, 11):

    p = TRAIN / f"train_rows_round_{r:02d}.jsonl"

    rows = readj(p)

    if not rows:
        print(f"round {r}: MISSING {p}")
        continue

    rewards = [
        float(x["reward"])
        for x in rows
    ]

    advs = [
        float(x["local_advantage"])
        for x in rows
    ]

    by_group = {}

    for x in rows:
        by_group.setdefault(
            x["group_id"],
            []
        ).append(
            float(x["reward"])
        )

    group_best = [
        max(v)
        for v in by_group.values()
    ]

    groups_positive = [
        any(x > 0 for x in v)
        for v in by_group.values()
    ]

    sign_violations = sum(
        (
            rwd < 0 and adv > 1e-12
        ) or (
            rwd > 0 and adv < -1e-12
        )
        for rwd, adv in zip(
            rewards,
            advs,
        )
    )

    topn = max(
        1,
        len(rewards) // 10,
    )

    top10 = sorted(
        rewards,
        reverse=True,
    )[:topn]

    print(
        f"round {r}: "
        f"N={len(rows):4d}  "
        f"meanR={mean(rewards):+.5f}  "
        f"positive={sum(x>0 for x in rewards)/len(rewards):.1%}  "
        f"group_has_positive={mean(groups_positive):.1%}  "
        f"mean_best50={mean(group_best):+.5f}  "
        f"top10%={mean(top10):+.5f}  "
        f"sign_violations={sign_violations}"
    )


print()
print("=" * 108)
print("PAIRED CLEAN DEV ROLLOUT")
print("same 8 systems · same initial belief · same 6-query budget")
print("=" * 108)


def round_metrics(root, rnd):

    p = root / f"metrics_dev_round_{rnd:02d}.jsonl"

    rows = readj(p)

    keys = [
        "role_log_loss",
        "role_brier",
        "role_auc",
        "true_false_margin",
    ]

    return {
        k: mean(
            [
                float(x[k])
                for x in rows
                if x.get(k) is not None
            ]
        )
        for k in keys
    }, rows


print()
print(
    " R | "
    "OLD LogLoss  NEW LogLoss  Delta   | "
    "OLD AUC  NEW AUC | "
    "OLD Brier NEW Brier | "
    "OLD Margin NEW Margin"
)

print("-" * 108)

for r in range(1, 7):

    om, _ = round_metrics(
        OLD,
        r,
    )

    nm, _ = round_metrics(
        NEW,
        r,
    )

    print(
        f"{r:2d} | "
        f"{om['role_log_loss']:.6f}  "
        f"{nm['role_log_loss']:.6f}  "
        f"{nm['role_log_loss']-om['role_log_loss']:+.6f} | "
        f"{om['role_auc']:.4f}   "
        f"{nm['role_auc']:.4f} | "
        f"{om['role_brier']:.5f}  "
        f"{nm['role_brier']:.5f} | "
        f"{om['true_false_margin']:+.5f}  "
        f"{nm['true_false_margin']:+.5f}"
    )


_, old_final = round_metrics(
    OLD,
    6,
)

_, new_final = round_metrics(
    NEW,
    6,
)

old_by = {
    x["session_id"]: x
    for x in old_final
}

new_by = {
    x["session_id"]: x
    for x in new_final
}

common = sorted(
    set(old_by) & set(new_by)
)

deltas = [
    float(new_by[s]["role_log_loss"])
    -
    float(old_by[s]["role_log_loss"])
    for s in common
]

print()
print("FINAL ROUND-6 PAIRED LOGLOSS")
print("-" * 108)

for sid, d in zip(
    common,
    deltas,
):
    print(
        f"{sid:28s} "
        f"NEW-OLD={d:+.6f}"
    )

print()
print(
    "systems            :",
    len(common),
)

print(
    "improved systems   :",
    sum(d < 0 for d in deltas),
    "/",
    len(deltas),
)

print(
    "mean NEW-OLD       :",
    f"{mean(deltas):+.6f}",
)

if deltas:

    final_old = mean(
        [
            float(old_by[s]["role_log_loss"])
            for s in common
        ]
    )

    final_new = mean(
        [
            float(new_by[s]["role_log_loss"])
            for s in common
        ]
    )

    print(
        "final old logloss :",
        f"{final_old:.6f}",
    )

    print(
        "final new logloss :",
        f"{final_new:.6f}",
    )

    print()

    if final_new < final_old:
        print(
            "VERDICT: G=50 continuation IMPROVED "
            "held-out sequential inference."
        )
    else:
        print(
            "VERDICT: G=50 continuation did NOT improve "
            "held-out final logloss."
        )

print("=" * 108)
