from pathlib import Path
from collections import defaultdict
import json
import statistics
import math

ROOT = Path("artifacts/v14_adaptive_role_v1")


def load(p):
    return [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ]


def avg(xs):
    return statistics.mean(xs) if xs else float("nan")


print("=" * 120)
print("V14 G=50 — TRUE TRAINING REWARD AUDIT")
print("=" * 120)

for rnd in range(7, 11):

    gp = ROOT / f"grpo_train_round_{rnd:02d}.jsonl"
    cp = ROOT / f"candidates_train_round_{rnd:02d}.jsonl"

    print()
    print("=" * 120)
    print(f"ROUND {rnd}")
    print("=" * 120)

    if not gp.exists():
        print("MISSING:", gp)
        continue

    rows = load(gp)
    cand = load(cp) if cp.exists() else []

    qmap = {
        (str(x["session_id"]), int(x["candidate_index"])): str(x.get("query", ""))
        for x in cand
    }

    print("file       :", gp)
    print("rows       :", len(rows))
    print("keys       :", sorted(rows[0]) if rows else [])

    rewards = [float(x["reward"]) for x in rows]
    advs = [float(x["advantage"]) for x in rows]

    groups = defaultdict(list)

    for x in rows:
        groups[str(x["group_id"])].append(x)

    best_rewards = []
    group_positive = []
    n_positive_per_group = []

    for g, xs in groups.items():
        rr = [float(x["reward"]) for x in xs]
        best_rewards.append(max(rr))
        group_positive.append(any(v > 0 for v in rr))
        n_positive_per_group.append(sum(v > 0 for v in rr))

    sign_violations = [
        x for x in rows
        if (
            float(x["reward"]) < 0 and
            float(x["advantage"]) > 1e-12
        ) or (
            float(x["reward"]) > 0 and
            float(x["advantage"]) < -1e-12
        )
    ]

    harmful_reinforced = [
        x for x in rows
        if float(x["reward"]) < 0
        and float(x["advantage"]) > 0
    ]

    helpful_punished = [
        x for x in rows
        if float(x["reward"]) > 0
        and float(x["advantage"]) < 0
    ]

    sr = sorted(rewards)
    p90 = sr[min(len(sr)-1, int(.90*(len(sr)-1)))]
    p95 = sr[min(len(sr)-1, int(.95*(len(sr)-1)))]
    p99 = sr[min(len(sr)-1, int(.99*(len(sr)-1)))]

    top_n = max(1, len(sr)//10)

    print()
    print("REWARD DISTRIBUTION")
    print(f"mean reward             = {avg(rewards):+.6f}")
    print(f"median reward           = {statistics.median(rewards):+.6f}")
    print(f"positive reward         = {sum(r>0 for r in rewards)}/{len(rewards)} "
          f"({sum(r>0 for r in rewards)/len(rewards):.1%})")
    print(f"p90 / p95 / p99         = {p90:+.6f} / {p95:+.6f} / {p99:+.6f}")
    print(f"top-10% mean reward     = {avg(sorted(rewards, reverse=True)[:top_n]):+.6f}")

    print()
    print("G=50 GROUP QUALITY")
    print(f"groups                  = {len(groups)}")
    print(f"group has >=1 positive  = {sum(group_positive)}/{len(group_positive)} "
          f"({avg(group_positive):.1%})")
    print(f"mean # positive / group = {avg(n_positive_per_group):.2f} / 50")
    print(f"mean BEST reward/group  = {avg(best_rewards):+.6f}")
    print(f"median BEST/group       = {statistics.median(best_rewards):+.6f}")
    print(f"min BEST/group          = {min(best_rewards):+.6f}")
    print(f"max BEST/group          = {max(best_rewards):+.6f}")

    print()
    print("ADVANTAGE SANITY")
    print(f"sign violations         = {len(sign_violations)}")
    print(f"harmful reinforced      = {len(harmful_reinforced)}")
    print(f"helpful punished        = {len(helpful_punished)}")

    # Best/worst real queries.
    ranked = sorted(
        rows,
        key=lambda x: float(x["reward"]),
        reverse=True,
    )

    print()
    print("TOP 5 PROBES")
    for x in ranked[:5]:
        key = (str(x["session_id"]), int(x["candidate_index"]))
        print(
            f"R={float(x['reward']):+.6f} "
            f"A={float(x['advantage']):+.3f} "
            f"q={qmap.get(key,'')!r}"
        )

    print()
    print("BOTTOM 5 PROBES")
    for x in ranked[-5:]:
        key = (str(x["session_id"]), int(x["candidate_index"]))
        print(
            f"R={float(x['reward']):+.6f} "
            f"A={float(x['advantage']):+.3f} "
            f"q={qmap.get(key,'')!r}"
        )

print()
print("=" * 120)
