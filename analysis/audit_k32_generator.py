from pathlib import Path
from collections import defaultdict
import json
import random
import numpy as np

ROOT = Path("artifacts/k32_generator_diagnostic")
F = ROOT / "responses_train_round_01.jsonl"

rows = [
    json.loads(x)
    for x in F.open()
    if x.strip()
]

by = defaultdict(list)

for x in rows:
    sid = x.get("system_id") or x.get("session_id")
    if sid:
        by[sid].append(x)


def selected(x):
    return set(
        x.get("selected_roles_training_only") or []
    )


def truth(x):
    return set(
        x.get("truth_roles_training_only") or []
    )


def greedy(rows, budget):
    remaining = list(rows)
    seen = set()

    for _ in range(min(budget, len(remaining))):
        best_i = None
        best_key = None

        for i, x in enumerate(remaining):
            s = selected(x)
            key = (
                len(s - seen),   # new agents
                len(s),          # total activated
                -int(x.get("candidate_index") or 0),
            )

            if best_key is None or key > best_key:
                best_key = key
                best_i = i

        x = remaining.pop(best_i)
        seen.update(selected(x))

    return seen


print("=" * 100)
print("K=6 LARGE-CANDIDATE GENERATOR DIAGNOSTIC")
print("=" * 100)
print("systems =", len(by))
print("rows    =", len(rows))

sizes = []
pool_exact = []
pool_cov = []

for sid, rr in by.items():
    rr.sort(
        key=lambda x:int(x.get("candidate_index") or 0)
    )

    gt = truth(rr[0])

    u = set()
    for x in rr:
        u.update(selected(x))

    sizes.append(len(rr))
    pool_exact.append(u == gt)
    pool_cov.append(len(u & gt) / len(gt))

print()
print("candidate counts:", sorted(set(sizes)))
print(
    "ALL-32 exact K/set =",
    np.mean(pool_exact)
)
print(
    "ALL-32 mean coverage =",
    np.mean(pool_cov)
)

print()
print("=" * 100)
print("ORACLE GREEDY — HOW MANY REAL QUERIES ARE NEEDED?")
print("=" * 100)

for b in range(1, 9):
    exact = []
    cov = []

    for sid, rr in by.items():
        gt = truth(rr[0])
        seen = greedy(rr, b)

        exact.append(seen == gt)
        cov.append(len(seen & gt) / len(gt))

    print(
        f"budget={b}: "
        f"exactK={np.mean(exact):.3f} "
        f"coverage={np.mean(cov):.3f}"
    )

print()
print("=" * 100)
print("PER SYSTEM")
print("=" * 100)

for sid, rr in sorted(by.items()):
    gt = truth(rr[0])

    u = set()
    for x in rr:
        u.update(selected(x))

    print(
        sid,
        f"K={len(gt)}",
        f"pool_seen={len(u)}",
        f"pool_exact={u == gt}",
        f"truth={sorted(gt)}",
        f"seen={sorted(u)}",
    )
