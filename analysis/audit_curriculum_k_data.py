from pathlib import Path
from collections import Counter, defaultdict
import json
import re

ROOT = Path("artifacts/adaptive_v11_curriculum_v2")

files = sorted(ROOT.rglob("responses_train_round_*.jsonl"))

print(f"response_files={len(files)}")

systems = {}
locations = defaultdict(set)
rounds = defaultdict(set)

for p in files:
    m = re.search(r"round_(\d+)", p.name)
    rnd = int(m.group(1)) if m else -1

    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            sid = r.get("system_id") or r.get("session_id")
            truth = list(r.get("truth_roles_training_only") or [])

            if not sid or len(truth) not in {3,4,5,6}:
                continue

            truth = tuple(sorted(truth))

            if sid in systems and systems[sid] != truth:
                print(
                    "WARNING truth mismatch:",
                    sid,
                    systems[sid],
                    truth,
                )
                continue

            systems[sid] = truth
            locations[sid].add(str(p.parent))
            rounds[sid].add(rnd)


by_k = Counter(len(t) for t in systems.values())

print()
print("=" * 80)
print("UNIQUE TRAINING SYSTEMS BY K")
print("=" * 80)

for k in (3,4,5,6):
    ss = [sid for sid,t in systems.items() if len(t) == k]

    print(
        f"K={k}: systems={len(ss)} "
        f"mean_rounds="
        f"{sum(len(rounds[s]) for s in ss)/max(1,len(ss)):.2f}"
    )


print()
print("=" * 80)
print("SYSTEMS WITH >= 4 ROUNDS")
print("=" * 80)

for k in (3,4,5,6):
    ss = [
        sid for sid,t in systems.items()
        if len(t) == k and len(rounds[sid]) >= 4
    ]

    print(f"K={k}: {len(ss)}")


print()
print("=" * 80)
print("ROUND DISTRIBUTION")
print("=" * 80)

for k in (3,4,5,6):
    c = Counter(
        len(rounds[sid])
        for sid,t in systems.items()
        if len(t) == k
    )

    print(f"K={k}: {dict(sorted(c.items()))}")


print()
print("=" * 80)
print("EXAMPLES")
print("=" * 80)

for k in (3,4,5,6):
    print(f"\nK={k}")

    n = 0

    for sid,t in systems.items():
        if len(t) != k:
            continue

        print(
            sid,
            f"rounds={len(rounds[sid])}",
            "truth=" + ",".join(t),
        )

        n += 1
        if n >= 5:
            break
