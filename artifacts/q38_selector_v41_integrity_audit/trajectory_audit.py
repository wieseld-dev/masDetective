import json
from pathlib import Path

ROOT = Path(
    "artifacts/v14_R7_q38selectorV41_G8_simple5_v1"
)

def read(p):
    return [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ] if p.exists() else []

rounds = {}

for r in range(1,9):
    p = ROOT / f"kboot_metrics_dev_round_{r:02d}.jsonl"

    if not p.exists():
        continue

    rounds[r] = {
        str(x["session_id"]): float(x["role_auc"])
        for x in read(p)
    }

systems = sorted({
    sid
    for d in rounds.values()
    for sid in d
})

print("SYSTEM".ljust(38), end="")

for r in rounds:
    print(f" R{r:>2}", end="")

print()

for sid in systems:
    print(sid.ljust(38), end="")

    prev = .5

    for r in rounds:
        cur = rounds[r].get(sid)

        if cur is None:
            print("      -", end="")
            continue

        d = cur - prev

        print(
            f" {cur:5.3f}"
            f"({d:+.3f})",
            end=""
        )

        prev = cur

    print()

print("\nMEAN AUC / DELTA")

prev = .5

for r,d in rounds.items():
    m = sum(d.values()) / len(d)

    print(
        f"R{r}: "
        f"AUC={m:.6f} "
        f"delta={m-prev:+.6f}"
    )

    prev = m
