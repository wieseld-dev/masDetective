from pathlib import Path
import argparse
import json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--rounds", type=int, required=True)
a = ap.parse_args()

root = Path(a.root)

print("=" * 100)
print("V14 ADAPTIVE ROLE TRAINING REPORT")
print("=" * 100)

for split in ("train", "dev"):

    print()
    print(split.upper())

    for r in range(1, a.rounds + 1):

        p = root / f"metrics_{split}_round_{r:02d}.jsonl"

        if not p.exists():
            print(f"round {r}: MISSING")
            continue

        rows = [
            json.loads(x)
            for x in p.read_text().splitlines()
            if x.strip()
        ]

        def mean(key):
            vals = [
                float(x[key])
                for x in rows
                if x.get(key) is not None
            ]
            return float(np.mean(vals)) if vals else float("nan")

        print(
            f"round {r}: "
            f"logloss={mean('role_log_loss'):.6f} "
            f"brier={mean('role_brier'):.6f} "
            f"auc={mean('role_auc'):.6f} "
            f"margin={mean('true_false_margin'):.6f}"
        )

print()
print(
    "FINAL POLICY:",
    root / "checkpoints"
    / f"round_{a.rounds:02d}"
    / "policy",
)

print("=" * 100)
