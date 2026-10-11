#!/bin/bash
#SBATCH --job-name=hyb-v2-summary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --output=logs/hyb-v2-summary-%j.out
#SBATCH --error=logs/hyb-v2-summary-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

"$HOME/.conda/envs/masdetective/bin/python" - <<'PY'
import json
from pathlib import Path

root = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/01_GENERATOR_HYBRID_V2/"
    "data/r1_selector_test"
)

baseline = {
    1: 0.5098039215686274,
    2: 0.5647058823529412,
    3: 0.6274509803921569,
    4: 0.6392156862745098,
    5: 0.6588235294117647,
    6: 0.7411764705882353,
    7: 0.7019607843137254,
    8: 0.7137254901960783,
}

vals = []

print("=" * 72)
print("HYBRID V2 vs CLEAN BASELINE")
print("=" * 72)

for r in range(1, 9):
    p = root / f"advance_dev_round_{r:02d}_summary.json"

    if not p.exists():
        print(f"R{r}: MISSING")
        continue

    x = json.loads(p.read_text())
    auc = float(x["mean_role_auc"])
    vals.append((r, auc))

    print(
        f"R{r}: "
        f"HYBRID={auc:.6f}  "
        f"BASE={baseline[r]:.6f}  "
        f"DELTA={auc-baseline[r]:+.6f}"
    )

print()

if vals:
    peak_r, peak_auc = max(vals, key=lambda z: z[1])
    final = dict(vals).get(8)

    print(f"HYBRID PEAK: R{peak_r} = {peak_auc:.6f}")

    if final is not None:
        print(f"HYBRID FINAL R8 = {final:.6f}")
        print(f"BASE FINAL R8   = {baseline[8]:.6f}")
        print(f"FINAL DELTA     = {final-baseline[8]:+.6f}")

print()
print("===== FRESHNESS =====")

total_hits = 0
total_rows = 0

for r in range(1, 9):
    p = root / f"responses_dev_round_{r:02d}.jsonl"

    if not p.exists():
        continue

    rows = [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ]

    hits = sum(bool(x.get("cache_hit", False)) for x in rows)

    total_hits += hits
    total_rows += len(rows)

    print(
        f"R{r}: responses={len(rows)} "
        f"cache_hits={hits}"
    )

print(
    f"TOTAL: responses={total_rows} "
    f"cache_hits={total_hits}"
)

print()
print("===== CACHE SIZE =====")

for name in [
    "target_public_cache.jsonl",
    "target_truth_cache_training_only.jsonl",
]:
    p = root / name
    n = sum(1 for x in p.read_text().splitlines() if x.strip())
    print(name, n)
PY
