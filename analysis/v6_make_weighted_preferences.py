import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

SRC = Path(
    "artifacts/v6_pref_final_hp_v5/preferences.jsonl"
)

OUT = Path(
    "artifacts/v6_pref_final_training"
)
OUT.mkdir(parents=True, exist_ok=True)

DST = OUT / "preferences_weighted.jsonl"
SUMMARY = OUT / "summary.json"


def load(p):
    return [
        json.loads(x)
        for x in p.open(encoding="utf-8")
        if x.strip()
    ]


rows = load(SRC)
assert len(rows) == 122

by_gold_family = defaultdict(list)

for i, x in enumerate(rows):
    by_gold_family[
        (int(x["gold_index"]), x["family"])
    ].append(i)


families_by_gold = defaultdict(set)

for g, fam in by_gold_family:
    families_by_gold[g].add(fam)


raw_weights = [0.0] * len(rows)

for (g, fam), idxs in by_gold_family.items():
    nfam = len(families_by_gold[g])

    # Total Gold mass = 1.
    # Families share that mass equally.
    family_mass = 1.0 / nfam

    pair_weight = family_mass / len(idxs)

    for i in idxs:
        raw_weights[i] = pair_weight


# Normalize mean pair weight to 1.
mean_w = sum(raw_weights) / len(raw_weights)

weights = [
    w / mean_w
    for w in raw_weights
]


out = []

for x, w in zip(rows, weights):
    y = dict(x)
    y["weight"] = float(w)
    out.append(y)


with DST.open("w", encoding="utf-8") as f:
    for x in out:
        f.write(
            json.dumps(x, ensure_ascii=False)
            + "\n"
        )


# --------------------------------------------------
# Audits
# --------------------------------------------------

gold_mass = defaultdict(float)
family_mass = defaultdict(float)
gold_family_mass = defaultdict(float)

for x in out:
    g = int(x["gold_index"])
    fam = x["family"]
    w = float(x["weight"])

    gold_mass[g] += w
    family_mass[fam] += w
    gold_family_mass[(g, fam)] += w


def sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


summary = {
    "pairs": len(out),

    "families":
        dict(Counter(x["family"] for x in out)),

    "gold_counts":
        dict(Counter(
            str(x["gold_index"])
            for x in out
        )),

    "weight_min":
        min(weights),

    "weight_max":
        max(weights),

    "weight_mean":
        sum(weights) / len(weights),

    "gold_mass": {
        str(g): v
        for g, v in sorted(gold_mass.items())
    },

    "gold_family_mass": {
        f"G{g:02d}:{fam}": v
        for (g, fam), v
        in sorted(gold_family_mass.items())
    },

    "source_sha256":
        sha256(SRC),

    "weighted_sha256":
        sha256(DST),
}


SUMMARY.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)


print("=" * 72)
print("V6 FINAL TRAINING PREFERENCES")
print("=" * 72)

print("pairs:", len(out))
print("families:", summary["families"])

print(
    "weights:",
    f"min={summary['weight_min']:.6f}",
    f"max={summary['weight_max']:.6f}",
    f"mean={summary['weight_mean']:.6f}",
)

print("\nGold total masses:")
for g, mass in sorted(gold_mass.items()):
    print(
        f"G{g:02d}",
        f"{mass:.6f}",
        "families=",
        sorted(families_by_gold[g]),
    )

print("\nSHA256:")
print("source  :", summary["source_sha256"])
print("weighted:", summary["weighted_sha256"])

print("\nWROTE:", DST)
print("WROTE:", SUMMARY)
