#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path("artifacts/adaptive_v11_curriculum_v2")
CANON = Path("artifacts/canonical_parallel_v1")

ROLES = [
    "code",
    "writing",
    "math",
    "planning",
    "research",
    "data",
    "design",
    "operations",
    "qa_testing",
    "translation",
    "financial_analysis",
    "database_design",
    "accessibility_review",
    "fact_checking",
    "requirements_analysis",
    "incident_management",
    "experiment_design",
    "risk_assessment",
    "documentation",
    "customer_support_triage",
]


def readj(path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sid(row):
    return row.get("system_id") or row.get("session_id")


def truth(row):
    return tuple(sorted(row.get("truth_roles_training_only") or []))


# ------------------------------------------------------------
# Systems already used for the 36-system canonical diagnostic.
# Keep them out of V13.7 initially.
# ------------------------------------------------------------

excluded = set()

for name in ["k3", "k4", "high"]:
    p = CANON / name / "meta.json"
    if not p.exists():
        continue

    meta = json.loads(p.read_text())
    excluded.update(meta.get("systems", []))


# ------------------------------------------------------------
# Build UNIQUE system inventory from TRAIN artifacts only.
# Do not inspect any final-test artifact.
# ------------------------------------------------------------

occurrences = defaultdict(list)

response_paths = sorted(
    BASE.glob("stage_*/cycle_*/responses_train_round_01.jsonl")
)

for rp in response_paths:
    cp = rp.parent / "candidates_train_round_01.jsonl"

    candidate_ids = set()
    if cp.exists():
        for x in readj(cp):
            s = x.get("session_id") or x.get("system_id")
            if s:
                candidate_ids.add(s)

    local_seen = set()

    for row in readj(rp):
        s = sid(row)

        if not s or s in local_seen:
            continue

        local_seen.add(s)

        roles = truth(row)
        if not roles:
            continue

        occurrences[s].append({
            "source_root": str(rp.parent),
            "response_path": str(rp),
            "candidate_path": str(cp),
            "has_template": s in candidate_ids,
            "K": len(roles),
            "mode": str(row.get("mode")),
            "roles": roles,
        })


# ------------------------------------------------------------
# Validate that repeated appearances of the same MAS have
# consistent hidden composition metadata.
# ------------------------------------------------------------

inventory = {}
inconsistent = []

for s, occs in occurrences.items():
    signatures = {
        (x["K"], x["mode"], x["roles"])
        for x in occs
    }

    if len(signatures) != 1:
        inconsistent.append((s, signatures))
        continue

    usable = [x for x in occs if x["has_template"]]

    if not usable:
        continue

    # Deterministic source selection.
    usable.sort(key=lambda x: x["source_root"])

    inventory[s] = usable[-1]


print("=" * 96)
print("V13.7 SYSTEM INVENTORY AUDIT")
print("=" * 96)

print("response files scanned :", len(response_paths))
print("unique seen systems    :", len(occurrences))
print("usable unique systems  :", len(inventory))
print("inconsistent systems   :", len(inconsistent))
print("canonical excluded     :", len(excluded))

pool = {
    s: x for s, x in inventory.items()
    if s not in excluded and 3 <= x["K"] <= 8
}

print("clean K3-K8 pool       :", len(pool))

if inconsistent:
    print("\nWARNING: inconsistent repeated-system metadata:")
    for s, sig in inconsistent[:10]:
        print(" ", s, sig)


# ------------------------------------------------------------
# Distribution
# ------------------------------------------------------------

print("\n" + "=" * 96)
print("K DISTRIBUTION")
print("=" * 96)

kc = Counter(x["K"] for x in pool.values())

for k in sorted(kc):
    print(f"K={k}: {kc[k]}")


print("\n" + "=" * 96)
print("MODE DISTRIBUTION")
print("=" * 96)

mc = Counter(x["mode"] for x in pool.values())

for m in sorted(mc):
    print(f"{m:12s}: {mc[m]}")


# ------------------------------------------------------------
# Per-role feasibility.
#
# pairable_pos:
# positives that live in a (K,mode) stratum containing >=1 negative.
#
# pairable_neg:
# negatives that live in a (K,mode) stratum containing >=1 positive.
#
# This tells us whether strict K/mode-matched hard-negative sampling
# is feasible.
# ------------------------------------------------------------

print("\n" + "=" * 96)
print("PER-ROLE MATCHED-SAMPLING FEASIBILITY")
print("=" * 96)

rows = []

for role in ROLES:
    pos = {
        s: x for s, x in pool.items()
        if role in x["roles"]
    }

    neg = {
        s: x for s, x in pool.items()
        if role not in x["roles"]
    }

    pos_strata = Counter(
        (x["K"], x["mode"])
        for x in pos.values()
    )

    neg_strata = Counter(
        (x["K"], x["mode"])
        for x in neg.values()
    )

    pairable_pos = sum(
        n for stratum, n in pos_strata.items()
        if neg_strata[stratum] > 0
    )

    pairable_neg = sum(
        n for stratum, n in neg_strata.items()
        if pos_strata[stratum] > 0
    )

    # Small-system availability is particularly important.
    small_pos = sum(
        1 for x in pos.values()
        if x["K"] in (3, 4)
    )

    small_neg = sum(
        1 for x in neg.values()
        if x["K"] in (3, 4)
    )

    rows.append((
        role,
        len(pos),
        len(neg),
        pairable_pos,
        pairable_neg,
        small_pos,
        small_neg,
    ))


print(
    f"{'role':27s}"
    f"{'pos':>7s}"
    f"{'neg':>7s}"
    f"{'match+':>9s}"
    f"{'match-':>9s}"
    f"{'K3/4+':>9s}"
    f"{'K3/4-':>9s}"
)

for r in rows:
    print(
        f"{r[0]:27s}"
        f"{r[1]:7d}"
        f"{r[2]:7d}"
        f"{r[3]:9d}"
        f"{r[4]:9d}"
        f"{r[5]:9d}"
        f"{r[6]:9d}"
    )


# ------------------------------------------------------------
# Conservative feasibility gate for the ~480-example design:
# roughly 12 positive + 12 negative examples / role overall.
# ------------------------------------------------------------

bad = [
    r for r in rows
    if r[3] < 12 or r[4] < 12
]

print("\n" + "=" * 96)
print("12 POS / 12 NEG MATCHED FEASIBILITY")
print("=" * 96)

if bad:
    print("NOT CLEANLY FEASIBLE for:")
    for r in bad:
        print(
            r[0],
            "pairable_pos=", r[3],
            "pairable_neg=", r[4],
        )
else:
    print("PASS: every role has >=12 pairable positives and negatives.")


# ------------------------------------------------------------
# Verify that excluded diagnostic systems really are excluded.
# ------------------------------------------------------------

overlap = set(pool) & excluded

print("\ncanonical overlap after filtering:", len(overlap))

if overlap:
    raise RuntimeError(
        f"Excluded canonical systems leaked into pool: {sorted(overlap)[:10]}"
    )

print("AUDIT COMPLETE")
