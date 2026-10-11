#!/usr/bin/env python3

import json
import random
import hashlib
import re

from pathlib import Path
from collections import Counter, defaultdict

SEED = 138

BASE = Path("artifacts/adaptive_v11_curriculum_v2")
LEGACY = Path("artifacts/canonical_parallel_v1")
OUT = Path("artifacts/v137_presence_v1")

OUT.mkdir(parents=True, exist_ok=True)

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
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sid(row):
    return row.get("system_id") or row.get("session_id")


def truth(row):
    return tuple(sorted(row.get("truth_roles_training_only") or []))


def stage_cycle_key(root):
    # .../stage_10/cycle_02
    sm = re.search(r"stage_(\d+)", str(root))
    cm = re.search(r"cycle_(\d+)", str(root))

    stage = int(sm.group(1)) if sm else -1
    cycle = int(cm.group(1)) if cm else -1

    return stage, cycle


def manifest_for_root(root):
    cycle = root.name.split("_")[-1]
    return root.parent / f"manifest_train_cycle_{cycle}.json"


# ------------------------------------------------------------------
# 1. Legacy canonical systems = TRAIN ONLY.
# ------------------------------------------------------------------

legacy_train = {}

for cohort in ["k3", "k4", "high"]:
    p = LEGACY / cohort / "meta.json"

    if not p.exists():
        raise FileNotFoundError(p)

    meta = json.loads(p.read_text())

    for s in meta["systems"]:
        info = meta["system_info"][s]

        if s in legacy_train:
            raise RuntimeError(f"legacy duplicate system: {s}")

        legacy_train[s] = {
            "system_id": s,
            "legacy_cohort": cohort,
            "K": int(info["K"]),
            "mode": str(info["mode"]),
            "roles": sorted(info["truth"]),
            "source_root": str(meta["source_root"]),
            "manifest": str(meta["manifest"]),
        }


# ------------------------------------------------------------------
# 2. Build fresh unique-system inventory from TRAIN artifacts only.
#    We do NOT inspect any untouched final-test artifact.
# ------------------------------------------------------------------

occurrences = defaultdict(list)

response_paths = sorted(
    BASE.glob("stage_*/cycle_*/responses_train_round_01.jsonl")
)

for rp in response_paths:
    root = rp.parent

    candidate_path = root / "candidates_train_round_01.jsonl"
    manifest = manifest_for_root(root)

    template_ids = set()

    if candidate_path.exists():
        for x in readj(candidate_path):
            s = sid(x)
            if s:
                template_ids.add(s)

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
            "system_id": s,
            "K": len(roles),
            "mode": str(row.get("mode")),
            "roles": roles,
            "source_root": str(root),
            "candidate_path": str(candidate_path),
            "manifest": str(manifest),
            "has_template": s in template_ids,
            "stage_cycle": stage_cycle_key(root),
        })


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

    usable = [
        x for x in occs
        if x["has_template"]
        and Path(x["manifest"]).exists()
    ]

    if not usable:
        continue

    # Deterministically use latest numeric stage/cycle source.
    usable.sort(key=lambda x: x["stage_cycle"])
    chosen = usable[-1]

    inventory[s] = {
        "system_id": s,
        "K": int(chosen["K"]),
        "mode": chosen["mode"],
        "roles": list(chosen["roles"]),
        "source_root": chosen["source_root"],
        "candidate_path": chosen["candidate_path"],
        "manifest": chosen["manifest"],
    }


if inconsistent:
    raise RuntimeError(
        f"{len(inconsistent)} systems have inconsistent composition metadata"
    )


legacy_ids = set(legacy_train)

fresh = {
    s: x
    for s, x in inventory.items()
    if s not in legacy_ids
    and 3 <= x["K"] <= 8
}


# ------------------------------------------------------------------
# 3. Deterministic stratified split by (K, mode).
#
#    Target approximately:
#       fresh train ~70%
#       dev         ~15%
#       test        ~15%
#
#    Every sufficiently populated stratum contributes to all splits.
# ------------------------------------------------------------------

strata = defaultdict(list)

for s, x in fresh.items():
    strata[(x["K"], x["mode"])].append(s)


fresh_train = {}
dev = {}
test = {}

rng = random.Random(SEED)

stratum_report = []

for stratum in sorted(strata):
    systems = sorted(strata[stratum])

    local_rng = random.Random(
        f"{SEED}|{stratum[0]}|{stratum[1]}"
    )
    local_rng.shuffle(systems)

    n = len(systems)

    if n < 3:
        raise RuntimeError(
            f"Stratum {stratum} has only {n} systems; "
            "cannot represent all three splits."
        )

    n_dev = max(1, round(0.15 * n))
    n_test = max(1, round(0.15 * n))

    # Protect at least one train system.
    while n_dev + n_test >= n:
        if n_test >= n_dev and n_test > 1:
            n_test -= 1
        elif n_dev > 1:
            n_dev -= 1
        else:
            raise RuntimeError(
                f"Cannot split stratum {stratum}, n={n}"
            )

    n_train = n - n_dev - n_test

    train_ids = systems[:n_train]
    dev_ids = systems[n_train:n_train + n_dev]
    test_ids = systems[n_train + n_dev:]

    for s in train_ids:
        fresh_train[s] = fresh[s]

    for s in dev_ids:
        dev[s] = fresh[s]

    for s in test_ids:
        test[s] = fresh[s]

    stratum_report.append({
        "K": stratum[0],
        "mode": stratum[1],
        "total": n,
        "train": len(train_ids),
        "dev": len(dev_ids),
        "test": len(test_ids),
    })


# ------------------------------------------------------------------
# 4. Integrity.
# ------------------------------------------------------------------

sets = {
    "legacy_train": set(legacy_train),
    "fresh_train": set(fresh_train),
    "dev": set(dev),
    "internal_test": set(test),
}

names = list(sets)

for i, a in enumerate(names):
    for b in names[i + 1:]:
        overlap = sets[a] & sets[b]

        if overlap:
            raise RuntimeError(
                f"SYSTEM LEAKAGE: {a} vs {b}: "
                f"{sorted(overlap)[:10]}"
            )


union_fresh = (
    sets["fresh_train"]
    | sets["dev"]
    | sets["internal_test"]
)

if union_fresh != set(fresh):
    missing = set(fresh) - union_fresh
    extra = union_fresh - set(fresh)

    raise RuntimeError(
        f"fresh split partition mismatch "
        f"missing={len(missing)} extra={len(extra)}"
    )


# ------------------------------------------------------------------
# 5. Coverage audit.
# ------------------------------------------------------------------

def audit(name, data):
    rows = list(data.values())

    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    print("systems:", len(rows))

    kc = Counter(x["K"] for x in rows)
    mc = Counter(x["mode"] for x in rows)

    print("K:", dict(sorted(kc.items())))
    print("mode:", dict(sorted(mc.items())))

    print()
    print(
        f"{'role':27s}"
        f"{'pos':>7s}"
        f"{'neg':>7s}"
        f"{'K3/4+':>9s}"
        f"{'K3/4-':>9s}"
    )

    role_stats = {}

    for role in ROLES:
        pos = [
            x for x in rows
            if role in x["roles"]
        ]

        neg = [
            x for x in rows
            if role not in x["roles"]
        ]

        small_pos = [
            x for x in pos
            if x["K"] in (3, 4)
        ]

        small_neg = [
            x for x in neg
            if x["K"] in (3, 4)
        ]

        role_stats[role] = {
            "positive": len(pos),
            "negative": len(neg),
            "small_positive": len(small_pos),
            "small_negative": len(small_neg),
        }

        print(
            f"{role:27s}"
            f"{len(pos):7d}"
            f"{len(neg):7d}"
            f"{len(small_pos):9d}"
            f"{len(small_neg):9d}"
        )

    mins = {
        "min_positive": min(
            x["positive"] for x in role_stats.values()
        ),
        "min_negative": min(
            x["negative"] for x in role_stats.values()
        ),
        "min_small_positive": min(
            x["small_positive"] for x in role_stats.values()
        ),
        "min_small_negative": min(
            x["small_negative"] for x in role_stats.values()
        ),
    }

    print("\nminimum coverage:", mins)

    return {
        "systems": len(rows),
        "K_counts": dict(sorted(kc.items())),
        "mode_counts": dict(sorted(mc.items())),
        "role_stats": role_stats,
        **mins,
    }


print("=" * 100)
print("V13.7 SYSTEM SPLIT FREEZE V1")
print("=" * 100)

print("seed:", SEED)
print("all unique curriculum systems:", len(inventory))
print("legacy TRAIN-only systems:", len(legacy_train))
print("fresh K3-K8 systems:", len(fresh))

print("\nSTRATA:")
for r in stratum_report:
    print(
        f"K={r['K']} mode={r['mode']:10s} "
        f"total={r['total']:3d} "
        f"train={r['train']:3d} "
        f"dev={r['dev']:3d} "
        f"test={r['test']:3d}"
    )


audit_results = {
    "legacy_train": audit(
        "LEGACY TRAIN ONLY",
        legacy_train,
    ),
    "fresh_train": audit(
        "FRESH TRAIN",
        fresh_train,
    ),
    "dev": audit(
        "DEV",
        dev,
    ),
    "internal_test": audit(
        "INTERNAL TEST",
        test,
    ),
}


# ------------------------------------------------------------------
# 6. Conservative gates.
#
# We have a large pool, so DEV/TEST should not have paper-thin
# role coverage. These are inventory counts, before any execution.
# ------------------------------------------------------------------

for split_name in ["dev", "internal_test"]:
    a = audit_results[split_name]

    if a["min_positive"] < 10:
        raise RuntimeError(
            f"{split_name}: some role has <10 positive systems "
            f"(minimum={a['min_positive']})"
        )

    if a["min_negative"] < 10:
        raise RuntimeError(
            f"{split_name}: some role has <10 negative systems "
            f"(minimum={a['min_negative']})"
        )

    if a["min_small_positive"] < 3:
        raise RuntimeError(
            f"{split_name}: some role has <3 K3/K4 positive systems "
            f"(minimum={a['min_small_positive']})"
        )


# ------------------------------------------------------------------
# 7. Save immutable-ish split manifest.
# ------------------------------------------------------------------

manifest = {
    "version": "v137_system_splits_v1",
    "seed": SEED,
    "protocol": {
        "legacy_canonical_systems": "TRAIN_ONLY",
        "fresh_split_unit": "system_id",
        "stratification": ["K", "mode"],
        "untouched_final_test_used": False,
        "system_id_is_feature": False,
        "truth_fields_at_inference": False,
    },
    "counts": {
        "legacy_train": len(legacy_train),
        "fresh_train": len(fresh_train),
        "dev": len(dev),
        "internal_test": len(test),
    },
    "strata": stratum_report,
    "audit": audit_results,
    "splits": {
        "legacy_train": legacy_train,
        "fresh_train": fresh_train,
        "dev": dev,
        "internal_test": test,
    },
}

out_path = OUT / "system_splits.json"

payload = json.dumps(
    manifest,
    indent=2,
    sort_keys=True,
    ensure_ascii=False,
)

out_path.write_text(payload)

sha = hashlib.sha256(
    payload.encode("utf-8")
).hexdigest()

(OUT / "system_splits.sha256").write_text(
    sha + "  system_splits.json\n"
)

print("\n" + "=" * 100)
print("FINAL")
print("=" * 100)

print("NO SYSTEM OVERLAP: PASS")
print("UNTOUCHED FINAL TEST USED: NO")
print("manifest:", out_path)
print("sha256:", sha)
print("SPLIT FREEZE AUDIT: PASS")
