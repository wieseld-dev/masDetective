#!/usr/bin/env python3

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("artifacts/v137_presence_v1")
SPLIT_PATH = ROOT / "system_splits.json"
SPLIT_SHA = ROOT / "system_splits.sha256"
PROBE_PATH = ROOT / "probe_bank_v1.json"
PROBE_SHA = ROOT / "probe_bank_v1.sha256"

OUT = ROOT / "execution_plan_v1"
OUT.mkdir(parents=True, exist_ok=True)

MIN_LEGACY_POS_PER_ROLE = 12
MIN_FRESH_PAIRS_PER_ROLE = 4


# ---------------------------------------------------------------------
# IO / integrity
# ---------------------------------------------------------------------

def load_json(path):
    return json.loads(path.read_text())


def verify_sha(path, sha_path):
    expected = sha_path.read_text().strip().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()

    if actual != expected:
        raise RuntimeError(
            f"SHA MISMATCH for {path}\n"
            f"expected={expected}\n"
            f"actual  ={actual}"
        )

    return actual


def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


split_sha = verify_sha(SPLIT_PATH, SPLIT_SHA)
probe_sha = verify_sha(PROBE_PATH, PROBE_SHA)

splits_doc = load_json(SPLIT_PATH)
probe_doc = load_json(PROBE_PATH)

splits = splits_doc["splits"]

legacy = splits["legacy_train"]
fresh_train = splits["fresh_train"]
dev = splits["dev"]
internal_test = splits["internal_test"]


# ---------------------------------------------------------------------
# Probe bank
# ---------------------------------------------------------------------

probe_bank = {}
descriptions = {}

for item in probe_doc["roles"]:
    role = item["role"]
    descriptions[role] = item["public_description"]
    probe_bank[role] = item["probes"]

ROLES = sorted(probe_bank)

if len(ROLES) != 20:
    raise RuntimeError(f"Expected 20 roles, got {len(ROLES)}")

for role in ROLES:
    for family in ["T0", "T1", "D", "T"]:
        q = probe_bank[role].get(family)
        if not isinstance(q, str) or not q.strip():
            raise RuntimeError(
                f"Missing probe: role={role} family={family}"
            )


# ---------------------------------------------------------------------
# Frozen split integrity
# ---------------------------------------------------------------------

split_sets = {
    "legacy": set(legacy),
    "fresh_train": set(fresh_train),
    "dev": set(dev),
    "internal_test": set(internal_test),
}

names = list(split_sets)

for i, a in enumerate(names):
    for b in names[i + 1:]:
        overlap = split_sets[a] & split_sets[b]

        if overlap:
            raise RuntimeError(
                f"Frozen split leakage: {a} vs {b}: "
                f"{sorted(overlap)[:10]}"
            )


# ---------------------------------------------------------------------
# Determine how many fresh T1 pairs each role needs.
#
# Existing T0 already supplies 36 examples / role.
#
# For role r:
#   target_pairs = max(
#       4,
#       12 - existing_positive_count
#   )
#
# Every new pair contributes:
#   one positive
#   one matched negative
#
# This gives the previously expected 95 pairs total.
# ---------------------------------------------------------------------

legacy_positive = {}

for role in ROLES:
    legacy_positive[role] = sum(
        role in set(x["roles"])
        for x in legacy.values()
    )

pair_need = {
    role: max(
        MIN_FRESH_PAIRS_PER_ROLE,
        MIN_LEGACY_POS_PER_ROLE - legacy_positive[role],
    )
    for role in ROLES
}

total_pairs_requested = sum(pair_need.values())


# ---------------------------------------------------------------------
# Pair candidate construction.
#
# POS and NEG must have:
#   same K
#   same execution mode
#
# Matching quality:
#   maximize overlap of OTHER installed roles.
#
# Exact-swap example:
#
# POS:
# {database_design, code, data, planning}
#
# NEG:
# {research, code, data, planning}
#
# intersection excluding target = K-1
# ---------------------------------------------------------------------

def roleset(x):
    return set(x["roles"])


def stratum(x):
    return int(x["K"]), str(x["mode"])


pair_candidates = {}

for role in ROLES:
    positives = [
        (sid, x)
        for sid, x in fresh_train.items()
        if role in roleset(x)
    ]

    negatives = [
        (sid, x)
        for sid, x in fresh_train.items()
        if role not in roleset(x)
    ]

    candidates = []

    for psid, p in positives:
        p_roles = roleset(p)
        p_other = p_roles - {role}

        for nsid, n in negatives:
            if psid == nsid:
                continue

            if stratum(p) != stratum(n):
                continue

            n_roles = roleset(n)

            inter = p_other & n_roles
            union = p_other | n_roles

            overlap_n = len(inter)
            jaccard = (
                len(inter) / len(union)
                if union else 1.0
            )

            K = int(p["K"])

            # Since POS has target and NEG does not,
            # K-1 shared other roles is the theoretical max.
            exact_swap = int(overlap_n == K - 1)

            candidates.append({
                "role": role,
                "positive_system": psid,
                "negative_system": nsid,
                "K": K,
                "mode": str(p["mode"]),
                "overlap_other_roles": overlap_n,
                "other_role_jaccard": jaccard,
                "exact_one_role_swap": exact_swap,
            })

    # Most confound-controlled pair first.
    candidates.sort(
        key=lambda x: (
            -x["exact_one_role_swap"],
            -x["overlap_other_roles"],
            -x["other_role_jaccard"],
            x["K"],
            x["mode"],
            x["positive_system"],
            x["negative_system"],
        )
    )

    if not candidates:
        raise RuntimeError(
            f"No matched candidate pairs for role={role}"
        )

    pair_candidates[role] = candidates


# ---------------------------------------------------------------------
# Greedy pair assignment.
#
# Strong preference:
#   each fresh TRAIN MAS used only once.
#
# If mathematically impossible, retry with cap=2 and REPORT IT.
#
# Roles needing the most pairs are allocated first.
# ---------------------------------------------------------------------

def allocate_pairs(max_system_uses):
    usage = Counter()
    global_strata = Counter()
    chosen = []

    role_order = sorted(
        ROLES,
        key=lambda r: (
            -pair_need[r],
            len(pair_candidates[r]),
            r,
        )
    )

    for role in role_order:
        local_strata = Counter()

        for pair_idx in range(pair_need[role]):
            feasible = []

            for c in pair_candidates[role]:
                p = c["positive_system"]
                n = c["negative_system"]

                if usage[p] >= max_system_uses:
                    continue

                if usage[n] >= max_system_uses:
                    continue

                s = (c["K"], c["mode"])

                # Dynamic preference:
                # 1. diversify this role across K/mode
                # 2. diversify globally
                # 3. then maximize semantic/composition matching
                key = (
                    local_strata[s],
                    global_strata[s],
                    -c["exact_one_role_swap"],
                    -c["overlap_other_roles"],
                    -c["other_role_jaccard"],
                    c["positive_system"],
                    c["negative_system"],
                )

                feasible.append((key, c))

            if not feasible:
                return None

            feasible.sort(key=lambda z: z[0])
            c = feasible[0][1].copy()

            c["pair_index"] = pair_idx
            c["pair_id"] = (
                f"T1::{role}::{pair_idx:02d}"
            )

            chosen.append(c)

            p = c["positive_system"]
            n = c["negative_system"]
            s = (c["K"], c["mode"])

            usage[p] += 1
            usage[n] += 1

            local_strata[s] += 1
            global_strata[s] += 1

    return chosen, usage


allocation = allocate_pairs(max_system_uses=1)

if allocation is not None:
    train_pairs, train_system_usage = allocation
    train_use_cap = 1
else:
    allocation = allocate_pairs(max_system_uses=2)

    if allocation is None:
        raise RuntimeError(
            "Could not allocate matched T1 pairs even with "
            "max_system_uses=2"
        )

    train_pairs, train_system_usage = allocation
    train_use_cap = 2


if len(train_pairs) != total_pairs_requested:
    raise RuntimeError(
        f"Pair count mismatch: "
        f"{len(train_pairs)} != {total_pairs_requested}"
    )


# ---------------------------------------------------------------------
# Expand TRAIN matched pairs to execution rows.
# ---------------------------------------------------------------------

train_rows = []

for pair in train_pairs:
    role = pair["role"]
    query = probe_bank[role]["T1"]

    for label, side, sid in [
        (1, "positive", pair["positive_system"]),
        (0, "negative", pair["negative_system"]),
    ]:
        x = fresh_train[sid]

        actual = int(role in roleset(x))

        if actual != label:
            raise RuntimeError(
                f"Label mismatch pair={pair['pair_id']} "
                f"system={sid}"
            )

        train_rows.append({
            "dataset_split": "train",
            "probe_family": "T1",
            "execute_now": True,

            "pair_id": pair["pair_id"],
            "pair_side": side,

            "system_id": sid,
            "target_role": role,
            "public_role_description": descriptions[role],
            "query": query,

            # TRAINING LABEL ONLY.
            # Never an inference feature.
            "label_target_installed": label,

            # Audit metadata only.
            "K_training_only": int(x["K"]),
            "mode_training_only": str(x["mode"]),

            "source_root": x["source_root"],
            "candidate_path": x["candidate_path"],
            "manifest": x["manifest"],

            "match_overlap_other_roles": (
                pair["overlap_other_roles"]
            ),
            "match_other_role_jaccard": (
                pair["other_role_jaccard"]
            ),
            "match_exact_one_role_swap": (
                pair["exact_one_role_swap"]
            ),
        })


# ---------------------------------------------------------------------
# DEV / TEST small-MAS panels.
#
# Use ALL frozen K=3/4 systems.
#
# This preserves the min >=3 positive systems / role gate that we
# explicitly required when freezing the split.
#
# DEV uses D wording.
# Internal TEST uses T wording and is NOT executed yet.
# ---------------------------------------------------------------------

def make_small_panel(split_name, systems, family, execute_now):
    rows = []

    small = {
        sid: x
        for sid, x in systems.items()
        if int(x["K"]) in (3, 4)
    }

    for sid in sorted(small):
        x = small[sid]
        installed = roleset(x)

        for role in ROLES:
            rows.append({
                "dataset_split": split_name,
                "probe_family": family,
                "execute_now": execute_now,

                "pair_id": None,
                "pair_side": None,

                "system_id": sid,
                "target_role": role,
                "public_role_description": descriptions[role],
                "query": probe_bank[role][family],

                # EVALUATION LABEL ONLY.
                "label_target_installed": int(
                    role in installed
                ),

                "K_training_only": int(x["K"]),
                "mode_training_only": str(x["mode"]),

                "source_root": x["source_root"],
                "candidate_path": x["candidate_path"],
                "manifest": x["manifest"],
            })

    return rows


dev_rows = make_small_panel(
    "dev_small",
    dev,
    "D",
    True,
)

test_rows = make_small_panel(
    "internal_test_small",
    internal_test,
    "T",
    False,
)


# ---------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------

if len(train_rows) != 2 * total_pairs_requested:
    raise RuntimeError("Unexpected TRAIN row count")

expected_dev_systems = sum(
    int(x["K"]) in (3, 4)
    for x in dev.values()
)

expected_test_systems = sum(
    int(x["K"]) in (3, 4)
    for x in internal_test.values()
)

if len(dev_rows) != expected_dev_systems * 20:
    raise RuntimeError("Unexpected DEV row count")

if len(test_rows) != expected_test_systems * 20:
    raise RuntimeError("Unexpected TEST row count")


train_ids = {x["system_id"] for x in train_rows}
dev_ids = {x["system_id"] for x in dev_rows}
test_ids = {x["system_id"] for x in test_rows}

if train_ids & dev_ids:
    raise RuntimeError("TRAIN/DEV system leakage")

if train_ids & test_ids:
    raise RuntimeError("TRAIN/TEST system leakage")

if dev_ids & test_ids:
    raise RuntimeError("DEV/TEST system leakage")

if train_ids & set(legacy):
    raise RuntimeError(
        "Fresh T1 TRAIN unexpectedly overlaps legacy T0 TRAIN"
    )


# Same pair must have exactly one 1 and one 0.
by_pair = defaultdict(list)

for row in train_rows:
    by_pair[row["pair_id"]].append(row)

for pid, rr in by_pair.items():
    if len(rr) != 2:
        raise RuntimeError(
            f"{pid}: expected 2 rows, got {len(rr)}"
        )

    labels = sorted(
        x["label_target_installed"]
        for x in rr
    )

    if labels != [0, 1]:
        raise RuntimeError(
            f"{pid}: expected labels [0,1], got {labels}"
        )

    if rr[0]["query"] != rr[1]["query"]:
        raise RuntimeError(
            f"{pid}: positive/negative query mismatch"
        )

    if rr[0]["K_training_only"] != rr[1]["K_training_only"]:
        raise RuntimeError(
            f"{pid}: K mismatch"
        )

    if rr[0]["mode_training_only"] != rr[1]["mode_training_only"]:
        raise RuntimeError(
            f"{pid}: mode mismatch"
        )


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def label_counts(rows):
    return Counter(
        x["label_target_installed"]
        for x in rows
    )


def role_positive_counts(rows):
    c = Counter()

    for x in rows:
        if x["label_target_installed"] == 1:
            c[x["target_role"]] += 1

    return c


def k_counts(rows):
    # Unique system count, not row count.
    d = {}

    for x in rows:
        d[x["system_id"]] = x["K_training_only"]

    return Counter(d.values())


def mode_counts(rows):
    d = {}

    for x in rows:
        d[x["system_id"]] = x["mode_training_only"]

    return Counter(d.values())


exact_swap_n = sum(
    p["exact_one_role_swap"]
    for p in train_pairs
)

mean_jaccard = (
    sum(p["other_role_jaccard"] for p in train_pairs)
    / len(train_pairs)
)

pair_k = Counter(
    p["K"]
    for p in train_pairs
)

pair_mode = Counter(
    p["mode"]
    for p in train_pairs
)


print("=" * 100)
print("V13.7 EXECUTION PLAN V1")
print("=" * 100)

print("frozen split SHA :", split_sha)
print("frozen probe SHA :", probe_sha)

print()
print("LEGACY T0 TRAIN")
print("  systems        :", len(legacy))
print("  existing rows  :", len(legacy) * 20)

print()
print("FRESH T1 TRAIN")
print("  requested pairs:", total_pairs_requested)
print("  execution rows :", len(train_rows))
print("  labels         :", dict(label_counts(train_rows)))
print("  unique systems :", len(train_ids))
print("  max system use :", max(train_system_usage.values()))
print("  allocation cap :", train_use_cap)
print("  exact swaps    :", exact_swap_n, "/", len(train_pairs))
print("  mean Jaccard   :", round(mean_jaccard, 4))
print("  pair K         :", dict(sorted(pair_k.items())))
print("  pair mode      :", dict(sorted(pair_mode.items())))

print()
print(
    f"{'role':27s}"
    f"{'legacy+':>9s}"
    f"{'T1pairs':>9s}"
    f"{'train+':>9s}"
)

for role in ROLES:
    print(
        f"{role:27s}"
        f"{legacy_positive[role]:9d}"
        f"{pair_need[role]:9d}"
        f"{legacy_positive[role] + pair_need[role]:9d}"
    )

print()
print("DEV SMALL")
print("  systems        :", len(dev_ids))
print("  K              :", dict(sorted(k_counts(dev_rows).items())))
print("  mode           :", dict(sorted(mode_counts(dev_rows).items())))
print("  execution rows :", len(dev_rows))
print("  labels         :", dict(label_counts(dev_rows)))

dev_pos = role_positive_counts(dev_rows)

print(
    "  min role positives:",
    min(dev_pos[r] for r in ROLES),
)

print()
print("INTERNAL TEST SMALL")
print("  systems        :", len(test_ids))
print("  K              :", dict(sorted(k_counts(test_rows).items())))
print("  mode           :", dict(sorted(mode_counts(test_rows).items())))
print("  planned rows   :", len(test_rows))
print("  EXECUTE NOW    : NO")

test_pos = role_positive_counts(test_rows)

print(
    "  min role positives:",
    min(test_pos[r] for r in ROLES),
)

print()
print("SYSTEM LEAKAGE")
print("  TRAIN ∩ DEV  :", len(train_ids & dev_ids))
print("  TRAIN ∩ TEST :", len(train_ids & test_ids))
print("  DEV ∩ TEST   :", len(dev_ids & test_ids))

print()
execute_now = [
    *train_rows,
    *dev_rows,
]

print("EXECUTION BUDGET")
print("  TRAIN T1 now :", len(train_rows))
print("  DEV D now    :", len(dev_rows))
print("  TOTAL NOW    :", len(execute_now))
print("  TEST held    :", len(test_rows))

print()
print("UNTOUCHED FINAL TEST USED: NO")
print("PLAN INTEGRITY: PASS")


# ---------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------

write_jsonl(
    OUT / "train_t1_plan.jsonl",
    train_rows,
)

write_jsonl(
    OUT / "dev_small_d_plan.jsonl",
    dev_rows,
)

write_jsonl(
    OUT / "internal_test_small_t_plan.jsonl",
    test_rows,
)

write_jsonl(
    OUT / "execute_now_plan.jsonl",
    execute_now,
)

write_jsonl(
    OUT / "train_pairs.jsonl",
    train_pairs,
)

meta = {
    "version": "v137_execution_plan_v1",
    "frozen_inputs": {
        "system_splits_sha256": split_sha,
        "probe_bank_sha256": probe_sha,
    },
    "counts": {
        "legacy_t0_existing_rows": len(legacy) * 20,
        "train_t1_pairs": len(train_pairs),
        "train_t1_rows": len(train_rows),
        "dev_small_systems": len(dev_ids),
        "dev_small_rows": len(dev_rows),
        "internal_test_small_systems": len(test_ids),
        "internal_test_small_rows": len(test_rows),
        "execute_now_rows": len(execute_now),
    },
    "protocol": {
        "train_T0": "existing legacy canonical responses",
        "train_T1": "fresh matched positive/negative pairs",
        "dev_D": "all frozen DEV K3/K4 systems",
        "test_T": "all frozen internal-test K3/K4 systems; held for later",
        "pair_match": "same K, same mode, maximize other-role overlap",
        "untouched_final_test_used": False,
    },
}

meta_text = json.dumps(
    meta,
    indent=2,
    sort_keys=True,
)

(OUT / "meta.json").write_text(meta_text)

plan_sha = hashlib.sha256(
    (OUT / "execute_now_plan.jsonl").read_bytes()
).hexdigest()

(OUT / "execute_now_plan.sha256").write_text(
    plan_sha + "  execute_now_plan.jsonl\n"
)

print()
print("OUTPUT:", OUT)
print("execute-now SHA256:", plan_sha)
