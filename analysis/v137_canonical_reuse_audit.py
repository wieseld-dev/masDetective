#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("artifacts/canonical_parallel_v1")
COHORTS = ["k3", "k4", "high"]

TARGET_POS = 12
TARGET_NEG = 12


def read_jsonl(path):
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def first(row, keys, default=None):
    for k in keys:
        if k in row and row[k] not in (None, "", [], {}):
            return row[k]
    return default


def to_roles(x):
    if x is None:
        return ()
    if isinstance(x, str):
        return (x,)
    if isinstance(x, dict):
        return tuple(sorted(k for k, v in x.items() if v))
    return tuple(sorted(str(v) for v in x))


def sid(row):
    return first(row, ("system_id", "session_id"))


truth_keys = (
    "truth_roles_training_only",
    "truth_roles",
)

selected_keys = (
    "selected_roles_training_only",
    "selected_roles",
)

possible_response_keys = (
    "response",
    "final_response",
    "visible_response",
    "final_answer",
    "answer",
    "mas_response",
    "output_text",
    "output",
)

all_rows = []
all_systems = set()
seen_cohort_systems = defaultdict(set)
response_key_counts = Counter()

print("=" * 100)
print("V13.7 EXISTING CANONICAL REUSE AUDIT")
print("=" * 100)

for cohort in COHORTS:
    root = ROOT / cohort
    meta_path = root / "meta.json"
    resp_path = root / "responses_train_round_01.jsonl"

    print(f"\n[{cohort}]")
    print("meta exists     :", meta_path.exists())
    print("responses exists:", resp_path.exists())

    if not meta_path.exists() or not resp_path.exists():
        continue

    meta = json.loads(meta_path.read_text())
    rows = read_jsonl(resp_path)

    probe_by_idx = {
        int(p["candidate_index"]): {
            "role": p["role"],
            "query": p["query"],
        }
        for p in meta["probes"]
    }

    expected_systems = set(meta["systems"])

    print("meta systems    :", len(expected_systems))
    print("response rows   :", len(rows))
    print("first-row keys  :", sorted(rows[0].keys()) if rows else [])

    for row in rows:
        s = sid(row)
        if not s:
            raise RuntimeError(f"{cohort}: response row missing system/session id")

        idx = int(row["candidate_index"])

        if idx not in probe_by_idx:
            raise RuntimeError(
                f"{cohort}: candidate_index={idx} missing from meta probes"
            )

        target = probe_by_idx[idx]["role"]
        query = probe_by_idx[idx]["query"]

        truth = to_roles(first(row, truth_keys))
        selected_raw = first(row, selected_keys)
        selected = to_roles(selected_raw)

        if not truth:
            raise RuntimeError(
                f"{cohort}/{s}/{idx}: missing truth role composition"
            )

        for key in possible_response_keys:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                response_key_counts[key] += 1

        all_rows.append({
            "cohort": cohort,
            "system_id": s,
            "candidate_index": idx,
            "target": target,
            "query": query,
            "truth": truth,
            "selected": selected,
            "selected_available": selected_raw is not None,
            "installed": int(target in truth),
            "activated": (
                int(target in selected)
                if selected_raw is not None
                else None
            ),
        })

        all_systems.add(s)
        seen_cohort_systems[cohort].add(s)

    missing = expected_systems - seen_cohort_systems[cohort]
    extra = seen_cohort_systems[cohort] - expected_systems

    print("seen systems    :", len(seen_cohort_systems[cohort]))
    print("missing systems :", len(missing))
    print("extra systems   :", len(extra))


print("\n" + "=" * 100)
print("GLOBAL INTEGRITY")
print("=" * 100)

print("rows total      :", len(all_rows))
print("unique systems  :", len(all_systems))

system_cohort_count = Counter()

for cohort, systems in seen_cohort_systems.items():
    for s in systems:
        system_cohort_count[s] += 1

cross_cohort = [
    s for s, n in system_cohort_count.items()
    if n > 1
]

print("cross-cohort systems:", len(cross_cohort))

if cross_cohort:
    raise RuntimeError(
        f"Same system appears in multiple canonical cohorts: {cross_cohort[:10]}"
    )


pair_seen = Counter(
    (r["system_id"], r["target"])
    for r in all_rows
)

duplicates = [
    k for k, n in pair_seen.items()
    if n != 1
]

print("system-target duplicates/missing:", len(duplicates))

if duplicates:
    print("first problems:", duplicates[:10])


print("\nPossible visible-response fields:")
for key in possible_response_keys:
    print(f"  {key:20s}: {response_key_counts[key]}")


# ------------------------------------------------------------------
# Per-role reusable TRAIN counts
# ------------------------------------------------------------------

by_role = defaultdict(list)

for r in all_rows:
    by_role[r["target"]].append(r)


print("\n" + "=" * 100)
print("PER-ROLE EXISTING CANONICAL COVERAGE")
print("=" * 100)

header = (
    f"{'role':27s}"
    f"{'n':>6s}"
    f"{'pos':>6s}"
    f"{'neg':>6s}"
    f"{'act+':>7s}"
    f"{'route%':>9s}"
    f"{'need+':>7s}"
    f"{'need-':>7s}"
)

print(header)

roles = sorted(by_role)
deficits = []

for role in roles:
    rr = by_role[role]

    pos = [x for x in rr if x["installed"] == 1]
    neg = [x for x in rr if x["installed"] == 0]

    routed_pos = [
        x for x in pos
        if x["selected_available"]
        and x["activated"] == 1
    ]

    selected_available_pos = [
        x for x in pos
        if x["selected_available"]
    ]

    if selected_available_pos:
        route_rate = len(routed_pos) / len(selected_available_pos)
    else:
        route_rate = None

    need_pos = max(0, TARGET_POS - len(pos))
    need_neg = max(0, TARGET_NEG - len(neg))

    deficits.append((role, need_pos, need_neg))

    rate_s = (
        f"{100*route_rate:8.1f}%"
        if route_rate is not None
        else "     N/A"
    )

    print(
        f"{role:27s}"
        f"{len(rr):6d}"
        f"{len(pos):6d}"
        f"{len(neg):6d}"
        f"{len(routed_pos):7d}"
        f"{rate_s:>9s}"
        f"{need_pos:7d}"
        f"{need_neg:7d}"
    )


print("\n" + "=" * 100)
print("NEW TRAINING EXECUTIONS NEEDED TO REACH 12+/12- PER ROLE")
print("=" * 100)

total_pos_deficit = sum(x[1] for x in deficits)
total_neg_deficit = sum(x[2] for x in deficits)

print("positive deficit total:", total_pos_deficit)
print("negative deficit total:", total_neg_deficit)
print("minimum targeted executions to fill deficits:",
      total_pos_deficit + total_neg_deficit)

for role, p, n in deficits:
    if p or n:
        print(
            f"  {role:27s} need_pos={p:2d} need_neg={n:2d}"
        )


print("\n" + "=" * 100)
print("ROUTING WARNINGS")
print("=" * 100)

warnings = []

for role in roles:
    pos = [x for x in by_role[role] if x["installed"] == 1]
    avail = [x for x in pos if x["selected_available"]]

    if not avail:
        warnings.append((role, None))
        continue

    rate = sum(x["activated"] == 1 for x in avail) / len(avail)

    if rate < 0.90:
        warnings.append((role, rate))

if not warnings:
    print("PASS: all measurable roles route on >=90% of installed targeted probes.")
else:
    for role, rate in warnings:
        if rate is None:
            print(role, "selected-role metadata unavailable")
        else:
            print(role, f"target-routing rate={rate:.3f}")


print("\nAUDIT COMPLETE")
