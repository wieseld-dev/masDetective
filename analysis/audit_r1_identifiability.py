from __future__ import annotations

import json
from pathlib import Path
from collections import Counter, defaultdict

FILES = [
    Path("artifacts/v13_probe_disambiguation.jsonl"),
    Path("artifacts/v11_probe_discovery.jsonl"),
]

TARGETS = {
    "database_design",
    "math",
    "incident_management",
    "financial_analysis",
}

def norm(x):
    return str(x or "").strip()

def first(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return None

rows = []

for p in FILES:
    if not p.exists():
        continue

    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                r = json.loads(line)
            except Exception:
                continue

            query = norm(first(r, ("query", "probe", "user_query")))
            target = norm(first(r, ("target_role", "candidate_role", "target")))

            # Direct row-level selected roles if available
            selected = first(
                r,
                (
                    "selected_roles_training_only",
                    "selected_roles",
                    "selected",
                )
            )

            if isinstance(selected, str):
                selected = [selected]

            # Some discovery rows contain branches
            branches = []

            for name in ("plus", "minus", "sham"):
                b = r.get(name)
                if isinstance(b, dict):
                    branches.append((name, b))

            if branches:
                for bname, b in branches:
                    out = norm(
                        first(
                            b,
                            (
                                "output",
                                "answer",
                                "final_output",
                                "response",
                            ),
                        )
                    )

                    sel = first(
                        b,
                        (
                            "selected_roles_training_only",
                            "selected_roles",
                            "selected",
                        ),
                    )

                    if isinstance(sel, str):
                        sel = [sel]

                    rows.append({
                        "source": p.name,
                        "branch": bname,
                        "query": query,
                        "target": target,
                        "selected": list(sel or []),
                        "output": out,
                    })

            elif selected is not None:
                out = norm(
                    first(
                        r,
                        (
                            "output",
                            "answer",
                            "final_output",
                            "response",
                        ),
                    )
                )

                rows.append({
                    "source": p.name,
                    "branch": "row",
                    "query": query,
                    "target": target,
                    "selected": list(selected or []),
                    "output": out,
                })

print("=" * 100)
print("R1 IDENTIFIABILITY / COVERAGE AUDIT")
print("=" * 100)
print(f"rows={len(rows)}")

target_counts = Counter()
target_selected = Counter()
mismatch_counts = Counter()

for r in rows:
    t = r["target"]
    sels = r["selected"]

    if t:
        target_counts[t] += 1

    for s in sels:
        target_selected[(t, s)] += 1

        if t and s != t:
            mismatch_counts[(t, s)] += 1

print()
print("===== TARGET COUNTS =====")
for t in sorted(TARGETS):
    print(f"{t:24s} {target_counts[t]}")

print()
print("===== IMPORTANT TARGET -> SELECTED COUNTS =====")
pairs = [
    ("database_design", "code"),
    ("database_design", "database_design"),
    ("math", "code"),
    ("math", "math"),
    ("incident_management", "code"),
    ("incident_management", "incident_management"),
    ("financial_analysis", "financial_analysis"),
]

for t, s in pairs:
    print(
        f"{t:24s} -> {s:24s} "
        f"{target_selected[(t,s)]}"
    )

print()
print("===== MOST COMMON MISMATCHES =====")
for (t, s), n in mismatch_counts.most_common(30):
    print(f"{t:24s} -> {s:24s} {n}")

print()
print("===== DATABASE_DESIGN TARGET, CODE SELECTED EXAMPLES =====")

hits = [
    r for r in rows
    if r["target"] == "database_design"
    and "code" in r["selected"]
]

print(f"count={len(hits)}")

for i, r in enumerate(hits[:20], 1):
    print()
    print(f"--- EXAMPLE {i} ---")
    print("SOURCE:", r["source"], r["branch"])
    print("QUERY :", r["query"])
    print("SELECT:", r["selected"])
    print("OUTPUT:", r["output"][:1200])

print()
print("===== MATH TARGET, CODE SELECTED =====")
hits = [
    r for r in rows
    if r["target"] == "math"
    and "code" in r["selected"]
]
print("count=", len(hits))

print()
print("===== INCIDENT TARGET, CODE SELECTED =====")
hits = [
    r for r in rows
    if r["target"] == "incident_management"
    and "code" in r["selected"]
]
print("count=", len(hits))
