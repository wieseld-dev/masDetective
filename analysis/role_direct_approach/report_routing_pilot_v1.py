#!/usr/bin/env python3

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(
    "artifacts/role_direct_routing_pilot_v1"
)

manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    x["session_id"]: set(x["roles"])
    for x in manifest["systems"]
    if x["split"] == "dev"
}

rows = [
    json.loads(x)
    for x in (
        ROOT / "responses_dev_round_01.jsonl"
    ).read_text().splitlines()
    if x.strip()
]

by_role = defaultdict(list)

for x in rows:

    role = x["target_role"]
    sid = x["session_id"]

    selected = set(
        x.get(
            "selected_roles_training_only",
            []
        )
    )

    present = role in truth[sid]

    by_role[role].append({
        "present": present,
        "hit": role in selected,
        "exclusive": selected == {role},
        "selected": sorted(selected),
    })


print("=" * 95)
print("ROLE DIRECT — ROUTING PILOT")
print("=" * 95)

all_present = []
all_absent = []

for role in sorted(by_role):

    rr = by_role[role]

    pos = [x for x in rr if x["present"]]
    neg = [x for x in rr if not x["present"]]

    pos_hit = (
        sum(x["hit"] for x in pos) / len(pos)
        if pos else None
    )

    pos_excl = (
        sum(x["exclusive"] for x in pos) / len(pos)
        if pos else None
    )

    neg_hit = (
        sum(x["hit"] for x in neg) / len(neg)
        if neg else None
    )

    if pos:
        all_present += pos

    if neg:
        all_absent += neg

    print()
    print(role)

    print(
        "  present_n=",
        len(pos),
        " target_selected=",
        None if pos_hit is None else round(pos_hit, 3),
        " exclusive=",
        None if pos_excl is None else round(pos_excl, 3),
    )

    print(
        "  absent_n=",
        len(neg),
        " target_selected=",
        None if neg_hit is None else round(neg_hit, 3),
    )

    if neg:
        c = Counter(
            r
            for x in neg
            for r in x["selected"]
        )

        print(
            "  absent_routes=",
            c.most_common(4),
        )


print()
print("=" * 95)
print("OVERALL")
print("=" * 95)

print(
    "P(target selected | installed) =",
    round(
        sum(x["hit"] for x in all_present)
        / len(all_present),
        4,
    ),
)

print(
    "P(exclusive target | installed) =",
    round(
        sum(x["exclusive"] for x in all_present)
        / len(all_present),
        4,
    ),
)

print(
    "P(target selected | absent) =",
    round(
        sum(x["hit"] for x in all_absent)
        / len(all_absent),
        4,
    ),
)

print(
    "present observations =",
    len(all_present),
)

print(
    "absent observations =",
    len(all_absent),
)
