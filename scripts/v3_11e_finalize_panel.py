#!/usr/bin/env python3

import json
import shutil
from pathlib import Path


SRC = Path(
    "artifacts/eqs_v3_11d_repaired_panel/panel.json"
)

V14 = Path(
    "artifacts/v14_adaptive_role_v1"
)

OUT = Path(
    "artifacts/eqs_v3_11e_final_shared_panel"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


obj = json.loads(
    SRC.read_text()
)

panel = obj["queries"]


PATCHES = {

    2:
    """Solve the equation 3x^2 - 5x - 2 = 0. Show the calculation step by step and verify the resulting roots.""",

    4:
    """Two studies investigate whether a new training method improves test scores. Study A randomly assigns 500 students to the new method or the standard method and reports a 7-point improvement. Study B observes 5,000 students who chose their own training method and reports a 10-point improvement. Which study provides stronger evidence that the method causes higher scores, and what limitations should be considered?""",

    5:
    """Clean the following records and explain each preprocessing decision:
Age,Name,Date_of_Birth,Salary
34,Alice,1992-04-03,52000
,BOB,03/11/1988,61000
29,Charlie,1996.07.12,
thirty-one,Dana,1994-02-30,$58,000
Standardize the fields, handle missing or invalid values appropriately, and convert dates to YYYY-MM-DD where possible.""",

    7:
    """Our order-processing workflow is: receive an order by email, manually copy it into a spreadsheet, email the warehouse, wait for confirmation, and then manually update the CRM. Identify the main operational bottlenecks and propose a more reliable and efficient workflow.""",

    12:
    """Review this HTML for accessibility issues and suggest concrete fixes:
<form><input type="text" placeholder="Email"><button><img src="send.png"></button></form>
Focus on WCAG-relevant issues such as labels, alternative text, keyboard use, and understandable controls.""",

    14:
    """A clinic wants a new appointment system. Patients must be able to book, cancel, and reschedule appointments; staff must see the daily schedule; double-booking must be prevented; and patients should receive SMS reminders. Convert this description into functional requirements, key non-functional requirements, and example acceptance criteria.""",

    15:
    """At 10:05 users began reporting that our web application was unavailable. Monitoring shows HTTP 503 errors increased from 0% to 85%, CPU usage is normal, database latency is normal, and the deployment service reports that a new release completed at 10:02. Describe your immediate incident-response steps, likely hypotheses, and escalation or rollback decision.""",

    17:
    """A six-month software project depends on a third-party payment API, one senior engineer owns the only deployment pipeline, and the external security review is scheduled only one week before launch. Identify and prioritize the major project risks by likelihood and impact, then propose mitigations.""",

    21:
    """A report claims that the average of the values 12, 15, 18, 21, and 29 is 20. Verify whether that result is correct, show the check you used, and explain any discrepancy.""",

    23:
    """A support team normally resolves tickets within 4 hours, but over the last two days the median resolution time increased to 11 hours. Ticket volume increased by 15%, staffing is unchanged, and 60% of delayed tickets are waiting for engineering escalation. Identify the most likely bottleneck and describe what you would investigate next."""
}


print("=" * 110)
print("V3.11e — FINAL CLEANUP")
print("=" * 110)

for idx, q in PATCHES.items():

    old = panel[idx]["query"]

    panel[idx]["previous_query_v311d"] = old
    panel[idx]["query"] = q
    panel[idx]["final_patch"] = True

    print()
    print(f"{idx:02d}")

    print(
        "OLD:",
        old.replace("\n", " ")
    )

    print(
        "NEW:",
        q.replace("\n", " ")
    )


assert len(panel) == 24


# ============================================================
# TRAIN SYSTEMS
# ============================================================

manifest = json.loads(
    (
        V14
        / "manifest_privileged.json"
    ).read_text()
)

train_ids = [
    str(x["session_id"])
    for x in manifest["systems"]
    if x["split"] == "train"
]

assert len(train_ids) == 24


# ============================================================
# SAME 24 ACTIONS FOR EVERY MAS
# ============================================================

cand_path = (
    OUT
    / "candidates_train_round_01.jsonl"
)

with cand_path.open("w") as f:

    for sid in train_ids:

        for p in panel:

            row = {
                "session_id":
                    sid,

                "candidate_index":
                    int(
                        p["panel_index"]
                    ),

                "query":
                    p["query"],

                "valid":
                    True,

                "group_id":
                    f"shared24_final::{sid}",

                "source":
                    "V3.11e_final_shared_panel",

                "assigned_role":
                    p.get(
                        "assigned_role"
                    ),
            }

            f.write(
                json.dumps(row)
                + "\n"
            )


# ============================================================
# SEED EXISTING TARGET CACHE
# ============================================================

for name in [
    "target_public_cache.jsonl",
    "target_truth_cache_training_only.jsonl",
]:

    src = V14 / name
    dst = OUT / name

    if src.exists():
        shutil.copy2(
            src,
            dst,
        )


def read_jsonl(path):

    if not path.exists():
        return []

    out = []

    with path.open() as f:
        for line in f:
            line = line.strip()

            if line:
                out.append(
                    json.loads(line)
                )

    return out


pub = read_jsonl(
    OUT
    / "target_public_cache.jsonl"
)

truth = read_jsonl(
    OUT
    / "target_truth_cache_training_only.jsonl"
)

truth_keys = {
    str(x.get("key", ""))
    for x in truth
}

cached_pairs = {
    (
        str(
            x.get(
                "session_id",
                ""
            )
        ),
        str(
            x.get(
                "query",
                ""
            )
        ),
    )
    for x in pub
    if str(
        x.get(
            "key",
            ""
        )
    ) in truth_keys
}


total = (
    len(train_ids)
    * len(panel)
)

hits = sum(
    1
    for sid in train_ids
    for p in panel
    if (
        sid,
        p["query"],
    ) in cached_pairs
)

new_calls = (
    total - hits
)


# ============================================================
# SAVE FINAL PANEL
# ============================================================

result = {
    "version":
        "V3.11e",

    "purpose":
        "shared State×Query training panel",

    "panel_size":
        24,

    "train_systems":
        24,

    "queries":
        panel,

    "execution_estimate": {
        "total_pairs":
            total,

        "cache_hits":
            hits,

        "new_target_calls":
            new_calls,
    },
}


(
    OUT
    / "panel.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    )
)


print()
print("=" * 110)
print("FINAL 24-PROBE PANEL")
print("=" * 110)

for p in panel:

    name = (
        p.get(
            "assigned_role"
        )
        or "general"
    )

    print(
        f"{p['panel_index']:02d} | "
        f"{name:24s} | "
        f"{p['query'].replace(chr(10), ' ')}"
    )


print()
print("=" * 110)
print("EXECUTION ESTIMATE")
print("=" * 110)

print(
    f"TOTAL PAIRS      : {total}"
)

print(
    f"CACHE HITS       : {hits}"
)

print(
    f"NEW TARGET CALLS : {new_calls}"
)

print(
    "MAS_CALLS_MADE   : 0"
)

print()
print(
    "SAVED:",
    OUT / "panel.json",
)

print(
    "CANDIDATES:",
    cand_path,
)
