from __future__ import annotations

import json
import random
import re
from pathlib import Path
from collections import Counter


SRC = Path(
    "artifacts/v7_historical_benign_audit_v1/"
    "strict_benign.jsonl"
)

OUT = Path(
    "artifacts/v7_benign_validity_v4"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
rng = random.Random(SEED)


# ============================================================
# IMPORTANT SEMANTICS
#
# This file judges ONLY VALIDITY / BENIGNNESS.
#
# It does NOT reject a query because it may have weak
# fingerprint utility.
#
# In particular, the following are ALLOWED:
#
#   "What information should I provide...?"
#   "Should I explain X first?"
#   "Would providing more context help?"
#   "Should I gather more information about X?"
#
# These may later receive low extraction utility, but they are
# legitimate user queries.
# ============================================================


# Explicit interrogation of the SYSTEM'S OWN capability,
# expertise, agents, tools, hidden structure, etc.
SYSTEM_INTROSPECTION = [
    r"\bdo you have experience (?:with|in)\b",
    r"\bhave you worked with\b",
    r"\bare you experienced (?:with|in)\b",
    r"\bare you knowledgeable (?:about|in)\b",

    r"\bwhat are your capabilities\b",
    r"\bwhat capabilities do you have\b",
    r"\bwhich capabilities do you have\b",

    r"\bwhat is your expertise\b",
    r"\bwhat are your strengths\b",
    r"\bwhat do you specialize in\b",

    r"\bwhat would you find .*challenging\b",
    r"\bwhich .* would you find .*challenging\b",

    r"\bwhich .* agent\b",
    r"\bwhat .* agent\b",
    r"\bdo you have an? .* agent\b",

    r"\bwhich tools? do you\b",
    r"\bwhat tools? do you\b",

    r"\byour internal architecture\b",
    r"\byour internal workflow\b",
    r"\byour hidden\b",

    # User's important distinction:
    # Asking what information to PROVIDE is okay.
    # Asking about system understanding/capability itself is not.
    r"\bimprove your understanding\b",
    r"\btest your understanding\b",
    r"\bassess your understanding\b",
]


# Explicit capability-routing/meta probes.
CAPABILITY_ROUTING = [
    r"\bwhich .* skill areas? (?:do you|would you)\b",
    r"\bwhat .* skill areas? (?:do you|would you)\b",

    r"\bwhich .* capability (?:would|could|should)\b",
    r"\bwhat .* capability (?:would|could|should)\b",

    r"\bwhich .* expertise (?:would|could|should)\b",

    r"\bwhich agent (?:would|should|could)\b",
    r"\bwhat agent (?:would|should|could)\b",

    r"\bwhich tool (?:would|should|could)\b",
    r"\bwhat tool (?:would|should|could)\b",
]


# Genuine assistant -> user reversal.
#
# Be careful:
# "Would providing more information help you answer?"
# is NOT rejected.
#
# We only reject cases where the speaker clearly offers
# assistance as though THEY are the assistant.
DIRECTION_REVERSAL = [
    r"^\s*what .* would you like (?:help|assistance) with\b",
    r"^\s*which .* would you like (?:help|assistance) with\b",

    r"^\s*what .* do you need (?:help|assistance) with\b",
    r"^\s*which .* do you need (?:help|assistance) with\b",

    r"^\s*how can i help you\b",
    r"^\s*how can i assist you\b",

    r"^\s*what can i help you with\b",
    r"^\s*what can i assist you with\b",

    r"^\s*would you like me to\b",
    r"^\s*do you want me to\b",
]


# Missing referent in a standalone query.
DANGLING = [
    r"\bthese tasks\b",
    r"\bthese questions\b",
    r"\bthese pieces\b",
    r"\bthese items\b",
    r"\bthese examples\b",
    r"\bthose tasks\b",
    r"\bthose questions\b",

    r"\bthe above\b",
    r"\bthe previous\b",

    r"\[[^\]]+\]",          # unresolved [placeholder]
    r"_{3,}",               # ______ placeholder
]


# ============================================================
# TAGS ONLY — NEVER REJECT FOR THESE.
# ============================================================

CLARIFICATION_TAG = [
    r"\bwhat information should i provide\b",
    r"\bwhat details should i provide\b",
    r"\bwhat context should i provide\b",

    r"\bwhat information do you need\b",
    r"\bwhat details do you need\b",

    r"\bwould providing .* information\b",
    r"\bwould providing .* details\b",
    r"\bwould more context\b",

    r"\bshould i provide\b",
    r"\bshould i explain\b",
    r"\bshould i describe\b",
    r"\bshould i share\b",
    r"\bshould i gather\b",
    r"\bshould i begin with explaining\b",
    r"\bshould i start by explaining\b",
]


SUBSTANTIVE_TAG = [
    r"\bhow can i\b",
    r"\bhow should i\b",
    r"\bhow do i\b",

    r"\bwhat should i\b",
    r"\bwhat would you recommend\b",
    r"\bwhat do you recommend\b",

    r"\bcan you explain\b",
    r"\bcan you analyze\b",
    r"\bcan you compare\b",

    r"\bexplain\b",
    r"\banalyze\b",
    r"\bcompare\b",
    r"\bevaluate\b",
    r"\brecommend\b",
    r"\bcalculate\b",
    r"\bidentify\b",
    r"\boptimize\b",
    r"\bdesign\b",
]


def norm(x):
    return " ".join(
        str(x or "")
        .strip()
        .lower()
        .split()
    )


def match(patterns, text):
    return [
        p
        for p in patterns
        if re.search(
            p,
            text,
            flags=re.I,
        )
    ]


def classify(query):
    t = norm(query)

    introspection = match(
        SYSTEM_INTROSPECTION,
        t,
    )

    routing = match(
        CAPABILITY_ROUTING,
        t,
    )

    reversal = match(
        DIRECTION_REVERSAL,
        t,
    )

    dangling = match(
        DANGLING,
        t,
    )

    invalid_reasons = []

    if introspection:
        invalid_reasons.append(
            "SYSTEM_INTROSPECTION"
        )

    if routing:
        invalid_reasons.append(
            "CAPABILITY_ROUTING"
        )

    if reversal:
        invalid_reasons.append(
            "DIRECTION_REVERSAL"
        )

    if dangling:
        invalid_reasons.append(
            "DANGLING"
        )

    clarification = bool(
        match(
            CLARIFICATION_TAG,
            t,
        )
    )

    substantive = bool(
        match(
            SUBSTANTIVE_TAG,
            t,
        )
    )

    # Note:
    # clarification=True is COMPLETELY VALID.
    valid = (
        len(invalid_reasons) == 0
    )

    if clarification and substantive:
        interaction_type = (
            "clarification_plus_task"
        )
    elif clarification:
        interaction_type = (
            "clarification"
        )
    elif substantive:
        interaction_type = (
            "substantive"
        )
    else:
        interaction_type = (
            "other_benign"
        )

    return {
        "valid_benign":
            valid,

        "invalid_reasons":
            invalid_reasons,

        "interaction_type":
            interaction_type,

        "clarification":
            clarification,

        "substantive":
            substantive,

        "matches": {
            "system_introspection":
                introspection,

            "capability_routing":
                routing,

            "direction_reversal":
                reversal,

            "dangling":
                dangling,
        },
    }


rows = []

with SRC.open(
    encoding="utf-8"
) as f:
    for line in f:
        if line.strip():
            x = json.loads(line)

            a = classify(
                x["query"]
            )

            rows.append({
                **x,
                "validity_v4":
                    a,
            })


valid = [
    x
    for x in rows
    if x[
        "validity_v4"
    ][
        "valid_benign"
    ]
]

invalid = [
    x
    for x in rows
    if not x[
        "validity_v4"
    ][
        "valid_benign"
    ]
]


reason_counts = Counter()
interaction_counts = Counter()
role_counts = Counter()
sessions = set()

for x in invalid:
    for r in x[
        "validity_v4"
    ][
        "invalid_reasons"
    ]:
        reason_counts[r] += 1


for x in valid:
    a = x["validity_v4"]

    interaction_counts[
        a["interaction_type"]
    ] += 1

    sessions.add(
        x["session_id"]
    )

    for role in x[
        "selected_roles"
    ]:
        role_counts[role] += 1


summary = {
    "input":
        len(rows),

    "valid_benign":
        len(valid),

    "valid_rate":
        (
            len(valid) / len(rows)
            if rows
            else None
        ),

    "invalid":
        len(invalid),

    "unique_sessions_valid":
        len(sessions),

    "invalid_reason_counts":
        dict(
            sorted(
                reason_counts.items()
            )
        ),

    "interaction_type_counts":
        dict(
            sorted(
                interaction_counts.items()
            )
        ),

    "selected_role_positive_counts_valid":
        dict(
            sorted(
                role_counts.items()
            )
        ),
}


def write_jsonl(path, xs):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in xs:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


write_jsonl(
    OUT / "valid_benign.jsonl",
    valid,
)

write_jsonl(
    OUT / "invalid.jsonl",
    invalid,
)


(
    OUT / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print(
    "===== VALIDITY V4 SUMMARY ====="
)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)


print(
    "\n===== 60 RANDOM VALID ====="
)

sample_valid = (
    valid
    if len(valid) <= 60
    else rng.sample(
        valid,
        60,
    )
)

for i, x in enumerate(
    sample_valid,
    1,
):
    a = x[
        "validity_v4"
    ]

    print(
        f"\n[{i:02d}] "
        f"type={a['interaction_type']} "
        f"selected={x['selected_roles']}"
    )

    print(
        "Q:",
        x["query"],
    )


print(
    "\n===== 30 RANDOM INVALID ====="
)

sample_invalid = (
    invalid
    if len(invalid) <= 30
    else rng.sample(
        invalid,
        30,
    )
)

for i, x in enumerate(
    sample_invalid,
    1,
):
    print(
        f"\n[{i:02d}] "
        f"reasons="
        f"{x['validity_v4']['invalid_reasons']}"
    )

    print(
        "Q:",
        x["query"],
    )


print(
    "\nSaved:"
)

print(
    " ",
    OUT / "summary.json",
)

print(
    " ",
    OUT / "valid_benign.jsonl",
)

print(
    " ",
    OUT / "invalid.jsonl",
)
