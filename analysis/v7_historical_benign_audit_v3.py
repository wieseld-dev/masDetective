from __future__ import annotations

import json
import random
import re
from pathlib import Path
from collections import Counter


SRC = Path(
    "artifacts/v7_historical_benign_audit_v2/"
    "sensor_candidates.jsonl"
)

OUT = Path(
    "artifacts/v7_historical_benign_audit_v3"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
rng = random.Random(SEED)


# ------------------------------------------------------------
# HIGH-PRECISION rejection rules.
#
# Goal:
#   keep ordinary user -> system substantive tasks
#
# Reject:
#   - system/capability interrogation
#   - assistant -> user direction reversal
#   - generic clarification / info-request probes
#   - dangling/vague references
#   - user offering to explain/provide/gather instead of asking
#     the system to solve/reason about the task
#
# We intentionally prefer PRECISION over RECALL here.
# ------------------------------------------------------------


SYSTEM_SELF = [
    r"^\s*do you\b",
    r"^\s*are you\b",
    r"^\s*have you\b",

    r"\bdo you frequently\b",
    r"\bdo you commonly\b",
    r"\bdo you typically\b",
    r"\bdo you usually\b",

    r"\bwhat do you specialize in\b",
    r"\bwhat are you good at\b",
    r"\bwhat are your strengths\b",
    r"\bwhat is your expertise\b",
    r"\bwhat are your capabilities\b",
    r"\bwhat capabilities do you\b",

    r"\bcan you handle\b",
    r"\bcan you perform\b",

    r"\byour experience\b",
    r"\byour expertise\b",
    r"\byour capabilities\b",
    r"\byour skills\b",
]


REVERSED_DIRECTION = [
    r"\bwould you like\b",
    r"\bwould you prefer\b",

    r"\bwhat .* would you like\b",
    r"\bwhich .* would you like\b",

    r"\bwhat .* do you need\b",
    r"\bwhich .* do you need\b",

    r"\bwhat .* would you require\b",
    r"\bwhich .* would you require\b",

    r"\bwould you require me\b",
    r"\bwould you need me\b",

    r"\bwhat kind of .* would you need\b",
    r"\bwhat type of .* would you need\b",
]


GENERIC_CLARIFICATION = [
    r"\bwhat information should i provide\b",
    r"\bwhat details should i provide\b",
    r"\bwhat context should i provide\b",

    r"\bwhat information do you need\b",
    r"\bwhat details do you need\b",
    r"\bwhat context do you need\b",

    r"\bwhat .* would you need from me\b",
    r"\bwhat .* do you need from me\b",

    r"\bwhat should i tell you\b",
    r"\bwhat should i send you\b",
    r"\bwhat should i share\b",

    r"\bwhat specific details would you need\b",
    r"\bwhat specific information would you need\b",
]


USER_OFFERS_INFO = [
    r"^\s*should i (?:start by |begin by )?"
    r"(?:provide|explain|describe|tell|share|give|gather|detail)\b",

    r"^\s*should i (?:provide|explain|describe|tell|share|give|gather|detail)\b",

    r"^\s*can i (?:provide|explain|describe|tell|share)\b",

    r"^\s*may i (?:provide|explain|describe|tell|share)\b",
]


CAPABILITY_ROUTING = [
    r"\bwhich .* skill\b",
    r"\bwhat .* skill\b",

    r"\bwhich .* capability\b",
    r"\bwhat .* capability\b",

    r"\bwhich .* expertise\b",
    r"\bwhat .* expertise\b",

    r"\brequire assistance with\b",
    r"\brequires assistance with\b",

    r"\bbest suited for\b",
    r"\bbest suited to\b",
]


# Single-turn benchmark:
# reject references that require missing previous conversational context.
DANGLING = [
    r"\bthese tasks\b",
    r"\bthese questions\b",
    r"\bthese pieces\b",
    r"\bthese items\b",
    r"\bthese examples\b",
    r"\bthese documents\b",
    r"\bthese datasets\b",
    r"\bthese requests\b",
    r"\bthose tasks\b",
    r"\bthose questions\b",

    r"\bthe above\b",
    r"\bthe previous\b",

    r"\bthis upcoming (?:problem|task|project|challenge)\b",
    r"\bthis current (?:problem|task|project|challenge)\b",
]


# Very vague prompts can produce generic answers and are poor
# fingerprint probes even when technically benign.
TOO_VAGUE = [
    r"\bmy current work\b",
    r"\bmy current projects?\b",
    r"\bthis issue\b",
    r"\bthis challenge\b",
    r"\bthis problem\b",

    r"^\s*which specific aspect of my\b",

    r"^\s*how can i best prepare for "
    r"(?:this|my) upcoming\b",
]


# Keep "Can you explain/analyze/compare..." because that is a normal
# user request. This catches only a few meta/task-selection patterns.
META_TASK_SELECTION = [
    r"\bwhat kind of .* task .* relate to\b",
    r"\bwhich .* task .* relate to\b",
    r"\bwhat type of .* task .* relate to\b",

    r"\bwhich .* phase do you need guidance on\b",
    r"\bwhat .* support would you need\b",
]


def norm(x):
    return " ".join(
        str(x or "")
        .strip()
        .lower()
        .split()
    )


def hits(patterns, text):
    return [
        p
        for p in patterns
        if re.search(
            p,
            text,
            flags=re.I,
        )
    ]


def audit(q):
    t = norm(q)

    groups = {
        "SYSTEM_SELF":
            hits(SYSTEM_SELF, t),

        "REVERSED_DIRECTION":
            hits(REVERSED_DIRECTION, t),

        "GENERIC_CLARIFICATION":
            hits(GENERIC_CLARIFICATION, t),

        "USER_OFFERS_INFO":
            hits(USER_OFFERS_INFO, t),

        "CAPABILITY_ROUTING":
            hits(CAPABILITY_ROUTING, t),

        "DANGLING":
            hits(DANGLING, t),

        "TOO_VAGUE":
            hits(TOO_VAGUE, t),

        "META_TASK_SELECTION":
            hits(META_TASK_SELECTION, t),
    }

    reasons = [
        name
        for name, ms in groups.items()
        if ms
    ]

    return {
        "keep":
            len(reasons) == 0,

        "reasons":
            reasons,

        "matches":
            groups,
    }


rows = []

with SRC.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(
                json.loads(line)
            )


kept = []
rejected = []

reason_counts = Counter()
role_counts = Counter()
session_ids = set()


for x in rows:
    a = audit(
        x["query"]
    )

    y = {
        **x,
        "audit_v3":
            a,
    }

    if a["keep"]:
        kept.append(y)

        session_ids.add(
            x["session_id"]
        )

        for r in x[
            "selected_roles"
        ]:
            role_counts[r] += 1

    else:
        rejected.append(y)

        for reason in a[
            "reasons"
        ]:
            reason_counts[
                reason
            ] += 1


summary = {
    "input_v2_candidates":
        len(rows),

    "high_precision_candidates":
        len(kept),

    "keep_rate":
        (
            len(kept) / len(rows)
            if rows
            else None
        ),

    "rejected":
        len(rejected),

    "unique_sessions":
        len(session_ids),

    "rejection_reason_counts":
        dict(
            sorted(
                reason_counts.items()
            )
        ),

    "selected_role_positive_counts":
        dict(
            sorted(
                role_counts.items()
            )
        ),
}


with (
    OUT / "high_precision_candidates.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:
    for x in kept:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


with (
    OUT / "rejected.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:
    for x in rejected:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


def sample(xs, n):
    if len(xs) <= n:
        return list(xs)

    return rng.sample(
        xs,
        n,
    )


audit_sample = {
    "kept_100":
        sample(
            kept,
            100,
        ),

    "rejected_50":
        sample(
            rejected,
            50,
        ),
}


(
    OUT / "human_audit.json"
).write_text(
    json.dumps(
        audit_sample,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
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
    "===== V3 HIGH-PRECISION SUMMARY ====="
)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)


print(
    "\n===== 50 RANDOM SURVIVORS ====="
)

for i, x in enumerate(
    sample(
        kept,
        50,
    ),
    1,
):
    print(
        f"\n[{i:02d}] "
        f"selected={x['selected_roles']}"
    )

    print(
        "Q:",
        x["query"],
    )


print(
    "\n===== 25 RANDOM REJECTED ====="
)

for i, x in enumerate(
    sample(
        rejected,
        25,
    ),
    1,
):
    print(
        f"\n[{i:02d}] "
        f"reasons={x['audit_v3']['reasons']}"
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
    OUT / "high_precision_candidates.jsonl",
)

print(
    " ",
    OUT / "human_audit.json",
)
