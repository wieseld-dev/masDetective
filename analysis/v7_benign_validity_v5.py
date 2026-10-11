from __future__ import annotations

import json
import random
import re
from pathlib import Path
from collections import Counter


SRC = Path(
    "artifacts/v7_benign_validity_v4/"
    "valid_benign.jsonl"
)

OUT = Path(
    "artifacts/v7_benign_validity_v5"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
rng = random.Random(SEED)


# ============================================================
# V5 PHILOSOPHY
#
# KEEP BY DEFAULT.
#
# We DO NOT reject:
#   - clarification
#   - asking what information to provide
#   - asking whether more context would help
#   - "Should I explain ...?"
#   - "Should I provide ...?"
#   - "Should I gather ...?"
#   - asking the AI what the user should focus on next
#
# Those may have HIGH or LOW fingerprint utility.
# Utility is NOT validity.
#
# Reject only clear violations:
#   1. direct interrogation of the AI/system's own capabilities
#   2. explicit hidden/internal agent/tool/system inspection
#   3. unresolved placeholders / clearly missing referents
# ============================================================


# ------------------------------------------------------------
# 1. DIRECT SYSTEM-CAPABILITY INTERROGATION
# ------------------------------------------------------------

DIRECT_CAPABILITY = [
    # "Do you know how to implement X?"
    r"^\s*do you know how\b",

    # "Do you typically/often/... analyze X?"
    r"^\s*do you (?:typically|usually|often|frequently|commonly|regularly)\b",

    # "Do you have experience with X?"
    r"^\s*do you have .*experience\b",
    r"^\s*do you have experience\b",

    # "Have you worked with X?"
    r"^\s*have you (?:worked|handled|performed|conducted|used)\b",

    # Explicit ability/expertise.
    r"^\s*are you (?:able|capable|experienced|familiar|knowledgeable)\b",

    r"^\s*can you handle\b",

    # Directly asks what the system can do.
    r"\bwhat are your capabilities\b",
    r"\bwhat capabilities do you have\b",
    r"\bwhich capabilities do you have\b",

    r"\bwhat is your expertise\b",
    r"\bwhat are your areas of expertise\b",
    r"\bwhat are your strengths\b",
    r"\bwhat do you specialize in\b",

    # Explicit system self-assessment.
    r"\bwhat would you find .*challenging\b",
    r"\bwhich .* would you find .*challenging\b",
    r"\bwhat do you find .*challenging\b",

    # Direct questions about whether the AI performs a domain/task.
    r"^\s*do you (?:perform|conduct|analyze|review|work on|engage in)\b",
]


# ------------------------------------------------------------
# 2. INTERNAL / HIDDEN SYSTEM PROBES
# ------------------------------------------------------------

INTERNAL_SYSTEM = [
    r"\bdo you have an? .*agent\b",
    r"\bwhich .*agent\b",
    r"\bwhat .*agent\b",

    r"\bwhich internal .*agent\b",
    r"\bwhat internal .*agent\b",

    r"\bwhat tools do you have\b",
    r"\bwhich tools do you have\b",
    r"\bwhat tool do you use\b",
    r"\bwhich tool do you use\b",

    r"\bsystem prompt\b",
    r"\bhidden prompt\b",
    r"\binternal prompt\b",

    r"\binternal architecture\b",
    r"\binternal workflow\b",
    r"\binternal routing\b",

    r"\bhidden architecture\b",
    r"\bhidden components\b",
    r"\bhidden roles\b",

    r"\bwhich role handles\b",
    r"\bwhat role handles\b",

    r"\bwhich capability handles\b",
    r"\bwhat capability handles\b",
]


# ------------------------------------------------------------
# 3. CLEARLY MISSING CONTEXT
#
# Keep this narrow.
# We do NOT reject ordinary "this project" / "my project".
# ------------------------------------------------------------

MISSING_CONTEXT = [
    # Explicit generated placeholders.
    r"\[[^\]]+\]",
    r"_{3,}",

    # Clear plural deictics with no supplied object in our
    # single-query text benchmark.
    r"^\s*do these tasks\b",
    r"^\s*do these questions\b",
    r"^\s*do these descriptions\b",
    r"^\s*do these pieces\b",
    r"^\s*do these examples\b",
    r"^\s*do these datasets\b",
    r"^\s*do these screenshots\b",

    r"\bthe above\b",
    r"\bthe previous question\b",
]


def normalize(x):
    return " ".join(
        str(x or "")
        .strip()
        .lower()
        .split()
    )


def matched(patterns, text):
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
    t = normalize(query)

    cap = matched(
        DIRECT_CAPABILITY,
        t,
    )

    internal = matched(
        INTERNAL_SYSTEM,
        t,
    )

    context = matched(
        MISSING_CONTEXT,
        t,
    )

    reasons = []

    if cap:
        reasons.append(
            "DIRECT_CAPABILITY"
        )

    if internal:
        reasons.append(
            "INTERNAL_SYSTEM"
        )

    if context:
        reasons.append(
            "MISSING_CONTEXT"
        )

    return {
        "valid_benign":
            len(reasons) == 0,

        "invalid_reasons":
            reasons,

        "matches": {
            "direct_capability":
                cap,

            "internal_system":
                internal,

            "missing_context":
                context,
        },
    }


rows = []

with SRC.open(
    encoding="utf-8"
) as f:
    for line in f:
        if not line.strip():
            continue

        x = json.loads(line)

        audit = classify(
            x["query"]
        )

        rows.append({
            **x,
            "validity_v5":
                audit,
        })


valid = [
    x
    for x in rows
    if x[
        "validity_v5"
    ][
        "valid_benign"
    ]
]

invalid = [
    x
    for x in rows
    if not x[
        "validity_v5"
    ][
        "valid_benign"
    ]
]


reason_counts = Counter()
role_counts = Counter()

for x in invalid:
    for r in x[
        "validity_v5"
    ][
        "invalid_reasons"
    ]:
        reason_counts[r] += 1


for x in valid:
    for r in x[
        "selected_roles"
    ]:
        role_counts[r] += 1


sessions = {
    x["session_id"]
    for x in valid
    if x.get("session_id")
}


summary = {
    "input_from_v4":
        len(rows),

    "valid_benign":
        len(valid),

    "valid_rate":
        (
            len(valid) / len(rows)
            if rows
            else None
        ),

    "rejected":
        len(invalid),

    "unique_sessions_valid":
        len(sessions),

    "rejection_reason_counts":
        dict(
            sorted(
                reason_counts.items()
            )
        ),

    "selected_role_positive_counts_valid":
        dict(
            sorted(
                role_counts.items()
            )
        ),
}


def write_jsonl(path, items):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for item in items:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


write_jsonl(
    OUT / "valid_benign.jsonl",
    valid,
)

write_jsonl(
    OUT / "rejected.jsonl",
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
    "===== VALIDITY V5 SUMMARY ====="
)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)


print(
    "\n===== 80 RANDOM VALID ====="
)

sample_valid = (
    valid
    if len(valid) <= 80
    else rng.sample(
        valid,
        80,
    )
)

for i, x in enumerate(
    sample_valid,
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
    "\n===== 40 RANDOM REJECTED ====="
)

sample_invalid = (
    invalid
    if len(invalid) <= 40
    else rng.sample(
        invalid,
        40,
    )
)

for i, x in enumerate(
    sample_invalid,
    1,
):
    print(
        f"\n[{i:02d}] "
        f"reasons="
        f"{x['validity_v5']['invalid_reasons']}"
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
    OUT / "rejected.jsonl",
)
