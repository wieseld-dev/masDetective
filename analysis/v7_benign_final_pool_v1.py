from __future__ import annotations

import json
import random
import re
from pathlib import Path
from collections import Counter


STRICT = Path(
    "artifacts/v7_historical_benign_audit_v1/"
    "strict_benign.jsonl"
)

REJECTED = Path(
    "artifacts/v7_historical_benign_audit_v1/"
    "rejected.jsonl"
)

OUT = Path(
    "artifacts/v7_benign_final_pool_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
rng = random.Random(SEED)


# ============================================================
# FINAL VALIDITY PHILOSOPHY
#
# KEEP BY DEFAULT.
#
# Validity != fingerprint utility.
#
# DO NOT reject:
#   - clarification questions
#   - "What information should I provide?"
#   - "Should I explain ...?"
#   - "Should I provide ...?"
#   - "Should I gather ...?"
#   - "Would more context help?"
#   - "Do you know how to ...?"
#   - normal task/domain questions
#
# Reject ONLY extremely explicit:
#   A. surveying the AI's standing abilities/experience/expertise
#   B. probing hidden/internal system structure
#   C. obvious synthetic template placeholders
# ============================================================


# ------------------------------------------------------------
# A. HIGH-CONFIDENCE DIRECT CAPABILITY SURVEY
# ------------------------------------------------------------

CAPABILITY_PATTERNS = [
    # Explicit experience / expertise survey.
    r"\bdo you have (?:any |much |extensive |prior |previous )?"
    r"(?:experience|expertise|skills?|background)\b",

    r"\bare you (?:experienced|skilled|proficient|knowledgeable)"
    r"(?: in| with| at)?\b",

    # "Do you work well with databases?"
    r"\bdo you work well with\b",

    # Standing / habitual behavior.
    r"\bdo you "
    r"(?:typically|usually|often|frequently|commonly|regularly) "
    r"(?:perform|conduct|analy[sz]e|review|handle|manage|"
    r"work on|engage in|deal with|create|design|develop|"
    r"collaborate on)\b",

    # Explicit self-description.
    r"\bwhat are your (?:capabilities|skills|strengths)\b",
    r"\bwhat are your areas of expertise\b",
    r"\bwhat is your expertise\b",
    r"\bwhat do you specialize in\b",

    # Explicit system difficulty self-assessment.
    r"\bwhat would you find .*challenging\b",
    r"\bwhich .* would you find .*challenging\b",
    r"\bwhat do you find .*challenging\b",
]


# ------------------------------------------------------------
# B. EXPLICIT INTERNAL SYSTEM PROBING
# ------------------------------------------------------------

INTERNAL_PATTERNS = [
    r"\bsystem prompt\b",
    r"\bhidden prompt\b",
    r"\binternal prompt\b",

    r"\binternal architecture\b",
    r"\bhidden architecture\b",
    r"\binternal workflow\b",
    r"\binternal routing\b",

    r"\bhidden components\b",
    r"\bhidden roles\b",

    r"\bwhich internal agent\b",
    r"\bwhat internal agent\b",

    r"\bwhich agent handles\b",
    r"\bwhat agent handles\b",

    r"\bdo you have an? .*agent\b",

    r"\bwhat internal tools\b",
    r"\bwhich internal tools\b",

    r"\bwhat tools do you have internally\b",
    r"\bwhich tools do you have internally\b",

    r"\bwhich hidden role\b",
    r"\bwhat hidden role\b",
]


# ------------------------------------------------------------
# C. OBVIOUS SYNTHETIC PLACEHOLDERS
#
# Natural "this dataset" / "these documents" is KEEP.
# ------------------------------------------------------------

PLACEHOLDER_PATTERNS = [
    r"\[[^\]\n]{1,120}\]",
    r"_{3,}",
    r"\{\{[^}]+\}\}",
    r"<insert [^>]+>",
]


def norm(x):
    return " ".join(
        str(x or "")
        .strip()
        .lower()
        .split()
    )


def matches(patterns, text):
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

    capability = matches(
        CAPABILITY_PATTERNS,
        t,
    )

    internal = matches(
        INTERNAL_PATTERNS,
        t,
    )

    placeholder = matches(
        PLACEHOLDER_PATTERNS,
        t,
    )

    reasons = []

    if capability:
        reasons.append(
            "DIRECT_CAPABILITY"
        )

    if internal:
        reasons.append(
            "INTERNAL_SYSTEM"
        )

    if placeholder:
        reasons.append(
            "SYNTHETIC_PLACEHOLDER"
        )

    return {
        "valid_benign":
            len(reasons) == 0,

        "invalid_reasons":
            reasons,

        "matches": {
            "capability":
                capability,

            "internal":
                internal,

            "placeholder":
                placeholder,
        },
    }


def read_jsonl(path):
    with path.open(
        encoding="utf-8"
    ) as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


rows = (
    read_jsonl(STRICT)
    + read_jsonl(REJECTED)
)


# ============================================================
# Reconstruct unique execution dataset.
# ============================================================

unique = {}

for x in rows:
    key = (
        x.get("session_id"),
        x.get("query"),
        x.get("final_output"),
        tuple(
            x.get(
                "selected_roles",
                []
            )
        ),
    )

    unique[key] = x


rows = list(
    unique.values()
)


print(
    "reconstructed unique:",
    len(rows)
)

if len(rows) != 26747:
    print(
        "WARNING expected 26747"
    )


classified = []

for x in rows:
    audit = classify(
        x["query"]
    )

    classified.append({
        **x,
        "final_validity": audit,
    })


valid = [
    x
    for x in classified
    if x[
        "final_validity"
    ][
        "valid_benign"
    ]
]

rejected = [
    x
    for x in classified
    if not x[
        "final_validity"
    ][
        "valid_benign"
    ]
]


# ============================================================
# Statistics
# ============================================================

reason_counts = Counter()
role_counts = Counter()

for x in rejected:
    for reason in x[
        "final_validity"
    ][
        "invalid_reasons"
    ]:
        reason_counts[
            reason
        ] += 1


for x in valid:
    for role in x.get(
        "selected_roles",
        []
    ):
        role_counts[
            role
        ] += 1


sessions = {
    x.get("session_id")
    for x in valid
    if x.get("session_id")
}


summary = {
    "input_unique":
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
        len(rejected),

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
    OUT / "classified.jsonl",
    classified,
)

write_jsonl(
    OUT / "valid_benign.jsonl",
    valid,
)

write_jsonl(
    OUT / "rejected.jsonl",
    rejected,
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


# ============================================================
# Human audit.
# ============================================================

keep_sample = (
    valid
    if len(valid) <= 100
    else rng.sample(
        valid,
        100
    )
)

reject_sample = (
    rejected
    if len(rejected) <= 60
    else rng.sample(
        rejected,
        60
    )
)


(
    OUT / "human_audit.json"
).write_text(
    json.dumps(
        {
            "keep_100":
                keep_sample,

            "reject_60":
                reject_sample,
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


print(
    "\n===== FINAL BENIGN POOL SUMMARY ====="
)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)


print(
    "\n===== 60 RANDOM KEEP ====="
)

for i, x in enumerate(
    keep_sample[:60],
    1
):
    print(
        f"\n[{i:02d}] "
        f"selected={x.get('selected_roles')}"
    )

    print(
        "Q:",
        x["query"]
    )


print(
    "\n===== 40 RANDOM REJECT ====="
)

for i, x in enumerate(
    reject_sample[:40],
    1
):
    print(
        f"\n[{i:02d}] "
        f"reasons="
        f"{x['final_validity']['invalid_reasons']}"
    )

    print(
        "Q:",
        x["query"]
    )


print(
    "\nSaved:"
)

print(
    " ",
    OUT / "valid_benign.jsonl"
)

print(
    " ",
    OUT / "summary.json"
)

print(
    " ",
    OUT / "human_audit.json"
)
