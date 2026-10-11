from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


SRC = Path(
    "artifacts/generator_v12_10/all_candidates.jsonl"
)

OUT = Path(
    "artifacts/generator_v12_10c"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


ROLES = [
    "code",
    "writing",
    "math",
    "planning",
    "research",
    "data",
    "design",
    "operations",
]


# These phrases mean the generated "scenario" has started
# talking TO the assistant instead of merely describing
# the user's situation.
ASSISTANT_DIRECTED = [
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bwould you\b",
    r"\bwill you\b",

    r"\bhelp me\b",
    r"\bhelp us\b",
    r"\bassist me\b",
    r"\bassist us\b",
    r"\bguide me\b",
    r"\bguide us\b",
    r"\btell me\b",
    r"\bshow me\b",

    r"\bneed assistance\b",
    r"\bneeds assistance\b",
    r"\bneed guidance\b",
    r"\bneeds guidance\b",
    r"\bseeking advice\b",
    r"\bseek advice\b",
    r"\bneed advice\b",
    r"\bneeds advice\b",
    r"\byour expertise\b",
    r"\byour help\b",
    r"\byour guidance\b",
    r"\byour advice\b",
]


# Important: V12.10 sometimes stripped the literal '?'
# while leaving the interrogative sentence behind.
QUESTIONISH = [
    r"\bwhat specific details\b",
    r"\bwhat details\b",
    r"\bwhat information\b",
    r"\bwhat steps\b",
    r"\bwhat step\b",
    r"\bwhat initial step\b",
    r"\bwhich step\b",
    r"\bwhich initial step\b",
    r"\bwhat questions\b",
    r"\bwhat evidence\b",
    r"\bwhat inputs\b",

    r"\bhow can i\b",
    r"\bhow can we\b",
    r"\bhow should i\b",
    r"\bhow should we\b",
    r"\bhow do i\b",
    r"\bhow do we\b",

    r"\bwhich information\b",
    r"\bwhich details\b",
    r"\bwhich steps\b",
    r"\bwhich evidence\b",
]


# These pre-answer the later "what should I provide?"
# elicitation question.
ANSWER_LEAKAGE = [
    r"\bi should start by\b",
    r"\bwe should start by\b",
    r"\bshould start by\b",
    r"\bfirst i should\b",
    r"\bfirst we should\b",

    r"\bneed to collect\b",
    r"\bneeds to collect\b",
    r"\bneed to gather\b",
    r"\bneeds to gather\b",

    r"\bmust collect\b",
    r"\bmust gather\b",

    r"\bcollect detailed logs\b",
    r"\breview maintenance records\b",
    r"\bcheck for missing values\b",
    r"\bchecking for missing values\b",
    r"\bhandling duplicates\b",
]


def matches(patterns, text):
    return any(
        re.search(
            p,
            text,
            flags=re.I,
        )
        for p in patterns
    )


def clean_gate(row):
    s = str(
        row.get(
            "scenario",
            ""
        )
    ).strip()

    low = s.lower()

    if not row.get(
        "surface_valid",
        False,
    ):
        return False, "old_surface_invalid"

    # Keep V12.9/V12.10 semantic quality requirement.
    sim = row.get(
        "target_similarity"
    )

    margin = row.get(
        "target_margin"
    )

    if (
        sim is None
        or sim < 0.28
    ):
        return False, "low_target_similarity"

    if (
        margin is None
        or margin < 0.03
    ):
        return False, "low_target_margin"

    if "?" in s:
        return False, "question_mark"

    if matches(
        ASSISTANT_DIRECTED,
        low,
    ):
        return False, "assistant_directed"

    if matches(
        QUESTIONISH,
        low,
    ):
        return False, "questionish_clause"

    if matches(
        ANSWER_LEAKAGE,
        low,
    ):
        return False, "answer_leakage"

    # Reject obvious deterministic truncation / incomplete endings.
    bad_endings = [
        " before making any business.",
        " to ensure.",
        " to make it.",
        " to start with",
        " to identify potential causes",
    ]

    if any(
        low.endswith(x)
        for x in bad_endings
    ):
        return False, "incomplete_ending"

    return True, ""


rows = [
    json.loads(line)
    for line in SRC.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]


for row in rows:

    ok, reason = clean_gate(
        row
    )

    row[
        "v12_10b_valid"
    ] = ok

    row[
        "v12_10b_reason"
    ] = reason


selected = []


for role in ROLES:

    candidates = [
        x
        for x in rows
        if (
            x["role"] == role
            and
            x["v12_10b_valid"]
        )
    ]

    candidates.sort(
        key=lambda x: (
            x.get(
                "score",
                -999,
            ),
            x.get(
                "target_margin",
                -999,
            ),
            x.get(
                "target_similarity",
                -999,
            ),
        ),
        reverse=True,
    )

    winners = []

    # Best overall candidate.
    if candidates:
        winners.append(
            candidates[0]
        )

    # Prefer diversity: second winner from another family.
    if winners:

        diverse = [
            x
            for x in candidates[1:]
            if (
                x["family"]
                != winners[0]["family"]
            )
        ]

        if diverse:
            winners.append(
                diverse[0]
            )

    # Fallback if only one family survived.
    if len(winners) < 2:

        for x in candidates:

            if x not in winners:
                winners.append(
                    x
                )

            if len(winners) == 2:
                break

    for rank, row in enumerate(
        winners,
        1,
    ):
        selected.append({
            **row,
            "selected_rank":
                rank,
        })


with (
    OUT
    / "all_candidates_reranked.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for row in rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


with (
    OUT
    / "selected_bank.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for row in selected:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "=" * 120
)

print(
    "V12.10c CLEAN RE-RANK"
)

print(
    "=" * 120
)


for role in ROLES:

    valid = [
        x
        for x in rows
        if (
            x["role"] == role
            and
            x["v12_10b_valid"]
        )
    ]

    winners = [
        x
        for x in selected
        if x["role"] == role
    ]

    print(
        f"\n### {role} "
        f"clean_candidates={len(valid)}/24 "
        f"selected={len(winners)}/2"
    )

    for x in winners:

        print(
            f"\nrank={x['selected_rank']} "
            f"family={x['family']} "
            f"rep={x['rep']} "
            f"target={x['target_similarity']:.4f} "
            f"other={x['best_other_role']}:"
            f"{x['best_other_similarity']:.4f} "
            f"margin={x['target_margin']:+.4f} "
            f"score={x['score']:.3f}"
        )

        print(
            "SCENARIO:",
            x["scenario"],
        )

        print(
            "QUERY:",
            x["query"],
        )


print(
    "\n"
    + "=" * 120
)

print(
    "SUMMARY"
)

print(
    "=" * 120
)

print(
    "TOTAL GENERATED:",
    len(rows),
)

print(
    "CLEAN VALID:",
    sum(
        x[
            "v12_10b_valid"
        ]
        for x in rows
    ),
)

print(
    "SELECTED:",
    len(selected),
    "/ 16",
)


print(
    "\nREJECTIONS:"
)

counts = Counter(
    x["v12_10b_reason"]
    for x in rows
    if not x[
        "v12_10b_valid"
    ]
)

for reason, n in counts.most_common():

    print(
        f"{reason:<28} {n}"
    )


print(
    "\nMISSING ROLES:"
)

missing = []

for role in ROLES:

    n = sum(
        x["role"] == role
        for x in selected
    )

    if n < 2:
        missing.append(
            (role, n)
        )


if not missing:

    print(
        "NONE"
    )

else:

    for role, n in missing:

        print(
            f"{role}: {n}/2"
        )
