from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


SRC = Path(
    "artifacts/generator_v12_10/all_candidates.jsonl"
)

OUT = Path(
    "artifacts/generator_v12_10d"
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


MIN_TARGET_SIM = 0.30
MIN_MARGIN = 0.05


ASSISTANT_DIRECTED = [
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bwould you\b",
    r"\bwill you\b",

    r"\bhelp me\b",
    r"\bhelp us\b",
    r"\byour help\b",

    r"\bassist me\b",
    r"\bassist us\b",
    r"\bneed assistance\b",
    r"\bneeds assistance\b",

    r"\bguide me\b",
    r"\bguide us\b",
    r"\bneed guidance\b",
    r"\bneeds guidance\b",
    r"\byour guidance\b",

    r"\btell me\b",
    r"\bshow me\b",
    r"\bseeking advice\b",
    r"\bneed advice\b",
    r"\bneeds advice\b",
    r"\byour advice\b",
    r"\byour expertise\b",
]


QUESTIONISH = [
    r"\bwhat specific details\b",
    r"\bwhat details\b",
    r"\bwhat information\b",
    r"\bwhat steps\b",
    r"\bwhat step\b",
    r"\bwhat initial step\b",
    r"\bwhat questions\b",
    r"\bwhat evidence\b",
    r"\bwhat inputs\b",

    r"\bwhich step\b",
    r"\bwhich initial step\b",
    r"\bwhich information\b",
    r"\bwhich details\b",
    r"\bwhich evidence\b",

    r"\bhow can i\b",
    r"\bhow can we\b",
    r"\bhow should i\b",
    r"\bhow should we\b",
    r"\bhow do i\b",
    r"\bhow do we\b",
]


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
]


# We do not want safety/high-stakes behavior to become
# a hidden confound in a role fingerprint experiment.
HIGH_STAKES = [
    r"\bpatient\b",
    r"\bpatients\b",
    r"\bdrug dosage\b",
    r"\bdosage\b",
    r"\bdisease\b",
    r"\bmedical diagnosis\b",
    r"\btreatment\b",
    r"\bclinical treatment\b",
    r"\bdoctor\b",

    r"\bstock price\b",
    r"\bstock prices\b",
    r"\bstock market\b",
    r"\btrading\b",
    r"\binvestment advice\b",

    r"\blegal advice\b",
    r"\bcriminal case\b",
]


# These are not forbidden words in general.
# They are cues that can pre-answer the later
# "what evidence should I provide?" question.
#
# We only reject when a scenario contains too many
# such cues for its intended role.
EVIDENCE_CUES = {
    "code": [
        "logs",
        "error message",
        "error messages",
        "stack trace",
        "stack traces",
        "reproduction steps",
        "environment details",
        "software version",
        "recent changes",
    ],

    "writing": [
        "target audience",
        "audience",
        "tone",
        "clarity",
        "structure",
        "readability",
        "style",
        "word limit",
    ],

    "math": [
        "sample size",
        "confidence level",
        "uncertainty",
        "variance",
        "units",
        "assumptions",
        "error rates",
        "calibration",
        "expected variability",
    ],

    "planning": [
        "goals",
        "milestones",
        "resources",
        "risks",
        "dependencies",
        "deadlines",
        "stakeholders",
        "budget",
    ],

    "research": [
        "sources",
        "studies",
        "evidence",
        "citations",
        "methodology",
        "credibility",
        "peer-reviewed",
        "sample size",
    ],

    "data": [
        "missing values",
        "duplicates",
        "schema",
        "column types",
        "metadata",
        "outliers",
        "preprocessing",
        "data types",
    ],

    "design": [
        "user feedback",
        "analytics",
        "wireframes",
        "prototype",
        "prototypes",
        "usability tests",
        "target audience",
        "interaction patterns",
    ],

    "operations": [
        "logs",
        "maintenance records",
        "failure history",
        "incident records",
        "downtime",
        "throughput",
        "sla",
        "monitoring metrics",
    ],
}


MAX_CUES = {
    "code": 1,
    "writing": 3,
    "math": 1,
    "planning": 3,
    "research": 3,
    "data": 1,
    "design": 2,
    "operations": 1,
}


BAD_ENDINGS = [
    " before making any business.",
    " to ensure.",
    " to make it.",
    " to start with",
    " to identify potential causes",
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


def cue_count(role, text):
    low = text.lower()

    return sum(
        int(cue in low)
        for cue in EVIDENCE_CUES[role]
    )


def gate(row):
    scenario = str(
        row.get(
            "scenario",
            ""
        )
    ).strip()

    low = scenario.lower()

    if not row.get(
        "surface_valid",
        False,
    ):
        return False, "old_surface_invalid", 0

    sim = row.get(
        "target_similarity"
    )

    margin = row.get(
        "target_margin"
    )

    if (
        sim is None
        or sim < MIN_TARGET_SIM
    ):
        return False, "low_target_similarity", 0

    if (
        margin is None
        or margin < MIN_MARGIN
    ):
        return False, "low_target_margin", 0

    if "?" in scenario:
        return False, "question_mark", 0

    if matches(
        ASSISTANT_DIRECTED,
        low,
    ):
        return False, "assistant_directed", 0

    if matches(
        QUESTIONISH,
        low,
    ):
        return False, "questionish_clause", 0

    if matches(
        ANSWER_LEAKAGE,
        low,
    ):
        return False, "answer_leakage", 0

    if matches(
        HIGH_STAKES,
        low,
    ):
        return False, "high_stakes_domain", 0

    if any(
        low.endswith(x)
        for x in BAD_ENDINGS
    ):
        return False, "incomplete_ending", 0

    n_cues = cue_count(
        row["role"],
        scenario,
    )

    if n_cues > MAX_CUES[
        row["role"]
    ]:
        return False, "evidence_cue_leakage", n_cues

    return True, "", n_cues


rows = [
    json.loads(line)
    for line in SRC.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]


for row in rows:

    ok, reason, cues = gate(
        row
    )

    row[
        "v12_10d_valid"
    ] = ok

    row[
        "v12_10d_reason"
    ] = reason

    row[
        "evidence_cue_count"
    ] = cues


selected = []


for role in ROLES:

    candidates = [
        x
        for x in rows
        if (
            x["role"] == role
            and
            x["v12_10d_valid"]
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

    if candidates:
        winners.append(
            candidates[0]
        )

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
    "V12.10d SCIENTIFIC CLEAN RE-RANK"
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
            x["v12_10d_valid"]
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
            f"cues={x['evidence_cue_count']} "
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
    "TOTAL:",
    len(rows),
)

print(
    "SCIENTIFIC CLEAN:",
    sum(
        x[
            "v12_10d_valid"
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
    x["v12_10d_reason"]
    for x in rows
    if not x[
        "v12_10d_valid"
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
