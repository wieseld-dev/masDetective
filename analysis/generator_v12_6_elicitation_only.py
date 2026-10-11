from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch

from analysis.benign_generator_ab_v1 import (
    strict_benign,
)

from adaptive_v11_active.io import (
    stable_int,
)

from adaptive_v12_benign.gate_v12_1 import (
    INTERNAL_PATTERNS,
    DIRECTION_REVERSAL_PATTERNS,
    _contains_any_pattern,
    _richness_score,
)

from adaptive_v12_benign.targeted_policy_v12_6 import (
    TargetedElicitationPolicyV12_6,
)

from v11.role_universe import (
    ROLE_DESCRIPTIONS,
    anchor_probe,
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


FAMILIES = [
    "ranked_intake",
    "diagnostic_evidence",
    "validation_evidence",
    "decision_questions",
]


TARGET_CUES = {
    "code": {
        "code", "software", "debug", "bug",
        "error", "exception", "stack trace",
        "logs", "runtime", "reproduction",
        "memory", "api", "function",
        "implementation",
    },

    "writing": {
        "writing", "draft", "essay", "article",
        "paragraph", "email", "tone", "clarity",
        "style", "audience", "structure",
        "reader", "voice", "revision",
    },

    "math": {
        "calculation", "equation", "formula",
        "measurement", "uncertainty", "variance",
        "probability", "confidence", "precision",
        "assumption", "units", "numerical",
        "statistical", "statistics",
    },

    "planning": {
        "plan", "planning", "roadmap",
        "timeline", "milestone", "deadline",
        "dependency", "dependencies",
        "priority", "resources", "stakeholder",
        "schedule", "risk",
    },

    "research": {
        "research", "source", "sources",
        "evidence", "claim", "claims",
        "study", "studies", "literature",
        "citation", "credibility",
        "peer-reviewed", "methodology",
        "publication",
    },

    "data": {
        "data", "dataset", "column", "columns",
        "schema", "metadata", "missing",
        "outlier", "distribution", "sample",
        "data quality", "cleaning",
        "preprocessing", "measurement",
        "transform",
    },

    "design": {
        "design", "interface", "screen",
        "ux", "user experience", "usability",
        "user flow", "layout", "navigation",
        "prototype", "wireframe", "interaction",
        "accessibility",
    },

    "operations": {
        "operations", "operational", "process",
        "workflow", "incident", "maintenance",
        "reliability", "failure", "handoff",
        "bottleneck", "monitoring", "throughput",
        "procedure", "downtime", "sla",
    },
}


PROVIDE_PATTERNS = [
    r"\bwhat .* should i (?:provide|send|collect|gather|prepare|share)\b",
    r"\bwhich .* should i (?:provide|send|collect|gather|prepare|share)\b",
    r"\bwhat .* would .* useful .* (?:provide|send|collect|gather|prepare|share)\b",
    r"\bwhat .* do you need from me\b",
    r"\bwhich .* do you need from me\b",
    r"\bbefore i .* what .* should i\b",
]


PRIORITY_PATTERNS = [
    r"\brank\b",
    r"\branked\b",
    r"\bprioritize\b",
    r"\bprioritise\b",
    r"\bpriority\b",
    r"\bmost important\b",
    r"\bmost useful\b",
    r"\bmost informative\b",
    r"\bhighest diagnostic value\b",
    r"\bin order of\b",
    r"\btop five\b",
    r"\bfive .* most\b",
]


WHY_PATTERNS = [
    r"\bexplain why\b",
    r"\bwhy each\b",
    r"\bwhy .* matters\b",
    r"\bwhat each .* (?:help|show|reveal|indicate|distinguish|rule)\b",
    r"\bhow each .* (?:help|change|affect|inform)\b",
    r"\bwhat .* could help .* (?:distinguish|rule out|validate|decide)\b",
    r"\bwhat .* would tell you\b",
]


FIVE_PATTERNS = [
    r"\bfive\b",
    r"\b5\b",
    r"\b4-6\b",
    r"\btop five\b",
]


GENERIC_ONLY_PATTERNS = [
    r"^what information do you need\??$",
    r"^what details do you need\??$",
    r"^what should i provide\??$",
    r"^could you clarify\b",
]


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def clean(x):
    s = str(
        x or ""
    ).strip()

    s = re.sub(
        r"^\s*(?:question|query|probe|task)\s*:\s*",
        "",
        s,
        flags=re.I,
    ).strip()

    if (
        len(s) >= 2
        and
        s[0] == s[-1]
        and
        s[0] in {"'", '"'}
    ):
        s = s[1:-1].strip()

    return s


def recover(sample):
    for field in (
        "query",
        "completion",
        "raw_completion",
    ):
        q = clean(
            getattr(
                sample,
                field,
                "",
            )
        )

        if q:
            return q, field

    return "", "empty"


def any_match(
    patterns,
    text,
):
    return any(
        re.search(
            p,
            text,
            flags=re.I,
        )
        for p in patterns
    )


def cue_hits(
    text,
    cues,
):
    low = text.lower()

    return sum(
        int(cue in low)
        for cue in cues
    )


def validate(
    role,
    q,
):
    q = str(
        q or ""
    ).strip()

    low = q.lower()

    words = re.findall(
        r"\b[\w'-]+\b",
        q,
    )

    n = len(words)

    if n < 25:
        return False, "too_short", {}

    if n > 125:
        return False, "too_long", {}

    ok, reason = strict_benign(
        q
    )

    # Imperative wording is not a safety failure.
    if (
        not ok
        and
        reason != "not_question"
    ):
        return (
            False,
            f"strict_benign:{reason}",
            {},
        )

    if _contains_any_pattern(
        low,
        INTERNAL_PATTERNS,
    ):
        return False, "internal_meta", {}

    if _contains_any_pattern(
        low,
        DIRECTION_REVERSAL_PATTERNS,
    ):
        return (
            False,
            "direction_reversal",
            {},
        )

    if any(
        x in low
        for x in (
            "<topic>",
            "<task>",
            "{topic}",
            "{task}",
            "<insert",
        )
    ):
        return (
            False,
            "template_residue",
            {},
        )

    if any_match(
        GENERIC_ONLY_PATTERNS,
        low,
    ):
        return (
            False,
            "generic_clarification",
            {},
        )

    target = cue_hits(
        q,
        TARGET_CUES[role],
    )

    asks_provide = any_match(
        PROVIDE_PATTERNS,
        low,
    )

    asks_priority = any_match(
        PRIORITY_PATTERNS,
        low,
    )

    asks_why = any_match(
        WHY_PATTERNS,
        low,
    )

    asks_five = any_match(
        FIVE_PATTERNS,
        low,
    )

    checks = {
        "words":
            n,
        "target_hits":
            target,
        "asks_provide":
            asks_provide,
        "asks_priority":
            asks_priority,
        "asks_why":
            asks_why,
        "asks_five":
            asks_five,
    }

    if target < 2:
        return (
            False,
            "weak_target_signal",
            checks,
        )

    if not asks_provide:
        return (
            False,
            "missing_help_me_help_you",
            checks,
        )

    if not asks_priority:
        return (
            False,
            "missing_prioritization",
            checks,
        )

    if not asks_why:
        return (
            False,
            "missing_why_each",
            checks,
        )

    if not asks_five:
        return (
            False,
            "missing_five_items",
            checks,
        )

    return True, "", checks


def score(
    role,
    q,
    checks,
):
    n = checks[
        "words"
    ]

    target = checks[
        "target_hits"
    ]

    richness = _richness_score(
        q
    )

    if 35 <= n <= 85:
        length_score = 4.0

    elif 30 <= n <= 100:
        length_score = 2.0

    else:
        length_score = 0.0

    low = q.lower()

    distinction_bonus = sum([
        2.0
        * int(
            "rule out"
            in low
        ),
        2.0
        * int(
            "distinguish"
            in low
        ),
        1.5
        * int(
            "change your"
            in low
        ),
        1.5
        * int(
            "diagnostic"
            in low
        ),
        1.0
        * int(
            "confidence"
            in low
        ),
        1.0
        * int(
            "validate"
            in low
        ),
    ])

    return float(
        3.0 * min(
            target,
            5,
        )
        + 1.5 * min(
            richness,
            4,
        )
        + length_score
        + distinction_bonus
    )


def main():
    out = Path(
        "artifacts/generator_v12_6"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = (
        TargetedElicitationPolicyV12_6(
            "Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path=None,
            trainable=False,
            use_4bit=False,
            max_seq_length=4096,
        )
    )

    rows = []
    selected = []

    for role in ROLES:

        role_rows = []

        for family in FAMILIES:

            for rep in range(3):

                seed = stable_int(
                    20260927,
                    "v12_6",
                    role,
                    family,
                    rep,
                )

                seed_all(
                    seed
                )

                previous = [
                    x["query"]
                    for x in role_rows
                    if x.get(
                        "query"
                    )
                ]

                obs = {
                    "candidate_role":
                        role,

                    "candidate_description":
                        ROLE_DESCRIPTIONS[
                            role
                        ],

                    "anchor_probe":
                        anchor_probe(
                            role
                        ),

                    "probe_family":
                        family,

                    "avoid_probes":
                        [
                            anchor_probe(
                                role
                            ),
                            *previous[-5:],
                        ],
                }

                sample = (
                    policy.sample_probe(
                        obs,
                        temperature=0.92,
                        top_p=0.96,
                        max_new_tokens=224,
                    )
                )

                q, source = recover(
                    sample
                )

                valid, reason, checks = (
                    validate(
                        role,
                        q,
                    )
                )

                s = (
                    score(
                        role,
                        q,
                        checks,
                    )
                    if valid
                    else -999.0
                )

                row = {
                    "role":
                        role,
                    "family":
                        family,
                    "rep":
                        rep,
                    "query":
                        q,
                    "source":
                        source,
                    "valid":
                        valid,
                    "reason":
                        reason,
                    "score":
                        s,
                    **checks,
                }

                rows.append(
                    row
                )

                role_rows.append(
                    row
                )

        valid_rows = [
            x
            for x in role_rows
            if x["valid"]
        ]

        valid_rows.sort(
            key=lambda x: (
                x["score"],
                x["target_hits"],
                -x["words"],
            ),
            reverse=True,
        )

        winners = []

        if valid_rows:
            winners.append(
                valid_rows[0]
            )

        if winners:
            different_family = [
                x
                for x in valid_rows[1:]
                if (
                    x["family"]
                    != winners[0]["family"]
                )
            ]

            if different_family:
                winners.append(
                    different_family[0]
                )

        if (
            len(winners) < 2
            and
            len(valid_rows) > 1
        ):
            for x in valid_rows:
                if x not in winners:
                    winners.append(
                        x
                    )

                if len(winners) == 2:
                    break

        for rank, x in enumerate(
            winners,
            1,
        ):
            selected.append({
                **x,
                "selected_rank":
                    rank,
            })

    with (
        out
        / "all_candidates.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (
        out
        / "selected_bank.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        for x in selected:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "\n"
        + "=" * 120
    )
    print(
        "V12.6 STRICT ELICITATION BANK"
    )
    print(
        "=" * 120
    )

    for role in ROLES:

        role_all = [
            x
            for x in rows
            if x["role"] == role
        ]

        role_valid = [
            x
            for x in role_all
            if x["valid"]
        ]

        print(
            f"\n### {role} "
            f"valid={len(role_valid)}/12"
        )

        winners = [
            x
            for x in selected
            if x["role"] == role
        ]

        if not winners:
            print(
                "NO VALID WINNERS"
            )

        for x in winners:

            print(
                f"\nrank={x['selected_rank']} "
                f"family={x['family']} "
                f"rep={x['rep']} "
                f"score={x['score']:.2f} "
                f"words={x['words']} "
                f"target={x['target_hits']}"
            )

            print(
                x["query"]
            )

    print(
        "\n"
        + "=" * 120
    )
    print(
        "REJECTION COUNTS"
    )
    print(
        "=" * 120
    )

    from collections import Counter

    c = Counter(
        x["reason"]
        for x in rows
        if not x["valid"]
    )

    for reason, n in c.most_common():
        print(
            f"{reason:<32} {n}"
        )

    print(
        "\nTOTAL:",
        len(rows),
    )

    print(
        "VALID:",
        sum(
            x["valid"]
            for x in rows
        ),
    )

    print(
        "SELECTED:",
        len(selected),
    )


if __name__ == "__main__":
    main()
