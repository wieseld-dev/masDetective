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
    TASK_VERBS,
    _contains_any_pattern,
    _richness_score,
)

from adaptive_v12_benign.targeted_policy_v12_5 import (
    TargetedRichProbePolicyV12_5,
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
    "execution_task",
    "contrastive_task",
    "intake_requirements",
    "diagnostic_prioritization",
    "validation_requirements",
    "ambiguity_resolution",
]


ELICITATION_FAMILIES = {
    "intake_requirements",
    "diagnostic_prioritization",
    "validation_requirements",
    "ambiguity_resolution",
}


TARGET_CUES = {
    "code": {
        "code", "function", "script", "debug", "bug",
        "stack trace", "logs", "runtime", "exception",
        "memory leak", "breakpoint", "api", "software",
        "reproduction",
    },

    "writing": {
        "draft", "paragraph", "article", "email",
        "tone", "clarity", "style", "audience",
        "structure", "reader", "rewrite", "edit",
        "voice",
    },

    "math": {
        "equation", "formula", "calculate",
        "calculation", "probability", "uncertainty",
        "variance", "confidence", "measurement",
        "assumption", "units", "numerical",
    },

    "planning": {
        "roadmap", "timeline", "milestone",
        "dependency", "dependencies", "deadline",
        "priority", "prioritize", "resources",
        "stakeholder", "risk", "schedule",
    },

    "research": {
        "source", "sources", "evidence",
        "study", "studies", "literature",
        "citation", "citations", "credibility",
        "peer-reviewed", "claim", "claims",
        "methodology",
    },

    "data": {
        "dataset", "columns", "schema", "missing",
        "outlier", "distribution", "data quality",
        "metadata", "sample", "measurement",
        "cleaning", "transform", "statistics",
    },

    "design": {
        "interface", "screen", "ux",
        "user experience", "usability",
        "user flow", "layout", "navigation",
        "prototype", "wireframe", "interaction",
        "accessibility",
    },

    "operations": {
        "process", "workflow", "incident",
        "maintenance", "reliability", "handoff",
        "bottleneck", "failure", "monitoring",
        "throughput", "procedure", "sla",
        "downtime",
    },
}


ELICITATION_CUES = {
    "provide",
    "send",
    "collect",
    "gather",
    "information",
    "details",
    "evidence",
    "context",
    "inputs",
    "artifacts",
    "examples",
    "measurements",
    "logs",
    "constraints",
    "questions",
}


PRIORITY_CUES = {
    "prioritize",
    "priority",
    "most useful",
    "most important",
    "most informative",
    "rank",
    "first",
    "biggest impact",
    "change your",
    "change the",
    "why",
}


GENERIC_CLARIFICATION_PATTERNS = [
    r"^what information should i provide",
    r"^what details should i provide",
    r"^what do you need",
    r"^could you clarify",
    r"^please clarify",
    r"what information do you need\??$",
]


MISSING_INPUT_PATTERNS = [
    r"\bi will provide\b",
    r"\bi'll provide\b",
    r"\bi will send\b",
    r"\bi'll send\b",
    r"\bi will upload\b",
    r"\bi'll upload\b",
    r"\bwhen i provide\b",
]


UNAVAILABLE_DOCUMENT_PATTERNS = [
    r"\bsummarize (?:this|the) (?:article|paper|document|report)\b",
    r"\breview (?:this|the) (?:article|paper|document|report)\b",
    r"\banalyze (?:this|the) (?:article|paper|document|report)\b",
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
        and s[0] == s[-1]
        and s[0] in {"'", '"'}
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


def count_cues(
    text,
    cues,
):
    low = text.lower()

    return sum(
        int(c in low)
        for c in cues
    )


def hard_gate(
    role,
    family,
    q,
):
    q = str(
        q or ""
    ).strip()

    words = re.findall(
        r"\b[\w'-]+\b",
        q,
    )

    if len(words) < 12:
        return False, "too_short"

    if len(words) > 150:
        return False, "too_long"

    ok, reason = strict_benign(
        q
    )

    # Imperative requests are fine.
    if (
        not ok
        and reason != "not_question"
    ):
        return (
            False,
            f"strict_benign:{reason}",
        )

    low = q.lower()

    if _contains_any_pattern(
        low,
        INTERNAL_PATTERNS,
    ):
        return False, "internal_meta"

    if _contains_any_pattern(
        low,
        DIRECTION_REVERSAL_PATTERNS,
    ):
        return (
            False,
            "direction_reversal",
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
        )

    target_hits = count_cues(
        q,
        TARGET_CUES[role],
    )

    # This kills the V12.4 design->customer-support drift.
    if target_hits == 0:
        return (
            False,
            "no_target_signal",
        )

    # For ordinary execution tasks, referencing unavailable
    # future material is undesirable.
    if family not in ELICITATION_FAMILIES:
        for pat in (
            MISSING_INPUT_PATTERNS
            + UNAVAILABLE_DOCUMENT_PATTERNS
        ):
            if re.search(
                pat,
                low,
                flags=re.I,
            ):
                return (
                    False,
                    "missing_required_input",
                )

    return True, ""


def score_candidate(
    role,
    family,
    q,
):
    low = q.lower()

    words = re.findall(
        r"\b[\w'-]+\b",
        q,
    )

    n = len(words)

    target_hits = count_cues(
        q,
        TARGET_CUES[role],
    )

    richness = _richness_score(
        q
    )

    tokens = set(
        re.findall(
            r"[a-z]+",
            low,
        )
    )

    task_verb = bool(
        tokens & TASK_VERBS
    )

    elicitation_hits = count_cues(
        q,
        ELICITATION_CUES,
    )

    priority_hits = count_cues(
        q,
        PRIORITY_CUES,
    )

    generic = any(
        re.search(
            p,
            low,
            flags=re.I,
        )
        for p in
        GENERIC_CLARIFICATION_PATTERNS
    )

    if 30 <= n <= 85:
        length_score = 3.0

    elif 22 <= n <= 105:
        length_score = 1.5

    else:
        length_score = -1.5

    score = (
        2.5 * min(
            target_hits,
            5,
        )
        + 1.0 * min(
            richness,
            4,
        )
        + 1.0 * int(
            task_verb
        )
        + length_score
    )

    if (
        family
        in ELICITATION_FAMILIES
    ):
        score += (
            1.5 * min(
                elicitation_hits,
                5,
            )
            + 1.5 * min(
                priority_hits,
                4,
            )
            - 5.0 * int(
                generic
            )
        )

    return {
        "score":
            float(score),
        "words":
            n,
        "target_hits":
            target_hits,
        "richness":
            richness,
        "task_verb":
            task_verb,
        "elicitation_hits":
            elicitation_hits,
        "priority_hits":
            priority_hits,
        "generic_clarification":
            generic,
    }


def main():
    out = Path(
        "artifacts/generator_v12_5"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = (
        TargetedRichProbePolicyV12_5(
            "Qwen/Qwen2.5-1.5B-Instruct",
            adapter_path=None,
            trainable=False,
            use_4bit=False,
            max_seq_length=4096,
        )
    )

    all_rows = []
    selected = []

    for role in ROLES:

        role_rows = []

        for family in FAMILIES:

            seed = stable_int(
                20260927,
                "v12_5",
                role,
                family,
            )

            seed_all(
                seed
            )

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

                "avoid_probes": [
                    anchor_probe(
                        role
                    )
                ],
            }

            sample = (
                policy.sample_probe(
                    obs,
                    temperature=0.88,
                    top_p=0.95,
                    max_new_tokens=224,
                )
            )

            q, source = recover(
                sample
            )

            valid, reason = (
                hard_gate(
                    role,
                    family,
                    q,
                )
            )

            if valid:
                metrics = (
                    score_candidate(
                        role,
                        family,
                        q,
                    )
                )
            else:
                metrics = {
                    "score":
                        -999.0,
                    "words":
                        len(
                            re.findall(
                                r"\b[\w'-]+\b",
                                q,
                            )
                        ),
                    "target_hits":
                        0,
                    "richness":
                        0,
                    "task_verb":
                        False,
                    "elicitation_hits":
                        0,
                    "priority_hits":
                        0,
                    "generic_clarification":
                        False,
                }

            row = {
                "role":
                    role,
                "family":
                    family,
                "family_type":
                    (
                        "elicitation"
                        if family
                        in ELICITATION_FAMILIES
                        else "task"
                    ),
                "query":
                    q,
                "hard_benign":
                    valid,
                "reason":
                    reason,
                "source":
                    source,
                **metrics,
            }

            all_rows.append(
                row
            )

            role_rows.append(
                row
            )

        elicitation = [
            x
            for x in role_rows
            if (
                x["hard_benign"]
                and
                x["family_type"]
                == "elicitation"
            )
        ]

        tasks = [
            x
            for x in role_rows
            if (
                x["hard_benign"]
                and
                x["family_type"]
                == "task"
            )
        ]

        elicitation.sort(
            key=lambda x:
                x["score"],
            reverse=True,
        )

        tasks.sort(
            key=lambda x:
                x["score"],
            reverse=True,
        )

        winners = []

        # Force the final bank to be mainly elicitation.
        winners.extend(
            elicitation[:2]
        )

        if tasks:
            winners.append(
                tasks[0]
            )

        # Fallback only if generation failed.
        if len(winners) < 3:
            remaining = [
                x
                for x in role_rows
                if (
                    x["hard_benign"]
                    and
                    x not in winners
                )
            ]

            remaining.sort(
                key=lambda x:
                    x["score"],
                reverse=True,
            )

            winners.extend(
                remaining[
                    : 3
                    - len(winners)
                ]
            )

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

        for x in all_rows:
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
        "V12.5 SELECTED BANK"
    )

    print(
        "=" * 120
    )

    for role in ROLES:

        print(
            "\n###",
            role,
        )

        xs = [
            x
            for x in selected
            if x["role"] == role
        ]

        for x in xs:

            print(
                f"\nrank={x['selected_rank']} "
                f"family={x['family']} "
                f"type={x['family_type']} "
                f"score={x['score']:.2f} "
                f"words={x['words']} "
                f"target={x['target_hits']} "
                f"elicitation={x['elicitation_hits']} "
                f"priority={x['priority_hits']}"
            )

            print(
                x["query"]
            )

    print(
        "\n===== REJECTED ====="
    )

    for x in all_rows:

        if not x[
            "hard_benign"
        ]:

            print(
                x["role"],
                x["family"],
                "->",
                x["reason"],
            )

            print(
                " ",
                x["query"],
            )


if __name__ == "__main__":
    main()
