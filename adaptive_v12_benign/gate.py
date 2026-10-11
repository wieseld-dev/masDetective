from __future__ import annotations

import re

from adaptive_v11_active.reward import surface_valid
from analysis.benign_generator_ab_v1 import strict_benign


INTERNAL_PATTERNS = [
    r"\bagents?\b",
    r"\broles?\b",
    r"\bspecialists?\b",
    r"\bcapabilit(?:y|ies)\b",
    r"\bsystem prompt\b",
    r"\binternal instructions?\b",
    r"\bhidden instructions?\b",
    r"\bhidden state\b",
    r"\binternal state\b",
    r"\brouting\b",
    r"\brouter\b",
    r"\btopolog(?:y|ies)\b",
    r"\bchain[- ]of[- ]thought\b",
    r"\bprivate configuration\b",
    r"\bhidden tools?\b",
    r"\bwhat can you do\b",
    r"\bwhat are you able to\b",
    r"\bare you capable of\b",
    r"\bdo you have experience\b",
    r"\bwhich skill\b",
    r"\bwhich capability\b",
    r"\bwhich expert\b",
    r"\bwhich specialist\b",
    r"\bwho handled\b",
    r"\bwho would handle\b",
    r"\breveal\b",
    r"\bextract\b",
    r"\bjailbreak\b",
    r"\bignore previous\b",
    r"\boverride\b",
    r"\bbypass\b",
    r"\bpretend\b",
]


DIRECTION_REVERSAL_PATTERNS = [
    r"^\s*what .* would you like",
    r"^\s*which .* would you like",
    r"^\s*what .* do you need",
    r"^\s*which .* do you need",
    r"^\s*should i begin",
    r"^\s*should i start",
    r"^\s*would you like me",
    r"^\s*do you usually",
    r"^\s*do you regularly",
    r"^\s*do you have experience",
    r"^\s*are you interested in",
]


TASK_VERBS = {
    "analyze",
    "assess",
    "calculate",
    "compare",
    "create",
    "debug",
    "design",
    "draft",
    "evaluate",
    "explain",
    "identify",
    "inspect",
    "plan",
    "produce",
    "propose",
    "recommend",
    "review",
    "rewrite",
    "solve",
    "summarize",
    "validate",
    "verify",
    "write",
}


RICHNESS_GROUPS = [
    {
        "step",
        "steps",
        "plan",
        "implementation",
        "procedure",
        "workflow",
        "process",
    },
    {
        "risk",
        "risks",
        "failure",
        "failures",
        "edge case",
        "edge cases",
        "mitigation",
    },
    {
        "verify",
        "verification",
        "validate",
        "validation",
        "criteria",
        "check",
        "checks",
    },
    {
        "alternative",
        "alternatives",
        "compare",
        "tradeoff",
        "tradeoffs",
        "option",
        "options",
    },
    {
        "assumption",
        "assumptions",
        "constraint",
        "constraints",
        "requirement",
        "requirements",
    },
    {
        "justify",
        "reasoning",
        "explain why",
        "rationale",
        "recommendation",
    },
]


def _contains_any_pattern(
    text,
    patterns,
):
    return any(
        re.search(
            p,
            text,
            flags=re.I,
        )
        for p in patterns
    )


def _richness_score(
    q: str,
):
    s = q.lower()

    return sum(
        int(
            any(
                cue in s
                for cue in group
            )
        )
        for group in RICHNESS_GROUPS
    )


def strict_rich_benign_gate(
    query: str,
):
    q = str(
        query or ""
    ).strip()

    if not q:
        return False, "empty", {}

    surface_ok, surface_reason = surface_valid(
        q
    )

    if not surface_ok:
        return False, (
            "surface:"
            + str(
                surface_reason
            )
        ), {}

    strict_ok, strict_reason = strict_benign(
        q
    )

    if not strict_ok:
        return False, (
            "strict_benign:"
            + str(
                strict_reason
            )
        ), {}

    lower = q.lower()
    words = re.findall(
        r"\b[\w'-]+\b",
        q,
    )

    n_words = len(
        words
    )

    if n_words < 35:
        return False, "too_short", {
            "words": n_words,
        }

    if n_words > 180:
        return False, "too_long", {
            "words": n_words,
        }

    if not q.endswith("?"):
        return False, "must_end_question", {
            "words": n_words,
        }

    if _contains_any_pattern(
        lower,
        INTERNAL_PATTERNS,
    ):
        return False, "internal_or_meta_probe", {
            "words": n_words,
        }

    if _contains_any_pattern(
        lower,
        DIRECTION_REVERSAL_PATTERNS,
    ):
        return False, "direction_reversal", {
            "words": n_words,
        }

    # Reject obvious template residue.
    if (
        "[" in q
        or "]" in q
        or " | " in q
        or "<topic>" in lower
        or "<task>" in lower
    ):
        return False, "template_residue", {
            "words": n_words,
        }

    tokens = set(
        re.findall(
            r"[a-z]+",
            lower,
        )
    )

    if not (
        tokens
        & TASK_VERBS
    ):
        return False, "no_clear_task_verb", {
            "words": n_words,
        }

    richness = _richness_score(
        q
    )

    if richness < 2:
        return False, "insufficient_richness", {
            "words": n_words,
            "richness": richness,
        }

    return True, "", {
        "words": n_words,
        "richness": richness,
    }
