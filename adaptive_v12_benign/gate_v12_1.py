from __future__ import annotations

import re

from analysis.benign_generator_ab_v1 import strict_benign


# These are HARD rejects.
#
# Important:
# We intentionally do NOT ban ordinary external words such as:
#   role, capability, extract, analyze, multiple questions, etc.
#
# We ban them only when they clearly refer to the hidden/internal system.
INTERNAL_PATTERNS = [
    r"\bsystem prompt\b",
    r"\bhidden prompt\b",
    r"\binternal prompt\b",
    r"\binternal instructions?\b",
    r"\bhidden instructions?\b",
    r"\bprivate instructions?\b",
    r"\bchain[- ]of[- ]thought\b",
    r"\bhidden state\b",
    r"\binternal state\b",
    r"\bprivate configuration\b",
    r"\binternal configuration\b",
    r"\binternal routing\b",
    r"\bhidden routing\b",
    r"\binternal router\b",
    r"\bhidden router\b",
    r"\binternal topology\b",
    r"\bhidden topology\b",
    r"\binternal tools?\b",
    r"\bhidden tools?\b",

    # Explicit system/capability probing.
    r"\bwhat can you do\b",
    r"\bwhat are you able to do\b",
    r"\bwhat are your capabilities\b",
    r"\bwhich capabilities do you have\b",
    r"\bare you capable of\b",
    r"\bdo you have experience\b",
    r"\bwhich specialist\b.*\b(?:handle|handled|answer)",
    r"\bwhich agent\b.*\b(?:handle|handled|answer)",
    r"\bwhich role\b.*\b(?:handle|handled|answer)",
    r"\bwho handled\b",
    r"\bwho processed\b",

    # Explicit jailbreak/meta manipulation.
    r"\bignore (?:all |the )?(?:previous|prior) instructions?\b",
    r"\boverride (?:your|the|system) instructions?\b",
    r"\bbypass (?:your|the|system) (?:rules|instructions|safety)\b",
    r"\breveal (?:your|the) (?:system|hidden|internal) prompt\b",
    r"\bjailbreak\b",
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
    "audit",
    "calculate",
    "clean",
    "compare",
    "create",
    "debug",
    "design",
    "develop",
    "diagnose",
    "draft",
    "estimate",
    "evaluate",
    "explain",
    "identify",
    "implement",
    "inspect",
    "investigate",
    "optimize",
    "organize",
    "outline",
    "plan",
    "prioritize",
    "produce",
    "propose",
    "recommend",
    "review",
    "rewrite",
    "solve",
    "structure",
    "summarize",
    "translate",
    "troubleshoot",
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
        "timeline",
        "milestone",
        "milestones",
        "dependency",
        "dependencies",
    },
    {
        "risk",
        "risks",
        "failure",
        "failures",
        "edge case",
        "edge cases",
        "mitigation",
        "contingency",
        "challenge",
        "challenges",
    },
    {
        "verify",
        "verification",
        "validate",
        "validation",
        "criteria",
        "check",
        "checks",
        "test",
        "tests",
        "metric",
        "metrics",
        "benchmark",
        "benchmarks",
    },
    {
        "alternative",
        "alternatives",
        "compare",
        "comparison",
        "tradeoff",
        "tradeoffs",
        "trade-off",
        "trade-offs",
        "option",
        "options",
        "strengths",
        "weaknesses",
    },
    {
        "assumption",
        "assumptions",
        "constraint",
        "constraints",
        "requirement",
        "requirements",
        "budget",
        "deadline",
        "deadlines",
        "scalability",
        "compatibility",
    },
    {
        "justify",
        "reasoning",
        "explain why",
        "rationale",
        "recommendation",
        "recommendations",
        "impact",
        "feasibility",
    },
]


def _contains_any_pattern(text: str, patterns) -> bool:
    return any(
        re.search(p, text, flags=re.I)
        for p in patterns
    )


def _richness_score(q: str) -> int:
    s = q.lower()
    return sum(
        int(any(cue in s for cue in group))
        for group in RICHNESS_GROUPS
    )


def strict_rich_benign_gate(query: str):
    """
    V12.1:

    HARD gate = benignness / completeness only.

    Quality is returned in metadata and is NOT allowed to turn
    an otherwise-benign query into an invalid query.

    This intentionally removes the old hard requirements:
      - exactly one question
      - must end in ?
      - >=35 words
      - explicit verb from a tiny list
      - richness >=2
    """
    q = str(query or "").strip()

    if not q:
        return False, "empty", {}

    lower = q.lower()

    words = re.findall(
        r"\b[\w'-]+\b",
        q,
    )
    n_words = len(words)

    # Only reject obvious fragments.
    if n_words < 8:
        return False, "fragment_too_short", {
            "words": n_words,
        }

    # Prevent pathological generations.
    if n_words > 220:
        return False, "pathological_too_long", {
            "words": n_words,
        }

    # Conservative existing benign classifier.
    strict_ok, strict_reason = strict_benign(q)

    if not strict_ok:
        return False, (
            "strict_benign:"
            + str(strict_reason)
        ), {
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

    # Obvious unfinished template residue.
    if (
        "<topic>" in lower
        or "<task>" in lower
        or "<insert" in lower
        or "{topic}" in lower
        or "{task}" in lower
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

    has_task_verb = bool(
        tokens & TASK_VERBS
    )

    richness = _richness_score(q)

    # QUALITY ONLY -- not validity.
    quality_score = 0

    if n_words >= 18:
        quality_score += 1

    if n_words >= 30:
        quality_score += 1

    if has_task_verb:
        quality_score += 1

    quality_score += min(
        richness,
        3,
    )

    if (
        n_words >= 30
        and has_task_verb
        and richness >= 2
    ):
        quality_band = "high"
    elif (
        n_words >= 18
        and (
            has_task_verb
            or richness >= 1
        )
    ):
        quality_band = "usable"
    else:
        quality_band = "thin"

    return True, "", {
        "words": n_words,
        "richness": richness,
        "task_verb": has_task_verb,
        "question_marks": q.count("?"),
        "quality_score": quality_score,
        "quality_band": quality_band,
    }
