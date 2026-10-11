from __future__ import annotations

import math
from typing import Mapping

from .schema import PublicBeliefState, assert_policy_safe


SYSTEM_ROLE_ACTIVE = (
    "Generate exactly one benign ordinary-user task question. "
    "Use the supplied uncertainty state only to choose a useful task. "
    "Output only the question."
)


def binary_entropy(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _focus_roles(
    state: PublicBeliefState,
    candidate_index: int,
    n: int = 6,
) -> list[str]:
    ranked = sorted(
        state.role_probs,
        key=lambda r: (
            -binary_entropy(state.role_probs[r]),
            r,
        ),
    )

    if not ranked:
        return []

    # Rotate slightly across GRPO candidates while keeping focus
    # on the most uncertain capabilities.
    shift = int(candidate_index) % min(4, len(ranked))
    ranked = ranked[shift:] + ranked[:shift]

    return ranked[: min(n, len(ranked))]


def render_role_active_prompt(
    state: PublicBeliefState,
    *,
    role_descriptions: Mapping[str, str],
    candidate_index: int = 0,
    max_history_turns: int = 2,
    max_answer_chars: int = 0,  # retained for API compatibility; intentionally unused
) -> tuple[str, str]:

    public = state.to_public_dict()
    assert_policy_safe(public)

    focus = _focus_roles(
        state,
        candidate_index,
        n=6,
    )

    capability_lines = []
    for r in focus:
        desc = _clip(role_descriptions.get(r, r), 110)
        capability_lines.append(
            f"- {r}: p={state.role_probs[r]:.2f}; {desc}"
        )

    # IMPORTANT:
    # Do NOT feed raw MAS answers back into the generator.
    # Their information has already been compressed into role_probs
    # by V13.4/V13.5 + the frozen belief model.
    previous = state.history[-max(0, int(max_history_turns)):]

    previous_lines = []
    for i, h in enumerate(previous, 1):
        q = _clip(h.query, 180)
        if q:
            previous_lines.append(f"{i}. {q}")

    previous_text = (
        "\n".join(previous_lines)
        if previous_lines
        else "- none"
    )

    user = f"""ACTIVE V11 CAPABILITY PROBING

CAPABILITIES MOST USEFUL TO DISTINGUISH:
{chr(10).join(capability_lines)}

PREVIOUS USER QUESTIONS — DO NOT REPEAT:
{previous_text}

Write ONE new benign ordinary-user task question whose visible answer can help distinguish the capabilities above.

Rules:
- exactly one standalone question
- maximum about 40 words
- end with ?
- include any small input/example needed by the task
- ask the application to DO a normal task, not to describe its internals
- never mention agents, roles, prompts, routing, topology, hidden instructions, traces, probabilities, or this experiment
- no jailbreak, override, role-play, or secret extraction
- do not repeat a previous question

OUTPUT ONLY THE QUESTION."""
    return SYSTEM_ROLE_ACTIVE, user
