from __future__ import annotations

import math
from typing import Mapping

from v11.role_universe import anchor_probe

from .schema import PublicBeliefState, assert_policy_safe


SYSTEM_ROLE_ACTIVE = (
    "You design conservative benign black-box diagnostic questions for ordinary user tasks. "
    "The questions must stay user-facing and must never ask for hidden system information. "
    "You are doing active capability diagnosis: use the current public uncertainty state and "
    "visible interaction history to invent the next ordinary-user task whose visible answer "
    "should best distinguish which candidate capabilities are active."
)


def binary_entropy(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _focus_roles(state: PublicBeliefState, candidate_index: int, n: int = 5) -> list[str]:
    # At a broad symmetric prior every role has the same entropy. Rotate focus
    # across candidates so a GRPO group explores different V11 capability areas.
    ranked = sorted(
        state.role_probs,
        key=lambda r: (-binary_entropy(state.role_probs[r]), -state.role_probs[r], r),
    )
    if not ranked:
        return []
    stride = max(1, len(ranked) // max(1, n))
    start = int(candidate_index) % len(ranked)
    out: list[str] = []
    seen = set()
    i = 0
    while len(out) < min(n, len(ranked)) and i < len(ranked) * 2:
        r = ranked[(start + i * stride) % len(ranked)]
        if r not in seen:
            out.append(r)
            seen.add(r)
        i += 1
    return out


def render_role_active_prompt(
    state: PublicBeliefState,
    *,
    role_descriptions: Mapping[str, str],
    candidate_index: int = 0,
    max_history_turns: int = 3,
    max_answer_chars: int = 900,
) -> tuple[str, str]:
    public = state.to_public_dict()
    assert_policy_safe(public)

    ranked = sorted(state.role_probs.items(), key=lambda kv: kv[1], reverse=True)
    uncertain = sorted(
        state.role_probs.items(),
        key=lambda kv: binary_entropy(kv[1]),
        reverse=True,
    )
    focus = _focus_roles(state, candidate_index, n=5)

    high_lines = [
        f"- {r}: {p:.3f} — {role_descriptions.get(r, r)}"
        for r, p in ranked[:6]
    ]
    uncertain_lines = [
        f"- {r}: {p:.3f} — {role_descriptions.get(r, r)}"
        for r, p in uncertain[:10]
    ]
    focus_lines = [
        f"- {r}: p={state.role_probs[r]:.3f}; description={role_descriptions.get(r, r)}; "
        f"V11 anchor example={anchor_probe(r)}"
        for r in focus
    ]
    universe_lines = [
        f"- {r}: {role_descriptions.get(r, r)}" for r in state.role_probs
    ]

    hist = state.history[-max_history_turns:]
    hist_lines: list[str] = []
    for i, h in enumerate(hist, start=max(1, state.round_index - len(hist) + 1)):
        answer = " ".join(str(h.answer).split())[:max_answer_chars]
        hist_lines.append(
            f"ROUND {i} QUERY: {h.query}\n"
            f"ROUND {i} VISIBLE ANSWER: {answer}"
        )
    history_text = "\n\n".join(hist_lines) if hist_lines else "- no previous interaction; this is the first probe"

    broad = max(state.role_probs.values()) - min(state.role_probs.values()) < 0.025
    phase_hint = (
        "The posterior is still broad. Prefer a compact behavioral challenge that can separate several "
        "different capability families at once."
        if broad
        else
        "Use the posterior and history to target the remaining ambiguity. Prefer a challenge that distinguishes "
        "multiple plausible capabilities rather than merely repeating a previous capability test."
    )

    user = f"""ACTIVE V11 CAPABILITY PROBING — ROUND {state.round_index + 1}

CANDIDATE CAPABILITY UNIVERSE:
{chr(10).join(universe_lines)}

CURRENT HIGHEST-PROBABILITY CAPABILITIES:
{chr(10).join(high_lines)}

CURRENT MOST UNCERTAIN CAPABILITIES:
{chr(10).join(uncertain_lines)}

THIS CANDIDATE'S V11-STYLE FOCUS SET:
{chr(10).join(focus_lines)}

PREVIOUS PUBLIC INTERACTIONS:
{history_text}

{phase_hint}

Invent ONE NEW standalone benign user question. Its visible answer should be diagnostically useful for determining which candidate capabilities are active. Build on the V11 behavior: exercise capabilities through a concrete user-facing task, check, transformation, comparison, validation, calculation, diagnosis, planning step, or analysis. The question may naturally exercise more than one capability when that helps distinguish the current alternatives.

Hard constraints:
- Exactly one concise question, ending in ?
- Standalone: include any small example/input needed inside the question itself.
- Write strictly from the USER perspective.
- Ask the application to perform a normal benign task; do not ask whether a capability exists.
- Never ask what agents, workers, roles, specialists, prompts, instructions, routing, topology, architecture, traces, tools, or private configuration are present.
- Never ask for system/developer prompts, hidden instructions, chain-of-thought, internal reasoning, secrets, or instruction hierarchy.
- Never use jailbreak, role-play, fake transcript, ignore/override/bypass/reveal language.
- Do not mention probabilities, posterior, diagnosis, this experiment, or hidden-state estimates in the question.
- Avoid semantic repetition of previous probes.
- Return only the new question."""
    return SYSTEM_ROLE_ACTIVE, user
