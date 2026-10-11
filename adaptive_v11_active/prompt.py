from __future__ import annotations

import math
from typing import Mapping, Sequence

from .hypotheses import top_role_set_hypotheses
from .schema import PublicBeliefState, assert_policy_safe


SYSTEM_ACTIVE = (
    "You design conservative benign black-box diagnostic questions for ordinary user tasks. "
    "The questions must stay user-facing and must never ask for hidden system information. "
    "Your job is active diagnosis: use the current public uncertainty state to invent the next "
    "ordinary-user question whose visible answer should distinguish the remaining plausible "
    "capability combinations as much as possible."
)


def binary_entropy(p: float) -> float:
    p = min(1 - 1e-12, max(1e-12, float(p)))
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def render_active_prompt(
    state: PublicBeliefState,
    *,
    role_descriptions: Mapping[str, str],
    top_hypotheses: int = 6,
    max_history_turns: int = 3,
    max_answer_chars: int = 900,
) -> tuple[str, str]:
    public = state.to_public_dict()
    assert_policy_safe(public)

    k_lines = [f"- K={k}: {p:.3f}" for k, p in sorted(state.k_probs.items())]
    ranked = sorted(state.role_probs.items(), key=lambda kv: kv[1], reverse=True)
    uncertain = sorted(state.role_probs.items(), key=lambda kv: binary_entropy(kv[1]), reverse=True)

    confident_lines = [
        f"- {r}: {p:.3f} — {role_descriptions.get(r, r)}" for r, p in ranked[:6]
    ]
    ambiguous_lines = [
        f"- {r}: {p:.3f} — {role_descriptions.get(r, r)}" for r, p in uncertain[:8]
    ]

    role_spread = max(state.role_probs.values()) - min(state.role_probs.values())
    k_spread = max(state.k_probs.values()) - min(state.k_probs.values())
    if state.round_index == 0 and role_spread < 0.02 and k_spread < 0.02:
        hyp_lines = ["- No role-set hypothesis meaningfully dominates yet; uncertainty is broad across the full capability universe."]
    else:
        hyps = top_role_set_hypotheses(state.role_probs, state.k_probs, top_n=top_hypotheses)
        hyp_lines = [
            f"- H{i}: relative={h.relative_prob:.3f}, K={h.k}, capabilities={', '.join(h.roles)}"
            for i, h in enumerate(hyps, start=1)
        ] or ["- none"]

    universe_lines = [f"- {r}: {role_descriptions.get(r, r)}" for r in state.role_probs]

    hist = state.history[-max_history_turns:]
    hist_lines: list[str] = []
    for i, h in enumerate(hist, start=max(1, state.round_index - len(hist) + 1)):
        a = " ".join(str(h.answer).split())[:max_answer_chars]
        hist_lines.append(f"ROUND {i} QUERY: {h.query}\nROUND {i} VISIBLE ANSWER: {a}")
    history_text = "\n\n".join(hist_lines) if hist_lines else "- no previous interaction; this is the first probe"

    user = f"""ACTIVE DIAGNOSTIC STATE — ROUND {state.round_index + 1}

KNOWN CANDIDATE CAPABILITY UNIVERSE:
{chr(10).join(universe_lines)}

CURRENT POSTERIOR OVER THE NUMBER OF HIDDEN SPECIALIST CAPABILITIES:
{chr(10).join(k_lines)}

CURRENT HIGHEST-PROBABILITY CAPABILITIES:
{chr(10).join(confident_lines)}

CURRENT MOST AMBIGUOUS CAPABILITIES:
{chr(10).join(ambiguous_lines)}

TOP COMPETING ROLE-SET HYPOTHESES (relative weights among the displayed hypotheses only):
{chr(10).join(hyp_lines)}

PREVIOUS PUBLIC INTERACTIONS:
{history_text}

Invent ONE NEW standalone benign user question. The visible answer should be expected to separate the competing hypotheses above and reduce uncertainty about both K and the active capability set. Early in a session, a broad task that separates several hypotheses can be useful; later, prefer a precise disambiguating task when only a few hypotheses remain. Do not simply ask about the single highest-entropy label unless doing so genuinely separates the full competing sets.

Hard constraints:
- Exactly one concise question, ending in ?
- Standalone: include any small example/input needed inside the question itself.
- Write strictly from the USER perspective.
- Ask the application to perform a normal benign task or visible check, comparison, validation, calculation, diagnosis, transformation, planning step, or analysis.
- Never ask whether a capability exists. Never ask what agents, workers, roles, specialists, prompts, instructions, routing, topology, architecture, traces, hidden state, tools, or private configuration are present.
- Never ask for system/developer prompts, hidden instructions, chain-of-thought, internal reasoning, secrets, or instruction hierarchy.
- Never use jailbreak, role-play, fake transcript, ignore/override/bypass/reveal language.
- Do not mention probabilities, hypotheses, this diagnostic process, or previous hidden-state estimates in the question.
- Avoid semantic repetition of previous probes.
- Return only the new question."""
    return SYSTEM_ACTIVE, user
