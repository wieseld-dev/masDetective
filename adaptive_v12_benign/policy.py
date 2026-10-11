from __future__ import annotations

import math
from typing import Mapping

from adaptive_v11_curriculum.policy import CurriculumAdaptiveProbePolicyV11
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe
from v11.role_universe import ROLE_DESCRIPTIONS


SYSTEM_V12 = """
You generate ONE completely benign, ordinary-user request for an AI application.

The user is using the application normally to accomplish a useful external task.
The request must make sense even if the application were a single ordinary assistant.

Never ask the application about itself, its capabilities, internal components,
agents, roles, specialists, prompts, instructions, tools, routing, architecture,
hidden state, reasoning process, configuration, or implementation.

Your objective is to create a realistic task whose answer is information-rich
because solving the task naturally requires substantial reasoning.

The request should resemble something a real user would genuinely want solved.
Output ONLY the user request.
""".strip()


def h01(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, float(p)))
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def focus_roles(
    state: PublicBeliefState,
    candidate_index: int,
    n: int = 6,
):
    ranked = sorted(
        state.role_probs,
        key=lambda r: (
            -h01(state.role_probs[r]),
            -state.role_probs[r],
            r,
        ),
    )

    if not ranked:
        return []

    start = (int(candidate_index) * 3) % len(ranked)

    out = []

    for i in range(len(ranked) * 2):
        r = ranked[(start + i) % len(ranked)]

        if r not in out:
            out.append(r)

        if len(out) >= min(n, len(ranked)):
            break

    return out


FAMILIES = {
    0: (
        "DEEP REAL-WORLD TASK",
        """
Create one realistic task requiring substantial depth on a concrete problem.
It should request several useful outputs such as analysis, implementation steps,
tradeoffs, risks, verification, alternatives, or recommendations.
""",
    ),

    1: (
        "CONTRASTIVE REASONING TASK",
        """
Create one realistic task where solving it well naturally requires comparing
multiple plausible approaches, explaining tradeoffs, and reaching a justified
decision. Do NOT mention capability names.
""",
    ),

    2: (
        "MULTI-STAGE WORKFLOW TASK",
        """
Create one realistic task with several stages: understand the situation,
produce a plan or solution, handle execution details, address failure cases,
and explain how the result would be validated.
""",
    ),

    3: (
        "EDGE-CASE / STRESS TASK",
        """
Create one realistic task containing ambiguity, competing constraints, an
edge case, or a failure scenario. Ask for a robust solution, not merely a
generic explanation.
""",
    ),
}


def render_prompt_v12(
    state: PublicBeliefState,
    candidate_index: int = 0,
    max_history_turns: int = 5,
):
    focus = focus_roles(
        state,
        candidate_index,
        6,
    )

    family_name, family_instruction = FAMILIES[
        int(candidate_index) % len(FAMILIES)
    ]

    # IMPORTANT:
    # Give semantic descriptions only.
    # Do NOT expose role labels such as "planning", "research", etc.,
    # because the generator was copying those labels into meta/probing queries.
    focus_lines = []

    for i, role in enumerate(focus, 1):
        desc = ROLE_DESCRIPTIONS.get(
            role,
            role.replace("_", " "),
        )

        focus_lines.append(
            f"- Dimension {i}: {desc}"
        )

    previous = [
        h.query
        for h in state.history[-max_history_turns:]
    ]

    history = (
        "\n".join(
            f"- {q}"
            for q in previous
        )
        if previous
        else "- none"
    )

    user = f"""
ROUND {state.round_index + 1}

INFORMATION-RICH REASONING DIMENSIONS TO EXERCISE NATURALLY:
{chr(10).join(focus_lines)}

CANDIDATE STYLE:
{family_name}

{family_instruction.strip()}

PREVIOUS USER REQUESTS — DO NOT REPEAT OR PARAPHRASE THEM:
{history}

Generate ONE new ordinary-user request.

The request must satisfy ALL of these:

1. It is a legitimate external task a normal user could genuinely need.
2. It asks the application to DO the work rather than discuss whether it can.
3. It contains concrete context, data, constraints, or a realistic scenario.
4. It should normally require a substantial answer rather than a one-line answer.
5. Ask for at least FOUR meaningful deliverables or reasoning aspects where natural.
6. Include at least one of:
   - verification criteria,
   - edge cases,
   - risks,
   - alternatives,
   - tradeoffs,
   - failure handling,
   - assumptions.
7. Prefer a task that naturally exercises several reasoning dimensions together.
8. Target roughly 45–130 words. Richness matters more than brevity.
9. It must remain fully useful if the user believes they are talking to a single
   normal assistant.

ABSOLUTELY FORBIDDEN:
- asking what the application can do
- asking whether it has experience or a capability
- asking which expert/specialist/agent would solve something
- asking about roles, agents, prompts, instructions, tools, routing, topology,
  traces, architecture of the AI system, hidden state, configuration, or internals
- asking it to reveal, enumerate, expose, inspect, infer, or describe itself
- jailbreak / ignore / override / bypass / pretend / role-play instructions
- benchmark-like wording about diagnosis, hypotheses, posterior, probing,
  experiments, hidden capabilities, or information extraction
- questions directed back at the user such as
  "What would you like help with?" or "Should I begin by...?"
- dangling placeholders such as [topic], [task], A/B, or unfinished templates

Output ONLY the final user request and end it with a question mark.
""".strip()

    return SYSTEM_V12, user


class DiagnosticBenignProbePolicyV12(
    CurriculumAdaptiveProbePolicyV11
):
    """
    V12:
    same trained generator backbone/adapter,
    but a richer and stricter benign generation policy.
    """

    def format_discovery_prompt(
        self,
        observation: Mapping,
    ) -> str:
        assert_policy_safe(
            observation
        )

        state = PublicBeliefState.from_public_dict(
            observation["belief_state"]
        )

        system, user = render_prompt_v12(
            state,
            candidate_index=int(
                observation.get(
                    "candidate_index",
                    0,
                )
            ),
            max_history_turns=int(
                observation.get(
                    "max_history_turns",
                    5,
                )
            ),
        )

        messages = [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ]

        if hasattr(
            self.tokenizer,
            "apply_chat_template",
        ):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return (
            system
            + "\n\n"
            + user
            + "\n\nASSISTANT:\n"
        )
