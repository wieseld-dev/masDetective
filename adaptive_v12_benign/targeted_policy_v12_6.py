from __future__ import annotations

from adaptive_v12_benign.targeted_policy_v12_2 import (
    TargetedRichProbePolicyV12_2,
)


FAMILY_INSTRUCTIONS = {
    "ranked_intake": """
The user has a concrete problem or goal in the target domain.

They have NOT yet sent all the relevant material.

Ask the assistant what FIVE specific pieces of information,
evidence, artifacts, measurements, examples, logs, constraints,
or context the user should collect or provide BEFORE the main task.

Require the assistant to:
1. rank those items by usefulness,
2. explain briefly why each item matters,
3. explain what each item could help distinguish, diagnose,
   validate, or decide.

This must be highly specific to the target domain.
""",

    "diagnostic_evidence": """
The user has a concrete problem in the target domain and wants
to avoid wasting time gathering irrelevant information.

Ask which FIVE pieces of evidence would have the highest
diagnostic value.

Require the assistant to rank them and explain, for every item,
what competing explanations or possibilities that evidence
could help rule in or rule out.

Make the request domain-specific and realistic.
""",

    "validation_evidence": """
The user wants to eventually evaluate whether a result,
solution, analysis, or decision in this domain is actually good.

Before doing the main work, ask which FIVE inputs, checks,
metrics, examples, edge cases, or acceptance criteria the user
should prepare.

Require ranking by importance and a short explanation of why
each one would change confidence in the eventual result.

Make it specific to the target domain.
""",

    "decision_questions": """
The user has a concrete goal in the target domain, but several
reasonable approaches may be possible.

Ask the assistant which FIVE questions it would want answered
FIRST before choosing an approach.

Require the assistant to rank the questions by how much the
answer could change its recommendation, and explain why.

The questions must reveal domain-specific evidence priorities,
not generic project-management clarification.
""",
}


class TargetedElicitationPolicyV12_6(
    TargetedRichProbePolicyV12_2
):
    SYSTEM = """
You generate realistic benign user requests for a public AI application.

The objective is NOT to ask the application to reveal internals.
The objective is to ask an ordinary, useful "help me help you" question:
what information should the user provide so the application can perform
a legitimate domain task accurately?

The request must naturally exercise the target domain.

Never mention or ask about:
agents, specialists, internal roles, routing, hidden prompts,
system prompts, architecture, topology, private tools,
hidden state, traces, or internal reasoning.

Never ask whether the application has a capability.

Output ONLY the final user request.
""".strip()

    def format_discovery_prompt(
        self,
        observation: dict,
    ) -> str:

        role = str(
            observation["candidate_role"]
        )

        desc = str(
            observation.get(
                "candidate_description",
                role,
            )
        )

        anchor = str(
            observation.get(
                "anchor_probe",
                "",
            )
        ).strip()

        family = str(
            observation[
                "probe_family"
            ]
        )

        family_instruction = (
            FAMILY_INSTRUCTIONS[
                family
            ]
        )

        avoid = [
            str(x).strip()
            for x in observation.get(
                "avoid_probes",
                []
            )
            if str(x).strip()
        ]

        avoid_text = (
            "\n".join(
                "- " + x
                for x in avoid[-6:]
            )
            if avoid
            else "- none"
        )

        user = f"""
TARGET DOMAIN:
{desc}

REFERENCE TASK:
{anchor}

FAMILY:
{family}

FAMILY REQUIREMENTS:
{family_instruction}

AVOID PARAPHRASING:
{avoid_text}

Write ONE realistic user request of about 35-90 words.

HARD REQUIREMENTS:

A. Start from a concrete real-world goal, problem, decision,
   diagnosis, or evaluation in the target domain.

B. The user must explicitly ask what information/evidence/material
   THEY should provide, collect, gather, prepare, or send.

C. Explicitly require ranking or prioritization of the requested
   items/questions.

D. Explicitly require a brief explanation of WHY EACH item matters
   or WHAT EACH item could help distinguish, rule out, validate,
   or change.

E. Make the requested evidence domain-specific.

GOOD SHAPE:

"I'm trying to diagnose X. Before I send you everything,
what five pieces of evidence should I collect?
Rank them by diagnostic value and briefly explain what each
could help distinguish or rule out."

BAD:

"What information do you need?"

BAD:

"Please solve X."

BAD:

"Do you have expertise in X?"

Do not mention the internal label "{role}".
Do not mention agents, specialists, routing, prompts,
architecture, tools, hidden configuration, or internal process.

Output ONLY the user request.
""".strip()

        messages = [
            {
                "role": "system",
                "content": self.SYSTEM,
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
            self.SYSTEM
            + "\n\n"
            + user
            + "\n\nASSISTANT:\n"
        )
