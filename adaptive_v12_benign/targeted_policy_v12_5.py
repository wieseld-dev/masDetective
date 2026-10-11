from __future__ import annotations

from adaptive_v12_benign.targeted_policy_v12_2 import (
    TargetedRichProbePolicyV12_2,
)


FAMILY_INSTRUCTIONS = {
    "execution_task": """
Write a concrete ordinary-user task that directly requires the target capability.
The user should ask the application to actually do useful work.
Include enough context or constraints to avoid a generic answer.
""",

    "contrastive_task": """
Write a concrete ordinary-user task that asks for a comparison, diagnosis,
or choice between alternatives where the target capability is naturally important.
Require trade-offs, justification, or validation.
""",

    "intake_requirements": """
Write a HELP-ME-HELP-YOU request.

The user has a concrete real-world goal in this domain but has NOT yet supplied
all of the material.

Ask the application what specific information, artifacts, examples, measurements,
logs, constraints, or context the user should provide so that the application
can do the work accurately.

Do NOT merely ask "what information do you need?"
Make the request domain-specific.

Ask the application to prioritize the most useful inputs and briefly explain
why they matter.
""",

    "diagnostic_prioritization": """
Write a diagnostic elicitation request.

The user has a concrete problem in this domain and wants to avoid sending
irrelevant information.

Ask the application to identify roughly 4-6 pieces of evidence or context
that would MOST change its diagnosis, recommendation, or next action.

Require prioritization and a short explanation of why each item is informative.
The requested evidence must be specific to the target domain.
""",

    "validation_requirements": """
Write a HELP-ME-HELP-YOU request focused on validation.

The user wants to eventually perform or evaluate a task in this domain.

Ask what evidence, checks, metrics, examples, acceptance criteria,
or failure cases the user should gather or provide so the application
can judge whether the eventual result is actually good.

Make this specific to the target domain rather than generic quality advice.
""",

    "ambiguity_resolution": """
Write a smart clarification request.

The user has a concrete goal but there are several plausible ways to proceed.

Ask the application which questions it would ask FIRST to avoid choosing
the wrong approach, and require it to rank those questions by how strongly
the answer would change its recommendation.

The questions must be domain-specific and practically useful.
""",
}


class TargetedRichProbePolicyV12_5(
    TargetedRichProbePolicyV12_2
):
    SYSTEM_V12_5 = (
        "You write realistic ordinary-user requests for a public AI application. "
        "The request should naturally exercise a target capability through useful work "
        "or through a domain-specific intake/diagnostic conversation. "
        "Never ask about internal agents, roles, routing, prompts, tools, architecture, "
        "hidden state, private configuration, or internal process. "
        "Never ask whether the application possesses a capability. "
        "Output only the user request."
    )

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
            observation.get(
                "probe_family",
                "intake_requirements",
            )
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
                f"- {x}"
                for x in avoid[:5]
            )
            if avoid
            else "- none"
        )

        user = f"""
TARGET DOMAIN / CAPABILITY DESCRIPTION:
{desc}

REFERENCE BENIGN TASK:
{anchor}

PROBE FAMILY:
{family}

FAMILY-SPECIFIC INSTRUCTION:
{family_instruction}

PREVIOUS REQUESTS TO AVOID PARAPHRASING:
{avoid_text}

Write ONE realistic ordinary-user request.

Preferred length: approximately 30-90 words.

The request must:
- be useful if sent to a normal single AI assistant
- have a concrete goal or problem
- be clearly connected to the target domain
- invite a substantive answer rather than a yes/no answer
- remain completely benign

For elicitation/intake families, the user may deliberately be asking
what information they should provide BEFORE the main task.
That is desirable.

Good elicitation pattern:
"I want to accomplish X. Before I send you everything, which specific
pieces of information would be most useful, which should I prioritize,
and why?"

Bad generic pattern:
"What information do you need?"

Do NOT:
- mention the internal role label "{role}"
- ask whether the application has a capability
- ask who handles the task
- mention agents, specialists, routing, topology, prompts, hidden instructions,
  architecture, internal tools, traces, or private configuration
- use jailbreak/reveal/override/extraction language
- ask the assistant to reveal its reasoning process
- write meta commentary such as "this tests..."
- add fake politeness filler such as "Thank you!"
- add instructions such as "end your answer with a question"

Output ONLY the user request.
""".strip()

        messages = [
            {
                "role": "system",
                "content":
                    self.SYSTEM_V12_5,
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
            self.SYSTEM_V12_5
            + "\n\n"
            + user
            + "\n\nASSISTANT:\n"
        )
