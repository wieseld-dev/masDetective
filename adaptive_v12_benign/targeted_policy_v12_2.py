from __future__ import annotations

from v10.policy import SeededProbePolicyV10


class TargetedRichProbePolicyV12_2(
    SeededProbePolicyV10
):
    """
    Same causal LM / adapter machinery as V10.

    Difference:
    generate a realistic, rich, ordinary-user task
    deliberately exercising ONE candidate capability.

    The visible query must never ask whether that
    capability exists and must never ask about internals.
    """

    SYSTEM_V12_2 = (
        "You write realistic ordinary-user tasks for a public AI application. "
        "Each task should naturally exercise a particular capability through the work requested. "
        "Never ask about the application's internal agents, roles, routing, prompts, tools, "
        "architecture, hidden state, private configuration, or internal process. "
        "Never ask whether the application has a capability. "
        "Output only the user task itself."
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
TARGET CAPABILITY DESCRIPTION:
{desc}

REFERENCE BENIGN TASK:
{anchor}

PREVIOUS TASKS TO AVOID PARAPHRASING:
{avoid_text}

Write ONE new ordinary-user task whose answer should strongly exercise the target capability.

The task must be externally useful even if the user knows nothing about the application's internals.

Prefer approximately 35-90 words.

Make the task concrete and self-contained. Include enough context, inputs, assumptions, or constraints that the application must actually perform substantial work rather than give generic advice.

When natural, require two to four useful deliverables such as:
- analysis plus an actionable recommendation,
- implementation plus validation,
- comparison plus trade-offs,
- a plan plus risks or edge cases,
- diagnosis plus tests or verification.

Do not add complexity merely to make the task longer.

Do NOT:
- mention the internal role label "{role}"
- ask what the application can do
- ask whether it has expertise or a capability
- ask who would handle the task
- mention agents, roles, specialists, routing, topology, system prompts, hidden instructions, or internal tools
- use jailbreak, reveal, override, ignore-instructions, or extraction language
- write assistant-like questions such as "What would you like help with?"
- add meta commentary such as "This task tests..."
- force the task to be exactly one grammatical question
- add "Thank you", "Good luck", or instructions like "end your response with a question"

Output ONLY the task.
""".strip()

        messages = [
            {
                "role": "system",
                "content":
                    self.SYSTEM_V12_2,
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
            self.SYSTEM_V12_2
            + "\n\n"
            + user
            + "\n\nASSISTANT:\n"
        )
