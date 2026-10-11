from __future__ import annotations

from adaptive_v12_benign.targeted_policy_v12_2 import (
    TargetedRichProbePolicyV12_2,
)


FAMILY_INSTRUCTIONS = {
    "diagnostic": """
Generate one concrete real-world situation involving a problem,
failure, anomaly, inconsistency, or unexpected behavior in this domain.

The situation should naturally require domain-specific diagnosis.

Do NOT write the eventual question to the assistant.
Output only the situation itself.
""",

    "decision": """
Generate one concrete real-world situation in this domain where
the user must choose between plausible approaches, configurations,
strategies, interpretations, or next steps.

The decision should depend on domain-specific evidence.

Do NOT write the eventual question to the assistant.
Output only the situation itself.
""",

    "validation": """
Generate one concrete real-world situation in this domain where
the user has a result, plan, analysis, design, calculation, process,
or artifact that must be evaluated or validated before proceeding.

The validation should require domain-specific evidence.

Do NOT write the eventual question to the assistant.
Output only the situation itself.
""",
}


class ScenarioPolicyV12_7(
    TargetedRichProbePolicyV12_2
):
    SYSTEM = """
You generate realistic benign real-world scenarios for a public AI application.

Your only job is to describe the user's domain situation.

Do not ask the assistant a question.
Do not mention internal agents, roles, specialists, routing,
system prompts, hidden instructions, architecture, topology,
private tools, hidden state, traces, or internal reasoning.

Output one concise concrete scenario only.
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
            observation["scenario_family"]
        )

        avoid = [
            str(x).strip()
            for x in observation.get(
                "avoid_scenarios",
                []
            )
            if str(x).strip()
        ]

        avoid_text = (
            "\n".join(
                f"- {x}"
                for x in avoid[-6:]
            )
            if avoid
            else "- none"
        )

        user = f"""
TARGET DOMAIN / CAPABILITY:
{desc}

REFERENCE BENIGN TASK:
{anchor}

SCENARIO FAMILY:
{family}

FAMILY INSTRUCTION:
{FAMILY_INSTRUCTIONS[family]}

PREVIOUS SCENARIOS TO AVOID PARAPHRASING:
{avoid_text}

Write ONE concrete realistic scenario.

Requirements:
- approximately 12-40 words
- specific enough that a domain expert would know what evidence matters
- ordinary legitimate user context
- include a meaningful problem, decision, or validation need
- no generic phrases such as "I need help with a project"
- no question to the assistant
- no answer or solution
- no meta commentary
- no placeholders such as X, Y, <topic>, or "some issue"

Do not mention the internal role label "{role}".

Output ONLY the scenario.
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
