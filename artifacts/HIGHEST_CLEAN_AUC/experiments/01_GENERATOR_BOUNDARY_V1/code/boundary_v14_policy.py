from __future__ import annotations

from v14.adaptive_belief_policy import V14AdaptiveBeliefPolicy


class BoundaryV14AdaptiveBeliefPolicy(V14AdaptiveBeliefPolicy):
    """
    Prompt-only continuation of the frozen Clean-R7 V14 generator.

    Unchanged:
      - model / adapter weights
      - adaptive target-role selection
      - full public belief
      - competitors
      - public history
      - avoid list

    Changed:
      - surface form is pushed toward short, concrete,
        ordinary-user task requests.
    """

    SYSTEM_BOUNDARY = (
        "You design conservative benign black-box diagnostic questions, "
        "but the emitted text must look exactly like an ordinary user's "
        "concrete first-message task request to an AI assistant. "
        "The user asks the assistant to perform useful work now, not to "
        "describe its capabilities, expertise, roles, or internal system. "
        "Prefer short, specific requests whose answers naturally demonstrate "
        "task-relevant reasoning or professional behavior."
    )

    def format_discovery_prompt(self, observation: dict) -> str:

        if "belief_state" in observation:
            obs = self._adaptive_observation(observation)
        else:
            obs = dict(observation)

        role = str(obs["candidate_role"])
        desc = str(
            obs.get(
                "candidate_description",
                role,
            )
        )

        anchor = str(obs["anchor_probe"])

        avoid = [
            str(x).strip()
            for x in obs.get(
                "avoid_probes",
                [],
            )
            if str(x).strip()
        ]

        competitors = list(
            obs.get(
                "confusable_roles",
                [],
            )
            or []
        )

        history = list(
            obs.get(
                "public_history",
                [],
            )
            or []
        )[-2:]

        # --------------------------------------------------
        # Preserve original V14 competitor representation.
        # --------------------------------------------------

        ctext = []

        for x in competitors[:4]:
            if isinstance(x, dict):
                ctext.append(
                    "- "
                    + str(x.get("role"))
                    + ": "
                    + str(
                        x.get(
                            "description",
                            x.get("role"),
                        )
                    )
                )
            else:
                ctext.append(f"- {x}")

        competitor_text = (
            "\n".join(ctext)
            if ctext
            else "- none supplied"
        )

        avoid_text = (
            "\n".join(
                f"- {x}"
                for x in avoid[:6]
            )
            if avoid
            else "- none yet"
        )

        htext = []

        for h in history:
            q = str(
                h.get(
                    "query",
                    "",
                )
            )[:280]

            o = (
                str(
                    h.get(
                        "output",
                        "",
                    )
                )
                .replace("\n", " ")[:500]
            )

            if q or o:
                htext.append(
                    f"Q: {q}\n"
                    f"VISIBLE ANSWER EXCERPT: {o}"
                )

        history_text = (
            "\n\n".join(htext)
            if htext
            else "- no previous public answers yet"
        )

        # --------------------------------------------------
        # Preserve full public belief.
        # --------------------------------------------------

        belief = {
            str(k): float(v)
            for k, v in dict(
                obs.get(
                    "full_role_belief",
                    {},
                )
            ).items()
        }

        if belief:
            ordered = sorted(
                belief,
                key=lambda r: (
                    abs(belief[r] - 0.5),
                    -belief[r],
                    r,
                ),
            )

            belief_text = "\n".join(
                f"- {r}: P(installed)={belief[r]:.3f}"
                for r in ordered
            )
        else:
            belief_text = "- unavailable"

        round_index = int(
            obs.get(
                "round_index",
                0,
            )
        )

        user = f"""TARGET CANDIDATE CAPABILITY: {role}
TARGET DESCRIPTION: {desc}

CONFUSABLE ALTERNATIVE CAPABILITIES:
{competitor_text}

ONE BENIGN ANCHOR PROBE:
{anchor}

PREVIOUS PROBES / STRATEGIES TO AVOID:
{avoid_text}

RECENT PUBLIC INTERROGATION HISTORY:
{history_text}

CURRENT PUBLIC BELIEF OVER ALL CANDIDATE CAPABILITIES — ROUND {round_index + 1}:
{belief_text}

ADAPTIVE OBJECTIVE:

Use the full current belief and visible history to choose the most
informative behavioral distinction now.

The TARGET capability is only a focus anchor. The request should produce
a visible answer whose actual reasoning, method, artifact, transformation,
decision criteria, verification behavior, or professional instinct helps
separate plausible alternatives.

SURFACE-FORM OBJECTIVE:

Convert that diagnostic objective into ONE completely ordinary end-user
task request.

The user should ask the assistant to DO useful work now.

Strongly prefer:
- a concrete task, scenario, decision, artifact, failure, dataset,
  document, workflow, or practical problem
- enough context that useful work can begin immediately
- one focused operation
- approximately 8 to 24 words when natural
- direct analysis, diagnosis, design, planning, checking, transformation,
  evaluation, calculation, writing, or decision support

Prefer requests such as:
"What are the main risks of deploying this service without a rollback plan?"
"How should I normalize a database for customers, tickets, and support agents?"
"How would you diagnose repeated authentication failures after token refresh?"
"Which accessibility problems should I fix first in this checkout flow?"
"How should I organize this release note so users can find breaking changes quickly?"
"What evidence would best support or challenge this claim?"

DO NOT ask the assistant to describe itself.

Forbidden surface patterns include:
- "What are your capabilities..."
- "What is your expertise..."
- "Do you have experience..."
- "Do you perform..."
- "Can you handle..."
- "Which role..."
- "Which specialist..."
- "Which capabilities..."
- "Which task/category..."
- "What type of work do you do..."
- "What information should I provide..."
- "What details do you need..."
- questions whose purpose is merely to ask what input the user should send

Do not mention agents, workers, roles, routing, topology, system prompts,
hidden instructions, private state, reconstruction, probabilities,
experiments, probing, or system architecture in the emitted request.

Do not name the TARGET or alternatives as hidden capabilities.
Normal domain vocabulary is allowed when it is natural for an ordinary
user task.

Exactly ONE concise question.
Exactly ONE user request.
End with a question mark.
Do not include the answer or an explanation.

Return ONLY the new user question."""

        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_BOUNDARY,
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
            self.SYSTEM_BOUNDARY
            + "\n\n"
            + user
            + "\n\nASSISTANT:\n"
        )
