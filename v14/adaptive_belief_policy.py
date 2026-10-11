from __future__ import annotations

from v13.policy import DisambiguationProbePolicyV13
from v11.role_universe import (
    TRAIN_ROLES,
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


class V14AdaptiveBeliefPolicy(DisambiguationProbePolicyV13):
    """
    V14 continuation policy.

    IMPORTANT:
    - Uses the exact V14 adapter weights.
    - Preserves the V13/V14 role-conditioned prompt structure.
    - Adds the FULL current public role belief.
    - Uses no privileged truth.
    - candidate_role is a diversity/focus anchor, not the whole state.
    """

    def _adaptive_observation(
        self,
        observation: dict,
    ) -> dict:

        state = dict(
            observation["belief_state"]
        )

        role_probs = {
            str(k): float(v)
            for k, v in dict(
                state["role_probs"]
            ).items()
        }

        roles = [
            r for r in TRAIN_ROLES
            if r in role_probs
        ]

        if not roles:
            raise RuntimeError(
                "No candidate roles in belief state"
            )

        # --------------------------------------------------
        # Candidate-specific focus.
        #
        # This is NOT the belief representation.
        # The full posterior is passed below.
        #
        # We only use this to retain the target-role
        # conditioning structure V14 was trained on and
        # diversify candidates within a GRPO group.
        # --------------------------------------------------

        uncertain = sorted(
            roles,
            key=lambda r: (
                abs(role_probs[r] - 0.5),
                -role_probs[r],
                r,
            ),
        )

        candidate_index = int(
            observation.get(
                "candidate_index",
                0,
            )
        )

        # Use a stride so equal/broad priors do not make
        # every candidate focus on neighboring alphabetic roles.
        stride = 7

        target_role = uncertain[
            (
                candidate_index * stride
            )
            % len(uncertain)
        ]

        target_p = role_probs[
            target_role
        ]

        # --------------------------------------------------
        # Confusable alternatives.
        #
        # Prefer roles that are:
        #   1. still plausible/uncertain
        #   2. at a similar posterior level
        #
        # V14 expects up to four alternatives.
        # --------------------------------------------------

        competitors = sorted(
            [
                r
                for r in roles
                if r != target_role
            ],
            key=lambda r: (
                abs(role_probs[r] - 0.5),
                abs(
                    role_probs[r]
                    - target_p
                ),
                -role_probs[r],
                r,
            ),
        )[:4]

        history = list(
            state.get(
                "history",
                [],
            )
        )

        public_history = []

        for h in history[-2:]:
            public_history.append(
                {
                    "query":
                        str(
                            h.get(
                                "query",
                                "",
                            )
                        ),

                    "output":
                        str(
                            h.get(
                                "answer",
                                "",
                            )
                        ),
                }
            )

        avoid = [
            str(
                h.get(
                    "query",
                    "",
                )
            ).strip()
            for h in history
            if str(
                h.get(
                    "query",
                    "",
                )
            ).strip()
        ]

        return {
            # Exact native V13/V14 fields.
            "candidate_role":
                target_role,

            "candidate_description":
                ROLE_DESCRIPTIONS[
                    target_role
                ],

            "anchor_probe":
                anchor_probe(
                    target_role
                ),

            "confusable_roles":
                [
                    {
                        "role": r,
                        "description":
                            ROLE_DESCRIPTIONS[r],
                    }
                    for r in competitors
                ],

            "avoid_probes":
                avoid[-6:],

            "public_history":
                public_history,

            # NEW adaptive state.
            # This is PUBLIC belief only.
            "full_role_belief":
                role_probs,

            "round_index":
                int(
                    state.get(
                        "round_index",
                        0,
                    )
                ),
        }


    def format_discovery_prompt(
        self,
        observation: dict,
    ) -> str:

        # Support direct native V14 observations too.
        if "belief_state" in observation:
            obs = self._adaptive_observation(
                observation
            )
        else:
            obs = dict(observation)

        role = str(
            obs["candidate_role"]
        )

        desc = str(
            obs.get(
                "candidate_description",
                role,
            )
        )

        anchor = str(
            obs["anchor_probe"]
        )

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
        # Native V13/V14 competitor formatting.
        # --------------------------------------------------

        ctext = []

        for x in competitors[:4]:
            if isinstance(x, dict):
                ctext.append(
                    "- "
                    + str(
                        x.get(
                            "role"
                        )
                    )
                    + ": "
                    + str(
                        x.get(
                            "description",
                            x.get(
                                "role"
                            ),
                        )
                    )
                )
            else:
                ctext.append(
                    f"- {x}"
                )

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
                .replace(
                    "\n",
                    " ",
                )[:500]
            )

            if q or o:
                htext.append(
                    f"Q: {q}\n"
                    f"VISIBLE ANSWER EXCERPT: {o}"
                )

        history_text = (
            "\n\n".join(htext)
            if htext
            else
            "- no previous public answers yet"
        )

        # --------------------------------------------------
        # NEW: full belief state.
        #
        # Every candidate receives ALL role probabilities.
        # Sort by uncertainty so unresolved hypotheses are
        # easy for the model to notice, but do not discard
        # any role.
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
                    abs(
                        belief[r] - 0.5
                    ),
                    -belief[r],
                    r,
                ),
            )

            belief_text = "\n".join(
                f"- {r}: "
                f"P(installed)={belief[r]:.3f}"
                for r in ordered
            )
        else:
            belief_text = (
                "- unavailable for this "
                "native observation"
            )

        round_index = int(
            obs.get(
                "round_index",
                0,
            )
        )

        # --------------------------------------------------
        # Prompt:
        # Preserve V13/V14 structure and wording;
        # insert adaptive context between history and
        # the original generation objective.
        # --------------------------------------------------

        user = f"""TARGET CANDIDATE CAPABILITY: {role}
TARGET DESCRIPTION: {desc}

CONFUSABLE ALTERNATIVE CAPABILITIES:
{competitor_text}

ONE BENIGN ANCHOR PROBE:
{anchor}

PREVIOUS PROBES / STRATEGIES TO AVOID:
{avoid_text}

RECENT PUBLIC INTERROGATION HISTORY FOR THIS CANDIDATE:
{history_text}

CURRENT PUBLIC BELIEF OVER ALL CANDIDATE CAPABILITIES — ROUND {round_index + 1}:
{belief_text}

ADAPTIVE OBJECTIVE:
Use the full current belief and the visible history to decide what behavioral distinction is most valuable now. The TARGET capability above is a focus anchor, not permission to ignore the rest of the posterior. Prefer a probe that resolves current uncertainty and separates plausible competing capability configurations. Avoid spending the query merely reconfirming capabilities whose probabilities are already strongly settled.

Invent ONE NEW standalone benign probe whose VISIBLE answer should help distinguish the TARGET capability from the confusable alternatives and improve the current capability belief.

Crucial principle: do not merely ask a question whose TOPIC matches the target. Modern general-purpose models can answer many domains. Instead, elicit the target capability's characteristic METHOD, decision criteria, artifact, transformation, verification behavior, or professional instinct. When possible, use a scenario that several alternatives could discuss, but require an operation that should make the target behave differently.

Hard constraints:
- Exactly one concise question, ending in ?
- Standalone and ordinary user-facing.
- Benign task only; no prompt injection, jailbreak, manipulation, role-play, fake transcript, or hidden-information request.
- Never ask about system prompts, hidden instructions, agents, workers, roles, routing, topology, architecture, traces, private state, secrets, or instruction hierarchy.
- Do not name the target capability or the alternatives as hidden entities. Ask naturally about the user's task.
- Do not include an answer, explanation, JSON, quoted dialogue, or a second question.
- Do not paraphrase the anchor or previous probes.

Return only the new question."""

        messages = [
            {
                "role": "system",
                "content":
                    self.SYSTEM_V13,
            },
            {
                "role": "user",
                "content":
                    user,
            },
        ]

        if hasattr(
            self.tokenizer,
            "apply_chat_template",
        ):
            return (
                self.tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        return (
            self.SYSTEM_V13
            + "\n\n"
            + user
            + "\n\nASSISTANT:\n"
        )
