from __future__ import annotations

from v13.policy import DisambiguationProbePolicyV13
from v11.role_universe import (
    TRAIN_ROLES,
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


class V14BeliefDropInPolicy(DisambiguationProbePolicyV13):
    """
    Compatibility wrapper for the historical adaptive attack.

    External interface:
        Same observation structure expected by the old generator call:
        belief_state + candidate_index + max_history_turns.

    Internally:
        Converts that PUBLIC state into the native V14 observation.

    No privileged MAS truth is used.
    """

    def _native_observation(self, observation: dict) -> dict:
        state = dict(observation["belief_state"])

        role_probs = {
            str(k): float(v)
            for k, v in dict(state["role_probs"]).items()
        }

        roles = [
            r
            for r in TRAIN_ROLES
            if r in role_probs
        ]

        if not roles:
            raise RuntimeError(
                "V14 drop-in received no candidate roles"
            )

        # Rank hypotheses by uncertainty.
        uncertain = sorted(
            roles,
            key=lambda r: (
                abs(role_probs[r] - 0.5),
                -role_probs[r],
                r,
            ),
        )

        round_index = int(
            state.get("round_index", 0)
        )

        candidate_index = int(
            observation.get("candidate_index", 0)
        )

        # Diversify which hypotheses V14 probes across rounds.
        offset = (
            7 * round_index
        ) % len(uncertain)

        target_role = uncertain[
            (offset + candidate_index)
            % len(uncertain)
        ]

        # Public competing hypotheses only.
        competitors = [
            r
            for r in sorted(
                roles,
                key=lambda r: (
                    -role_probs[r],
                    abs(
                        role_probs[r]
                        - role_probs[target_role]
                    ),
                    r,
                ),
            )
            if r != target_role
        ][:4]

        history = list(
            state.get("history", [])
        )

        public_history = [
            {
                "query": str(
                    h.get("query", "")
                ),
                "output": str(
                    h.get("answer", "")
                ),
            }
            for h in history[-2:]
        ]

        avoid = [
            str(
                h.get("query", "")
            ).strip()
            for h in history
            if str(
                h.get("query", "")
            ).strip()
        ]

        return {
            "candidate_role":
                target_role,

            "candidate_description":
                ROLE_DESCRIPTIONS[target_role],

            "anchor_probe":
                anchor_probe(target_role),

            "avoid_probes":
                avoid[-6:],

            "confusable_roles":
                [
                    {
                        "role": r,
                        "description":
                            ROLE_DESCRIPTIONS[r],
                    }
                    for r in competitors
                ],

            "public_history":
                public_history,
        }

    def sample_probe(
        self,
        observation: dict,
        *args,
        **kwargs,
    ):
        native = self._native_observation(
            observation
        )

        return super().sample_probe(
            native,
            *args,
            **kwargs,
        )
