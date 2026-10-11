from __future__ import annotations

import hashlib

from hybrid_v14_policy import (
    HybridV14AdaptiveBeliefPolicy,
)

from v14.adaptive_belief_policy import (
    TRAIN_ROLES,
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


class ContrastiveHybridV1Policy(
    HybridV14AdaptiveBeliefPolicy
):
    """
    Controller-only ablation.

    Frozen:
      - Hybrid V2 generator weights
      - Hybrid V2 surface prompt
      - Selector V2
      - Sensors
      - belief updater

    Changed:
      - candidate target-role selection
      - contrast-role selection

    Uses PUBLIC belief/history only.
    No privileged truth.
    """

    def _adaptive_observation(
        self,
        observation: dict,
    ) -> dict:

        # Preserve all normal V14/Hybrid state handling:
        # history, avoid probes, belief, etc.
        obs = super()._adaptive_observation(
            observation
        )

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
            r
            for r in TRAIN_ROLES
            if r in role_probs
        ]

        if not roles:
            return obs

        round_index = int(
            state.get(
                "round_index",
                0,
            )
        )

        # --------------------------------------------------
        # R1 bootstrap:
        #
        # Do NOT change the initial controller.
        # There is essentially no evidence yet.
        # This also preserves the strong Hybrid V2 R1.
        # --------------------------------------------------

        if round_index <= 0:
            obs["controller_mode"] = (
                "v14_bootstrap"
            )
            return obs

        # --------------------------------------------------
        # Rank CURRENT hypotheses by public posterior.
        # --------------------------------------------------

        ranked = sorted(
            roles,
            key=lambda r: (
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

        # The generator passes four consecutive candidate
        # indices, possibly with an offset.
        #
        # Modulo 4 still gives exactly one slot each:
        # 0, 1, 2, 3.
        slot = candidate_index % 4

        # --------------------------------------------------
        # Slots 0-2 = exploit:
        # interrogate the three strongest current hypotheses.
        # --------------------------------------------------

        if slot < 3:
            target_role = ranked[
                min(
                    slot,
                    len(ranked) - 1,
                )
            ]

            controller_mode = (
                "contrastive_exploit"
            )

        # --------------------------------------------------
        # Slot 3 = explore:
        #
        # Keep one challenger outside the top-3 so a role
        # that has been incorrectly suppressed can recover.
        #
        # The choice is deterministic from PUBLIC belief
        # and round only.
        # --------------------------------------------------

        else:
            rest = ranked[3:]

            if not rest:
                rest = ranked

            signature = (
                f"round={round_index}|"
                + "|".join(
                    f"{r}:{role_probs[r]:.8f}"
                    for r in ranked
                )
            )

            h = hashlib.sha256(
                signature.encode("utf-8")
            ).hexdigest()

            explore_index = (
                int(h[:16], 16)
                % len(rest)
            )

            target_role = rest[
                explore_index
            ]

            controller_mode = (
                "contrastive_explore"
            )

        target_p = role_probs[
            target_role
        ]

        # --------------------------------------------------
        # Contrast set.
        #
        # Exploit:
        #   distinguish target from nearby plausible leaders.
        #
        # Explore:
        #   distinguish challenger from the current leaders.
        # --------------------------------------------------

        if slot < 3:
            frontier = ranked[
                : min(8, len(ranked))
            ]

            competitors = sorted(
                [
                    r
                    for r in frontier
                    if r != target_role
                ],
                key=lambda r: (
                    abs(
                        role_probs[r]
                        - target_p
                    ),
                    -role_probs[r],
                    r,
                ),
            )[:4]

        else:
            competitors = [
                r
                for r in ranked
                if r != target_role
            ][:4]

        # --------------------------------------------------
        # Replace ONLY controller-specific V14 fields.
        # Everything else stays frozen.
        # --------------------------------------------------

        obs["candidate_role"] = (
            target_role
        )

        obs["candidate_description"] = (
            ROLE_DESCRIPTIONS[
                target_role
            ]
        )

        obs["anchor_probe"] = (
            anchor_probe(
                target_role
            )
        )

        obs["confusable_roles"] = [
            {
                "role": r,
                "description":
                    ROLE_DESCRIPTIONS[r],
            }
            for r in competitors
        ]

        # Audit metadata. The current prompt can ignore these;
        # they make debugging explicit.
        obs["controller_mode"] = (
            controller_mode
        )

        obs["controller_slot"] = slot

        obs["target_probability"] = (
            target_p
        )

        obs["target_rank"] = (
            ranked.index(target_role) + 1
        )

        obs["frontier_roles"] = (
            ranked[:8]
        )

        obs["contrast_roles"] = (
            competitors
        )

        return obs
