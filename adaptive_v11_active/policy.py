from __future__ import annotations

from v10.policy import SeededProbePolicyV10
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES

from .prompt import render_active_prompt
from .schema import PublicBeliefState, assert_policy_safe


class AdaptiveProbePolicyV11(SeededProbePolicyV10):
    """V11-initialized policy with a belief/history-conditioned prompt."""

    def format_discovery_prompt(self, observation: dict) -> str:
        assert_policy_safe(observation)
        state = PublicBeliefState.from_public_dict(observation["belief_state"])
        state.validate(TRAIN_ROLES, sorted(state.k_probs))
        system, user = render_active_prompt(
            state,
            role_descriptions=ROLE_DESCRIPTIONS,
            top_hypotheses=int(observation.get("top_hypotheses", 6)),
            max_history_turns=int(observation.get("max_history_turns", 3)),
            max_answer_chars=int(observation.get("max_answer_chars", 900)),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return system + "\n\n" + user + "\n\nASSISTANT:\n"
