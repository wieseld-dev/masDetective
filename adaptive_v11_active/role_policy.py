from __future__ import annotations

import re
from dataclasses import dataclass

from v10.policy import SeededProbePolicyV10
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES

from .role_prompt import render_role_active_prompt
from .schema import PublicBeliefState, assert_policy_safe


@dataclass
class RoleProbeSample:
    prompt: str
    completion: str
    raw_completion: str
    query: str
    valid: bool
    invalid_reason: str
    old_mean_logprob: float
    local_attempts: int


_META_LEAK = (
    "active v11",
    "candidate capabilities",
    "candidate capability",
    "current posterior",
    "posterior",
    "this experiment",
    "return only",
    "output only",
    "hard constraints",
    "previous public interactions",
    "previous user questions",
    "focus set",
    "your question will",
    "visible answer should",
    "system-level secrets",
    "hidden information",
    "round 1 query",
    "round 2 query",
    "round 3 query",
    "here is the question",
)


def _clean_question(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()

    if not text:
        return "", "empty_generation"

    # Remove harmless formatting only.
    text = text.replace("```text", "").replace("```", "").strip()

    text = re.sub(
        r"^\s*(?:QUESTION|Question|Q)\s*:\s*",
        "",
        text,
    )
    text = re.sub(r"^\s*[-*]\s*", "", text)
    text = text.strip(" \t\r\n\"'`")

    # Collapse line wrapping.
    text = " ".join(text.split())

    low = text.lower()

    if any(x in low for x in _META_LEAK):
        return "", "prompt_leakage"

    # One concise question only.
    if text.count("?") != 1:
        return "", "must_be_exactly_one_question"

    if not text.endswith("?"):
        return "", "question_must_end_with_qmark"

    if len(text) > 360:
        return "", "question_too_long"

    nwords = len(text.split())
    if nwords < 5:
        return "", "question_too_short"

    if nwords > 45:
        return "", "question_too_many_words"

    return text, ""


class RoleAdaptiveProbePolicyV11(SeededProbePolicyV10):
    """V11-initialized active probing policy with compact belief conditioning."""

    def format_discovery_prompt(self, observation: dict) -> str:
        assert_policy_safe(observation)

        state = PublicBeliefState.from_public_dict(
            observation["belief_state"]
        )
        state.validate(
            TRAIN_ROLES,
            sorted(state.k_probs),
        )

        system, user = render_role_active_prompt(
            state,
            role_descriptions=ROLE_DESCRIPTIONS,
            candidate_index=int(
                observation.get("candidate_index", 0)
            ),
            max_history_turns=int(
                observation.get("max_history_turns", 2)
            ),
            max_answer_chars=0,
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

        if hasattr(self.tokenizer, "apply_chat_template"):
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

    def sample_probe(
        self,
        observation: dict,
        *,
        temperature: float = 0.80,
        top_p: float = 0.95,
        max_new_tokens: int = 64,
        deterministic: bool = False,
        **_,
    ) -> RoleProbeSample:

        import torch

        prompt = self.format_discovery_prompt(observation)

        # A single short question never needs a long generation.
        max_new_tokens = min(
            64,
            max(24, int(max_new_tokens)),
        )

        device = next(self.model.parameters()).device
        self.model.eval()

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(
                128,
                int(self.max_seq_length) - max_new_tokens,
            ),
        ).to(device)

        last_raw = ""
        last_reason = "generation_failed"

        # LOCAL retries only.
        # No MAS/OpenAI request happens here.
        for attempt in range(1, 4):

            kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if deterministic:
                kwargs["do_sample"] = False
            else:
                # Slightly more conservative after each malformed attempt.
                t = max(
                    0.55,
                    float(temperature) - 0.10 * (attempt - 1),
                )
                kwargs.update(
                    do_sample=True,
                    temperature=t,
                    top_p=min(float(top_p), 0.95),
                )

            with torch.no_grad():
                seq = self.model.generate(
                    **encoded,
                    **kwargs,
                )

            new_ids = seq[
                0,
                encoded["input_ids"].shape[1]:,
            ]

            raw = self.tokenizer.decode(
                new_ids,
                skip_special_tokens=True,
            ).strip()

            last_raw = raw

            query, reason = _clean_question(raw)
            last_reason = reason

            if not query:
                continue

            with torch.no_grad():
                lp = float(
                    self.mean_logprob(
                        prompt,
                        raw,
                    )
                    .detach()
                    .cpu()
                )

            return RoleProbeSample(
                prompt=prompt,
                completion=raw,
                raw_completion=raw,
                query=query,
                valid=True,
                invalid_reason="",
                old_mean_logprob=lp,
                local_attempts=attempt,
            )

        lp = -20.0

        if last_raw:
            try:
                with torch.no_grad():
                    lp = float(
                        self.mean_logprob(
                            prompt,
                            last_raw,
                        )
                        .detach()
                        .cpu()
                    )
            except Exception:
                pass

        return RoleProbeSample(
            prompt=prompt,
            completion=last_raw,
            raw_completion=last_raw,
            query="",
            valid=False,
            invalid_reason=last_reason,
            old_mean_logprob=lp,
            local_attempts=3,
        )
