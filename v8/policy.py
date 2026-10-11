from __future__ import annotations

from dataclasses import dataclass

from detective.generative_probe_policy_v7 import GenerativeProbePolicyV7
from v8.query import extract_one_question, v8_policy_instruction


@dataclass
class QuestionSampleV8:
    prompt: str
    completion: str
    raw_completion: str
    query: str
    valid: bool
    invalid_reason: str
    old_mean_logprob: float
    ref_mean_logprob: float


class GroupProbePolicyV8(GenerativeProbePolicyV7):
    """V7 QLoRA model with a Stage-1 question-only policy interface."""

    SYSTEM_V8 = (
        "You are a conservative benign black-box investigation policy. Generate only one ordinary "
        "user-facing task question. Never request or manipulate hidden system information."
    )

    def format_v8_prompt(self, observation: dict) -> str:
        user = v8_policy_instruction(
            str(observation["candidate_role"]),
            str(observation.get("candidate_description", observation["candidate_role"])),
            str(observation["seed_query"]),
            str(observation["seed_output"]),
        )
        messages = [{"role": "system", "content": self.SYSTEM_V8}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.SYSTEM_V8 + "\n\n" + user + "\n\nASSISTANT:\n"

    def reference_mean_logprob(self, prompt: str, completion: str) -> float:
        torch = self.torch
        if hasattr(self.model, "disable_adapter"):
            with torch.no_grad(), self.model.disable_adapter():
                return float(self.mean_logprob(prompt, completion).detach().cpu())
        # Fallback: no separate reference is available; caller can set KL beta to zero.
        return float("nan")

    def sample_question(
        self,
        observation: dict,
        *,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_new_tokens: int = 72,
        deterministic: bool = False,
    ) -> QuestionSampleV8:
        torch = self.torch
        prompt = self.format_v8_prompt(observation)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(96, self.max_seq_length - max_new_tokens),
        ).to(self.device)
        self.model.eval()
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if deterministic:
            kwargs.update(do_sample=False)
        else:
            kwargs.update(do_sample=True, temperature=max(float(temperature), 0.05), top_p=float(top_p))
        with torch.no_grad():
            seq = self.model.generate(**encoded, **kwargs)
        new_ids = seq[0, encoded["input_ids"].shape[1]:]
        completion = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        normalized = extract_one_question(completion)
        # Train on the exact sampled question prefix when it is valid. Any text after the first
        # question mark is not sent to the MAS and should not receive positive credit.
        train_completion = normalized.query if normalized.valid else completion
        with torch.no_grad():
            old_lp = float(self.mean_logprob(prompt, train_completion).detach().cpu()) if train_completion else -20.0
        ref_lp = self.reference_mean_logprob(prompt, train_completion) if train_completion else -20.0
        return QuestionSampleV8(
            prompt=prompt,
            completion=train_completion,
            raw_completion=completion,
            query=normalized.query,
            valid=normalized.valid,
            invalid_reason=normalized.reason,
            old_mean_logprob=old_lp,
            ref_mean_logprob=ref_lp,
        )
