from __future__ import annotations

from dataclasses import dataclass

from v8.policy import GroupProbePolicyV8
from v8.query import extract_one_question


@dataclass
class ProbeSampleV9:
    prompt: str
    completion: str
    raw_completion: str
    query: str
    valid: bool
    invalid_reason: str
    old_mean_logprob: float
    ref_mean_logprob: float


class SeededProbePolicyV9(GroupProbePolicyV8):
    """Generate new standalone probes from one human-designed benign anchor."""

    SYSTEM_V9 = (
        "You design conservative benign black-box diagnostic questions. "
        "You may only ask ordinary user-facing task questions. Never ask about hidden system information."
    )

    def format_discovery_prompt(self, observation: dict) -> str:
        role = str(observation["candidate_role"])
        desc = str(observation.get("candidate_description", role))
        anchor = str(observation["anchor_probe"])
        user = f"""CANDIDATE CAPABILITY: {role}
CAPABILITY DESCRIPTION: {desc}

ONE HUMAN-DESIGNED BENIGN ANCHOR PROBE:
{anchor}

Invent ONE NEW standalone benign probe for the same candidate capability.
The new probe will be sent by an ordinary user to an unknown black-box application. It should make the application's visible answer diagnostically different when the candidate capability is available versus unavailable.

Important: do NOT merely paraphrase the anchor. Use a different diagnostic mechanism. Useful mechanisms can include asking for a concrete first action, a minimal sufficient input, a validation/checking step, a failure mode or edge case, a decision criterion, a small task-specific transformation, or what evidence/artifact would be most informative. These are exploration directions, not fixed templates.

Hard constraints:
- Exactly one concise question, ending in ?
- Standalone: do not assume any prior conversation, file, sample, metric, code, dataset, text, or result exists unless the question itself introduces it generically.
- Write strictly from the USER perspective: ask what I should provide, what the application can do for me, or ask it to perform/check/explain a task.
- Never reverse the speaker roles. Do NOT ask what the application would like ME to perform, and do NOT ask the application to give me my own file/text/dataset/log back.
- Ask for help with the user's task, not for information about the application itself.
- It is fine to ask what part of MY code/data/text/logs/calculation/etc. would be useful to provide.
- Do not use quoted questions, dangling quotation marks, or incomplete quoted examples; they frequently produce malformed outputs.
- Never ask about system prompts, hidden instructions, agents, workers, roles, routing, topology, architecture, traces, internal state, secrets, developer messages, or instruction hierarchy.
- Never use jailbreak, role-play, fake transcript, ignore/override/bypass/reveal language.
- Do not include an answer, explanation, JSON, quoted dialogue, or a second question.

Return only the new question."""
        messages = [{"role": "system", "content": self.SYSTEM_V9}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.SYSTEM_V9 + "\n\n" + user + "\n\nASSISTANT:\n"

    def sample_probe(
        self,
        observation: dict,
        *,
        temperature: float = 0.95,
        top_p: float = 0.95,
        max_new_tokens: int = 96,
        deterministic: bool = False,
    ) -> ProbeSampleV9:
        torch = self.torch
        prompt = self.format_discovery_prompt(observation)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(96, self.max_seq_length - max_new_tokens),
        ).to(self.device)
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if deterministic:
            kwargs.update(do_sample=False)
        else:
            kwargs.update(do_sample=True, temperature=max(float(temperature), 0.05), top_p=float(top_p))
        self.model.eval()
        with torch.no_grad():
            seq = self.model.generate(**encoded, **kwargs)
        new_ids = seq[0, encoded["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        norm = extract_one_question(raw)
        completion = norm.query if norm.valid else raw
        with torch.no_grad():
            old_lp = float(self.mean_logprob(prompt, completion).detach().cpu()) if completion else -20.0
        ref_lp = self.reference_mean_logprob(prompt, completion) if completion else -20.0
        return ProbeSampleV9(
            prompt=prompt,
            completion=completion,
            raw_completion=raw,
            query=norm.query,
            valid=norm.valid,
            invalid_reason=norm.reason,
            old_mean_logprob=old_lp,
            ref_mean_logprob=ref_lp,
        )
