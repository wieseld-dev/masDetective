from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from probes.role_probe_bank import primary_probe


def available_roles(observation: Mapping) -> Tuple[str, ...]:
    candidates = tuple(str(r) for r in observation.get("candidate_roles", ()))
    tested = set(str(r) for r in observation.get("tested_roles", ()))
    remain = tuple(r for r in candidates if r not in tested)
    return remain or candidates


def build_role_choice_prompt(observation: Mapping, choices: Sequence[str] | None = None) -> str:
    choices = tuple(choices or available_roles(observation))
    desc = observation.get("role_descriptions", {})
    belief = observation.get("role_belief", {})
    lines = []
    for i, role in enumerate(choices):
        label = chr(ord("A") + i)
        p = float(belief.get(role, 0.5))
        uncertainty = 1.0 - 2.0 * abs(p - 0.5)
        lines.append(
            f"{label} = {role} | P_present={p:.3f} | uncertainty={uncertainty:.3f} | "
            f"capability={desc.get(role, role)}"
        )
    return f"""Choose the next capability to test with a fixed benign diagnostic probe.

APPLICATION: {observation.get('app_name','application')}
The candidate list is deliberately overcomplete: some capabilities exist and some do not.
P_present is a BLACK-BOX belief based only on previous external outputs.
The goal is to choose ONE untested capability whose next probe is most useful for improving the
present/absent reconstruction under the remaining query budget. You do NOT write the probe.

REMAINING PROBE BUDGET: {observation.get('remaining_budget', 0)}
AVAILABLE ACTIONS:
{chr(10).join(lines)}

Return only the single letter of your choice."""


def fixed_query_for_role(role: str) -> str:
    return primary_probe(role)


@dataclass
class RoleChoiceSample:
    prompt: str
    focus_role: str
    query: str
    action_index: int
    choices: Tuple[str, ...]
    old_logprob: float


class RoleChoicePolicyV63:
    """Qwen+LoRA categorical policy over roles; no free-text probe generation."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        use_4bit: bool = True,
        adapter_path: str | None = None,
        train_adapter: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        max_seq_length: int = 1024,
    ):
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.model_name = model_name
        self.max_seq_length = int(max_seq_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {}
        if use_4bit:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs.update(quantization_config=qcfg, device_map={"": 0})
        else:
            kwargs.update(torch_dtype=torch.float16, device_map={"": 0})
        base = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        if adapter_path:
            self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=train_adapter)
            if train_adapter:
                self.model.gradient_checkpointing_enable()
                self.model.enable_input_require_grads()
        else:
            if use_4bit:
                base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
            else:
                base.gradient_checkpointing_enable()
                base.enable_input_require_grads()
            lora = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.model = get_peft_model(base, lora)
        self.model.config.use_cache = False

        self._label_token_ids = []
        for label in "ABCDEFGH":
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"policy label {label!r} is not one token: {ids}")
            self._label_token_ids.append(int(ids[0]))
        if len(set(self._label_token_ids)) != 8:
            raise RuntimeError("action labels are not unique tokenizer tokens")

    @property
    def device(self):
        return next(self.model.parameters()).device

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _chat_prompt(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": "Choose exactly one available action label. Return only that letter."},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _choice_logprobs(self, prompt: str, n_choices: int):
        torch = self.torch
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.device)
        out = self.model(**encoded)
        logits = out.logits[0, -1, :]
        ids = torch.tensor(self._label_token_ids[:n_choices], dtype=torch.long, device=self.device)
        return torch.log_softmax(logits.index_select(0, ids), dim=0)

    def action_logprob_and_entropy(self, prompt: str, n_choices: int, action_index: int):
        logps = self._choice_logprobs(prompt, n_choices)
        probs = self.torch.exp(logps)
        entropy = -(probs * logps).sum()
        return logps[int(action_index)], entropy

    def sample(self, observation: Mapping, *, temperature: float = 1.0) -> RoleChoiceSample:
        torch = self.torch
        choices = available_roles(observation)
        raw = build_role_choice_prompt(observation, choices)
        prompt = self._chat_prompt(raw)
        self.model.eval()
        with torch.no_grad():
            logps = self._choice_logprobs(prompt, len(choices))
            if temperature <= 0:
                idx = int(torch.argmax(logps).item())
            else:
                probs = torch.softmax(logps / float(temperature), dim=0)
                idx = int(torch.multinomial(probs, 1).item())
            old = float(logps[idx].detach().cpu())
        role = choices[idx]
        return RoleChoiceSample(prompt, role, fixed_query_for_role(role), idx, tuple(choices), old)

    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
