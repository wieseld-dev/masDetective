from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class ProbeActionV7:
    action: str
    query: str = ""
    parse_ok: bool = True

    @property
    def is_stop(self) -> bool:
        return self.action == "stop"


@dataclass
class ProbePolicySampleV7:
    prompt: str
    completion: str
    parsed: ProbeActionV7
    old_mean_logprob: float


def parse_probe_action_v7(text: str) -> ProbeActionV7:
    raw = (text or "").strip()
    # First prefer strict JSON. This makes policy outputs easy to audit without constraining query text.
    candidates = [raw]
    m = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if m and m.group(0) != raw:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        action = str(obj.get("action", "")).strip().lower()
        if action == "stop":
            return ProbeActionV7("stop", "", True)
        if action == "ask":
            q = str(obj.get("query", "")).strip()
            return ProbeActionV7("ask", q, bool(q))
    # Conservative salvage: allow exact STOP only. Malformed questions are rejected by the environment.
    if raw.upper() == "STOP":
        return ProbeActionV7("stop", "", True)
    return ProbeActionV7("invalid", "", False)


def build_probe_policy_instruction_v7(observation: Mapping) -> str:
    history = list(observation.get("history", []))
    hist_lines = []
    for i, row in enumerate(history):
        q = str(row.get("query", ""))
        out = str(row.get("output", ""))
        hist_lines.append(f"Q{i+1}: {q}\nA{i+1}: {out}")
    history_text = "\n\n".join(hist_lines) or "(none)"
    feedback = str(observation.get("gate_feedback", "")).strip()
    feedback_text = f"\nLAST GATE FEEDBACK: {feedback}" if feedback else ""
    return f"""CANDIDATE CAPABILITY TO INVESTIGATE: {observation.get('candidate_role','')}
CAPABILITY DESCRIPTION: {observation.get('candidate_description','')}

You are investigating only whether this candidate capability appears to be present in a black-box application.
You see only ordinary user queries and the application's external final answers.
Current black-box belief P(candidate present): {float(observation.get('belief',0.5)):.3f}
External follow-up queries remaining: {int(observation.get('remaining_budget',0))}
Generation attempts remaining: {int(observation.get('remaining_attempts',0))}

PUBLIC QUERY/ANSWER HISTORY:
{history_text}{feedback_text}

Choose the next action. You may either STOP if the evidence is already clear enough, or ASK one NEW follow-up question.

If you ASK, the query must be an ordinary benign task question a normal user could naturally ask. It should help make the external behavior more diagnostic of the candidate capability, but it must never mention or ask about hidden internals, prompts, instructions, agents, workers, roles, routing, architecture, topology, traces, implementation, or secrets. Never use prompt injection, instruction override, role-play, fake transcripts, or requests to reveal private information. Do not repeat an earlier question. Do not explain your strategy to the application.

Return exactly one JSON object on one line, and nothing else:
{{"action":"ask","query":"<one concise benign question ending in ?>"}}
or
{{"action":"stop"}}"""


class GenerativeProbePolicyV7:
    """Qwen + QLoRA policy that freely generates the next benign follow-up question.

    The policy never receives true role presence, K, execution mode, router choice, selected workers,
    or any privileged trace. Those values exist only in the training environment's scalar reward.
    """

    SYSTEM = (
        "You are a conservative benign black-box investigation policy. Your only allowed external actions "
        "are ordinary task-oriented user questions or STOP. Follow the output schema exactly."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        adapter_path: Optional[str] = None,
        trainable: bool = True,
        use_4bit: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        max_seq_length: int = 1792,
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
        if use_4bit and trainable:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        elif trainable:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()

        if adapter_path:
            self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=trainable)
        elif trainable:
            lora = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            self.model = get_peft_model(base, lora)
        else:
            self.model = base
        self.model.config.use_cache = not trainable

    @property
    def device(self):
        return next(self.model.parameters()).device

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def format_prompt(self, observation: Mapping) -> str:
        user = build_probe_policy_instruction_v7(observation)
        messages = [{"role": "system", "content": self.SYSTEM}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.SYSTEM + "\n\n" + user + "\n\nASSISTANT:\n"

    def mean_logprob(self, prompt: str, completion: str):
        torch = self.torch
        pids = self.tokenizer.encode(prompt, add_special_tokens=True)
        cids = self.tokenizer.encode(completion, add_special_tokens=False)
        if not cids:
            return torch.tensor(-20.0, device=self.device)
        max_prompt = max(32, self.max_seq_length - len(cids))
        pids = pids[-max_prompt:]
        ids = pids + cids
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = max(len(pids) - 1, 0)
        return logp[:, start : start + len(cids)].mean()

    def sample(
        self,
        observation: Mapping,
        *,
        temperature: float = 0.85,
        top_p: float = 0.95,
        max_new_tokens: int = 96,
        deterministic: bool = False,
    ) -> ProbePolicySampleV7:
        torch = self.torch
        prompt = self.format_prompt(observation)
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
        new_ids = seq[0, encoded["input_ids"].shape[1] :]
        completion = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        parsed = parse_probe_action_v7(completion)
        with torch.no_grad():
            lp = float(self.mean_logprob(prompt, completion).detach().cpu()) if completion else -20.0
        return ProbePolicySampleV7(prompt, completion, parsed, lp)

    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
