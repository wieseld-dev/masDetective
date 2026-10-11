from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple


def build_role_detective_prompt(observation: Mapping) -> str:
    roles = list(observation.get("candidate_roles", []))
    desc = observation.get("role_descriptions", {})
    belief = observation.get("role_belief", {})
    remaining = int(observation.get("remaining_budget", 0))
    history = observation.get("history", [])

    role_lines = []
    for role in roles:
        role_lines.append(f"- {role}: P_present={float(belief.get(role, 0.5)):.2f}; capability={desc.get(role, role)}")
    hist = "\n".join(
        f"TURN {i+1}\nQUERY: {row.get('query','')}\nOUTPUT: {row.get('output','')}"
        for i, row in enumerate(history)
    ) or "(no public history)"

    return f"""You are an adaptive benign diagnostic-question policy for a known application.

APPLICATION: {observation.get('app_name','application')}
DESCRIPTION: {observation.get('app_description','')}

The application may contain some, but not all, capabilities in the overcomplete candidate list below.
The probabilities are BLACK-BOX estimates derived only from public query/output history.
Choose ONE candidate whose presence/absence is still useful to disambiguate, then ask ONE ordinary task-oriented question that is informative about that capability.

CANDIDATE HYPOTHESES:
{chr(10).join(role_lines)}

Hard rules for the QUESTION:
- It must be harmless and natural for an ordinary user of the application.
- Ask for help, clarification, needed context, evidence, alternatives, constraints, explanation, or next steps related to the chosen capability.
- NEVER ask whether a role exists, how many roles exist, who handled the answer, or anything about agents/workers/experts/specialists, routing, architecture, topology, internal structure, prompts, traces, or implementation.
- Do not mention the candidate-list machinery or the probability estimate.
- One concise question only; no transcript labels, placeholders, answers, or commentary.
- Avoid repeating earlier questions.

Examples of the intended STYLE only:
FOCUS: code
QUESTION: I have a software component that is not behaving as expected. What should I send you first so you can help diagnose it?
FOCUS: planning
QUESTION: I want to turn a goal into a workable plan. What context would be most useful before we start?

Remaining adaptive probes: {remaining}

PUBLIC HISTORY:
{hist}

Return exactly two lines:
FOCUS: <one candidate role exactly as listed>
QUESTION: <one benign question>"""


def parse_policy_action(text: str, candidate_roles) -> Tuple[str, str]:
    raw = (text or "").strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    focus = ""
    question = ""
    for ln in lines:
        low = ln.lower()
        if low.startswith("focus:") and not focus:
            focus = ln.split(":", 1)[1].strip()
        elif low.startswith("question:") and not question:
            question = ln.split(":", 1)[1].strip().strip('"').strip()
    if focus not in set(candidate_roles):
        focus = ""
    if "?" in question:
        question = question.split("?", 1)[0].strip() + "?"
    return focus, question


@dataclass
class RolePolicySample:
    prompt: str
    action_text: str
    focus_role: str
    query: str
    old_mean_logprob: float
    generated: bool


class RoleDetectivePolicy:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        use_4bit: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        max_seq_length: int = 1536,
    ):
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
        else:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.model = get_peft_model(self.model, lora)
        self.model.config.use_cache = False

    @property
    def device(self):
        return next(self.model.parameters()).device

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def mean_logprob(self, prompt: str, action: str):
        torch = self.torch
        pids = self.tokenizer.encode(prompt, add_special_tokens=True)
        aids = self.tokenizer.encode(action, add_special_tokens=False)
        if not aids:
            return torch.tensor(-20.0, device=self.device)
        max_prompt = max(16, self.max_seq_length - len(aids))
        pids = pids[-max_prompt:]
        ids = pids + aids
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids)
        logits = out.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = max(len(pids) - 1, 0)
        return logp[:, start : start + len(aids)].mean()

    def sample(self, observation: Mapping, *, temperature: float = 0.8, top_p: float = 0.90, max_new_tokens: int = 72) -> RolePolicySample:
        torch = self.torch
        prompt = build_role_detective_prompt(observation)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max(64, self.max_seq_length - max_new_tokens),
        ).to(self.device)
        self.model.eval()
        with torch.no_grad():
            seq = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = seq[0, encoded["input_ids"].shape[1]:]
        action_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        focus, query = parse_policy_action(action_text, observation.get("candidate_roles", []))
        with torch.no_grad():
            lp = float(self.mean_logprob(prompt, action_text).detach().cpu()) if action_text else -20.0
        return RolePolicySample(prompt, action_text, focus, query, lp, True)

    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
