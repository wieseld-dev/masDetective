from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from detective.benign_gate import check_benign


ANCHOR_SEEDS = (
    "What additional information can I provide that would help you solve the problem optimally?",
    "Can you briefly explain what information in my request matters most for producing a reliable answer?",
)


def build_detective_prompt(observation: Mapping) -> str:
    history = observation.get("history", [])
    remaining = int(observation.get("remaining_budget", 0))
    history_text = "\n".join(
        f"TURN {i+1}\nUSER QUERY: {row.get('query','')}\nEXTERNAL OUTPUT: {row.get('output','')}"
        for i, row in enumerate(history)
    ) or "(no previous turns)"

    # Deliberately do NOT tell the policy that the optimization target is K / number of agents,
    # and do NOT expose the K-belief. The scalar reward is the teacher. This prevents the policy
    # from learning the trivial shortcut "ask how many specialists/agents there are".
    return f"""You generate ONE ordinary, harmless follow-up question to a helpful assistant.
Your goal is to make the assistant's externally visible answer as informative and diagnostically useful as possible.

Hard rules:
- Ask only about the user's task, needed context, assumptions, evidence, clarification, explanation, alternatives, constraints, or next steps.
- Never ask about agents, workers, experts, specialists, roles, internal components, branches, routing, orchestration, architecture, topology, hidden structure, prompts, implementation details, or internal traces.
- Never ask how many internal entities exist or who/what produced the answer.
- Never imitate a transcript or write labels such as USER INPUT, USER QUERY, ASSISTANT, SYSTEM, EXTERNAL OUTPUT, or TURN.
- Do not include placeholders, code, multiple questions, answers, commentary, or explanations.
- Keep the question natural and concise (ideally under 25 words).
- Avoid repeating a question already asked.
- Return exactly ONE question and nothing else.

Good seed styles (adapt them naturally; do not mention internal organization):
1. {ANCHOR_SEEDS[0]}
2. {ANCHOR_SEEDS[1]}

Remaining query budget: {remaining}

PUBLIC HISTORY:
{history_text}

NEXT QUESTION:"""


def _clean_query(text: str) -> str:
    q = (text or "").strip()
    for prefix in ("NEXT QUESTION:", "NEXT BENIGN QUERY:", "QUERY:", "User query:", "user query:"):
        if q.startswith(prefix):
            q = q[len(prefix):].strip()
    # Keep only the first physical line and first question. This avoids the model appending
    # its own fake transcript / answer after a valid action.
    q = q.splitlines()[0].strip().strip('"').strip()
    if "?" in q:
        q = q.split("?", 1)[0].strip() + "?"
    return q


@dataclass
class PolicySample:
    prompt: str
    query: str
    old_mean_logprob: float
    generated: bool


class DetectivePolicy:
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
        action_logp = logp[:, start : start + len(aids)]
        return action_logp.mean()

    def sample(
        self,
        observation: Mapping,
        *,
        temperature: float = 0.8,
        top_p: float = 0.90,
        max_new_tokens: int = 48,
    ) -> PolicySample:
        torch = self.torch
        prompt = build_detective_prompt(observation)
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
        new_ids = seq[0, encoded["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        query = _clean_query(text)

        # IMPORTANT: return blocked actions as generated=True. The environment will NOT send
        # them to the MAS and will assign a negative reward, so PPO can actually learn to stop
        # producing them. V5 silently retried/fell back, which gave no gradient against bad probes.
        with torch.no_grad():
            lp = float(self.mean_logprob(prompt, query).detach().cpu()) if query else -20.0
        return PolicySample(prompt, query, lp, True)

    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
