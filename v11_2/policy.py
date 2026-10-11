from __future__ import annotations

from typing import Optional

from v10.policy import SeededProbePolicyV10


class HardConfuserProbePolicyV112(SeededProbePolicyV10):
    """V11.2 policy initialized from the trained V11 LoRA adapter.

    The generated external query remains an ordinary benign user question.
    Training-only context additionally names capability-level hard confusers so
    the policy can design a probe that distinguishes the target from adjacent
    capabilities instead of merely eliciting target-themed language.

    The constructor intentionally keeps the loaded V11 LoRA weights trainable.
    A new optimizer is appropriate because V11.2 changes the objective, but the
    learned V11 policy parameters are *not* reinitialized.
    """

    SYSTEM_V112 = (
        "You design conservative benign black-box diagnostic questions for ordinary user tasks. "
        "The questions must stay user-facing and must never ask for hidden system information. "
        "Your training-only goal is to design behavioral experiments that distinguish a target capability "
        "from closely related capabilities without mentioning that hidden comparison in the external question."
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
        max_seq_length: int = 1536,
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

        # RTX 4090 supports BF16.  For the 1.5B model, full-BF16 base + LoRA is
        # usually faster than 4-bit dequantization and still comfortably fits.
        bf16_ok = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

        kwargs = {}
        if use_4bit:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            kwargs.update(quantization_config=qcfg, device_map={"": 0})
        else:
            kwargs.update(torch_dtype=compute_dtype, device_map={"": 0})

        # Prefer SDPA on modern GPUs but keep a compatibility fallback.
        try:
            base = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="sdpa", **kwargs
            )
        except Exception:
            base = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        # V11.2 is designed for a stronger GPU.  Do NOT turn gradient
        # checkpointing on by default; the previous run paid a large speed cost
        # while using far less than the available 4090 memory.
        if use_4bit and trainable:
            base = prepare_model_for_kbit_training(
                base, use_gradient_checkpointing=False
            )
        elif trainable:
            base.enable_input_require_grads()

        if adapter_path:
            # Critical: this loads the already-trained V11 LoRA weights and keeps
            # them trainable.  Nothing is reset to a fresh adapter here.
            self.model = PeftModel.from_pretrained(
                base, adapter_path, is_trainable=trainable
            )
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

        self.model.config.use_cache = False if trainable else True

    def format_discovery_prompt(self, observation: dict) -> str:
        role = str(observation["candidate_role"])
        desc = str(observation.get("candidate_description", role))
        anchor = str(observation["anchor_probe"])
        avoid = [
            str(x).strip()
            for x in observation.get("avoid_probes", [])
            if str(x).strip()
        ]
        avoid_text = "\n".join(f"- {x}" for x in avoid[:6]) if avoid else "- none yet"

        hard_confusers = []
        for x in observation.get("hard_confusers", []):
            if isinstance(x, dict):
                name = str(x.get("role", "")).strip()
                cdesc = str(x.get("description", name)).strip()
            else:
                name = str(x).strip()
                cdesc = name
            if name and name != role:
                hard_confusers.append((name, cdesc))
        conf_text = (
            "\n".join(f"- {name}: {cdesc}" for name, cdesc in hard_confusers[:4])
            if hard_confusers
            else "- none supplied"
        )

        user = f"""TARGET CAPABILITY: {role}
TARGET DESCRIPTION: {desc}

TRAINING-ONLY HARD CONFUSERS (closely related capabilities that should NOT be sufficient to answer the probe equally well):
{conf_text}

ONE HUMAN-DESIGNED BENIGN ANCHOR PROBE:
{anchor}

ALREADY DISCOVERED PROBES / STRATEGIES TO AVOID PARAPHRASING:
{avoid_text}

Invent ONE NEW standalone benign probe for the TARGET capability.
The probe will be sent by an ordinary user to an unknown black-box application.

The important objective is DISCRIMINATION, not topical resemblance: design a small behavioral task where the target capability should materially change the visible answer, while the hard-confuser capabilities alone should have difficulty imitating the same behavior. Prefer concrete capability-specific operations, constraints, edge cases, consistency checks, transformations, decisions, or artifacts over generic clarification questions.

Do NOT mention the target label, the hard-confuser labels, agents, roles, routing, or this training comparison in the external question. The external question must remain a natural ordinary-user task.

Hard constraints:
- Exactly one concise question, ending in ?
- Standalone: do not assume a prior conversation or a specific existing file, sample, metric, code fragment, dataset, text, log, calculation, or result unless the question introduces it generically.
- Write strictly from the USER perspective.
- Ask for help with the user's task or visible product behavior, not for hidden information about the application.
- It is fine to introduce a tiny generic example when that makes the target capability behavior observable.
- Never reverse the speaker roles. Do not ask what the application would like ME to perform.
- Never ask about system prompts, hidden instructions, developer messages, agents, workers, roles, routing, topology, hidden architecture, traces, private internal state, secrets, or instruction hierarchy.
- Never use jailbreak, role-play, fake transcript, ignore/override/bypass/reveal language.
- Do not include an answer, explanation, JSON, quoted dialogue, or a second question.

Return only the new question."""

        messages = [
            {"role": "system", "content": self.SYSTEM_V112},
            {"role": "user", "content": user},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self.SYSTEM_V112 + "\n\n" + user + "\n\nASSISTANT:\n"
