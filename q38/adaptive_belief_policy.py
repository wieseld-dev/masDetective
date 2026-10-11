from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from peft import PeftModel

from v14.adaptive_belief_policy import V14AdaptiveBeliefPolicy


class _Q38TokenizerProxy:
    """
    Reuse the exact V14 prompt construction while forcing Qwen3.8
    into non-thinking text-only chat mode.
    """

    def __init__(self, processor):
        object.__setattr__(self, "_processor", processor)
        tok = getattr(processor, "tokenizer", processor)
        object.__setattr__(self, "_tok", tok)

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __setattr__(self, name, value):
        if name in {"_processor", "_tok"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._tok, name, value)

    def __call__(self, *args, **kwargs):
        return self._tok(*args, **kwargs)

    def apply_chat_template(self, messages, *args, **kwargs):
        converted = []

        for m in messages:
            content = m.get("content", "")

            if isinstance(content, str):
                content = [
                    {
                        "type": "text",
                        "text": content,
                    }
                ]

            converted.append(
                {
                    "role": m["role"],
                    "content": content,
                }
            )

        kwargs.pop("enable_thinking", None)

        return self._processor.apply_chat_template(
            converted,
            *args,
            enable_thinking=False,
            **kwargs,
        )

    def save_pretrained(self, path):
        return self._tok.save_pretrained(path)


class Q38AdaptiveBeliefPolicy(V14AdaptiveBeliefPolicy):
    """
    Qwen3.8-27B implementation of the exact V14 adaptive-belief policy.

    Algorithmic state/prompting comes from V14AdaptiveBeliefPolicy.
    Only model loading/tokenization is Q38-specific.
    """

    def __init__(
        self,
        model_name: str,
        *,
        adapter_path: str,
        reference_adapter_path: str | None = None,
        trainable: bool = False,
        use_4bit: bool = False,
        max_seq_length: int = 3072,
        **kwargs,
    ):
        if use_4bit:
            raise RuntimeError(
                "Q38 online run is intentionally BF16; "
                "do not enable 4-bit."
            )

        self.torch = torch
        self.model_name = model_name
        self.max_seq_length = int(max_seq_length)

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

        raw_tok = getattr(
            self.processor,
            "tokenizer",
            self.processor,
        )

        if raw_tok.pad_token_id is None:
            raw_tok.pad_token = raw_tok.eos_token

        self.tokenizer = _Q38TokenizerProxy(
            self.processor
        )

        print(
            "[Q38-policy] loading BF16 base:",
            model_name,
            flush=True,
        )

        base = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

        if hasattr(base.config, "use_cache"):
            base.config.use_cache = not trainable

        if trainable:
            base.gradient_checkpointing_enable()

            if hasattr(
                base,
                "enable_input_require_grads",
            ):
                base.enable_input_require_grads()

        self.model = PeftModel.from_pretrained(
            base,
            str(adapter_path),
            is_trainable=trainable,
        )

        self.reference_adapter_path = (
            str(reference_adapter_path)
            if reference_adapter_path
            else None
        )

        self.has_reference = False

        if reference_adapter_path:
            self.model.load_adapter(
                str(reference_adapter_path),
                adapter_name="reference",
                is_trainable=False,
            )
            self.has_reference = True

        self.model.set_adapter("default")

        if trainable:
            self.model.train()
        else:
            self.model.eval()

        print(
            "[Q38-policy] adapter=",
            adapter_path,
            "reference=",
            reference_adapter_path,
            "trainable=",
            trainable,
            "VRAM=",
            f"{torch.cuda.memory_allocated()/1024**3:.2f} GiB",
            flush=True,
        )

    def sample_probe(self, *args, **kwargs):
        self.model.set_adapter("default")
        return super().sample_probe(
            *args,
            **kwargs,
        )

    def mean_logprob(
        self,
        prompt: str,
        completion: str,
        *,
        adapter: str = "current",
    ):
        if (
            adapter == "reference"
            and self.has_reference
        ):
            name = "reference"
        else:
            name = "default"

        self.model.set_adapter(name)

        try:
            return super().mean_logprob(
                prompt,
                completion,
            )
        finally:
            self.model.set_adapter("default")

    def save_adapter(self, path: str) -> None:
        path = Path(path)
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.set_adapter("default")

        self.model.save_pretrained(
            str(path)
        )

        # Checkpoints only require tokenizer files;
        # processor itself is always loaded from the base.
        getattr(
            self.processor,
            "tokenizer",
            self.processor,
        ).save_pretrained(
            str(path)
        )
