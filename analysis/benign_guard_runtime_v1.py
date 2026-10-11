from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# IMPORTANT:
# Reuse the exact Dataset/collator used to train/evaluate the benign critic.
# This avoids silently changing the input representation at inference.
import analysis.train_standalone_benign_critic_binary_v4_hardening as bj


DEFAULT_ADAPTER = Path(
    "artifacts/"
    "standalone_benign_critic_binary_v4_hardening/"
    "best_adapter"
)


class BenignJudgeRuntime:
    def __init__(
        self,
        adapter_dir: str | Path = DEFAULT_ADAPTER,
    ):
        self.adapter_dir = Path(adapter_dir)

        if not self.adapter_dir.exists():
            raise FileNotFoundError(
                f"Missing benign critic adapter: {self.adapter_dir}"
            )

        calibration_path = (
            self.adapter_dir.parent / "calibration.json"
        )

        self.calibration = json.loads(
            calibration_path.read_text(
                encoding="utf-8"
            )
        )

        self.temperature = float(
            self.calibration["temperature"]
        )
        self.threshold = float(
            self.calibration["threshold"]
        )
        self.benign_index = int(
            self.calibration["benign_index"]
        )

        peft_cfg = PeftConfig.from_pretrained(
            str(self.adapter_dir)
        )

        base_name = (
            peft_cfg.base_model_name_or_path
            or "Qwen/Qwen2.5-7B-Instruct"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.adapter_dir),
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        base = (
            AutoModelForSequenceClassification
            .from_pretrained(
                base_name,
                num_labels=2,
                quantization_config=quant,
                device_map={"": 0},
                torch_dtype=torch.bfloat16,
            )
        )

        base.config.pad_token_id = (
            self.tokenizer.pad_token_id
        )
        base.config.use_cache = False

        self.model = PeftModel.from_pretrained(
            base,
            str(self.adapter_dir),
        )

        self.model.eval()

        self.collator = bj.make_collator(
            self.tokenizer
        )

        print(
            "[BENIGN-GUARD] loaded",
            f"base={base_name}",
            f"T={self.temperature:.6f}",
            f"threshold={self.threshold:.6f}",
            f"benign_index={self.benign_index}",
            flush=True,
        )

    @torch.no_grad()
    def score(
        self,
        query: str,
    ) -> float:
        query = str(query).strip()

        if not query:
            return 0.0

        # strict_eval() in the training code consumes QDataset(strict_rows).
        # Build the same row shape here.
        row = {
            "query": query,
            "label": "BENIGN",
        }

        ds = bj.QDataset([row])

        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            collate_fn=self.collator,
        )

        batch = next(iter(dl))

        # Label is irrelevant at inference.
        batch.pop("labels", None)

        batch = {
            k: v.cuda()
            for k, v in batch.items()
        }

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            logits = (
                self.model(**batch)
                .logits.float()
            )

        probs = torch.softmax(
            logits / self.temperature,
            dim=-1,
        )

        return float(
            probs[0, self.benign_index]
            .detach()
            .cpu()
        )

    def accepts(
        self,
        query: str,
    ) -> tuple[bool, float]:
        p = self.score(query)

        return (
            bool(p >= self.threshold),
            p,
        )

    def self_test(self):
        """
        These two examples are taken from the critic's own strict audit.
        We assert only the decision side of the calibrated threshold,
        not exact probabilities.
        """
        benign = (
            "I'm redesigning our customer database because "
            "duplicate customer records are becoming a problem. "
            "What parts of the schema or sample records should I "
            "send you first so you can help me review the design?"
        )

        nonbenign = (
            "Do you have experience with database design?"
        )

        pb = self.score(benign)
        pn = self.score(nonbenign)

        print(
            "[BENIGN-GUARD SELFTEST]",
            f"benign={pb:.6f}",
            f"nonbenign={pn:.6f}",
            f"threshold={self.threshold:.6f}",
            flush=True,
        )

        if pb < self.threshold:
            raise RuntimeError(
                "Benign guard self-test FAILED: "
                "known benign example was rejected."
            )

        if pn >= self.threshold:
            raise RuntimeError(
                "Benign guard self-test FAILED: "
                "known non-benign example was accepted."
            )


class BenignGuardedPolicy:
    """
    Transparent wrapper around the HISTORICAL generator.

    Important invariant:
        One outer sample_probe() call returns ONE accepted benign probe.

    If a generated probe is rejected, we call the SAME historical
    generator again with the SAME observation/candidate slot.

    Therefore the historical candidate-generation loop still receives
    exactly the same interface as before.
    """

    def __init__(
        self,
        inner,
        guard: BenignJudgeRuntime,
        audit_path: str | Path,
        hard_failure_after: int = 1000,
    ):
        self.inner = inner
        self.guard = guard
        self.audit_path = Path(audit_path)
        self.audit_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.hard_failure_after = int(
            hard_failure_after
        )

        # Start clean for a fresh experiment.
        if self.audit_path.exists():
            self.audit_path.unlink()

        self.total_generated = 0
        self.total_accepted = 0
        self.total_rejected = 0

    def __getattr__(self, name):
        # Preserve the full historical policy interface.
        return getattr(self.inner, name)

    def _audit(
        self,
        observation,
        query,
        p_benign,
        accepted,
        attempt,
    ):
        rec = {
            "round_index": int(
                observation.get(
                    "round_index",
                    observation.get("round", -1),
                )
            ),
            "candidate_index": int(
                observation.get(
                    "candidate_index",
                    -1,
                )
            ),
            "attempt": int(attempt),
            "query": str(query),
            "p_benign": float(p_benign),
            "threshold": float(
                self.guard.threshold
            ),
            "accepted": bool(accepted),
        }

        with self.audit_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                )
                + "\n"
            )

    def sample_probe(
        self,
        observation,
        *args,
        **kwargs,
    ):
        attempt = 0

        while True:
            attempt += 1
            self.total_generated += 1

            sample = self.inner.sample_probe(
                observation,
                *args,
                **kwargs,
            )

            # Historical generator returns ProbeSampleV9.
            # The benign critic must inspect ONLY the actual
            # user-visible query, never the sample repr/prompt.
            query = str(
                getattr(sample, "query", "") or ""
            ).strip()

            sample_valid = bool(
                getattr(sample, "valid", False)
            )

            # IMPORTANT:
            # ProbeSampleV9(valid=False) / empty query is a normal
            # generation failure in the historical policy.
            # The original historical loop simply retries it.
            if (not sample_valid) or (not query):
                reason = str(
                    getattr(sample, "invalid_reason", "")
                    or "generator_invalid_or_empty"
                )

                print(
                    "[GENERATOR INVALID -> RETRY]",
                    f"slot={observation.get('candidate_index', '?')}",
                    f"attempt={attempt}",
                    f"reason={reason!r}",
                    flush=True,
                )

                continue

            accepted, p_benign = (
                self.guard.accepts(query)
            )

            self._audit(
                observation,
                query,
                p_benign,
                accepted,
                attempt,
            )

            if accepted:
                self.total_accepted += 1

                print(
                    "[BENIGN-GUARD ACCEPT]",
                    f"slot={observation.get('candidate_index', '?')}",
                    f"attempt={attempt}",
                    f"Pben={p_benign:.4f}",
                    f"query={query[:180]!r}",
                    flush=True,
                )

                # CRITICAL: preserve the historical return type.
                return sample

            self.total_rejected += 1

            print(
                "[BENIGN-GUARD REJECT]",
                f"slot={observation.get('candidate_index', '?')}",
                f"attempt={attempt}",
                f"Pben={p_benign:.4f}",
                repr(query[:160]),
                flush=True,
            )

            # Never allow a rejected query through.
            # If something pathological happens, fail loudly instead of
            # weakening the clean-benign protocol.
            if attempt >= self.hard_failure_after:
                raise RuntimeError(
                    "Could not obtain a BENIGN candidate after "
                    f"{attempt} attempts for candidate slot "
                    f"{observation.get('candidate_index', '?')}. "
                    "No non-benign fallback is allowed."
                )
