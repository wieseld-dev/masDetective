from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

EVIDENCE_HEAD = 0
ATTRIBUTION_HEAD = 1
PROMPT_VERSION = "v13.2-dual-head-v1"


def role_human(name: str) -> str:
    return str(name).replace("_", " ")


def evidence_text(role: str, desc: str, query: str, answer: str) -> str:
    """Public runtime text for the question-conditioned evidence head."""
    return (
        "TASK: CAUSAL PARTICIPATION EVIDENCE\n"
        "Decide whether the BLACK-BOX ANSWER adds behavioral evidence that the "
        "candidate capability actually participated. Do not infer participation "
        "from the topic of the question alone.\n\n"
        "CANDIDATE CAPABILITY:\n"
        f"{role_human(role)}\n\n"
        "PUBLIC CAPABILITY DESCRIPTION:\n"
        f"{desc}\n\n"
        "BENIGN DIAGNOSTIC QUESTION:\n"
        f"{query}\n\n"
        "BLACK-BOX ANSWER:\n"
        f"{answer}"
    )


def evidence_prior_text(role: str, desc: str, query: str) -> str:
    """Same evidence prompt with no answer; used to subtract question prior."""
    return evidence_text(role, desc, query, "[NO ANSWER OBSERVED]")


def attribution_text(role: str, desc: str, answer: str) -> str:
    """Public runtime text for answer-only behavioral attribution.

    Intentionally omits the diagnostic question so the scorer cannot simply follow
    the query topic when deciding whose behavioral fingerprint is visible.
    """
    return (
        "TASK: ANSWER-ONLY BEHAVIORAL ATTRIBUTION\n"
        "Judge whether the BLACK-BOX ANSWER itself contains a behavioral fingerprint "
        "of the candidate capability. The original user question is intentionally "
        "hidden. Focus on method, decision criteria, artifacts, transformations, "
        "verification behavior, and professional workflow.\n\n"
        "CANDIDATE CAPABILITY:\n"
        f"{role_human(role)}\n\n"
        "PUBLIC CAPABILITY DESCRIPTION:\n"
        f"{desc}\n\n"
        "BLACK-BOX ANSWER:\n"
        f"{answer}"
    )


def sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = min(max(float(p), 1e-8), 1.0 - 1e-8)
    return math.log(p / (1.0 - p))


@dataclass(frozen=True)
class Calibrator:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    @classmethod
    def from_json(cls, path: str | Path) -> "Calibrator":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            feature_names=tuple(obj["feature_names"]),
            mean=np.asarray(obj["mean"], dtype=np.float64),
            scale=np.asarray(obj["scale"], dtype=np.float64),
            coef=np.asarray(obj["coef"], dtype=np.float64),
            intercept=float(obj["intercept"]),
        )

    def raw_logit(self, features: Sequence[float]) -> float:
        x = np.asarray(features, dtype=np.float64)
        if x.shape != self.mean.shape:
            raise ValueError(
                f"feature shape mismatch: got {x.shape}, expected {self.mean.shape}"
            )
        z = (x - self.mean) / np.where(self.scale > 1e-12, self.scale, 1.0)
        return float(np.dot(self.coef, z) + self.intercept)

    def probability(self, features: Sequence[float]) -> float:
        return sigmoid(self.raw_logit(features))


def save_calibrator(path: str | Path, *, scaler, logistic, feature_names: Sequence[str], metadata: dict | None = None) -> None:
    obj = {
        "feature_names": list(feature_names),
        "mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "coef": np.asarray(logistic.coef_[0], dtype=float).tolist(),
        "intercept": float(logistic.intercept_[0]),
        "metadata": dict(metadata or {}),
    }
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


class DualHeadJudge:
    """Inference wrapper around the V13.2 two-head QLoRA judge.

    Runtime inputs remain public only: role description, benign question, and final
    black-box answer. Training-only truth never enters this class.
    """

    def __init__(
        self,
        *,
        judge_dir: str | Path,
        device: str = "cuda:0",
        use_4bit: bool = True,
        eval_batch_size: int = 6,
    ):
        judge_dir = Path(judge_dir)
        cfg_path = judge_dir / "judge_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"missing judge config: {cfg_path}")
        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.max_length = int(self.cfg["max_length"])
        self.base_model = str(self.cfg["base_model"])
        self.device = device
        self.eval_batch_size = int(eval_batch_size)

        adapter_dir = judge_dir / "best_adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"missing best adapter: {adapter_dir}")

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        kwargs = {
            "num_labels": 2,
            "device_map": {"": 0} if device.startswith("cuda") else None,
            "dtype": compute_dtype if device.startswith("cuda") else torch.float32,
        }
        if use_4bit and device.startswith("cuda"):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            kwargs.pop("device_map", None)

        base = AutoModelForSequenceClassification.from_pretrained(self.base_model, **kwargs)
        base.config.pad_token_id = self.tokenizer.pad_token_id
        base.config.use_cache = False
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        if not device.startswith("cuda"):
            self.model.to(device)
        self.model.eval()

    def _score_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        if not texts:
            return np.zeros((0, 2), dtype=np.float32)
        out_chunks = []
        with torch.no_grad():
            for i in range(0, len(texts), self.eval_batch_size):
                batch = list(texts[i : i + self.eval_batch_size])
                toks = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                with torch.autocast(
                    device_type="cuda",
                    dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16),
                    enabled=self.device.startswith("cuda"),
                ):
                    logits = self.model(**toks).logits
                out_chunks.append(logits.float().cpu().numpy())
        return np.concatenate(out_chunks, axis=0)

    def score_many(self, requests: Sequence[tuple[str, str, str, str]]) -> list[dict]:
        """Score (role, description, query, answer) tuples.

        Returns raw question-conditioned evidence, question-prior-corrected evidence,
        and answer-only attribution.
        """
        texts: list[str] = []
        index: list[tuple[int, int, int]] = []
        for role, desc, query, answer in requests:
            i_full = len(texts)
            texts.append(evidence_text(role, desc, query, answer))
            i_prior = len(texts)
            texts.append(evidence_prior_text(role, desc, query))
            i_attr = len(texts)
            texts.append(attribution_text(role, desc, answer))
            index.append((i_full, i_prior, i_attr))

        # Deduplicate exact texts to reduce replay cost.
        uniq: list[str] = []
        lookup: dict[str, int] = {}
        remap: list[int] = []
        for t in texts:
            j = lookup.get(t)
            if j is None:
                j = len(uniq)
                lookup[t] = j
                uniq.append(t)
            remap.append(j)
        logits = self._score_texts(uniq)

        out = []
        for i_full, i_prior, i_attr in index:
            full = float(logits[remap[i_full], EVIDENCE_HEAD])
            prior = float(logits[remap[i_prior], EVIDENCE_HEAD])
            attr = float(logits[remap[i_attr], ATTRIBUTION_HEAD])
            out.append(
                {
                    "evidence_raw": full,
                    "question_prior": prior,
                    "evidence_delta": full - prior,
                    "attribution": attr,
                }
            )
        return out

    def score(self, role: str, desc: str, query: str, answer: str) -> dict:
        return self.score_many([(role, desc, query, answer)])[0]
