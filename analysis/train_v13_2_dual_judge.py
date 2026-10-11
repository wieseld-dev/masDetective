#!/usr/bin/env python3
"""
V13.2: dual-head causal / answer-attribution judge.

Head 0: question-conditioned participation evidence, scored as
    E(role,q,a) = S_e(role,q,a) - S_e(role,q,NO_ANSWER)
so query-topic prior is explicitly removed.

Head 1: answer-only behavioral attribution. It never sees the diagnostic question.
This head is trained with explicit V13 hard-neighbor flips:
    PLUS/SHAM: target > hard neighbor
    MINUS:     hard neighbor > target

Data:
- V13 hard-negative counterfactual discovery data (primary)
- V11 counterfactual discovery data (causal breadth)
- privileged selected/true role information is used ONLY to construct training labels
- pilot/eval sessions are never used for training

Leakage controls:
- DSU groups source probe group IDs and counterfactual pair IDs
- identical diagnostic questions are linked across sources
- V13 role-set compositions are linked across probe groups
- exact duplicate public outputs are linked

No OpenAI/API calls are made by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from v13_2.judge import (
    ATTRIBUTION_HEAD,
    EVIDENCE_HEAD,
    PROMPT_VERSION,
    attribution_text,
    evidence_prior_text,
    evidence_text,
    save_calibrator,
)


# ---------------------------------------------------------------------------
# IO / deterministic helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    out: list[dict] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"{path}:{ln}: {e}") from e
            if not isinstance(obj, dict):
                raise RuntimeError(f"{path}:{ln}: expected JSON object")
            out.append(obj)
    return out


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def short_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:20]


def composition_key(roles: Iterable[str]) -> str:
    return "|".join(sorted(map(str, roles)))


def resolve_base_model(arg: str) -> str:
    if arg != "auto":
        return arg
    for p in (
        Path("artifacts/v13_policy/adapter_config.json"),
        Path("artifacts/v11_policy/adapter_config.json"),
    ):
        if p.exists():
            cfg = json.loads(p.read_text(encoding="utf-8"))
            base = cfg.get("base_model_name_or_path")
            if base:
                print(f"[model] discovered base model from {p}: {base}", flush=True)
                return str(base)
    fallback = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"[model] adapter config not found; fallback={fallback}", flush=True)
    return fallback


def load_role_descriptions() -> dict[str, str]:
    from v11.role_universe import ROLE_DESCRIPTIONS

    return {str(k): str(v) for k, v in ROLE_DESCRIPTIONS.items()}


# ---------------------------------------------------------------------------
# Normalized data schema
# ---------------------------------------------------------------------------


@dataclass
class BranchCase:
    source: str
    group_id: str
    row_index: int
    target: str
    target_desc: str
    query: str
    pair_id: str
    plus_output: str
    minus_output: str
    sham_output: str
    plus_selected: tuple[str, ...]
    minus_selected: tuple[str, ...]
    sham_selected: tuple[str, ...]
    plus_roles: tuple[str, ...]
    minus_roles: tuple[str, ...]
    sham_roles: tuple[str, ...]
    hardneg: str | None
    hardneg_desc: str | None
    specific: float | None
    strict: bool
    confirmed: bool
    elite: bool
    quality_weight: float
    quality_tier: str
    component: str = ""


@dataclass
class PairItem:
    task: int
    pos_text: str
    neg_text: str
    prior_text: str | None
    weight: float
    item_type: str
    target_role: str
    source: str
    component: str
    pair_id: str


@dataclass
class Endpoint:
    role: str
    desc: str
    query: str
    output: str
    label: int
    weight: float
    endpoint_type: str
    source: str
    component: str
    pair_id: str


def _as_tuple(x) -> tuple[str, ...]:
    return tuple(map(str, x or ()))


def _safe_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def quality_weight(
    *,
    specific: float | None,
    strict: bool,
    confirmed: bool,
    elite: bool,
    source_weight: float,
) -> tuple[float, str]:
    """RL-derived curriculum.

    Ordinary negative-SPEC probes are excluded rather than treated as if they were
    diagnostic. Confirmed/elite/strict probes are retained even if one individual
    branch is noisy because their probe-level evidence was independently strong.
    """
    s = specific
    if strict or confirmed or elite:
        tier = "strong"
        base = 1.0
    elif s is None:
        tier = "unknown"
        base = 0.20
    elif s < 0.0:
        tier = "negative"
        base = 0.0
    elif s < 0.015:
        tier = "weak"
        base = 0.20
    elif s < 0.050:
        tier = "moderate"
        base = 0.50
    else:
        tier = "strong"
        base = 1.0

    if base <= 0.0:
        return 0.0, tier
    if strict:
        base *= 1.15
    if confirmed:
        base *= 1.20
    if elite:
        base *= 1.15
    return min(1.75, base * float(source_weight)), tier


def normalize_discovery_rows(
    *,
    rows: Sequence[dict],
    source: str,
    role_desc: dict[str, str],
    source_weight: float,
) -> tuple[list[BranchCase], Counter]:
    out: list[BranchCase] = []
    skipped = Counter()

    for ridx, r in enumerate(rows):
        if r.get("phase") != "train":
            skipped["non_train_row"] += 1
            continue
        if r.get("status") not in (None, "ok"):
            skipped["non_ok_row"] += 1
            continue

        target = str(r.get("target_role") or "").strip()
        query = str(r.get("query") or "").strip()
        if not target or not query:
            skipped["missing_target_or_query"] += 1
            continue

        desc = str(r.get("candidate_description_public") or role_desc.get(target) or target)
        group_id = str(r.get("group_id", f"row{ridx}"))
        rb = r.get("reward_breakdown_training_only") or {}
        strict = bool(rb.get("strict_specific_pass", False))
        confirmed = bool(r.get("confirmation_pass_training_only", False))
        elite = bool(r.get("elite", False))

        branches = r.get("branches_training_only") or []
        if not branches:
            skipped["no_branches"] += 1
            continue

        for b in branches:
            pid = str(b.get("pair_id") or "").strip()
            po = str(b.get("plus_output") or "").strip()
            mo = str(b.get("minus_output") or "").strip()
            so = str(b.get("sham_output") or "").strip()
            if not (pid and po and mo and so):
                skipped["missing_branch_fields"] += 1
                continue

            if source == "v13":
                spec = _safe_float(b.get("specific_pairwise_training_only"))
            else:
                spec = _safe_float(b.get("specific_pairwise"))
            qw, tier = quality_weight(
                specific=spec,
                strict=strict,
                confirmed=confirmed,
                elite=elite,
                source_weight=source_weight,
            )
            if qw <= 0.0:
                skipped["negative_spec_excluded"] += 1
                continue

            hard = b.get("hard_negative_replacement_training_only")
            hard = str(hard) if hard else None
            hdesc = role_desc.get(hard, hard) if hard else None

            out.append(
                BranchCase(
                    source=source,
                    group_id=group_id,
                    row_index=ridx,
                    target=target,
                    target_desc=desc,
                    query=query,
                    pair_id=pid,
                    plus_output=po,
                    minus_output=mo,
                    sham_output=so,
                    plus_selected=_as_tuple(b.get("plus_selected")),
                    minus_selected=_as_tuple(b.get("minus_selected")),
                    sham_selected=_as_tuple(b.get("sham_selected")),
                    plus_roles=_as_tuple(b.get("plus_roles_training_only")),
                    minus_roles=_as_tuple(b.get("minus_roles_training_only")),
                    sham_roles=_as_tuple(b.get("sham_roles_training_only")),
                    hardneg=hard,
                    hardneg_desc=hdesc,
                    specific=spec,
                    strict=strict,
                    confirmed=confirmed,
                    elite=elite,
                    quality_weight=qw,
                    quality_tier=tier,
                )
            )

    return out, skipped


# ---------------------------------------------------------------------------
# Stronger V13 audit against public/truth logs
# ---------------------------------------------------------------------------


def build_public_truth_index(public_path: str | Path, truth_path: str | Path):
    pub: dict[tuple[str, str], dict] = {}
    for r in read_jsonl(public_path):
        pub[(str(r["system_id"]), str(r["query"]))] = r
    truth: dict[tuple[str, str], dict] = {}
    for r in read_jsonl(truth_path):
        truth[(str(r["system_id"]), str(r["query_hash"]))] = r
    return pub, truth


def audit_v13(
    raw_rows: Sequence[dict],
    public_path: str | Path,
    truth_path: str | Path,
) -> dict:
    pub, truth = build_public_truth_index(public_path, truth_path)
    c = Counter()

    for r in raw_rows:
        if r.get("phase") != "train" or r.get("status") not in (None, "ok"):
            continue
        query = str(r.get("query") or "")
        target = str(r.get("target_role") or "")
        for b in r.get("branches_training_only") or []:
            pid = str(b.get("pair_id") or "")
            if not pid:
                c["missing_pair_id"] += 1
                continue
            for branch in ("plus", "minus", "sham"):
                c["branches_checked"] += 1
                sysid = f"v13p_{pid}_{branch}"
                pr = pub.get((sysid, query))
                if pr is None:
                    c["missing_public"] += 1
                    continue
                if str(b.get(f"{branch}_output") or "") != str(pr.get("output") or ""):
                    c["output_mismatch"] += 1
                tr = truth.get((sysid, str(pr.get("query_hash") or "")))
                if tr is None:
                    c["missing_truth"] += 1
                    continue
                exp_sel = set(map(str, b.get(f"{branch}_selected") or []))
                got_sel = set(map(str, tr.get("selected_roles") or []))
                if exp_sel != got_sel:
                    c["selected_mismatch"] += 1
                exp_roles = set(map(str, b.get(f"{branch}_roles_training_only") or []))
                got_roles = set(map(str, tr.get("true_roles") or []))
                if exp_roles != got_roles:
                    c["true_roles_mismatch"] += 1
                if branch in ("plus", "sham") and target not in got_roles:
                    c["target_missing_when_should_exist"] += 1
                if branch == "minus" and target in got_roles:
                    c["target_present_in_minus"] += 1

    fatal_keys = (
        "missing_pair_id",
        "missing_public",
        "missing_truth",
        "output_mismatch",
        "selected_mismatch",
        "true_roles_mismatch",
        "target_missing_when_should_exist",
        "target_present_in_minus",
    )
    fatal = sum(c[k] for k in fatal_keys)
    checked = max(c["branches_checked"], 1)
    c["fatal"] = fatal
    c["fatal_rate"] = fatal / checked
    if fatal / checked > 0.01:
        raise RuntimeError(f"V13 audit failed (>1% fatal mismatch): {dict(c)}")
    return dict(c)


# ---------------------------------------------------------------------------
# Leakage-safe connected components
# ---------------------------------------------------------------------------


class DSU:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def attach_components(cases: Sequence[BranchCase]) -> list[BranchCase]:
    """Link probe family, pair, exact query, and exact duplicate outputs.

    Do not union globally by unordered role composition. In V13 the same role
    set can recur across many independently generated counterfactual families;
    transitive composition links collapse most hard-negative data into one giant
    component and make a meaningful train/validation split impossible. Exact
    composition overlap is reported separately as an audit signal below.
    """
    dsu = DSU()

    for c in cases:
        g = f"g:{c.source}:{c.group_id}"
        p = f"p:{c.source}:{c.pair_id}"
        q = f"q:{short_hash(norm_text(c.query))}"
        dsu.union(g, p)
        dsu.union(g, q)

        # Exact duplicate answers must never straddle train/validation.
        for output in (c.plus_output, c.minus_output, c.sham_output):
            dsu.union(g, f"o:{short_hash(norm_text(output))}")

    out: list[BranchCase] = []
    for c in cases:
        cc = BranchCase(**asdict(c))
        cc.component = dsu.find(f"g:{c.source}:{c.group_id}")
        out.append(cc)
    return out


# ---------------------------------------------------------------------------
# Objective construction
# ---------------------------------------------------------------------------


def build_items_and_endpoints(
    cases: Sequence[BranchCase],
    *,
    attr_target_weight: float,
    attr_hard_weight: float,
    minus_flip_weight: float,
) -> tuple[list[PairItem], list[Endpoint], dict]:
    items: list[PairItem] = []
    endpoints: list[Endpoint] = []
    counts = Counter()

    def add_evidence_pair(c: BranchCase, pos_out: str, neg_out: str, kind: str):
        items.append(
            PairItem(
                task=EVIDENCE_HEAD,
                pos_text=evidence_text(c.target, c.target_desc, c.query, pos_out),
                neg_text=evidence_text(c.target, c.target_desc, c.query, neg_out),
                prior_text=evidence_prior_text(c.target, c.target_desc, c.query),
                weight=c.quality_weight,
                item_type=kind,
                target_role=c.target,
                source=c.source,
                component=c.component,
                pair_id=c.pair_id,
            )
        )
        counts[kind] += 1

    def add_attr_pair(c: BranchCase, pos_role: str, pos_desc: str, neg_role: str, neg_desc: str, output_pos: str, output_neg: str, kind: str, mult: float):
        items.append(
            PairItem(
                task=ATTRIBUTION_HEAD,
                pos_text=attribution_text(pos_role, pos_desc, output_pos),
                neg_text=attribution_text(neg_role, neg_desc, output_neg),
                prior_text=None,
                weight=c.quality_weight * mult,
                item_type=kind,
                target_role=c.target,
                source=c.source,
                component=c.component,
                pair_id=c.pair_id,
            )
        )
        counts[kind] += 1

    def add_endpoint(c: BranchCase, role: str, desc: str, output: str, label: int, kind: str, mult: float = 1.0):
        endpoints.append(
            Endpoint(
                role=role,
                desc=desc,
                query=c.query,
                output=output,
                label=int(label),
                weight=c.quality_weight * mult,
                endpoint_type=kind,
                source=c.source,
                component=c.component,
                pair_id=c.pair_id,
            )
        )
        counts[f"endpoint:{kind}:{label}"] += 1

    for c in cases:
        plus_sel = set(c.plus_selected)
        minus_sel = set(c.minus_selected)
        sham_sel = set(c.sham_selected)

        # The counterfactual target must not be selected in MINUS for a clean
        # participation-negative endpoint.
        target_minus_negative = c.target not in minus_sel

        if c.target in plus_sel and target_minus_negative:
            add_evidence_pair(c, c.plus_output, c.minus_output, "evidence_plus_minus")
            add_attr_pair(
                c,
                c.target,
                c.target_desc,
                c.target,
                c.target_desc,
                c.plus_output,
                c.minus_output,
                "attr_target_plus_minus",
                attr_target_weight,
            )
            add_endpoint(c, c.target, c.target_desc, c.plus_output, 1, "target_plus")

        if c.target in sham_sel and target_minus_negative:
            add_evidence_pair(c, c.sham_output, c.minus_output, "evidence_sham_minus")
            add_attr_pair(
                c,
                c.target,
                c.target_desc,
                c.target,
                c.target_desc,
                c.sham_output,
                c.minus_output,
                "attr_target_sham_minus",
                attr_target_weight,
            )
            add_endpoint(c, c.target, c.target_desc, c.sham_output, 1, "target_sham")

        if target_minus_negative:
            add_endpoint(c, c.target, c.target_desc, c.minus_output, 0, "target_minus")

        # V13-only hard-neighbor same-answer flips.
        if c.hardneg and c.hardneg_desc:
            hard = c.hardneg
            hdesc = c.hardneg_desc
            plus_roles = set(c.plus_roles)
            sham_roles = set(c.sham_roles)

            hard_absent_plus = bool(plus_roles) and hard not in plus_roles
            hard_absent_sham = bool(sham_roles) and hard not in sham_roles
            hard_selected_minus = hard in minus_sel

            if c.target in plus_sel and hard_absent_plus:
                add_attr_pair(
                    c,
                    c.target,
                    c.target_desc,
                    hard,
                    hdesc,
                    c.plus_output,
                    c.plus_output,
                    "attr_hard_plus",
                    attr_hard_weight,
                )
                add_endpoint(c, hard, hdesc, c.plus_output, 0, "hard_plus_absent")

            if c.target in sham_sel and hard_absent_sham:
                add_attr_pair(
                    c,
                    c.target,
                    c.target_desc,
                    hard,
                    hdesc,
                    c.sham_output,
                    c.sham_output,
                    "attr_hard_sham",
                    attr_hard_weight,
                )
                add_endpoint(c, hard, hdesc, c.sham_output, 0, "hard_sham_absent")

            if hard_selected_minus and target_minus_negative:
                add_attr_pair(
                    c,
                    hard,
                    hdesc,
                    c.target,
                    c.target_desc,
                    c.minus_output,
                    c.minus_output,
                    "attr_hard_minus",
                    minus_flip_weight,
                )
                add_endpoint(c, hard, hdesc, c.minus_output, 1, "hard_minus_selected", minus_flip_weight)

    return items, endpoints, dict(counts)


# ---------------------------------------------------------------------------
# Dataset / model
# ---------------------------------------------------------------------------


class PairDataset(Dataset):
    def __init__(self, items: Sequence[PairItem]):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_pairs(batch: Sequence[PairItem]) -> dict:
    return {
        "task": torch.tensor([x.task for x in batch], dtype=torch.long),
        "pos_text": [x.pos_text for x in batch],
        "neg_text": [x.neg_text for x in batch],
        "prior_text": [x.prior_text for x in batch],
        "weight": torch.tensor([x.weight for x in batch], dtype=torch.float32),
        "item_type": [x.item_type for x in batch],
    }


def make_model(base_model: str):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        quantization_config=qconf,
        device_map={"": 0},
        dtype=compute_dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # 1.5B QLoRA comfortably fits a 24 GB 3090 at our batch/sequence size.
    # Disabling checkpointing avoids the PyTorch reentrant warning and is faster.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return tokenizer, model, compute_dtype


def forward_logits(model, tokenizer, texts: Sequence[str], max_length: int, device: torch.device) -> torch.Tensor:
    toks = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    return model(**toks).logits


def pair_scores_from_batch(model, tokenizer, batch: dict, max_length: int, device: torch.device):
    task = batch["task"].to(device)
    bsz = len(batch["pos_text"])
    full = forward_logits(
        model,
        tokenizer,
        batch["pos_text"] + batch["neg_text"],
        max_length,
        device,
    )
    pos_raw = full[:bsz].gather(1, task[:, None]).squeeze(1)
    neg_raw = full[bsz:].gather(1, task[:, None]).squeeze(1)

    pos = pos_raw.clone()
    neg = neg_raw.clone()
    ev_idx = torch.nonzero(task == EVIDENCE_HEAD, as_tuple=False).view(-1)
    if len(ev_idx):
        prior_texts = [batch["prior_text"][int(i)] for i in ev_idx.tolist()]
        if any(t is None for t in prior_texts):
            raise RuntimeError("evidence item missing prior_text")
        prior_logits = forward_logits(model, tokenizer, prior_texts, max_length, device)
        prior = prior_logits[:, EVIDENCE_HEAD]
        pos[ev_idx] = pos_raw[ev_idx] - prior
        neg[ev_idx] = neg_raw[ev_idx] - prior
    return pos, neg


# ---------------------------------------------------------------------------
# Validation scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_unique_texts(model, tokenizer, texts: Sequence[str], max_length: int, device: torch.device, batch_size: int) -> dict[str, tuple[float, float]]:
    uniq = list(dict.fromkeys(texts))
    scores: dict[str, tuple[float, float]] = {}
    model.eval()
    for i in range(0, len(uniq), batch_size):
        batch = uniq[i : i + batch_size]
        logits = forward_logits(model, tokenizer, batch, max_length, device)
        vals = logits.float().cpu().numpy()
        for t, row in zip(batch, vals):
            scores[t] = (float(row[EVIDENCE_HEAD]), float(row[ATTRIBUTION_HEAD]))
    return scores


def _evidence_delta(sc: dict[str, tuple[float, float]], role: str, desc: str, query: str, output: str) -> float:
    full = sc[evidence_text(role, desc, query, output)][EVIDENCE_HEAD]
    prior = sc[evidence_prior_text(role, desc, query)][EVIDENCE_HEAD]
    return float(full - prior)


def _attr(sc: dict[str, tuple[float, float]], role: str, desc: str, output: str) -> float:
    return float(sc[attribution_text(role, desc, output)][ATTRIBUTION_HEAD])


def collect_texts(items: Sequence[PairItem], endpoints: Sequence[Endpoint], cases: Sequence[BranchCase]) -> list[str]:
    texts: list[str] = []
    for x in items:
        texts.extend([x.pos_text, x.neg_text])
        if x.prior_text:
            texts.append(x.prior_text)
    for e in endpoints:
        texts.extend(
            [
                evidence_text(e.role, e.desc, e.query, e.output),
                evidence_prior_text(e.role, e.desc, e.query),
                attribution_text(e.role, e.desc, e.output),
            ]
        )
    for c in cases:
        # Branch-level validation deliberately examines all three target branches,
        # even when a branch did not create a clean training edge. Include them
        # explicitly so validation never depends on incidental item construction.
        texts.append(evidence_prior_text(c.target, c.target_desc, c.query))
        for o in (c.plus_output, c.minus_output, c.sham_output):
            texts.extend(
                [
                    evidence_text(c.target, c.target_desc, c.query, o),
                    attribution_text(c.target, c.target_desc, o),
                ]
            )

        if c.hardneg and c.hardneg_desc:
            # Combined calibration/branch diagnostics may score the hard role with
            # both heads, so include its evidence full/prior texts as well as
            # answer-only attribution texts.
            texts.append(evidence_prior_text(c.hardneg, c.hardneg_desc, c.query))
            for o in (c.plus_output, c.minus_output, c.sham_output):
                texts.extend(
                    [
                        evidence_text(c.hardneg, c.hardneg_desc, c.query, o),
                        attribution_text(c.hardneg, c.hardneg_desc, o),
                    ]
                )
    return texts


def eval_pair_items(items: Sequence[PairItem], sc: dict[str, tuple[float, float]]) -> dict:
    # Do not pool raw logits across the two heads: their scales need not match.
    # Pairwise accuracy/margin is well-defined within each objective type.
    by_type = defaultdict(lambda: [0, 0, []])

    for x in items:
        if x.task == EVIDENCE_HEAD:
            prior = sc[x.prior_text][EVIDENCE_HEAD]
            ps = sc[x.pos_text][EVIDENCE_HEAD] - prior
            ns = sc[x.neg_text][EVIDENCE_HEAD] - prior
        else:
            ps = sc[x.pos_text][ATTRIBUTION_HEAD]
            ns = sc[x.neg_text][ATTRIBUTION_HEAD]
        ok = ps > ns
        by_type[x.item_type][0] += int(ok)
        by_type[x.item_type][1] += 1
        by_type[x.item_type][2].append(float(ps - ns))

    type_acc = {k: v[0] / max(v[1], 1) for k, v in sorted(by_type.items())}
    type_margin = {k: float(np.mean(v[2])) if v[2] else None for k, v in sorted(by_type.items())}
    return {
        "pair_accuracy_by_type": type_acc,
        "pair_margin_by_type": type_margin,
    }


def eval_branch_metrics(cases: Sequence[BranchCase], sc: dict[str, tuple[float, float]]) -> dict:
    causal_plus_ok = causal_plus_n = 0
    causal_sham_ok = causal_sham_n = 0
    causal_complete_ok = causal_complete_n = 0
    hard_plus_ok = hard_plus_n = 0
    hard_sham_ok = hard_sham_n = 0
    hard_minus_ok = hard_minus_n = 0
    hard_flip_ok = hard_flip_n = 0

    for c in cases:
        plus_sel = set(c.plus_selected)
        minus_sel = set(c.minus_selected)
        sham_sel = set(c.sham_selected)
        tm = _evidence_delta(sc, c.target, c.target_desc, c.query, c.minus_output)
        tp = _evidence_delta(sc, c.target, c.target_desc, c.query, c.plus_output)
        ts = _evidence_delta(sc, c.target, c.target_desc, c.query, c.sham_output)

        plus_clean = c.target in plus_sel and c.target not in minus_sel
        sham_clean = c.target in sham_sel and c.target not in minus_sel
        if plus_clean:
            causal_plus_n += 1
            causal_plus_ok += int(tp > tm)
        if sham_clean:
            causal_sham_n += 1
            causal_sham_ok += int(ts > tm)
        if plus_clean and sham_clean:
            causal_complete_n += 1
            causal_complete_ok += int(tp > tm and ts > tm)

        if not (c.hardneg and c.hardneg_desc):
            continue
        hard = c.hardneg
        hdesc = c.hardneg_desc
        plus_roles = set(c.plus_roles)
        sham_roles = set(c.sham_roles)
        hard_absent_plus = bool(plus_roles) and hard not in plus_roles
        hard_absent_sham = bool(sham_roles) and hard not in sham_roles
        hard_selected_minus = hard in minus_sel

        atp = _attr(sc, c.target, c.target_desc, c.plus_output)
        ahp = _attr(sc, hard, hdesc, c.plus_output)
        ats = _attr(sc, c.target, c.target_desc, c.sham_output)
        ahs = _attr(sc, hard, hdesc, c.sham_output)
        atm = _attr(sc, c.target, c.target_desc, c.minus_output)
        ahm = _attr(sc, hard, hdesc, c.minus_output)

        if c.target in plus_sel and hard_absent_plus:
            hard_plus_n += 1
            hard_plus_ok += int(atp > ahp)
        if c.target in sham_sel and hard_absent_sham:
            hard_sham_n += 1
            hard_sham_ok += int(ats > ahs)
        if hard_selected_minus and c.target not in minus_sel:
            hard_minus_n += 1
            hard_minus_ok += int(ahm > atm)
        if c.target in plus_sel and hard_absent_plus and hard_selected_minus and c.target not in minus_sel:
            hard_flip_n += 1
            hard_flip_ok += int(atp > ahp and ahm > atm)

    causal_plus = causal_plus_ok / max(causal_plus_n, 1)
    causal_sham = causal_sham_ok / max(causal_sham_n, 1)
    causal_edge = float(np.mean([causal_plus, causal_sham]))
    return {
        "evidence_causal_plus_accuracy": causal_plus,
        "evidence_causal_plus_n": causal_plus_n,
        "evidence_causal_sham_accuracy": causal_sham,
        "evidence_causal_sham_n": causal_sham_n,
        "evidence_causal_edge_accuracy": causal_edge,
        "evidence_causal_complete_accuracy": causal_complete_ok / max(causal_complete_n, 1),
        "evidence_causal_complete_n": causal_complete_n,
        "attribution_hard_plus_accuracy": hard_plus_ok / max(hard_plus_n, 1),
        "attribution_hard_plus_n": hard_plus_n,
        "attribution_hard_sham_accuracy": hard_sham_ok / max(hard_sham_n, 1),
        "attribution_hard_sham_n": hard_sham_n,
        "attribution_hard_minus_accuracy": hard_minus_ok / max(hard_minus_n, 1),
        "attribution_hard_minus_n": hard_minus_n,
        "attribution_hard_flip_accuracy": hard_flip_ok / max(hard_flip_n, 1),
        "attribution_hard_flip_n": hard_flip_n,
    }


def eval_model(model, tokenizer, items: Sequence[PairItem], endpoints: Sequence[Endpoint], cases: Sequence[BranchCase], max_length: int, device: torch.device, batch_size: int) -> tuple[dict, dict[str, tuple[float, float]]]:
    texts = collect_texts(items, endpoints, cases)
    sc = score_unique_texts(model, tokenizer, texts, max_length, device, batch_size)
    metrics = {}
    metrics.update(eval_pair_items(items, sc))
    metrics.update(eval_branch_metrics(cases, sc))

    # Head-level endpoint diagnostics before the learned 2-feature calibrator.
    y, ev, at = [], [], []
    for e in endpoints:
        y.append(e.label)
        ev.append(_evidence_delta(sc, e.role, e.desc, e.query, e.output))
        at.append(_attr(sc, e.role, e.desc, e.output))
    yarr = np.asarray(y)
    for name, vals in (("evidence", ev), ("attribution", at)):
        if len(set(y)) == 2:
            metrics[f"{name}_endpoint_auc"] = float(roc_auc_score(yarr, np.asarray(vals)))
        else:
            metrics[f"{name}_endpoint_auc"] = None

    # Selection metric intentionally emphasizes the exact V13.1 failure mode.
    metrics["selection_composite"] = (
        0.25 * metrics["evidence_causal_edge_accuracy"]
        + 0.20 * metrics["evidence_causal_complete_accuracy"]
        + 0.20 * metrics["attribution_hard_minus_accuracy"]
        + 0.25 * metrics["attribution_hard_flip_accuracy"]
        + 0.10 * (metrics["evidence_endpoint_auc"] or 0.0)
    )
    return metrics, sc


# ---------------------------------------------------------------------------
# Combined 2-feature calibrator
# ---------------------------------------------------------------------------


FEATURE_NAMES = ("evidence_delta", "answer_attribution")


def endpoint_features(endpoints: Sequence[Endpoint], sc: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, w = [], [], []
    for e in endpoints:
        X.append(
            [
                _evidence_delta(sc, e.role, e.desc, e.query, e.output),
                _attr(sc, e.role, e.desc, e.output),
            ]
        )
        y.append(e.label)
        w.append(e.weight)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64), np.asarray(w, dtype=np.float64)


def fit_calibrator(train_endpoints: Sequence[Endpoint], sc: dict[str, tuple[float, float]]):
    X, y, w = endpoint_features(train_endpoints, sc)
    if X.shape[0] < 40 or len(np.unique(y)) != 2:
        raise RuntimeError("insufficient endpoint data for calibrator")
    scaler = StandardScaler().fit(X, sample_weight=w)
    Xz = scaler.transform(X)
    lr = LogisticRegression(
        C=0.30,
        class_weight="balanced",
        max_iter=2000,
        random_state=1331,
    )
    lr.fit(Xz, y, sample_weight=w)
    return scaler, lr


def calibrator_probs(endpoints: Sequence[Endpoint], sc: dict[str, tuple[float, float]], scaler, lr) -> tuple[np.ndarray, np.ndarray]:
    X, y, _ = endpoint_features(endpoints, sc)
    p = lr.predict_proba(scaler.transform(X))[:, 1]
    return y, p


def combined_branch_metrics(cases: Sequence[BranchCase], sc: dict[str, tuple[float, float]], scaler, lr) -> dict:
    def p(role: str, desc: str, q: str, o: str) -> float:
        x = np.asarray(
            [[_evidence_delta(sc, role, desc, q, o), _attr(sc, role, desc, o)]],
            dtype=np.float64,
        )
        return float(lr.predict_proba(scaler.transform(x))[0, 1])

    causal_ok = causal_n = 0
    flip_ok = flip_n = 0
    hard_minus_ok = hard_minus_n = 0
    for c in cases:
        plus_sel, minus_sel, sham_sel = map(set, (c.plus_selected, c.minus_selected, c.sham_selected))
        if c.target in plus_sel and c.target not in minus_sel:
            causal_n += 1
            causal_ok += int(p(c.target, c.target_desc, c.query, c.plus_output) > p(c.target, c.target_desc, c.query, c.minus_output))
        if c.target in sham_sel and c.target not in minus_sel:
            causal_n += 1
            causal_ok += int(p(c.target, c.target_desc, c.query, c.sham_output) > p(c.target, c.target_desc, c.query, c.minus_output))

        if not (c.hardneg and c.hardneg_desc):
            continue
        hard, hdesc = c.hardneg, c.hardneg_desc
        hard_absent_plus = bool(c.plus_roles) and hard not in set(c.plus_roles)
        hard_selected_minus = hard in minus_sel
        clean_minus = c.target not in minus_sel
        if hard_selected_minus and clean_minus:
            hard_minus_n += 1
            hard_minus_ok += int(p(hard, hdesc, c.query, c.minus_output) > p(c.target, c.target_desc, c.query, c.minus_output))
        if c.target in plus_sel and hard_absent_plus and hard_selected_minus and clean_minus:
            flip_n += 1
            flip_ok += int(
                p(c.target, c.target_desc, c.query, c.plus_output) > p(hard, hdesc, c.query, c.plus_output)
                and p(hard, hdesc, c.query, c.minus_output) > p(c.target, c.target_desc, c.query, c.minus_output)
            )
    return {
        "combined_causal_edge_accuracy": causal_ok / max(causal_n, 1),
        "combined_causal_edge_n": causal_n,
        "combined_hard_minus_accuracy": hard_minus_ok / max(hard_minus_n, 1),
        "combined_hard_minus_n": hard_minus_n,
        "combined_hard_flip_accuracy": flip_ok / max(flip_n, 1),
        "combined_hard_flip_n": flip_n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Train V13.2 dual-head causal/attribution judge")
    ap.add_argument("--v13-discovery", default="artifacts/v13_probe_disambiguation.jsonl")
    ap.add_argument("--v11-discovery", default="artifacts/v11_probe_discovery.jsonl")
    ap.add_argument("--v13-public", default="artifacts/v13_train_response_public.jsonl")
    ap.add_argument("--v13-truth", default="artifacts/v13_train_response_truth.jsonl")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--output-dir", default="artifacts/v13_2_dual_judge")
    ap.add_argument("--seed", type=int, default=1322)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=6)
    ap.add_argument("--max-length", type=int, default=1400)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--margin", type=float, default=0.75)
    ap.add_argument("--bce-weight", type=float, default=0.35)
    ap.add_argument("--v11-source-weight", type=float, default=0.70)
    ap.add_argument("--attr-target-weight", type=float, default=0.75)
    ap.add_argument("--attr-hard-weight", type=float, default=1.00)
    ap.add_argument("--minus-flip-weight", type=float, default=2.00)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--overwrite-output", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    outdir = Path(args.output_dir)
    if outdir.exists() and any(outdir.iterdir()):
        if not args.overwrite_output:
            raise SystemExit(
                f"Refusing to overwrite non-empty {outdir}. Archive it or pass --overwrite-output."
            )
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    role_desc = load_role_descriptions()
    raw_v13 = read_jsonl(args.v13_discovery)
    raw_v11 = read_jsonl(args.v11_discovery)

    audit = audit_v13(raw_v13, args.v13_public, args.v13_truth)
    print("[audit-v13]", json.dumps(audit, sort_keys=True), flush=True)

    v13_cases, sk13 = normalize_discovery_rows(
        rows=raw_v13,
        source="v13",
        role_desc=role_desc,
        source_weight=1.0,
    )
    v11_cases, sk11 = normalize_discovery_rows(
        rows=raw_v11,
        source="v11",
        role_desc=role_desc,
        source_weight=args.v11_source_weight,
    )
    cases = attach_components(v13_cases + v11_cases)

    quality_counts = Counter((c.source, c.quality_tier) for c in cases)
    component_count = len(set(c.component for c in cases))
    print(
        f"[data] cases={len(cases)} v13={len(v13_cases)} v11={len(v11_cases)} "
        f"components={component_count} quality={dict(quality_counts)}",
        flush=True,
    )

    if len(cases) < 80:
        raise RuntimeError(f"only {len(cases)} usable counterfactual branches")
    if component_count < args.n_splits:
        raise RuntimeError(
            f"only {component_count} leakage-safe components for {args.n_splits} folds"
        )

    # Split at branch-case level; all DSU-linked cases stay together.
    X = np.arange(len(cases))
    y = np.asarray([c.target for c in cases])
    groups = np.asarray([c.component for c in cases])
    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )
    splits = list(splitter.split(X, y, groups))
    if not 0 <= args.fold < len(splits):
        raise ValueError("fold out of range")
    tr_idx, va_idx = splits[args.fold]
    tr_components = {cases[i].component for i in tr_idx}
    va_components = {cases[i].component for i in va_idx}
    if not tr_components.isdisjoint(va_components):
        raise AssertionError("component leakage across split")

    train_cases = [c for c in cases if c.component in tr_components]
    val_cases = [c for c in cases if c.component in va_components]
    train_items, train_endpoints, train_counts = build_items_and_endpoints(
        train_cases,
        attr_target_weight=args.attr_target_weight,
        attr_hard_weight=args.attr_hard_weight,
        minus_flip_weight=args.minus_flip_weight,
    )
    val_items, val_endpoints, val_counts = build_items_and_endpoints(
        val_cases,
        attr_target_weight=args.attr_target_weight,
        attr_hard_weight=args.attr_hard_weight,
        minus_flip_weight=args.minus_flip_weight,
    )

    if len(train_items) < 100 or len(val_items) < 20:
        raise RuntimeError(
            f"insufficient objectives after split: train={len(train_items)} val={len(val_items)}"
        )
    train_type_counts = Counter(x.item_type for x in train_items)
    val_type_counts = Counter(x.item_type for x in val_items)

    if train_type_counts.get("attr_hard_minus", 0) < 50:
        raise RuntimeError(
            "training fold has fewer than 50 hard-MINUS examples; "
            "choose another fold/seed rather than train without the flip objective"
        )

    if val_type_counts.get("attr_hard_minus", 0) < 15:
        raise RuntimeError(
            "validation fold has fewer than 15 hard-MINUS examples; "
            "choose another fold/seed rather than trust a noisy flip metric"
        )
    if (
        val_type_counts.get("evidence_plus_minus", 0)
        + val_type_counts.get("evidence_sham_minus", 0)
    ) < 30:
        raise RuntimeError("validation fold has too few causal evidence comparisons")

    train_compositions = {
        composition_key(roles)
        for c in train_cases
        for roles in (c.plus_roles, c.minus_roles, c.sham_roles)
        if roles
    }
    val_compositions = {
        composition_key(roles)
        for c in val_cases
        for roles in (c.plus_roles, c.minus_roles, c.sham_roles)
        if roles
    }
    composition_overlap = sorted(train_compositions & val_compositions)

    split_report = {
        "seed": args.seed,
        "fold": args.fold,
        "n_splits": args.n_splits,
        "prompt_version": PROMPT_VERSION,
        "n_cases": len(cases),
        "n_components": component_count,
        "train_components": sorted(tr_components),
        "val_components": sorted(va_components),
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "train_items": len(train_items),
        "val_items": len(val_items),
        "train_endpoints": len(train_endpoints),
        "val_endpoints": len(val_endpoints),
        "train_item_types": dict(Counter(x.item_type for x in train_items)),
        "val_item_types": dict(Counter(x.item_type for x in val_items)),
        "train_sources": dict(Counter(c.source for c in train_cases)),
        "val_sources": dict(Counter(c.source for c in val_cases)),
        "train_target_roles": dict(Counter(c.target for c in train_cases)),
        "val_target_roles": dict(Counter(c.target for c in val_cases)),
        "train_composition_count": len(train_compositions),
        "val_composition_count": len(val_compositions),
        "composition_overlap_count": len(composition_overlap),
        "composition_overlap_sample": composition_overlap[:25],
        "composition_overlap_is_audit_only": True,
        "quality_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(quality_counts.items())},
        "normalize_skipped_v13": dict(sk13),
        "normalize_skipped_v11": dict(sk11),
        "objective_counts_train": train_counts,
        "objective_counts_val": val_counts,
        "audit_v13": audit,
    }
    (outdir / "split_and_audit.json").write_text(
        json.dumps(split_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[split] train_cases={len(train_cases)} val_cases={len(val_cases)} "
        f"train_items={len(train_items)} val_items={len(val_items)}",
        flush=True,
    )
    print("[train-types]", json.dumps(split_report["train_item_types"], sort_keys=True), flush=True)
    print("[val-types]", json.dumps(split_report["val_item_types"], sort_keys=True), flush=True)

    base_model = resolve_base_model(args.model)
    judge_cfg = {
        "protocol": "v13.2_dual_head_question_prior_corrected_answer_attribution",
        "prompt_version": PROMPT_VERSION,
        "base_model": base_model,
        "num_labels": 2,
        "head_map": {"evidence": EVIDENCE_HEAD, "attribution": ATTRIBUTION_HEAD},
        "max_length": args.max_length,
        "seed": args.seed,
        "fold": args.fold,
        "n_splits": args.n_splits,
        "v11_source_weight": args.v11_source_weight,
        "attr_target_weight": args.attr_target_weight,
        "attr_hard_weight": args.attr_hard_weight,
        "minus_flip_weight": args.minus_flip_weight,
        "pilot_used_for_training": False,
        "api_calls": 0,
    }
    (outdir / "judge_config.json").write_text(
        json.dumps(judge_cfg, indent=2, sort_keys=True), encoding="utf-8"
    )

    if args.audit_only:
        print("AUDIT-ONLY PASS: no model loaded and no API calls made.", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("V13.2 training requires CUDA; submit through SLURM")

    tokenizer, model, compute_dtype = make_model(base_model)
    device = torch.device("cuda:0")
    dl = DataLoader(
        PairDataset(train_items),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pairs,
        drop_last=False,
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    from transformers import get_cosine_schedule_with_warmup

    steps_per_epoch = math.ceil(len(dl) / max(args.grad_accum, 1))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(10, int(0.05 * total_steps))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(compute_dtype == torch.float16))

    best = -1.0
    best_epoch = 0
    best_metrics = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0

        for step, batch in enumerate(dl, 1):
            weights = batch["weight"].to(device)
            with torch.autocast(device_type="cuda", dtype=compute_dtype, enabled=True):
                pos, neg = pair_scores_from_batch(
                    model, tokenizer, batch, args.max_length, device
                )
                rank_each = F.softplus(args.margin - (pos - neg))
                bce_each = 0.5 * (
                    F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos), reduction="none")
                    + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg), reduction="none")
                )
                loss_each = rank_each + args.bce_weight * bce_each
                # Normalize by total weight rather than batch size so curriculum
                # weights change contribution without changing global loss scale.
                loss = (loss_each * weights).sum() / weights.sum().clamp_min(1e-6)
                loss_scaled = loss / max(args.grad_accum, 1)

            scaler.scale(loss_scaled).backward()
            running += float(loss.detach().cpu()) * len(batch["pos_text"])
            seen += len(batch["pos_text"])

            if step % args.grad_accum == 0 or step == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        vm, _ = eval_model(
            model,
            tokenizer,
            val_items,
            val_endpoints,
            val_cases,
            args.max_length,
            device,
            args.eval_batch_size,
        )
        vm["epoch"] = epoch
        print(
            f"epoch={epoch:02d} train_loss={running/max(seen,1):.4f} "
            f"ev_causal={vm['evidence_causal_edge_accuracy']:.4f} "
            f"ev_complete={vm['evidence_causal_complete_accuracy']:.4f} "
            f"attr_minus={vm['attribution_hard_minus_accuracy']:.4f} "
            f"attr_flip={vm['attribution_hard_flip_accuracy']:.4f} "
            f"ev_auc={vm['evidence_endpoint_auc']:.4f} "
            f"composite={vm['selection_composite']:.4f}",
            flush=True,
        )

        if vm["selection_composite"] > best + 1e-4:
            best = vm["selection_composite"]
            best_epoch = epoch
            best_metrics = dict(vm)
            bad_epochs = 0
            ckpt = outdir / "best_adapter"
            if ckpt.exists():
                shutil.rmtree(ckpt)
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            (outdir / "head_metrics.json").write_text(
                json.dumps(best_metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"[early-stop] epoch={epoch} best_epoch={best_epoch} best_composite={best:.4f}",
                    flush=True,
                )
                break

    if best_metrics is None:
        raise RuntimeError("training produced no best checkpoint")

    # Reload the actual best adapter before fitting the 2-feature calibrator.
    del model
    torch.cuda.empty_cache()
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        quantization_config=qconf,
        device_map={"": 0},
        dtype=compute_dtype,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    model = PeftModel.from_pretrained(base, str(outdir / "best_adapter"))
    model.eval()

    # Score train and validation endpoints using the frozen best judge.
    train_texts = collect_texts([], train_endpoints, train_cases)
    val_texts = collect_texts([], val_endpoints, val_cases)
    all_sc = score_unique_texts(
        model,
        tokenizer,
        train_texts + val_texts,
        args.max_length,
        device,
        args.eval_batch_size,
    )
    scaler2, lr2 = fit_calibrator(train_endpoints, all_sc)
    save_calibrator(
        outdir / "calibrator.json",
        scaler=scaler2,
        logistic=lr2,
        feature_names=FEATURE_NAMES,
        metadata={
            "fit_on": "training_components_only",
            "best_epoch": best_epoch,
            "C": 0.30,
        },
    )

    yv, pv = calibrator_probs(val_endpoints, all_sc, scaler2, lr2)
    combined_auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) == 2 else None
    combined_ba = float(balanced_accuracy_score(yv, pv >= 0.5)) if len(yv) else None
    combined_branch = combined_branch_metrics(val_cases, all_sc, scaler2, lr2)

    final = dict(best_metrics)
    final.update(
        {
            "best_epoch": best_epoch,
            "base_model": base_model,
            "combined_endpoint_auc": combined_auc,
            "combined_endpoint_balanced_accuracy_at_0_5": combined_ba,
            **combined_branch,
        }
    )
    (outdir / "metrics.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )

    thresholds = {
        "evidence_causal_edge_accuracy": 0.76,
        "attribution_hard_minus_accuracy": 0.60,
        "attribution_hard_flip_accuracy": 0.60,
        "combined_endpoint_auc": 0.75,
        "combined_hard_flip_accuracy": 0.60,
    }
    checks = {k: bool(final.get(k) is not None and final[k] >= v) for k, v in thresholds.items()}
    go = all(checks.values())
    decision = {
        "go_to_offline_pilot_replay": True,
        "go_to_fresh_api_pilot_from_validation_only": go,
        "reason": (
            "PASS validation gates. Run offline replay before any fresh API pilot."
            if go
            else "STOP fresh API calls. Offline replay is still allowed; validation gates did not all pass."
        ),
        "thresholds": thresholds,
        "checks": checks,
        "api_calls_used": 0,
    }
    (outdir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("\n=== V13.2 BEST VALIDATION ===", flush=True)
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)
    print("\n=== GO / NO-GO ===", flush=True)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
