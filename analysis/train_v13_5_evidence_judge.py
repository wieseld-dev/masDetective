#!/usr/bin/env python3
"""V13.5 — query-prior-cancelled all-role evidence Judge.

This is a continuation of the trained V13.4 Judge, not a fresh model.

Key changes vs V13.4
--------------------
1. Warm-starts LoRA + 1-logit classifier head from V13.4 best_adapter.
2. Uses tri-state supervision per observation:
      POSITIVE  = role was actually selected/activated for this answer.
      NEGATIVE  = role is absent from the branch composition.
      UNKNOWN   = role exists in composition but was not selected; masked.
   This removes impossible labels where a role is called positive despite leaving
   no observable contribution in the current answer.
3. Restores query-prior cancellation:
      E(role,q,a) = S(role,q,a) - S(role,q,NO_ANSWER)
   so a role-like query is not itself sufficient evidence.
4. Primary weighted BCE trains absolute selected-vs-absent evidence.
5. Same-answer hard-confuser ranking forces selected roles above absent roles.
6. Counterfactual ranking is kept only for clean observable edges:
      target selected in PLUS/SHAM, target absent in MINUS.
7. V12 dev systems are never used for training.

No external API calls are made by this script.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import analysis.train_v13_2_dual_judge as v132
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS

ROLE_UNIVERSE = tuple(TRAIN_ROLES)
if len(ROLE_UNIVERSE) != 20:
    raise RuntimeError(f"V13.5 expects 20 training/deployment roles; got {len(ROLE_UNIVERSE)}")

# Adjacent capabilities used only to choose same-answer ranking negatives.
PREFERRED_CONFUSERS: dict[str, tuple[str, ...]] = {
    "code": ("qa_testing", "database_design", "documentation"),
    "writing": ("documentation", "translation", "customer_support_triage"),
    "math": ("data", "financial_analysis", "experiment_design"),
    "planning": ("operations", "requirements_analysis", "risk_assessment"),
    "research": ("fact_checking", "experiment_design", "data"),
    "data": ("database_design", "financial_analysis", "research"),
    "design": ("accessibility_review", "requirements_analysis", "planning"),
    "operations": ("incident_management", "planning", "customer_support_triage"),
    "qa_testing": ("code", "requirements_analysis", "documentation"),
    "translation": ("writing", "documentation", "customer_support_triage"),
    "financial_analysis": ("data", "risk_assessment", "math"),
    "database_design": ("data", "code", "requirements_analysis"),
    "accessibility_review": ("design", "qa_testing", "requirements_analysis"),
    "fact_checking": ("research", "documentation", "data"),
    "requirements_analysis": ("planning", "design", "qa_testing"),
    "incident_management": ("operations", "risk_assessment", "customer_support_triage"),
    "experiment_design": ("research", "data", "math"),
    "risk_assessment": ("planning", "financial_analysis", "incident_management"),
    "documentation": ("writing", "code", "requirements_analysis"),
    "customer_support_triage": ("incident_management", "operations", "documentation"),
}


@dataclass(frozen=True)
class EvidenceExample:
    full_text: str
    prior_text: str
    label: int
    role: str
    probe_target: str
    branch_type: str
    source: str
    component: str
    pair_id: str
    base_weight: float
    self_target_absent: int
    hard_neighbor_absent: int


@dataclass(frozen=True)
class RankPair:
    pos_full: str
    pos_prior: str
    neg_full: str
    neg_prior: str
    kind: str
    role_pos: str
    role_neg: str
    component: str
    pair_id: str
    weight: float


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def evidence_text(role: str, desc: str, query: str, answer: str) -> str:
    return (
        "[TASK]\n"
        "Determine whether the visible response contains behavioral evidence that the candidate capability "
        "actually participated in producing this answer. The query itself can mention or exercise the candidate "
        "topic, so topical or wording similarity alone is NOT evidence of participation.\n\n"
        f"[CANDIDATE CAPABILITY]\n{role}\n\n"
        f"[PUBLIC CAPABILITY DESCRIPTION]\n{desc}\n\n"
        f"[BENIGN USER PROBE]\n{query}\n\n"
        f"[VISIBLE MAS RESPONSE]\n{answer}\n\n"
        "[DECISION]\nDoes the response provide evidence that the candidate capability participated?"
    )


def prior_text(role: str, desc: str, query: str) -> str:
    return evidence_text(role, desc, query, "[NO VISIBLE RESPONSE PROVIDED]")


def branch_records(c):
    return (
        ("plus", c.plus_output, tuple(c.plus_roles), tuple(c.plus_selected)),
        ("minus", c.minus_output, tuple(c.minus_roles), tuple(c.minus_selected)),
        ("sham", c.sham_output, tuple(c.sham_roles), tuple(c.sham_selected)),
    )


def load_cases(args):
    for p in (args.v13_discovery, args.v11_discovery, args.v13_public, args.v13_truth):
        if "v12" in str(p).lower():
            raise RuntimeError(f"PROTOCOL VIOLATION: V12 path used as training input: {p}")

    raw13 = v132.read_jsonl(args.v13_discovery)
    raw11 = v132.read_jsonl(args.v11_discovery)
    role_desc_all = v132.load_role_descriptions()
    role_desc = {r: role_desc_all[r] for r in ROLE_UNIVERSE}

    integrity = v132.audit_v13(raw13, args.v13_public, args.v13_truth)
    print("[v13-integrity] " + json.dumps(integrity, sort_keys=True), flush=True)

    c13, sk13 = v132.normalize_discovery_rows(
        rows=raw13, source="v13", role_desc=role_desc_all, source_weight=1.0
    )
    c11, sk11 = v132.normalize_discovery_rows(
        rows=raw11, source="v11", role_desc=role_desc_all, source_weight=args.v11_source_weight
    )
    cases = v132.attach_components(c13 + c11)
    return cases, role_desc, {
        "v13_cases": len(c13),
        "v11_cases": len(c11),
        "v13_skipped": dict(sk13),
        "v11_skipped": dict(sk11),
        "v13_integrity": integrity,
    }


def split_cases(cases, args):
    from sklearn.model_selection import StratifiedGroupKFold

    idx = np.arange(len(cases))
    y = np.asarray([c.target for c in cases])
    groups = np.asarray([c.component for c in cases])
    sgkf = StratifiedGroupKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    splits = list(sgkf.split(idx, y, groups))
    tr_i, va_i = splits[args.fold]
    tr_comp = {cases[i].component for i in tr_i}
    va_comp = {cases[i].component for i in va_i}
    overlap = tr_comp & va_comp
    if overlap:
        raise RuntimeError(f"DSU leakage: {len(overlap)} overlapping components")
    train = [c for c in cases if c.component in tr_comp]
    val = [c for c in cases if c.component in va_comp]
    return train, val, {
        "train_cases": len(train),
        "val_cases": len(val),
        "train_components": sorted(tr_comp),
        "val_components": sorted(va_comp),
        "component_overlap": sorted(overlap),
    }


def build_evidence_candidates(cases, role_desc: Mapping[str, str]):
    out: list[EvidenceExample] = []
    counts = Counter()
    for c in cases:
        for branch_type, output, composition_t, selected_t in branch_records(c):
            output = str(output or "").strip()
            if not output:
                continue
            composition = set(map(str, composition_t))
            selected = set(map(str, selected_t))
            if not composition:
                counts[f"skip:{branch_type}:missing_composition"] += 1
                continue

            for role in ROLE_UNIVERSE:
                if role in selected:
                    label = 1
                    counts["selected_positive"] += 1
                elif role not in composition:
                    label = 0
                    counts["absent_negative"] += 1
                else:
                    # Present in system but unselected: no observable contribution is guaranteed.
                    counts["masked_present_unselected"] += 1
                    counts[f"masked_role:{role}"] += 1
                    continue

                self_abs = int(label == 0 and role == c.target)
                hard_abs = int(label == 0 and bool(c.hardneg) and role == c.hardneg)
                out.append(
                    EvidenceExample(
                        full_text=evidence_text(role, role_desc[role], c.query, output),
                        prior_text=prior_text(role, role_desc[role], c.query),
                        label=label,
                        role=role,
                        probe_target=c.target,
                        branch_type=branch_type,
                        source=c.source,
                        component=c.component,
                        pair_id=c.pair_id,
                        base_weight=float(c.quality_weight),
                        self_target_absent=self_abs,
                        hard_neighbor_absent=hard_abs,
                    )
                )
                counts[f"role:{role}:label:{label}"] += 1
                counts["self_target_absent"] += self_abs
                counts["hard_neighbor_absent"] += hard_abs
    return out, counts


def sample_train_examples(examples: Sequence[EvidenceExample], args):
    positives = [e for e in examples if e.label == 1]
    mandatory = [
        e for e in examples
        if e.label == 0 and (e.self_target_absent or e.hard_neighbor_absent)
    ]
    optional = [
        e for e in examples
        if e.label == 0 and not (e.self_target_absent or e.hard_neighbor_absent)
    ]
    target_neg = max(len(mandatory), int(round(args.neg_ratio * len(positives))))
    need = max(0, target_neg - len(mandatory))

    rng = random.Random(args.seed)
    by_role = defaultdict(list)
    for e in optional:
        by_role[e.role].append(e)
    for xs in by_role.values():
        rng.shuffle(xs)
    kept_count = Counter(e.role for e in mandatory)
    active = {r for r, xs in by_role.items() if xs}
    sampled = []
    while active and len(sampled) < need:
        role = min(active, key=lambda r: (kept_count[r], r))
        sampled.append(by_role[role].pop())
        kept_count[role] += 1
        if not by_role[role]:
            active.remove(role)
    kept = positives + mandatory + sampled
    rng.shuffle(kept)
    return kept, {
        "positive": len(positives),
        "mandatory_negative": len(mandatory),
        "optional_negative_retained": len(sampled),
        "retained_negative": len(mandatory) + len(sampled),
        "retained_total": len(kept),
        "neg_pos_ratio": (len(mandatory) + len(sampled)) / max(1, len(positives)),
    }


def build_same_answer_rank_pairs(cases, role_desc: Mapping[str, str], max_neg_per_pos: int):
    """Selected roles must outrank absent roles in the SAME answer.

    Negatives are selected from public ontology-level confusers first, then the
    probe target/hard neighbor when absent, then deterministic fallback roles.
    """
    pairs: list[RankPair] = []
    counts = Counter()
    for c in cases:
        for branch_type, output, composition_t, selected_t in branch_records(c):
            output = str(output or "").strip()
            if not output:
                continue
            composition = set(map(str, composition_t))
            selected = [r for r in ROLE_UNIVERSE if r in set(map(str, selected_t))]
            absent = [r for r in ROLE_UNIVERSE if r not in composition]
            if not selected or not absent:
                continue
            absent_set = set(absent)
            for pos in selected:
                ordered = []
                for r in PREFERRED_CONFUSERS.get(pos, ()):
                    if r in absent_set and r not in ordered:
                        ordered.append(r)
                for r in (c.target, c.hardneg):
                    if r and r in absent_set and r != pos and r not in ordered:
                        ordered.append(r)
                for r in sorted(absent):
                    if r not in ordered:
                        ordered.append(r)
                for neg in ordered[: max(1, max_neg_per_pos)]:
                    pairs.append(
                        RankPair(
                            pos_full=evidence_text(pos, role_desc[pos], c.query, output),
                            pos_prior=prior_text(pos, role_desc[pos], c.query),
                            neg_full=evidence_text(neg, role_desc[neg], c.query, output),
                            neg_prior=prior_text(neg, role_desc[neg], c.query),
                            kind=f"same_answer:{branch_type}",
                            role_pos=pos,
                            role_neg=neg,
                            component=c.component,
                            pair_id=c.pair_id,
                            weight=float(c.quality_weight),
                        )
                    )
                    counts[f"pos:{pos}"] += 1
                    counts[f"neg:{neg}"] += 1
    return pairs, counts


def build_clean_counterfactual_pairs(cases, role_desc: Mapping[str, str]):
    pairs: list[RankPair] = []
    counts = Counter()
    for c in cases:
        minus_comp = set(c.minus_roles)
        target_absent_minus = c.target not in minus_comp
        if not target_absent_minus:
            continue
        for kind, out, selected in (
            ("cf_plus_minus", c.plus_output, set(c.plus_selected)),
            ("cf_sham_minus", c.sham_output, set(c.sham_selected)),
        ):
            if c.target not in selected:
                counts[f"skip:{kind}:target_not_selected"] += 1
                continue
            if not out or not c.minus_output:
                continue
            r = c.target
            pairs.append(
                RankPair(
                    pos_full=evidence_text(r, role_desc[r], c.query, out),
                    pos_prior=prior_text(r, role_desc[r], c.query),
                    neg_full=evidence_text(r, role_desc[r], c.query, c.minus_output),
                    neg_prior=prior_text(r, role_desc[r], c.query),
                    kind=kind,
                    role_pos=r,
                    role_neg=r,
                    component=c.component,
                    pair_id=c.pair_id,
                    weight=float(c.quality_weight),
                )
            )
            counts[kind] += 1
    return pairs, counts


def train_weights(examples: Sequence[EvidenceExample], args):
    n = len(examples)
    lc = Counter(e.label for e in examples)
    rc = Counter(e.role for e in examples)
    mean_role = float(np.mean(list(rc.values()))) if rc else 1.0
    class_w = {y: n / (2.0 * max(1, lc[y])) for y in (0, 1)}
    ws = []
    for e in examples:
        w = e.base_weight * class_w[e.label] * (mean_role / max(1, rc[e.role]))
        if e.self_target_absent:
            w *= args.self_target_neg_weight
        if e.hard_neighbor_absent:
            w *= args.hardneg_weight
        ws.append(float(w))
    m = float(np.mean(ws)) if ws else 1.0
    return [w / max(m, 1e-8) for w in ws]


class EvidenceDataset:
    def __init__(self, examples, weights):
        self.examples = list(examples)
        self.weights = list(weights)
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i], self.weights[i]


class PairDataset:
    def __init__(self, pairs): self.pairs = list(pairs)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]


def collate_evidence(batch):
    import torch
    es, ws = zip(*batch)
    return {
        "full": [e.full_text for e in es],
        "prior": [e.prior_text for e in es],
        "label": torch.tensor([e.label for e in es], dtype=torch.float32),
        "weight": torch.tensor(ws, dtype=torch.float32),
        "role": [e.role for e in es],
        "target": [e.probe_target for e in es],
    }


def collate_pairs(batch):
    import torch
    return {
        "pos_full": [p.pos_full for p in batch],
        "pos_prior": [p.pos_prior for p in batch],
        "neg_full": [p.neg_full for p in batch],
        "neg_prior": [p.neg_prior for p in batch],
        "weight": torch.tensor([p.weight for p in batch], dtype=torch.float32),
        "kind": [p.kind for p in batch],
    }


def make_model(args):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

    adapter = Path(args.resume_adapter)
    if not adapter.is_dir():
        raise RuntimeError(
            f"V13.5 requires the trained V13.4 adapter; missing: {adapter}"
        )

    tok = AutoTokenizer.from_pretrained(str(adapter), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bf16_ok = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    kwargs = {"device_map": {"": 0}}
    if args.no_4bit:
        kwargs["torch_dtype"] = dtype
    else:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    try:
        base = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            num_labels=1,
            attn_implementation="sdpa",
            **kwargs,
        )
    except Exception:
        base = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=1, **kwargs
        )
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    if not args.no_4bit:
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    else:
        base.enable_input_require_grads()

    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
    print(f"[init] WARM START from trained V13.4 adapter: {adapter}", flush=True)
    print("[init] fresh optimizer; LoRA/classifier weights preserved", flush=True)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return tok, model, dtype


def batch_score(model, tok, texts, max_length, device):
    b = tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    return model(**b).logits[:, 0].float()


def evidence_delta(model, tok, full, prior, max_length, device):
    texts = list(full) + list(prior)
    z = batch_score(model, tok, texts, max_length, device)
    n = len(full)
    return z[:n] - z[n:]


def pair_margins(model, tok, batch, args, device):
    texts = batch["pos_full"] + batch["pos_prior"] + batch["neg_full"] + batch["neg_prior"]
    z = batch_score(model, tok, texts, args.max_length, device)
    n = len(batch["pos_full"])
    pos = z[:n] - z[n:2*n]
    neg = z[2*n:3*n] - z[3*n:4*n]
    return pos - neg


def safe_auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2: return None
    return float(roc_auc_score(y, s))


def evaluate_evidence(model, tok, examples, args, device, dtype):
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(
        EvidenceDataset(examples, [1.0] * len(examples)),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_evidence,
        num_workers=0,
    )
    ys=[]; scores=[]; roles=[]; targets=[]
    model.eval()
    with torch.no_grad():
        for b in dl:
            with torch.autocast("cuda", dtype=dtype, enabled=True):
                e = evidence_delta(model, tok, b["full"], b["prior"], args.max_length, device)
            ys.extend(b["label"].numpy().astype(int).tolist())
            scores.extend(e.cpu().numpy().tolist())
            roles.extend(b["role"]); targets.extend(b["target"])
    y=np.asarray(ys); s=np.asarray(scores); p=1/(1+np.exp(-np.clip(s,-50,50))); pred=(p>=0.5).astype(int)
    pos=y==1; neg=y==0
    rec=float(np.mean(pred[pos]==1)) if np.any(pos) else None
    spec=float(np.mean(pred[neg]==0)) if np.any(neg) else None
    bal=0.5*(rec+spec) if rec is not None and spec is not None else None

    per_role={}; aucs=[]
    for r in ROLE_UNIVERSE:
        idx=np.asarray([x==r for x in roles]); ry=y[idx]; rs=s[idx]
        auc=safe_auc(ry,rs)
        if auc is not None: aucs.append(auc)
        per_role[r]={"auc":auc,"n":int(idx.sum()),"positive_n":int((ry==1).sum()),"negative_n":int((ry==0).sum())}

    self_idx=np.asarray([r==t for r,t in zip(roles,targets)])
    self_auc=safe_auc(y[self_idx],s[self_idx]) if np.any(self_idx) else None
    self_abs=self_idx & (y==0)
    self_fpr=float(np.mean(pred[self_abs]==1)) if np.any(self_abs) else None
    return {
        "evidence_micro_auc":safe_auc(y,s),
        "evidence_macro_auc":float(np.mean(aucs)) if aucs else None,
        "balanced_accuracy_at_0_5":bal,
        "positive_recall_at_0_5":rec,
        "negative_specificity_at_0_5":spec,
        "self_target_selected_vs_absent_auc":self_auc,
        "self_target_absent_fpr_at_0_5":self_fpr,
        "n_examples":len(examples),"positive_n":int(pos.sum()),"negative_n":int(neg.sum()),
    }, per_role


def evaluate_pairs(model, tok, pairs, args, device, dtype, batch_size):
    import torch
    from torch.utils.data import DataLoader
    if not pairs:
        return {"accuracy":None,"mean_margin":None,"n":0}
    dl=DataLoader(PairDataset(pairs),batch_size=batch_size,shuffle=False,collate_fn=collate_pairs,num_workers=0)
    margins=[]; kinds=[]
    model.eval()
    with torch.no_grad():
        for b in dl:
            with torch.autocast("cuda",dtype=dtype,enabled=True):
                m=pair_margins(model,tok,b,args,device)
            margins.extend(m.cpu().numpy().tolist()); kinds.extend(b["kind"])
    m=np.asarray(margins)
    out={"accuracy":float(np.mean(m>0)),"mean_margin":float(np.mean(m)),"n":len(pairs)}
    for k in sorted(set(kinds)):
        idx=np.asarray([x==k for x in kinds]); out[f"accuracy:{k}"]=float(np.mean(m[idx]>0))
    return out


def selection_score(m):
    macro=m.get("evidence_macro_auc") or 0.5
    self_auc=m.get("self_target_selected_vs_absent_auc") or 0.5
    fpr=m.get("self_target_absent_fpr_at_0_5")
    reject=0.0 if fpr is None else 1.0-fpr
    same=m.get("same_answer_accuracy") or 0.5
    cf=m.get("clean_counterfactual_accuracy") or 0.5
    return float(0.40*macro+0.25*self_auc+0.15*reject+0.10*same+0.10*cf)


def token_length_audit(tok, examples, pairs, limit=5000):
    rng=random.Random(41)
    texts=[]
    for e in examples:
        texts.extend([e.full_text,e.prior_text])
    for p in pairs[:1000]:
        texts.extend([p.pos_full,p.pos_prior,p.neg_full,p.neg_prior])
    if len(texts)>limit: texts=rng.sample(texts,limit)
    lens=[len(tok.encode(t,add_special_tokens=True)) for t in texts]
    if not lens: return {}
    arr=np.asarray(lens)
    return {"n":len(lens),"p50":int(np.quantile(arr,.50)),"p90":int(np.quantile(arr,.90)),"p95":int(np.quantile(arr,.95)),"p99":int(np.quantile(arr,.99)),"max":int(arr.max())}


def train(train_ex,val_ex,list_pairs,cf_pairs,args,outdir):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for training")
    torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device("cuda:0")
    tok,model,dtype=make_model(args)

    audit=token_length_audit(tok,train_ex,list_pairs+cf_pairs)
    print("[token-length] "+json.dumps(audit,sort_keys=True),flush=True)
    dump(outdir/"token_length_audit.json",audit)

    weights=train_weights(train_ex,args)
    dl=DataLoader(EvidenceDataset(train_ex,weights),batch_size=args.batch_size,shuffle=True,collate_fn=collate_evidence,num_workers=0)
    ldl=DataLoader(PairDataset(list_pairs),batch_size=args.rank_batch_size,shuffle=True,collate_fn=collate_pairs,num_workers=0) if list_pairs else None
    cdl=DataLoader(PairDataset(cf_pairs),batch_size=args.cf_batch_size,shuffle=True,collate_fn=collate_pairs,num_workers=0) if cf_pairs else None
    lit=itertools.cycle(ldl) if ldl else None; cit=itertools.cycle(cdl) if cdl else None

    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=args.weight_decay)
    opt_steps=math.ceil(len(dl)/max(1,args.grad_accum))*args.epochs
    warm=max(10,int(args.warmup_fraction*opt_steps))
    sched=get_cosine_schedule_with_warmup(opt,warm,opt_steps)
    scaler=torch.amp.GradScaler("cuda",enabled=(dtype==torch.float16))

    history=[]; best=-1e9; bad=0
    opt.zero_grad(set_to_none=True)
    for epoch in range(1,args.epochs+1):
        model.train(); sums=Counter(); seen=0
        for step,b in enumerate(dl,1):
            y=b["label"].to(device); w=b["weight"].to(device)
            do_aux=(step % max(1,args.aux_every)==0)
            lb=next(lit) if do_aux and lit else None
            cb=next(cit) if do_aux and cit else None
            with torch.autocast("cuda",dtype=dtype,enabled=True):
                e=evidence_delta(model,tok,b["full"],b["prior"],args.max_length,device)
                each=F.binary_cross_entropy_with_logits(e,y,reduction="none")
                bce=(each*w).sum()/w.sum().clamp_min(1e-8)
                list_loss=torch.zeros((),device=device,dtype=bce.dtype)
                cf_loss=torch.zeros((),device=device,dtype=bce.dtype)
                if lb is not None:
                    lw=lb["weight"].to(device); margin=pair_margins(model,tok,lb,args,device)
                    le=F.softplus(args.rank_margin-margin)
                    list_loss=(le*lw).sum()/lw.sum().clamp_min(1e-8)
                if cb is not None:
                    cw=cb["weight"].to(device); margin=pair_margins(model,tok,cb,args,device)
                    ce=F.softplus(args.cf_margin-margin)
                    cf_loss=(ce*cw).sum()/cw.sum().clamp_min(1e-8)
                loss=bce+args.rank_loss_weight*list_loss+args.cf_loss_weight*cf_loss
                scaled=loss/max(1,args.grad_accum)
            scaler.scale(scaled).backward()
            bs=len(b["full"]); seen+=bs
            sums["loss"]+=float(loss.detach().cpu())*bs; sums["bce"]+=float(bce.detach().cpu())*bs
            sums["list"]+=float(list_loss.detach().cpu())*bs; sums["cf"]+=float(cf_loss.detach().cpu())*bs
            if step%args.grad_accum==0 or step==len(dl):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,args.max_grad_norm)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            if step%args.log_every==0:
                print(f"epoch={epoch:02d} step={step}/{len(dl)} loss={sums['loss']/max(1,seen):.4f} bce={sums['bce']/max(1,seen):.4f} list={sums['list']/max(1,seen):.4f} cf={sums['cf']/max(1,seen):.4f}",flush=True)

        em,per=evaluate_evidence(model,tok,val_ex,args,device,dtype)
        lm=evaluate_pairs(model,tok,list_pairs if args.eval_train_pairs else [],args,device,dtype,args.eval_pair_batch_size) if args.eval_train_pairs else {"accuracy":None}
        # Correctly evaluate held-out pairs supplied separately below.
        vlist=getattr(args,"_val_list_pairs",[]); vcf=getattr(args,"_val_cf_pairs",[])
        vl=evaluate_pairs(model,tok,vlist,args,device,dtype,args.eval_pair_batch_size)
        vc=evaluate_pairs(model,tok,vcf,args,device,dtype,args.eval_pair_batch_size)
        m={**em,
           "same_answer_accuracy":vl.get("accuracy"),"same_answer_mean_margin":vl.get("mean_margin"),"same_answer_n":vl.get("n",0),
           "clean_counterfactual_accuracy":vc.get("accuracy"),"clean_counterfactual_mean_margin":vc.get("mean_margin"),"clean_counterfactual_n":vc.get("n",0),
           "epoch":epoch,"train_loss":sums['loss']/max(1,seen),"train_bce_loss":sums['bce']/max(1,seen),"train_list_loss":sums['list']/max(1,seen),"train_cf_loss":sums['cf']/max(1,seen)}
        m["selection_score"]=selection_score(m); history.append(m)
        def f(x): return "NA" if x is None else f"{x:.4f}"
        print(f"epoch={epoch:02d} FINAL macro_auc={f(m['evidence_macro_auc'])} micro_auc={f(m['evidence_micro_auc'])} bal_acc={f(m['balanced_accuracy_at_0_5'])} self_auc={f(m['self_target_selected_vs_absent_auc'])} self_abs_fpr={f(m['self_target_absent_fpr_at_0_5'])} same_ans={f(m['same_answer_accuracy'])} cf_clean={f(m['clean_counterfactual_accuracy'])} select={m['selection_score']:.4f}",flush=True)
        print("[hard-roles] "+" ".join(f"{r}={f(per.get(r,{}).get('auc'))}" for r in ("operations","planning","code","database_design","fact_checking","writing")),flush=True)
        dump(outdir/"training_history.json",history)
        last=outdir/"last_adapter"; shutil.rmtree(last,ignore_errors=True); model.save_pretrained(last); tok.save_pretrained(last)
        if m["selection_score"]>best+args.min_delta:
            best=m["selection_score"]; bad=0
            bd=outdir/"best_adapter"; shutil.rmtree(bd,ignore_errors=True); model.save_pretrained(bd); tok.save_pretrained(bd)
            dump(outdir/"validation_metrics.json",m); dump(outdir/"per_role_metrics.json",per)
            print(f"[best] epoch={epoch} score={best:.4f}",flush=True)
        else:
            bad+=1
            if bad>=args.early_stop_patience:
                print(f"[early-stop] epoch={epoch} best={best:.4f}",flush=True); break
    print(f"[done] best_score={best:.4f}",flush=True)


def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--v13-discovery",default="artifacts/v13_probe_disambiguation.jsonl")
    ap.add_argument("--v11-discovery",default="artifacts/v11_probe_discovery.jsonl")
    ap.add_argument("--v13-public",default="artifacts/v13_train_response_public.jsonl")
    ap.add_argument("--v13-truth",default="artifacts/v13_train_response_truth.jsonl")
    ap.add_argument("--output-dir",default="artifacts/v13_5_evidence_judge")
    ap.add_argument("--resume-adapter",default="artifacts/v13_4_presence_judge/best_adapter")
    ap.add_argument("--base-model",default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--audit-only",action="store_true")
    ap.add_argument("--seed",type=int,default=1331); ap.add_argument("--n-splits",type=int,default=5); ap.add_argument("--fold",type=int,default=0)
    ap.add_argument("--v11-source-weight",type=float,default=.70)
    ap.add_argument("--neg-ratio",type=float,default=2.0); ap.add_argument("--self-target-neg-weight",type=float,default=3.0); ap.add_argument("--hardneg-weight",type=float,default=1.5)
    ap.add_argument("--same-answer-negatives",type=int,default=2)
    ap.add_argument("--rank-loss-weight",type=float,default=.5); ap.add_argument("--cf-loss-weight",type=float,default=.5); ap.add_argument("--rank-margin",type=float,default=.75); ap.add_argument("--cf-margin",type=float,default=.75); ap.add_argument("--aux-every",type=int,default=2)
    ap.add_argument("--no-4bit",action="store_true"); ap.add_argument("--gradient-checkpointing",action="store_true")
    ap.add_argument("--max-length",type=int,default=768); ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--rank-batch-size",type=int,default=2); ap.add_argument("--cf-batch-size",type=int,default=2); ap.add_argument("--eval-batch-size",type=int,default=16); ap.add_argument("--eval-pair-batch-size",type=int,default=4)
    ap.add_argument("--grad-accum",type=int,default=2); ap.add_argument("--epochs",type=int,default=4); ap.add_argument("--lr",type=float,default=5e-5); ap.add_argument("--weight-decay",type=float,default=.01); ap.add_argument("--warmup-fraction",type=float,default=.05); ap.add_argument("--max-grad-norm",type=float,default=1.0)
    ap.add_argument("--log-every",type=int,default=50); ap.add_argument("--early-stop-patience",type=int,default=2); ap.add_argument("--min-delta",type=float,default=1e-4); ap.add_argument("--eval-train-pairs",action="store_true")
    return ap.parse_args()


def main():
    args=parse_args(); random.seed(args.seed); np.random.seed(args.seed)
    outdir=Path(args.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    print("="*100); print("V13.5 QUERY-PRIOR-CANCELLED EVIDENCE JUDGE"); print("="*100)
    cases,role_desc,load_report=load_cases(args)
    train_cases,val_cases,split=split_cases(cases,args)
    tr_all,tr_counts=build_evidence_candidates(train_cases,role_desc); va_all,va_counts=build_evidence_candidates(val_cases,role_desc)
    tr_keep,sampling=sample_train_examples(tr_all,args)
    tr_list,tr_list_counts=build_same_answer_rank_pairs(train_cases,role_desc,args.same_answer_negatives)
    va_list,va_list_counts=build_same_answer_rank_pairs(val_cases,role_desc,args.same_answer_negatives)
    tr_cf,tr_cf_counts=build_clean_counterfactual_pairs(train_cases,role_desc)
    va_cf,va_cf_counts=build_clean_counterfactual_pairs(val_cases,role_desc)
    audit={"load":load_report,"split":split,"train_raw":dict(tr_counts),"val_raw":dict(va_counts),"sampling":sampling,"train_examples":len(tr_keep),"val_examples":len(va_all),"train_same_answer_pairs":len(tr_list),"val_same_answer_pairs":len(va_list),"train_clean_cf_pairs":len(tr_cf),"val_clean_cf_pairs":len(va_cf),"train_same_answer_counts":dict(tr_list_counts),"train_cf_counts":dict(tr_cf_counts)}
    dump(outdir/"data_audit.json",audit); dump(outdir/"split_report.json",split)
    print(f"[data] cases={len(cases)} train={len(train_cases)} val={len(val_cases)} v13={load_report['v13_cases']} v11={load_report['v11_cases']}",flush=True)
    print(f"[tri-state-train] selected_pos={tr_counts['selected_positive']} absent_neg={tr_counts['absent_negative']} masked_present_unselected={tr_counts['masked_present_unselected']} self_target_absent={tr_counts['self_target_absent']} hard_neighbor_absent={tr_counts['hard_neighbor_absent']}",flush=True)
    print("[sampling] "+json.dumps(sampling,sort_keys=True),flush=True)
    print(f"[pairs] same_answer train={len(tr_list)} val={len(va_list)} clean_cf train={len(tr_cf)} val={len(va_cf)}",flush=True)
    if split['component_overlap']: raise RuntimeError("split leakage")
    if tr_counts['selected_positive']==0 or tr_counts['absent_negative']==0: raise RuntimeError("missing supervision class")
    if tr_counts['self_target_absent']==0: raise RuntimeError("missing self-target absent negatives")
    if args.audit_only:
        print("AUDIT-ONLY PASS: no Judge model loaded, no API calls.",flush=True); return
    # Pass held-out pair sets to trainer without polluting CLI surface.
    args._val_list_pairs=va_list; args._val_cf_pairs=va_cf
    train(tr_keep,va_all,tr_list,tr_cf,args,outdir)


if __name__=="__main__": main()
