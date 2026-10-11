#!/usr/bin/env python3
"""
V13.1-STRONG: shared causal evidence judge.

Goal:
    Learn ONE role-conditioned evidence function:
        s(role_description, diagnostic_question, black_box_answer)
    that says whether the public answer contains evidence that the candidate
    role actually participated.

Important:
- Uses privileged selected_roles/true_roles ONLY to construct TRAIN labels.
- Runtime scorer input is only public candidate description + public question
  + public answer.
- V13/V11 RL checkpoints are never modified.
- The pilot system is never used for training.
- Strict split blocks BOTH shared probe groups and shared counterfactual pair IDs.
"""

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ----------------------------- IO -------------------------------------------

def read_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"{path}:{ln}: {e}")
    return out


def resolve_base_model(arg):
    if arg != "auto":
        return arg

    candidates = [
        Path("artifacts/v13_policy/adapter_config.json"),
        Path("artifacts/v11_policy/adapter_config.json"),
    ]
    for p in candidates:
        if p.exists():
            cfg = json.loads(p.read_text())
            base = cfg.get("base_model_name_or_path")
            if base:
                print(f"[model] discovered base model from {p}: {base}")
                return base

    fallback = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"[model] adapter config not found; fallback={fallback}")
    return fallback


# ----------------------------- audit ----------------------------------------

def build_public_truth_index(public_path, truth_path):
    pub = {}
    for r in read_jsonl(public_path):
        pub[(r["system_id"], r["query"])] = r

    truth = {}
    for r in read_jsonl(truth_path):
        truth[(r["system_id"], r["query_hash"])] = r
    return pub, truth


def audit_v13(rows, public_path, truth_path):
    pub, truth = build_public_truth_index(public_path, truth_path)
    c = Counter()
    details = []

    for r in rows:
        if r.get("phase") != "train":
            continue
        query = r.get("query")
        target = r.get("target_role")
        for b in r.get("branches_training_only") or []:
            pid = b.get("pair_id")
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

                expected_output = b.get(f"{branch}_output")
                if expected_output != pr.get("output"):
                    c["output_mismatch"] += 1

                tr = truth.get((sysid, pr["query_hash"]))
                if tr is None:
                    c["missing_truth"] += 1
                    continue

                exp_selected = set(b.get(f"{branch}_selected") or [])
                got_selected = set(tr.get("selected_roles") or [])
                if exp_selected != got_selected:
                    c["selected_mismatch"] += 1

                exp_roles = set(
                    b.get(f"{branch}_roles_training_only") or []
                )
                got_roles = set(tr.get("true_roles") or [])
                if exp_roles != got_roles:
                    c["true_roles_mismatch"] += 1

                # Counterfactual invariants.
                if branch in ("plus", "sham") and target not in got_roles:
                    c["target_missing_when_should_exist"] += 1
                if branch == "minus" and target in got_roles:
                    c["target_present_in_minus"] += 1

    fatal = (
        c["output_mismatch"]
        + c["selected_mismatch"]
        + c["true_roles_mismatch"]
        + c["target_missing_when_should_exist"]
        + c["target_present_in_minus"]
    )
    checked = max(c["branches_checked"], 1)
    if fatal / checked > 0.01:
        raise RuntimeError(
            "V13 audit failed: >1% semantic mismatches. "
            f"counts={dict(c)}"
        )
    return dict(c)


# ----------------------------- leakage-safe grouping ------------------------

class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def attach_components(rows):
    """
    Connect a probe group to every pair_id it touched.
    If a pair_id is reused by another probe group, they become one component.
    This prevents:
      - same probe family in train and val
      - same counterfactual system trio in train and val
    """
    dsu = DSU()
    train_rows = []

    for i, r in enumerate(rows):
        if r.get("phase") != "train":
            continue
        gid = str(r.get("group_id", f"row{i}"))
        gnode = f"g:{gid}"
        dsu.find(gnode)
        pair_ids = []
        for b in r.get("branches_training_only") or []:
            pid = b.get("pair_id")
            if pid:
                pnode = f"p:{pid}"
                dsu.union(gnode, pnode)
                pair_ids.append(pid)

        rr = dict(r)
        rr["_group_node"] = gnode
        rr["_pair_ids"] = pair_ids
        train_rows.append(rr)

    for r in train_rows:
        r["_component"] = dsu.find(r["_group_node"])
    return train_rows


# ----------------------------- edge construction ----------------------------

def role_human(name):
    return str(name).replace("_", " ")


def qweight(spec, elite=False, confirmed=False):
    # Keep weak/noisy pairs, but do not let them dominate the ranking objective.
    # Range approx [0.75, 1.25], with small bonuses for confirmed/elite probes.
    if spec is None:
        w = 1.0
    else:
        x = max(-0.20, min(0.20, float(spec)))
        w = 0.75 + 0.50 / (1.0 + math.exp(-x / 0.05))
    if elite:
        w *= 1.10
    if confirmed:
        w *= 1.10
    return min(w, 1.45)


def make_text(role, desc, query, output):
    return (
        "CANDIDATE ROLE:\n"
        f"{role_human(role)}\n\n"
        "PUBLIC ROLE DESCRIPTION:\n"
        f"{desc}\n\n"
        "DIAGNOSTIC QUESTION ASKED TO THE BLACK-BOX MAS:\n"
        f"{query}\n\n"
        "BLACK-BOX FINAL ANSWER:\n"
        f"{output}\n\n"
        "DECISION TARGET:\n"
        "Does this answer contain behavioral evidence that the candidate role "
        "actually participated in producing the answer? Judge participation "
        "evidence, not whether a generic language model could answer the topic."
    )


@dataclass
class Edge:
    pos_text: str
    neg_text: str
    weight: float
    edge_type: str
    target_role: str
    component: str
    probe_id: str
    pair_id: str


@dataclass
class BranchEval:
    target: str
    hardneg: str
    query: str
    target_desc: str
    hardneg_desc: str
    plus_output: str
    minus_output: str
    sham_output: str
    plus_selected: bool
    sham_selected: bool
    hard_minus_selected: bool
    hard_absent_plus: bool
    hard_absent_sham: bool
    component: str
    pair_id: str


def build_edges(train_rows):
    role_desc = {}
    for r in train_rows:
        t = r.get("target_role")
        d = r.get("candidate_description_public")
        if t and d:
            role_desc[t] = d

    # Every target role should have a public description somewhere.
    all_roles = set(role_desc)

    edges = []
    branch_evals = []
    skip = Counter()

    for ridx, r in enumerate(train_rows):
        target = r.get("target_role")
        query = r.get("query")
        if not target or not query:
            skip["missing_target_or_query"] += 1
            continue

        tdesc = role_desc.get(target, role_human(target))
        elite = bool(r.get("elite"))
        confirmed = bool(r.get("confirmation_pass_training_only"))
        probe_id = f'{r.get("group_id","na")}:{r.get("round","na")}:{ridx}'

        for b in r.get("branches_training_only") or []:
            pid = str(b.get("pair_id", ""))
            plus_out = b.get("plus_output")
            minus_out = b.get("minus_output")
            sham_out = b.get("sham_output")
            if not (pid and plus_out and minus_out and sham_out):
                skip["missing_branch_fields"] += 1
                continue

            plus_roles = set(b.get("plus_roles_training_only") or [])
            minus_roles = set(b.get("minus_roles_training_only") or [])
            sham_roles = set(b.get("sham_roles_training_only") or [])
            plus_sel = set(b.get("plus_selected") or [])
            minus_sel = set(b.get("minus_selected") or [])
            sham_sel = set(b.get("sham_selected") or [])

            if target not in plus_roles or target in minus_roles or target not in sham_roles:
                skip["bad_counterfactual_truth"] += 1
                continue

            hard = b.get("hard_negative_replacement_training_only")
            hdesc = role_desc.get(hard, role_human(hard)) if hard else ""
            spec = b.get("specific_pairwise_training_only")
            w = qweight(spec, elite=elite, confirmed=confirmed)
            comp = r["_component"]

            pt = make_text(target, tdesc, query, plus_out)
            mt = make_text(target, tdesc, query, minus_out)
            st = make_text(target, tdesc, query, sham_out)

            # Cleanest causal edges: same target, same question, matched systems.
            if target in plus_sel:
                edges.append(Edge(
                    pt, mt, w, "causal_plus_minus",
                    target, comp, probe_id, pid
                ))
            if target in sham_sel:
                edges.append(Edge(
                    st, mt, w, "causal_sham_minus",
                    target, comp, probe_id, pid
                ))

            # Hard-neighbor role swap edges.
            if hard:
                hp = make_text(hard, hdesc, query, plus_out)
                hm = make_text(hard, hdesc, query, minus_out)
                hs = make_text(hard, hdesc, query, sham_out)

                if target in plus_sel and hard not in plus_roles:
                    edges.append(Edge(
                        pt, hp, w, "hard_same_output_plus",
                        target, comp, probe_id, pid
                    ))
                if target in sham_sel and hard not in sham_roles:
                    edges.append(Edge(
                        st, hs, w, "hard_same_output_sham",
                        target, comp, probe_id, pid
                    ))
                if hard in minus_sel:
                    edges.append(Edge(
                        hm, mt, w, "hard_same_output_minus",
                        target, comp, probe_id, pid
                    ))

                branch_evals.append(BranchEval(
                    target=target,
                    hardneg=hard,
                    query=query,
                    target_desc=tdesc,
                    hardneg_desc=hdesc,
                    plus_output=plus_out,
                    minus_output=minus_out,
                    sham_output=sham_out,
                    plus_selected=(target in plus_sel),
                    sham_selected=(target in sham_sel),
                    hard_minus_selected=(hard in minus_sel),
                    hard_absent_plus=(hard not in plus_roles),
                    hard_absent_sham=(hard not in sham_roles),
                    component=comp,
                    pair_id=pid,
                ))

    return edges, branch_evals, role_desc, dict(skip)


# ----------------------------- data -----------------------------------------

class EdgeDataset(Dataset):
    def __init__(self, edges):
        self.edges = edges

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, i):
        e = self.edges[i]
        return {
            "pos_text": e.pos_text,
            "neg_text": e.neg_text,
            "weight": e.weight,
            "edge_type": e.edge_type,
            "target_role": e.target_role,
            "pair_id": e.pair_id,
        }


def collate(batch):
    return {
        "pos_text": [x["pos_text"] for x in batch],
        "neg_text": [x["neg_text"] for x in batch],
        "weight": torch.tensor(
            [x["weight"] for x in batch], dtype=torch.float32
        ),
        "edge_type": [x["edge_type"] for x in batch],
        "target_role": [x["target_role"] for x in batch],
        "pair_id": [x["pair_id"] for x in batch],
    }


# ----------------------------- model ----------------------------------------

def make_model(base_model, max_length):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, use_fast=True
    )
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
        num_labels=1,
        quantization_config=qconf,
        device_map={"": 0},
        torch_dtype=compute_dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return tokenizer, model, compute_dtype


def forward_texts(model, tokenizer, texts, max_length, device):
    toks = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    out = model(**toks)
    return out.logits.view(-1)


# ----------------------------- eval -----------------------------------------

@torch.no_grad()
def score_unique_texts(
    model, tokenizer, texts, max_length, device, batch_size
):
    uniq = list(dict.fromkeys(texts))
    scores = {}
    model.eval()
    for i in range(0, len(uniq), batch_size):
        batch = uniq[i:i + batch_size]
        logits = forward_texts(
            model, tokenizer, batch, max_length, device
        )
        vals = logits.float().cpu().tolist()
        for t, s in zip(batch, vals):
            scores[t] = float(s)
    return scores


def eval_all(
    model, tokenizer, edges, branches,
    max_length, device, eval_batch_size
):
    all_texts = []
    for e in edges:
        all_texts.extend([e.pos_text, e.neg_text])

    # Branch metrics need some texts that may not have appeared as an edge.
    for b in branches:
        all_texts.extend([
            make_text(b.target, b.target_desc, b.query, b.plus_output),
            make_text(b.target, b.target_desc, b.query, b.minus_output),
            make_text(b.target, b.target_desc, b.query, b.sham_output),
            make_text(b.hardneg, b.hardneg_desc, b.query, b.plus_output),
            make_text(b.hardneg, b.hardneg_desc, b.query, b.minus_output),
            make_text(b.hardneg, b.hardneg_desc, b.query, b.sham_output),
        ])

    sc = score_unique_texts(
        model, tokenizer, all_texts,
        max_length, device, eval_batch_size
    )

    y, logits, weights = [], [], []
    by_type = defaultdict(lambda: [0, 0])
    margins = []

    for e in edges:
        p = sc[e.pos_text]
        n = sc[e.neg_text]
        ok = p > n
        by_type[e.edge_type][0] += int(ok)
        by_type[e.edge_type][1] += 1
        margins.append(p - n)

        y.extend([1, 0])
        logits.extend([p, n])
        weights.extend([e.weight, e.weight])

    probs = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits), -30, 30)))
    yarr = np.asarray(y)
    auc = float(roc_auc_score(yarr, probs)) if len(set(y)) == 2 else None
    ba05 = float(balanced_accuracy_score(yarr, probs >= 0.5))

    type_acc = {
        k: v[0] / max(v[1], 1)
        for k, v in sorted(by_type.items())
    }

    causal_complete_ok = causal_complete_n = 0
    hard_flip_ok = hard_flip_n = 0
    hard_plus_ok = hard_plus_n = 0
    hard_minus_ok = hard_minus_n = 0

    for b in branches:
        tp = sc[make_text(
            b.target, b.target_desc, b.query, b.plus_output
        )]
        tm = sc[make_text(
            b.target, b.target_desc, b.query, b.minus_output
        )]
        ts = sc[make_text(
            b.target, b.target_desc, b.query, b.sham_output
        )]
        hp = sc[make_text(
            b.hardneg, b.hardneg_desc, b.query, b.plus_output
        )]
        hm = sc[make_text(
            b.hardneg, b.hardneg_desc, b.query, b.minus_output
        )]

        if b.plus_selected and b.sham_selected:
            causal_complete_n += 1
            causal_complete_ok += int(tp > tm and ts > tm)

        if b.plus_selected and b.hard_absent_plus:
            hard_plus_n += 1
            hard_plus_ok += int(tp > hp)

        if b.hard_minus_selected:
            hard_minus_n += 1
            hard_minus_ok += int(hm > tm)

        if (
            b.plus_selected
            and b.hard_absent_plus
            and b.hard_minus_selected
        ):
            hard_flip_n += 1
            hard_flip_ok += int(tp > hp and hm > tm)

    causal_types = [
        type_acc.get("causal_plus_minus"),
        type_acc.get("causal_sham_minus"),
    ]
    causal_types = [x for x in causal_types if x is not None]
    hard_types = [
        type_acc.get("hard_same_output_plus"),
        type_acc.get("hard_same_output_sham"),
        type_acc.get("hard_same_output_minus"),
    ]
    hard_types = [x for x in hard_types if x is not None]

    causal_edge = float(np.mean(causal_types)) if causal_types else 0.0
    hard_edge = float(np.mean(hard_types)) if hard_types else 0.0

    # Composite intentionally prioritizes counterfactual evidence and the
    # difficult role-swap behavior, not raw BCE calibration.
    composite = (
        0.35 * causal_edge
        + 0.25 * hard_edge
        + 0.20 * (causal_complete_ok / max(causal_complete_n, 1))
        + 0.20 * (hard_flip_ok / max(hard_flip_n, 1))
    )

    return {
        "endpoint_auc": auc,
        "endpoint_balanced_accuracy_at_0_5": ba05,
        "mean_pair_margin": float(np.mean(margins)) if margins else None,
        "edge_accuracy_by_type": type_acc,
        "causal_edge_accuracy": causal_edge,
        "hard_edge_accuracy": hard_edge,
        "causal_complete_accuracy": (
            causal_complete_ok / max(causal_complete_n, 1)
        ),
        "causal_complete_n": causal_complete_n,
        "hard_plus_accuracy": hard_plus_ok / max(hard_plus_n, 1),
        "hard_plus_n": hard_plus_n,
        "hard_minus_accuracy": hard_minus_ok / max(hard_minus_n, 1),
        "hard_minus_n": hard_minus_n,
        "hard_flip_accuracy": hard_flip_ok / max(hard_flip_n, 1),
        "hard_flip_n": hard_flip_n,
        "composite": composite,
        "n_edges": len(edges),
    }


# ----------------------------- train ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--discovery",
        default="artifacts/v13_probe_disambiguation.jsonl",
    )
    ap.add_argument(
        "--public",
        default="artifacts/v13_train_response_public.jsonl",
    )
    ap.add_argument(
        "--truth",
        default="artifacts/v13_train_response_truth.jsonl",
    )
    ap.add_argument("--model", default="auto")
    ap.add_argument(
        "--output-dir",
        default="artifacts/v13_1_strong_judge",
    )
    ap.add_argument("--seed", type=int, default=1317)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1400)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.75)
    ap.add_argument("--bce-weight", type=float, default=0.50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "V13.1-STRONG requires a GPU. Submit through SLURM."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(args.discovery)
    audit = audit_v13(rows, args.public, args.truth)
    print("[audit]", json.dumps(audit, sort_keys=True))
    (outdir / "data_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True)
    )

    train_rows = attach_components(rows)
    edges, branches, role_desc, skipped = build_edges(train_rows)

    if len(edges) < 100:
        raise RuntimeError(
            f"Only {len(edges)} training edges were constructed; "
            "refusing to train."
        )

    components = sorted(set(r["_component"] for r in train_rows))
    if len(components) < args.n_splits:
        raise RuntimeError(
            f"Only {len(components)} leakage-safe components; "
            f"cannot do {args.n_splits}-fold split."
        )

    # Every row gets target-role stratum + leakage-safe component.
    y = np.asarray([r["target_role"] for r in train_rows])
    groups = np.asarray([r["_component"] for r in train_rows])
    X = np.arange(len(train_rows))

    sgkf = StratifiedGroupKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )
    splits = list(sgkf.split(X, y, groups))
    if args.fold >= len(splits):
        raise ValueError("fold out of range")
    tr_row_idx, va_row_idx = splits[args.fold]

    train_components = {
        train_rows[i]["_component"] for i in tr_row_idx
    }
    val_components = {
        train_rows[i]["_component"] for i in va_row_idx
    }
    assert train_components.isdisjoint(val_components)

    tr_edges = [e for e in edges if e.component in train_components]
    va_edges = [e for e in edges if e.component in val_components]
    tr_branches = [
        b for b in branches if b.component in train_components
    ]
    va_branches = [
        b for b in branches if b.component in val_components
    ]

    print(
        f"[data] rows={len(train_rows)} roles={len(role_desc)} "
        f"components={len(components)} edges={len(edges)} "
        f"train_edges={len(tr_edges)} val_edges={len(va_edges)} "
        f"skipped={skipped}"
    )

    if len(va_edges) < 20:
        raise RuntimeError(
            "Validation split too small after leakage-safe grouping."
        )

    split_report = {
        "seed": args.seed,
        "fold": args.fold,
        "n_splits": args.n_splits,
        "train_components": sorted(train_components),
        "val_components": sorted(val_components),
        "train_target_roles": dict(Counter(
            r["target_role"] for i, r in enumerate(train_rows)
            if i in set(tr_row_idx.tolist())
        )),
        "val_target_roles": dict(Counter(
            r["target_role"] for i, r in enumerate(train_rows)
            if i in set(va_row_idx.tolist())
        )),
        "n_train_edges": len(tr_edges),
        "n_val_edges": len(va_edges),
        "edge_types_train": dict(Counter(e.edge_type for e in tr_edges)),
        "edge_types_val": dict(Counter(e.edge_type for e in va_edges)),
    }
    (outdir / "split.json").write_text(
        json.dumps(split_report, indent=2, sort_keys=True)
    )

    base_model = resolve_base_model(args.model)
    tokenizer, model, compute_dtype = make_model(
        base_model, args.max_length
    )
    device = torch.device("cuda:0")

    ds = EdgeDataset(tr_edges)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=0.01
    )

    steps_per_epoch = math.ceil(
        len(dl) / max(args.grad_accum, 1)
    )
    total_steps = steps_per_epoch * args.epochs
    warmup = max(10, int(0.05 * total_steps))
    sched = get_cosine_schedule_with_warmup(
        opt, warmup, total_steps
    )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=(compute_dtype == torch.float16)
    )

    best = -1.0
    best_epoch = 0
    bad_epochs = 0
    best_metrics = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        seen = 0

        for step, batch in enumerate(dl, 1):
            texts = batch["pos_text"] + batch["neg_text"]
            weights = batch["weight"].to(device)
            bsz = len(batch["pos_text"])

            with torch.autocast(
                device_type="cuda",
                dtype=compute_dtype,
                enabled=True,
            ):
                logits = forward_texts(
                    model, tokenizer, texts,
                    args.max_length, device
                )
                pos = logits[:bsz]
                neg = logits[bsz:]

                rank_each = F.softplus(
                    args.margin - (pos - neg)
                )

                labels = torch.cat([
                    torch.ones_like(pos),
                    torch.zeros_like(neg),
                ])
                bce_each = F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                )
                bce_each = 0.5 * (
                    bce_each[:bsz] + bce_each[bsz:]
                )

                loss_each = (
                    rank_each
                    + args.bce_weight * bce_each
                )
                loss = (loss_each * weights).mean()
                loss_scaled = loss / args.grad_accum

            scaler.scale(loss_scaled).backward()
            running += float(loss.detach().cpu()) * bsz
            seen += bsz

            if (
                step % args.grad_accum == 0
                or step == len(dl)
            ):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    trainable, 1.0
                )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        vm = eval_all(
            model, tokenizer,
            va_edges, va_branches,
            args.max_length, device,
            args.eval_batch_size,
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={running/max(seen,1):.4f} "
            f"val_auc={vm['endpoint_auc']:.4f} "
            f"causal={vm['causal_edge_accuracy']:.4f} "
            f"hard={vm['hard_edge_accuracy']:.4f} "
            f"causal_complete={vm['causal_complete_accuracy']:.4f} "
            f"hard_flip={vm['hard_flip_accuracy']:.4f} "
            f"composite={vm['composite']:.4f}"
        )

        if vm["composite"] > best + 1e-4:
            best = vm["composite"]
            best_epoch = epoch
            bad_epochs = 0
            best_metrics = dict(vm)
            best_metrics["best_epoch"] = epoch
            best_metrics["base_model"] = base_model

            ckpt = outdir / "best_adapter"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            (outdir / "metrics.json").write_text(
                json.dumps(best_metrics, indent=2, sort_keys=True)
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"[early-stop] epoch={epoch} "
                    f"best_epoch={best_epoch} "
                    f"best_composite={best:.4f}"
                )
                break

    print("\n=== V13.1-STRONG BEST VALIDATION ===")
    print(json.dumps(best_metrics, indent=2, sort_keys=True))

    # Explicit go/no-go recommendation based on the metrics that matter.
    go = bool(
        best_metrics
        and best_metrics["causal_edge_accuracy"] >= 0.78
        and best_metrics["hard_edge_accuracy"] >= 0.75
        and best_metrics["hard_flip_accuracy"] >= 0.65
    )
    decision = {
        "go_to_integration_pilot": go,
        "reason": (
            "PASS: scorer learned causal and hard-neighbor discrimination."
            if go else
            "STOP: do not spend API calls yet; scorer still fails "
            "causal/hard-neighbor discrimination."
        ),
        "thresholds": {
            "causal_edge_accuracy": 0.78,
            "hard_edge_accuracy": 0.75,
            "hard_flip_accuracy": 0.65,
        },
    }
    (outdir / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True)
    )
    print("\n=== GO / NO-GO ===")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
