#!/usr/bin/env python3

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import analysis.train_v13_6b_activation_judge as tr
import analysis.v137_calibration_audit as ca


V137 = Path("artifacts/v13_7_fp3b_overnight/best_adapter")
CALROOT = Path("artifacts/v13_7_calibration_audit_v1")

PAIR_TYPES = (
    "activation_cf_plus_minus",
    "activation_cf_sham_minus",
    "within_answer_hard_neighbor_absent",
    "within_answer_probe_target_absent",
)

TYPE_MASS = {
    "activation_cf_plus_minus": 0.225,
    "activation_cf_sham_minus": 0.225,
    "within_answer_hard_neighbor_absent": 0.350,
    "within_answer_probe_target_absent": 0.200,
}

if abs(sum(TYPE_MASS.values()) - 1.0) > 1e-9:
    raise RuntimeError("TYPE_MASS must sum to 1.0")


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def base_type(t):
    return t.split(":", 1)[0]


def candidate_role(text):
    marker = "[CANDIDATE ROLE]\n"
    if marker not in text:
        raise RuntimeError("missing candidate role")
    return text.split(marker, 1)[1].split("\n", 1)[0].strip()


def split_data():
    rd = dict(tr.ROLE_DESCRIPTIONS)

    r13 = tr.read_jsonl(ca.V13)
    r11 = tr.read_jsonl(ca.V11)

    c13, _ = tr.normalize_rows(r13, "v13", rd, 1.0)
    c11, _ = tr.normalize_rows(r11, "v11", rd, ca.V11_WEIGHT)

    cases = tr.attach_components(c13 + c11)

    args = SimpleNamespace(
        seed=ca.SEED,
        n_splits=ca.N_SPLITS,
        fold=ca.FOLD,
    )

    train_cases, val_cases, split_report = tr.split_cases(
        cases, args
    )

    old_summary = json.loads(
        (CALROOT / "summary.json").read_text()
    )
    salt = int(old_summary["split_salt"])

    cal_cases = []
    dev_cases = []

    for c in val_cases:
        b = ca.stable_bucket(ca.component_value(c), salt)

        if b < int(ca.CAL_FRACTION * 100):
            cal_cases.append(c)
        else:
            dev_cases.append(c)

    return rd, train_cases, cal_cases, dev_cases, split_report


def pair_meta(p):
    typ = base_type(p.pair_type)

    if typ not in PAIR_TYPES:
        raise RuntimeError(f"unknown pair type {typ}")

    pos_role = candidate_role(p.pos_text)
    neg_role = candidate_role(p.neg_text)

    if typ.startswith("activation_cf_"):
        if pos_role != neg_role:
            raise RuntimeError(
                f"activation pair role mismatch: {pos_role} {neg_role}"
            )
        role_key = pos_role
        old_mult = 1.0

    elif typ == "within_answer_probe_target_absent":
        role_key = neg_role
        old_mult = 3.0

    elif typ == "within_answer_hard_neighbor_absent":
        role_key = neg_role
        old_mult = 2.0

    else:
        raise RuntimeError(typ)

    # Remove old V13.6b type multiplier.
    # Keep only underlying case/source quality.
    quality = float(p.weight) / old_mult

    return {
        "pair": p,
        "type": typ,
        "pos_role": pos_role,
        "neg_role": neg_role,
        "role_key": role_key,
        "quality": quality,
    }


def build_meta(cases, rd):
    pairs, counts = tr.build_pairs(cases, rd)
    meta = [pair_meta(p) for p in pairs]
    return pairs, meta, counts


def sampler_weights(meta):
    by_type_role = Counter(
        (x["type"], x["role_key"])
        for x in meta
    )

    roles_per_type = defaultdict(set)

    for x in meta:
        roles_per_type[x["type"]].add(x["role_key"])

    weights = []

    for x in meta:
        t = x["type"]
        r = x["role_key"]

        g = len(roles_per_type[t])
        n = by_type_role[(t, r)]

        # Explicit total mass per pair type.
        # Within each type:
        #   equal mass per role group,
        #   equal probability per example inside that role group.
        weights.append(
            TYPE_MASS[t]
            / (max(g, 1) * max(n, 1))
        )

    return weights, by_type_role, roles_per_type


def audit(args):
    rd, train_cases, cal_cases, dev_cases, split_report = split_data()

    tp, tm, tc = build_meta(train_cases, rd)
    cp, cm, cc = build_meta(cal_cases, rd)
    dp, dm, dc = build_meta(dev_cases, rd)

    sw, group_counts, groups = sampler_weights(tm)

    print("=" * 92)
    print("V13.8B BALANCED CAUSAL REPAIR AUDIT")
    print("=" * 92)

    print("train cases :", len(train_cases))
    print("cal cases   :", len(cal_cases))
    print("dev87 cases :", len(dev_cases))
    print()

    print("train pairs :", len(tp))
    print("cal pairs   :", len(cp))
    print("dev87 pairs :", len(dp))

    print("\nTRAIN PAIR TYPES")
    for k in PAIR_TYPES:
        print(f"{k:42s} {tc.get(k, 0)}")

    print("\nCAL PAIR TYPES")
    for k in PAIR_TYPES:
        print(f"{k:42s} {cc.get(k, 0)}")

    print("\nDEV87 PAIR TYPES")
    for k in PAIR_TYPES:
        print(f"{k:42s} {dc.get(k, 0)}")

    print("\nBALANCING GROUPS")
    for typ in PAIR_TYPES:
        rs = sorted(groups[typ])
        vals = [
            group_counts[(typ, r)]
            for r in rs
        ]

        print(
            f"{typ:42s} "
            f"role_groups={len(rs):2d} "
            f"min={min(vals) if vals else 0:3d} "
            f"max={max(vals) if vals else 0:3d}"
        )

    print("\nNORMALIZED SAMPLER TYPE MASS")

    total_sw = float(sum(sw))

    for typ in PAIR_TYPES:
        actual = sum(
            w
            for x, w in zip(tm, sw)
            if x["type"] == typ
        ) / total_sw

        print(
            f"{typ:42s} "
            f"target={TYPE_MASS[typ]:.3f} "
            f"actual={actual:.3f}"
        )

    print("\nSAMPLER")
    print("samples/epoch :", args.samples_per_epoch)
    print("batch size    :", args.batch_size)
    print("grad accum    :", args.grad_accum)
    print(
        "optimizer steps/epoch :",
        math.ceil(
            math.ceil(args.samples_per_epoch / args.batch_size)
            / args.grad_accum
        ),
    )

    assert len(dev_cases) == 87
    assert len(dp) == 798
    assert dc["activation_cf_plus_minus"] == 209
    assert dc["activation_cf_sham_minus"] == 233
    assert dc["within_answer_hard_neighbor_absent"] == 171
    assert dc["within_answer_probe_target_absent"] == 185

    for typ in PAIR_TYPES:
        if tc.get(typ, 0) <= 0:
            raise RuntimeError(f"zero training pairs: {typ}")

    if not V137.exists():
        raise FileNotFoundError(V137)

    report = {
        "protocol": "v13.8b_balanced_causal_pairwise_repair",
        "initial_adapter": str(V137),
        "train_cases": len(train_cases),
        "cal_cases": len(cal_cases),
        "dev87_cases": len(dev_cases),
        "train_pairs": len(tp),
        "cal_pairs": len(cp),
        "dev87_pairs": len(dp),
        "train_pair_counts": dict(tc),
        "cal_pair_counts": dict(cc),
        "dev87_pair_counts": dict(dc),
        "samples_per_epoch": args.samples_per_epoch,
        "pair_types": list(PAIR_TYPES),
        "model_selection_split": "cal86_only",
        "dev87_used_for_model_selection": False,
        "api_calls": 0,
        "mas_calls": 0,
    }

    return rd, tm, cm, dm, sw, report


class RepairDataset:
    def __init__(self, meta):
        self.meta = list(meta)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        return self.meta[i]


def collate(batch):
    import torch

    return {
        "pos_text": [
            x["pair"].pos_text
            for x in batch
        ],
        "neg_text": [
            x["pair"].neg_text
            for x in batch
        ],
        "quality": torch.tensor(
            [x["quality"] for x in batch],
            dtype=torch.float32,
        ),
        "type": [x["type"] for x in batch],
        "pos_role": [x["pos_role"] for x in batch],
        "neg_role": [x["neg_role"] for x in batch],
    }


def load_model():
    import torch

    from peft import (
        PeftModel,
        prepare_model_for_kbit_training,
    )

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    tok = AutoTokenizer.from_pretrained(
        V137,
        use_fast=True,
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    base = AutoModelForSequenceClassification.from_pretrained(
        ca.BASE_MODEL,
        num_labels=1,
        quantization_config=qconf,
        device_map={"": 0},
        dtype=dtype,
    )

    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
    )

    model = PeftModel.from_pretrained(
        base,
        V137,
        is_trainable=True,
    )

    model.config.use_cache = False

    print(
        "[init] continuing V13.7 epoch-1 adapter "
        "with V13.8b balanced causal repair",
        flush=True,
    )

    model.print_trainable_parameters()

    return tok, model, dtype


def forward_logits(model, tok, texts, max_length, device):
    x = tok(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    x = {
        k: v.to(device)
        for k, v in x.items()
    }

    return model(**x).logits[:, 0].float()


def eval_meta(model, tok, meta, args, device, dtype):
    import torch
    from torch.utils.data import DataLoader

    dl = DataLoader(
        RepairDataset(meta),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    records = []

    model.eval()

    with torch.no_grad():
        for b in dl:
            texts = b["pos_text"] + b["neg_text"]

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                z = forward_logits(
                    model,
                    tok,
                    texts,
                    args.max_length,
                    device,
                )

            n = len(b["pos_text"])
            margin = (
                z[:n] - z[n:]
            ).detach().cpu().numpy()

            for i, m in enumerate(margin):
                records.append({
                    "type": b["type"][i],
                    "pos_role": b["pos_role"][i],
                    "neg_role": b["neg_role"][i],
                    "margin": float(m),
                })

    def stats(rows):
        x = np.asarray(
            [r["margin"] for r in rows],
            dtype=float,
        )

        return {
            "n": int(len(x)),
            "accuracy": float(np.mean(x > 0)),
            "mean_margin": float(np.mean(x)),
            "median_margin": float(np.median(x)),
        }

    by_type = {}

    for typ in PAIR_TYPES:
        rows = [
            r for r in records
            if r["type"] == typ
        ]

        by_type[typ] = stats(rows)

    cf_rows = [
        r for r in records
        if r["type"].startswith("activation_cf_")
    ]

    cf_role = {}

    for role in tr.ROLE_DESCRIPTIONS:
        rr = [
            r for r in cf_rows
            if r["pos_role"] == role
        ]

        if len(rr) >= 3:
            cf_role[role] = stats(rr)

    cf_role_macro = (
        float(np.mean([
            x["accuracy"]
            for x in cf_role.values()
        ]))
        if cf_role
        else None
    )

    target_by_absent = {}

    for role in tr.ROLE_DESCRIPTIONS:
        rr = [
            r for r in records
            if r["type"]
               == "within_answer_probe_target_absent"
            and r["neg_role"] == role
        ]

        if rr:
            target_by_absent[role] = stats(rr)

    # Hard-neighbor performance grouped by the ABSENT role.
    hard_by_absent = {}

    for role in tr.ROLE_DESCRIPTIONS:
        rr = [
            r for r in records
            if r["type"]
               == "within_answer_hard_neighbor_absent"
            and r["neg_role"] == role
        ]

        if rr:
            hard_by_absent[role] = stats(rr)

    target_role_macro = (
        float(np.mean([
            x["accuracy"]
            for x in target_by_absent.values()
        ]))
        if target_by_absent
        else None
    )

    hard_role_macro = (
        float(np.mean([
            x["accuracy"]
            for x in hard_by_absent.values()
        ]))
        if hard_by_absent
        else None
    )

    vals = [
        by_type[t]["accuracy"]
        for t in PAIR_TYPES
    ]

    cf_micro = 0.5 * (
        by_type["activation_cf_plus_minus"]["accuracy"]
        + by_type["activation_cf_sham_minus"]["accuracy"]
    )

    pair_floor = min(vals)
    pair_mean = float(np.mean(vals))

    # Model selection emphasizes ROLE-MACRO behavior so a few
    # frequent/easy roles cannot hide planning/research/etc failures.
    selection_score = (
        0.30 * cf_role_macro
        + 0.30 * target_role_macro
        + 0.15 * hard_role_macro
        + 0.10 * cf_micro
        + 0.10 * pair_floor
        + 0.05 * pair_mean
    )

    return {
        "by_type": by_type,
        "same_role_cf_micro": cf_micro,
        "same_role_cf_role_macro": cf_role_macro,
        "same_role_cf_by_role": cf_role,
        "target_absent_role_macro": target_role_macro,
        "target_absent_by_absent_role":
            target_by_absent,
        "hard_neighbor_role_macro": hard_role_macro,
        "hard_neighbor_by_absent_role":
            hard_by_absent,
        "pair_floor": pair_floor,
        "pair_mean": pair_mean,
        "selection_score": selection_score,
    }


def print_metrics(title, m):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)

    for typ in PAIR_TYPES:
        x = m["by_type"][typ]

        print(
            f"{typ:42s} "
            f"n={x['n']:4d} "
            f"acc={x['accuracy']:.4f} "
            f"mean={x['mean_margin']:+.4f}"
        )

    print(
        "same-role role-macro :",
        f"{m['same_role_cf_role_macro']:.4f}",
    )
    print(
        "target-abs role-macro:",
        f"{m['target_absent_role_macro']:.4f}",
    )
    print(
        "hard-neg role-macro  :",
        f"{m['hard_neighbor_role_macro']:.4f}",
    )
    print(
        "pair floor           :",
        f"{m['pair_floor']:.4f}",
    )
    print(
        "pair mean            :",
        f"{m['pair_mean']:.4f}",
    )
    print(
        "selection score      :",
        f"{m['selection_score']:.4f}",
    )

    print("\nweak-role same-role CF:")

    for role in (
        "planning",
        "incident_management",
        "customer_support_triage",
        "research",
        "writing",
    ):
        x = m["same_role_cf_by_role"].get(role)

        if x:
            print(
                f"  {role:26s} "
                f"n={x['n']:3d} "
                f"acc={x['accuracy']:.4f} "
                f"mean={x['mean_margin']:+.4f}"
            )


def train(args):
    import torch
    import torch.nn.functional as F

    from torch.utils.data import (
        DataLoader,
        WeightedRandomSampler,
    )

    from transformers import (
        get_cosine_schedule_with_warmup,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for training"
        )

    (
        rd,
        train_meta,
        cal_meta,
        dev_meta,
        sampler_w,
        report,
    ) = audit(args)

    out = Path(args.output_dir)

    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"Refusing non-empty output dir: {out}"
        )

    out.mkdir(parents=True, exist_ok=True)
    dump(out / "data_audit.json", report)

    device = torch.device("cuda:0")

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    tok, model, dtype = load_model()

    # --------------------------------------------------------
    # Baseline guard: score the ORIGINAL V13.7 checkpoint on
    # CAL86 before a single optimizer update.
    # --------------------------------------------------------
    baseline_cal = eval_meta(
        model,
        tok,
        cal_meta,
        args,
        device,
        dtype,
    )

    print_metrics(
        "BASELINE V13.7 — CAL86",
        baseline_cal,
    )

    dump(
        out / "baseline_cal_metrics.json",
        baseline_cal,
    )

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    opt = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=0.01,
    )

    gen = torch.Generator()
    gen.manual_seed(args.seed)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sampler_w,
            dtype=torch.double,
        ),
        num_samples=args.samples_per_epoch,
        replacement=True,
        generator=gen,
    )

    dl = DataLoader(
        RepairDataset(train_meta),
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,
    )

    steps_per_epoch = math.ceil(
        len(dl) / args.grad_accum
    )

    total_steps = (
        steps_per_epoch * args.epochs
    )

    warmup = max(
        10,
        int(0.05 * total_steps),
    )

    sched = get_cosine_schedule_with_warmup(
        opt,
        warmup,
        total_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(dtype == torch.float16),
    )

    history = []
    best = baseline_cal["selection_score"]
    improved = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        run = 0.0
        rank_run = 0.0
        bce_run = 0.0
        seen = 0

        for step, b in enumerate(dl, 1):
            quality = b["quality"].to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                z = forward_logits(
                    model,
                    tok,
                    b["pos_text"] + b["neg_text"],
                    args.max_length,
                    device,
                )

                n = len(b["pos_text"])

                pos = z[:n]
                neg = z[n:]

                rank_each = F.softplus(
                    args.margin - (pos - neg)
                )

                bce_each = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        pos,
                        torch.ones_like(pos),
                        reduction="none",
                    )
                    +
                    F.binary_cross_entropy_with_logits(
                        neg,
                        torch.zeros_like(neg),
                        reduction="none",
                    )
                )

                loss_each = (
                    rank_each
                    + args.bce_weight * bce_each
                )

                loss = (
                    (loss_each * quality).sum()
                    / quality.sum().clamp_min(1e-8)
                )

            scaler.scale(
                loss / args.grad_accum
            ).backward()

            bs = n
            run += float(loss.detach().cpu()) * bs
            rank_run += (
                float(rank_each.mean().detach().cpu())
                * bs
            )
            bce_run += (
                float(bce_each.mean().detach().cpu())
                * bs
            )
            seen += bs

            if (
                step % args.grad_accum == 0
                or step == len(dl)
            ):
                scaler.unscale_(opt)

                torch.nn.utils.clip_grad_norm_(
                    params,
                    1.0,
                )

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            if step % 100 == 0:
                print(
                    f"[train] epoch={epoch} "
                    f"step={step}/{len(dl)} "
                    f"loss={run/max(seen,1):.4f}",
                    flush=True,
                )

        cal = eval_meta(
            model,
            tok,
            cal_meta,
            args,
            device,
            dtype,
        )

        row = {
            "epoch": epoch,
            "train_loss": run / max(seen, 1),
            "train_rank_loss":
                rank_run / max(seen, 1),
            "train_bce_loss":
                bce_run / max(seen, 1),
            "cal": cal,
        }

        history.append(row)
        dump(out / "training_history.json", history)

        print_metrics(
            f"EPOCH {epoch} — CAL86",
            cal,
        )

        last = out / "last_adapter"
        shutil.rmtree(last, ignore_errors=True)
        model.save_pretrained(last)
        tok.save_pretrained(last)

        if cal["selection_score"] > best + args.min_delta:
            best = cal["selection_score"]
            improved = True

            best_dir = out / "best_adapter"
            shutil.rmtree(
                best_dir,
                ignore_errors=True,
            )

            model.save_pretrained(best_dir)
            tok.save_pretrained(best_dir)

            dump(
                out / "best_cal_metrics.json",
                cal,
            )

    # For the one-epoch pilot this is the current/best model.
    # DEV87 is diagnostic only and NEVER used for selection.
    dev = eval_meta(
        model,
        tok,
        dev_meta,
        args,
        device,
        dtype,
    )

    print_metrics(
        "FINAL DIAGNOSTIC — DEV87 "
        "(NOT MODEL SELECTION)",
        dev,
    )

    dump(
        out / "dev87_diagnostic.json",
        dev,
    )

    pilot_summary = {
        "baseline_cal_selection_score":
            baseline_cal["selection_score"],
        "final_cal_selection_score":
            history[-1]["cal"]["selection_score"],
        "best_selection_score":
            best,
        "improved_over_baseline":
            improved,
        "best_adapter_exists":
            (out / "best_adapter").exists(),
        "dev87_used_for_model_selection":
            False,
    }

    dump(
        out / "pilot_summary.json",
        pilot_summary,
    )

    print("\nDONE")
    print(
        "baseline CAL selection score:",
        baseline_cal["selection_score"],
    )
    print(
        "final CAL selection score   :",
        history[-1]["cal"]["selection_score"],
    )
    print("best CAL selection score    :", best)
    print("improved over baseline      :", improved)
    print("wrote:", out)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--output-dir",
        default="artifacts/v13_8b_balanced_repair_pilot",
    )

    p.add_argument(
        "--audit-only",
        action="store_true",
    )

    p.add_argument("--seed", type=int, default=1381)
    p.add_argument(
        "--samples-per-epoch",
        type=int,
        default=1024,
    )

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1024)

    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--bce-weight", type=float, default=0.15)
    p.add_argument("--min-delta", type=float, default=1e-4)

    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.audit_only:
        _, _, _, _, _, report = audit(args)

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        dump(
            out / "data_audit.json",
            report,
        )

        print("\nAUDIT-ONLY PASS")
        print("GPU: 0")
        print("training: 0")
        print("MAS/API calls: 0")
        return

    train(args)


if __name__ == "__main__":
    main()
