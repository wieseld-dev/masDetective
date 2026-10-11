#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import analysis.train_v13_6b_activation_judge as tr


START_ADAPTER = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/best_adapter"
)
DATA_ROOT = Path("artifacts/v139_worker_outputs_final")

ROLES = list(tr.ROLE_DESCRIPTIONS)
ROLE_DESC = dict(tr.ROLE_DESCRIPTIONS)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_source_rows():
    rows = []

    for p in sorted(DATA_ROOT.glob("pair_*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                x = json.loads(line)
                split = str(x["split"])

                # Do not use locked test here.
                if split not in {"train", "val"}:
                    continue

                q = str(x["query"])
                qid = str(x["query_id"])
                ra = str(x["role_a"])
                rb = str(x["role_b"])

                rows.append({
                    "id": qid + "::a",
                    "query_id": qid,
                    "split": split,
                    "true_role": ra,
                    "paired_role": rb,
                    "query": q,
                    "answer": str(x["output_a"]),
                    "semantic_stratum": x.get("semantic_stratum"),
                })

                rows.append({
                    "id": qid + "::b",
                    "query_id": qid,
                    "split": split,
                    "true_role": rb,
                    "paired_role": ra,
                    "query": q,
                    "answer": str(x["output_b"]),
                    "semantic_stratum": x.get("semantic_stratum"),
                })

    return rows


def build_text(role, row):
    return tr.presence_text(
        role,
        ROLE_DESC[role],
        role,  # canonical SELF target encoding
        row["query"],
        row["answer"],
    )


def load_model(trainable=True):
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

    cfg = json.loads(
        (START_ADAPTER / "adapter_config.json").read_text()
    )
    base_name = cfg["base_model_name_or_path"]

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    try:
        tok = AutoTokenizer.from_pretrained(
            START_ADAPTER,
            use_fast=True,
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained(
            base_name,
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
        base_name,
        num_labels=1,
        quantization_config=qconf,
        device_map={"": 0},
        dtype=dtype,
    )

    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    if trainable:
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=True,
        )

    model = PeftModel.from_pretrained(
        base,
        str(START_ADAPTER),
        is_trainable=trainable,
    )

    model.config.use_cache = False

    if trainable:
        model.print_trainable_parameters()

    return tok, model, dtype


def score_roles(
    model,
    tok,
    dtype,
    row,
    roles,
    max_length,
    chunk_size,
    grad,
):
    import torch

    vals = []

    ctx = torch.enable_grad() if grad else torch.no_grad()

    with ctx:
        for start in range(0, len(roles), chunk_size):
            rr = roles[start:start + chunk_size]

            texts = [
                build_text(role, row)
                for role in rr
            ]

            x = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            x = {
                k: v.to("cuda:0")
                for k, v in x.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                z = model(**x).logits[:, 0].float()

            vals.append(z)

    return torch.cat(vals, dim=0)


def metrics_from_records(records):
    if not records:
        return {}

    ranks = np.asarray(
        [r["rank"] for r in records],
        dtype=int,
    )

    by_role = {}
    roles_seen = sorted(set(r["true_role"] for r in records))

    for role in roles_seen:
        rr = [
            r for r in records
            if r["true_role"] == role
        ]
        ar = np.asarray([x["rank"] for x in rr])

        by_role[role] = {
            "n": len(rr),
            "top1": float(np.mean(ar == 1)),
            "top3": float(np.mean(ar <= 3)),
            "mrr": float(np.mean(1.0 / ar)),
            "mean_rank": float(np.mean(ar)),
        }

    confusion = defaultdict(Counter)

    for r in records:
        confusion[r["true_role"]][r["pred_role"]] += 1

    confusion = {
        a: dict(b)
        for a, b in confusion.items()
    }

    return {
        "n": len(records),
        "top1": float(np.mean(ranks == 1)),
        "top3": float(np.mean(ranks <= 3)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "pairwise_accuracy": float(np.mean([
            r["true_score"] > r["paired_score"]
            for r in records
        ])),
        "rank_histogram": {
            str(k): int(v)
            for k, v in sorted(Counter(ranks).items())
        },
        "by_true_role": by_role,
        "confusion": confusion,
    }


def evaluate(
    model,
    tok,
    dtype,
    rows,
    max_length,
    chunk_size,
    label,
):
    model.eval()

    role_index = {
        r: i
        for i, r in enumerate(ROLES)
    }

    records = []

    for i, row in enumerate(rows, 1):
        z = score_roles(
            model,
            tok,
            dtype,
            row,
            ROLES,
            max_length,
            chunk_size,
            grad=False,
        )

        s = z.detach().cpu().numpy()

        true_i = role_index[row["true_role"]]
        pair_i = role_index[row["paired_role"]]

        order = np.argsort(-s)
        rank = int(
            np.where(order == true_i)[0][0]
        ) + 1

        pred = ROLES[int(order[0])]

        records.append({
            "id": row["id"],
            "query_id": row["query_id"],
            "true_role": row["true_role"],
            "paired_role": row["paired_role"],
            "pred_role": pred,
            "rank": rank,
            "true_score": float(s[true_i]),
            "paired_score": float(s[pair_i]),
            "scores": {
                r: float(s[j])
                for j, r in enumerate(ROLES)
            },
        })

        if i % 20 == 0 or i == len(rows):
            print(
                f"{label}: {i}/{len(rows)}",
                flush=True,
            )

    return metrics_from_records(records), records


def balanced_sample(rows, n, seed):
    rng = random.Random(seed)

    by = defaultdict(list)

    for x in rows:
        by[x["true_role"]].append(x)

    roles = sorted(
        by,
        key=lambda r: ROLES.index(r),
    )

    out = []

    while len(out) < n:
        for role in roles:
            if len(out) >= n:
                break
            out.append(rng.choice(by[role]))

    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output-dir",
        default="artifacts/v13_11_listwise_source_v1",
    )
    ap.add_argument("--train-samples", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--score-chunk", type=int, default=4)
    ap.add_argument("--eval-chunk", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--anchor-weight", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=13111)

    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    out = Path(args.output_dir)

    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"Refusing non-empty output dir: {out}"
        )

    out.mkdir(parents=True, exist_ok=True)

    rows = load_source_rows()

    train = [
        x for x in rows
        if x["split"] == "train"
    ]
    val = [
        x for x in rows
        if x["split"] == "val"
    ]

    train_counts = Counter(
        x["true_role"] for x in train
    )
    val_counts = Counter(
        x["true_role"] for x in val
    )

    eligible_roles = [
        r for r in ROLES
        if train_counts[r] > 0
    ]
    missing_positive_roles = [
        r for r in ROLES
        if train_counts[r] == 0
    ]

    print("ROLE COUNT:", len(ROLES), flush=True)
    print("TRAIN outputs:", len(train), flush=True)
    print("VAL outputs:", len(val), flush=True)
    print(
        "ELIGIBLE LISTWISE ROLES:",
        eligible_roles,
        flush=True,
    )
    print(
        "MISSING POSITIVE ROLES:",
        missing_positive_roles,
        flush=True,
    )
    print(
        "TRAIN SOURCE COUNTS:",
        dict(train_counts),
        flush=True,
    )

    dump(
        out / "data_audit.json",
        {
            "roles": ROLES,
            "train_outputs": len(train),
            "val_outputs": len(val),
            "train_source_counts": dict(train_counts),
            "val_source_counts": dict(val_counts),
            "eligible_roles": eligible_roles,
            "missing_positive_roles": missing_positive_roles,
            "locked_test_used": False,
        },
    )

    train_sample = balanced_sample(
        train,
        args.train_samples,
        args.seed,
    )

    dump(
        out / "train_sample_ids.json",
        [x["id"] for x in train_sample],
    )

    tok, model, dtype = load_model(trainable=True)

    # ------------------------------------------------------------
    # Baseline global-20 VAL before any optimizer update.
    # ------------------------------------------------------------
    print(
        "\n===== BASELINE GLOBAL20 VAL =====",
        flush=True,
    )

    baseline_val, baseline_records = evaluate(
        model,
        tok,
        dtype,
        val,
        args.max_length,
        args.eval_chunk,
        "baseline",
    )

    print(
        json.dumps(
            baseline_val,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    dump(
        out / "baseline_val.json",
        baseline_val,
    )

    # ------------------------------------------------------------
    # Freeze initial scores ONLY for roles with no positive source
    # examples, to stop listwise training from arbitrarily drifting
    # those three roles.
    # ------------------------------------------------------------
    print(
        "\n===== PRECOMPUTE MISSING-ROLE ANCHORS =====",
        flush=True,
    )

    model.eval()
    anchors = {}

    if missing_positive_roles:
        for i, row in enumerate(train_sample, 1):
            z = score_roles(
                model,
                tok,
                dtype,
                row,
                missing_positive_roles,
                args.max_length,
                args.eval_chunk,
                grad=False,
            )

            anchors[row["id"]] = (
                z.detach().cpu().float()
            )

            if i % 100 == 0 or i == len(train_sample):
                print(
                    f"anchor {i}/{len(train_sample)}",
                    flush=True,
                )

    # ------------------------------------------------------------
    # TRUE LISTWISE SOURCE RANKING
    # Each visible source output is scored against every source role
    # that actually has positive TRAIN examples.
    # ------------------------------------------------------------
    role_to_eligible = {
        r: i
        for i, r in enumerate(eligible_roles)
    }

    opt = torch.optim.AdamW(
        [
            p for p in model.parameters()
            if p.requires_grad
        ],
        lr=args.lr,
        weight_decay=0.01,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(dtype == torch.float16),
    )

    rng = random.Random(args.seed)

    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_rows = list(train_sample)
        rng.shuffle(epoch_rows)

        model.train()
        opt.zero_grad(set_to_none=True)

        losses = []
        ce_losses = []
        anchor_losses = []

        for step, row in enumerate(epoch_rows, 1):
            # We score ALL 20 candidate roles.
            z20 = score_roles(
                model,
                tok,
                dtype,
                row,
                ROLES,
                args.max_length,
                args.score_chunk,
                grad=True,
            )

            idx20 = {
                r: i
                for i, r in enumerate(ROLES)
            }

            eligible_idx = torch.tensor(
                [
                    idx20[r]
                    for r in eligible_roles
                ],
                device=z20.device,
                dtype=torch.long,
            )

            z = z20.index_select(
                0,
                eligible_idx,
            ) / args.temperature

            target = torch.tensor(
                [role_to_eligible[row["true_role"]]],
                device=z.device,
                dtype=torch.long,
            )

            ce = F.cross_entropy(
                z.unsqueeze(0),
                target,
            )

            if missing_positive_roles:
                missing_idx = torch.tensor(
                    [
                        idx20[r]
                        for r in missing_positive_roles
                    ],
                    device=z20.device,
                    dtype=torch.long,
                )

                current_missing = z20.index_select(
                    0,
                    missing_idx,
                )

                baseline_missing = anchors[
                    row["id"]
                ].to(
                    current_missing.device
                )

                anchor = F.mse_loss(
                    current_missing,
                    baseline_missing,
                )
            else:
                anchor = z20.sum() * 0.0

            loss = (
                ce
                + args.anchor_weight * anchor
            )

            scaler.scale(
                loss / args.grad_accum
            ).backward()

            if (
                step % args.grad_accum == 0
                or step == len(epoch_rows)
            ):
                scaler.unscale_(opt)

                torch.nn.utils.clip_grad_norm_(
                    [
                        p for p in model.parameters()
                        if p.requires_grad
                    ],
                    1.0,
                )

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()))
            ce_losses.append(float(ce.detach().cpu()))
            anchor_losses.append(
                float(anchor.detach().cpu())
            )

            if step % 50 == 0 or step == len(epoch_rows):
                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(epoch_rows)} "
                    f"loss={np.mean(losses):.4f} "
                    f"ce={np.mean(ce_losses):.4f} "
                    f"anchor={np.mean(anchor_losses):.4f}",
                    flush=True,
                )

        # --------------------------------------------------------
        # Global20 validation.
        # --------------------------------------------------------
        print(
            f"\n===== EPOCH {epoch} GLOBAL20 VAL =====",
            flush=True,
        )

        val_metrics, val_records = evaluate(
            model,
            tok,
            dtype,
            val,
            args.max_length,
            args.eval_chunk,
            f"epoch{epoch}",
        )

        print(
            json.dumps(
                val_metrics,
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )

        row = {
            "epoch": epoch,
            "mean_loss": float(np.mean(losses)),
            "mean_ce": float(np.mean(ce_losses)),
            "mean_anchor": float(
                np.mean(anchor_losses)
            ),
            "val": val_metrics,
        }

        history.append(row)

        epoch_dir = out / f"epoch_{epoch}_adapter"
        model.save_pretrained(epoch_dir)
        tok.save_pretrained(epoch_dir)

        dump(
            out / f"epoch_{epoch}_val_records.json",
            val_records,
        )

        dump(
            out / "training_history.json",
            history,
        )

    # Pick epoch by GLOBAL20 Top1, MRR tie-break.
    best = max(
        history,
        key=lambda x: (
            x["val"]["top1"],
            x["val"]["mrr"],
        ),
    )

    best_epoch = int(best["epoch"])
    best_src = out / f"epoch_{best_epoch}_adapter"
    best_dst = out / "best_adapter"

    if best_dst.exists():
        shutil.rmtree(best_dst)

    shutil.copytree(
        best_src,
        best_dst,
    )

    summary = {
        "protocol": "v13_11_listwise_source_v1",
        "initial_adapter": str(START_ADAPTER),
        "train_outputs_available": len(train),
        "train_samples_used": len(train_sample),
        "val_outputs": len(val),
        "eligible_listwise_roles": eligible_roles,
        "missing_positive_roles": missing_positive_roles,
        "listwise_denominator": len(eligible_roles),
        "baseline_val": baseline_val,
        "best_epoch": best_epoch,
        "best_val": best["val"],
        "delta_top1": (
            best["val"]["top1"]
            - baseline_val["top1"]
        ),
        "delta_mrr": (
            best["val"]["mrr"]
            - baseline_val["mrr"]
        ),
        "locked_test_used": False,
        "api_calls": 0,
        "mas_calls": 0,
    }

    dump(
        out / "summary.json",
        summary,
    )

    print(
        "\n" + "=" * 100,
        flush=True,
    )
    print(
        "FINAL V13.11 PILOT SUMMARY",
        flush=True,
    )
    print(
        "=" * 100,
        flush=True,
    )
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
