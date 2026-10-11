#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from peft import PeftConfig, PeftModel
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import train_v7_benign_output_sensor_pilot_v1 as bos
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS


ROLES = list(TRAIN_ROLES)

DATA_ROOT = ROOT / "artifacts/v7_benign_final_pool_v2"

DEFAULT_ADAPTER = (
    ROOT
    / "artifacts/v7_benign_output_sensor_pilot_v1/best_adapter"
)

WEAK_POSITIVES = {
    "financial_analysis",
    "accessibility_review",
    "database_design",
    "writing",
    "math",
    "experiment_design",
    "code",
    "requirements_analysis",
}

GLOBAL_FP = [
    "research",
    "operations",
    "fact_checking",
    "customer_support_triage",
    "planning",
]

CONFUSIONS = {
    "research": [
        "operations",
        "fact_checking",
        "planning",
        "customer_support_triage",
    ],
    "customer_support_triage": [
        "fact_checking",
        "operations",
        "requirements_analysis",
    ],
    "requirements_analysis": [
        "research",
        "planning",
        "customer_support_triage",
    ],
    "documentation": [
        "code",
        "writing",
        "qa_testing",
    ],
    "design": [
        "writing",
        "accessibility_review",
        "requirements_analysis",
    ],
    "code": [
        "documentation",
        "qa_testing",
        "database_design",
    ],
    "financial_analysis": [
        "data",
        "math",
        "planning",
        "research",
    ],
    "database_design": [
        "code",
        "data",
        "requirements_analysis",
    ],
    "accessibility_review": [
        "design",
        "writing",
        "requirements_analysis",
    ],
    "writing": [
        "documentation",
        "design",
        "translation",
    ],
    "math": [
        "data",
        "financial_analysis",
        "experiment_design",
    ],
    "experiment_design": [
        "research",
        "data",
        "math",
    ],
    "operations": [
        "incident_management",
        "planning",
        "customer_support_triage",
    ],
    "fact_checking": [
        "research",
        "customer_support_triage",
    ],
    "planning": [
        "requirements_analysis",
        "risk_assessment",
        "operations",
    ],
}


STOP_REQUESTED = False


def signal_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        f"[signal] received {signum}; "
        "will checkpoint at next safe optimizer boundary",
        flush=True,
    )


signal.signal(signal.SIGUSR1, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def read_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )
    return rows


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
    )
    os.close(fd)

    try:
        Path(tmp).write_text(
            json.dumps(
                obj,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def selected(row):
    return list(
        map(
            str,
            row.get("selected_roles") or [],
        )
    )


def answer(row):
    return str(
        row.get("final_output") or ""
    )


def load_model(adapter_path):
    adapter_path = Path(adapter_path)

    cfg = PeftConfig.from_pretrained(
        str(adapter_path)
    )

    base_name = cfg.base_model_name_or_path

    print(
        "[model] base:",
        base_name,
        flush=True,
    )
    print(
        "[model] adapter:",
        adapter_path,
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(
        str(adapter_path)
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"
    tok.truncation_side = "right"

    base = (
        AutoModelForSequenceClassification
        .from_pretrained(
            base_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        )
    )

    base.config.pad_token_id = (
        tok.pad_token_id
    )
    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        is_trainable=True,
    )

    # --------------------------------------------------------
    # Memory-safe training on 24 GB GPUs.
    # Gradient checkpointing dramatically reduces activation
    # memory. Because the base embeddings are frozen under
    # PEFT/LoRA, inputs must require grads for checkpointing
    # to propagate gradients into LoRA modules correctly.
    # --------------------------------------------------------
    core = (
        model.get_base_model()
        if hasattr(model, "get_base_model")
        else model
    )

    core.config.use_cache = False

    if hasattr(core, "enable_input_require_grads"):
        core.enable_input_require_grads()

    if hasattr(core, "gradient_checkpointing_enable"):
        try:
            core.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                }
            )
        except TypeError:
            core.gradient_checkpointing_enable()

    print(
        "[model] gradient checkpointing: ENABLED",
        flush=True,
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"[model] trainable {trainable:,}/{total:,} "
        f"({100*trainable/total:.4f}%)",
        flush=True,
    )

    return tok, model, base_name


def build_role_buckets(rows):
    buckets = {
        r: []
        for r in ROLES
    }

    for i, row in enumerate(rows):
        for r in selected(row):
            if r in buckets:
                buckets[r].append(i)

    missing = [
        r
        for r, ids in buckets.items()
        if not ids
    ]

    if missing:
        raise RuntimeError(
            f"roles missing positives in train split: {missing}"
        )

    return buckets


def balanced_epoch_indices(
    buckets,
    n_samples,
    seed,
):
    rng = random.Random(seed)

    role_cycle = []

    while len(role_cycle) < n_samples:
        block = list(ROLES)
        rng.shuffle(block)
        role_cycle.extend(block)

    role_cycle = role_cycle[:n_samples]

    ids = []

    for role in role_cycle:
        ids.append(
            (
                rng.choice(
                    buckets[role]
                ),
                role,
            )
        )

    return ids


def choose_negatives(
    positives,
    n_neg,
    rng,
):
    positives = set(positives)

    ordered = []

    def add(role):
        if (
            role in ROLES
            and role not in positives
            and role not in ordered
        ):
            ordered.append(role)

    for p in positives:
        for role in CONFUSIONS.get(
            p,
            [],
        ):
            add(role)

    for role in GLOBAL_FP:
        add(role)

    rest = [
        r for r in ROLES
        if r not in positives
        and r not in ordered
    ]

    rng.shuffle(rest)

    for role in rest:
        add(role)

    return ordered[:n_neg]


def make_example(
    row,
    n_neg,
    seed,
    focus_role=None,
):
    all_positives = [
        r for r in selected(row)
        if r in ROLES
    ]

    if not all_positives:
        return None

    if focus_role is not None:
        if focus_role not in all_positives:
            raise RuntimeError(
                f"focus role {focus_role!r} is not active "
                f"in row positives={all_positives}"
            )

        # One explicitly balanced positive role per training
        # example. This fixes GPU memory and makes the ranking
        # objective cleaner.
        positives = [
            focus_role
        ]
    else:
        positives = all_positives

    rng = random.Random(seed)

    # CRITICAL:
    # A row can have multiple genuinely active roles.
    # Only the focus role is optimized as the positive for
    # this step, but NONE of the other active roles may ever
    # become a negative.
    negatives = choose_negatives(
        all_positives,
        n_neg,
        rng,
    )

    out = answer(row)

    roles = (
        positives
        + negatives
    )

    texts = [
        bos.sensor_text(
            role,
            ROLE_DESCRIPTIONS[role],
            out,
        )
        for role in roles
    ]

    labels = np.asarray(
        [1] * len(positives)
        + [0] * len(negatives),
        dtype=np.float32,
    )

    pos_weights = np.asarray(
        [
            1.50
            if r in WEAK_POSITIVES
            else 1.0
            for r in positives
        ],
        dtype=np.float32,
    )

    return {
        "texts": texts,
        "labels": labels,
        "n_pos": len(positives),
        "positive_roles": positives,
        "negative_roles": negatives,
        "positive_weights": pos_weights,
    }


def compute_loss(
    logits,
    example,
    margin,
    bce_weight,
    worst_weight,
):
    n_pos = example["n_pos"]

    pos = logits[:n_pos]
    neg = logits[n_pos:]

    if len(neg) == 0:
        raise RuntimeError(
            "example has no negatives"
        )

    pw = torch.tensor(
        example["positive_weights"],
        dtype=torch.float32,
        device=logits.device,
    )

    pair_matrix = F.softplus(
        margin
        - (
            pos.unsqueeze(1)
            - neg.unsqueeze(0)
        )
    )

    rank_loss = (
        pair_matrix
        * pw.unsqueeze(1)
    ).mean()

    targets = torch.tensor(
        example["labels"],
        dtype=torch.float32,
        device=logits.device,
    )

    bce_example_weight = torch.ones_like(
        targets
    )

    bce_example_weight[:n_pos] = pw

    bce_raw = (
        F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
    )

    bce_loss = (
        bce_raw
        * bce_example_weight
    ).mean()

    hardest_negative = torch.max(
        neg
    )

    worst_losses = F.softplus(
        margin
        - pos
        + hardest_negative
    )

    worst_loss = (
        worst_losses
        * pw
    ).mean()

    loss = (
        rank_loss
        + bce_weight * bce_loss
        + worst_weight * worst_loss
    )

    return (
        loss,
        rank_loss.detach(),
        bce_loss.detach(),
        worst_loss.detach(),
    )


def save_adapter_atomic(
    model,
    tok,
    dest,
):
    dest = Path(dest)
    dest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = dest.parent / (
        dest.name
        + ".tmp"
    )

    shutil.rmtree(
        tmp,
        ignore_errors=True,
    )

    model.save_pretrained(
        tmp
    )
    tok.save_pretrained(
        tmp
    )

    shutil.rmtree(
        dest,
        ignore_errors=True,
    )

    os.replace(
        tmp,
        dest,
    )


def save_checkpoint(
    model,
    tok,
    opt,
    scheduler,
    out,
    epoch,
    sample_pos,
    optimizer_steps,
    best_dev,
):
    ckdir = (
        out
        / f"checkpoint_step_{optimizer_steps:06d}"
    )

    ckdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_adapter_atomic(
        model,
        tok,
        ckdir / "adapter",
    )

    state = {
        "epoch":
            int(epoch),
        "sample_pos":
            int(sample_pos),
        "optimizer_steps":
            int(optimizer_steps),
        "best_dev":
            float(best_dev),
        "optimizer":
            opt.state_dict(),
        "scheduler":
            scheduler.state_dict(),
    }

    torch.save(
        state,
        ckdir / "trainer_state.pt",
    )

    (out / "LATEST_CHECKPOINT").write_text(
        str(ckdir.resolve()) + "\n"
    )

    print(
        "[checkpoint]",
        ckdir,
        flush=True,
    )

    return ckdir


def eval_dev(
    model,
    tok,
    dev_rows,
    device,
    args,
):
    print(
        "\n===== DEV EVAL =====",
        flush=True,
    )

    metrics = bos.evaluate(
        model,
        tok,
        dev_rows,
        ROLES,
        dict(ROLE_DESCRIPTIONS),
        device,
        args.max_length,
        args.eval_batch_size,
        args.dev_limit,
        args.seed + 991,
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        ),
        flush=True,
    )

    return metrics


def run_smoke(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "training smoke requires CUDA compute node"
        )

    tok, model, _ = load_model(
        args.initial_adapter
    )

    device = torch.device("cuda")
    model.to(device)
    model.train()

    rows = read_jsonl(
        DATA_ROOT / "train.jsonl"
    )

    row = next(
        x for x in rows
        if selected(x)
    )

    ex = make_example(
        row,
        min(4, args.negatives_per_row),
        args.seed,
    )

    enc = tok(
        ex["texts"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=min(
            args.max_length,
            384,
        ),
    )

    enc = {
        k: v.to(device)
        for k, v in enc.items()
    }

    logits = model(
        **enc
    ).logits.float().view(-1)

    loss, rank, bce, worst = compute_loss(
        logits,
        ex,
        args.margin,
        args.bce_weight,
        args.worst_weight,
    )

    loss.backward()

    print(
        "[smoke] loss=",
        float(loss.detach()),
        "rank=",
        float(rank),
        "bce=",
        float(bce),
        "worst=",
        float(worst),
        flush=True,
    )

    print(
        "[smoke] PASS",
        flush=True,
    )


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; do not run training on login node"
        )

    torch.manual_seed(
        args.seed
    )
    np.random.seed(
        args.seed
    )
    random.seed(
        args.seed
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out = ROOT / args.out
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_rows = read_jsonl(
        DATA_ROOT / "train.jsonl"
    )

    dev_rows = read_jsonl(
        DATA_ROOT / "dev.jsonl"
    )

    print(
        "[data] train/dev:",
        len(train_rows),
        len(dev_rows),
        flush=True,
    )

    buckets = build_role_buckets(
        train_rows
    )

    resume_state = None
    adapter_to_load = Path(
        args.initial_adapter
    )

    if args.resume_dir:
        resume = Path(
            args.resume_dir
        )

        adapter_to_load = (
            resume / "adapter"
        )

        resume_state = torch.load(
            resume / "trainer_state.pt",
            map_location="cpu",
            weights_only=False,
        )

        print(
            "[resume]",
            resume,
            flush=True,
        )

    tok, model, base_name = load_model(
        adapter_to_load
    )

    device = torch.device("cuda")
    model.to(device)

    # --------------------------------------------------------
    # Evaluate the CURRENT GOOD V7 adapter before training.
    # A new checkpoint is allowed to replace it only if DEV
    # within-output ranking actually improves.
    # --------------------------------------------------------
    print("\n===== BASELINE DEV BEFORE TRAINING =====", flush=True)

    baseline_metrics = eval_dev(
        model,
        tok,
        dev_rows,
        device,
        args,
    )

    baseline_score = float(
        baseline_metrics["within_output_auc"]
    )

    atomic_json(
        out / "baseline.json",
        {
            "initial_adapter": str(adapter_to_load),
            "dev_metrics": baseline_metrics,
        },
    )

    save_adapter_atomic(
        model,
        tok,
        out / "best_adapter",
    )

    print(
        f"[baseline] protected DEV within-output AUC="
        f"{baseline_score:.6f}",
        flush=True,
    )

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    opt = AdamW(
        params,
        lr=args.lr,
        weight_decay=0.01,
    )

    opt_steps_per_epoch = math.ceil(
        args.train_rows_per_epoch
        / args.grad_accum
    )

    total_opt_steps = (
        opt_steps_per_epoch
        * args.epochs
    )

    warmup_steps = max(
        1,
        int(
            total_opt_steps
            * args.warmup_ratio
        ),
    )

    scheduler = (
        get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_opt_steps,
        )
    )

    start_epoch = 1
    start_sample_pos = 0
    optimizer_steps = 0
    best_dev = baseline_score

    if resume_state is not None:
        opt.load_state_dict(
            resume_state["optimizer"]
        )

        scheduler.load_state_dict(
            resume_state["scheduler"]
        )

        start_epoch = int(
            resume_state["epoch"]
        )

        start_sample_pos = int(
            resume_state["sample_pos"]
        )

        optimizer_steps = int(
            resume_state["optimizer_steps"]
        )

        best_dev = float(
            resume_state["best_dev"]
        )

    config = {
        "experiment":
            "v7_hardrank_night",
        "initial_adapter":
            str(args.initial_adapter),
        "base_model":
            base_name,
        "query_used":
            False,
        "train_split_only":
            True,
        "dev_for_selection_only":
            True,
        "roles":
            ROLES,
        "weak_positive_roles":
            sorted(WEAK_POSITIVES),
        "global_hard_negative_roles":
            GLOBAL_FP,
        "epochs":
            args.epochs,
        "train_rows_per_epoch":
            args.train_rows_per_epoch,
        "negatives_per_row":
            args.negatives_per_row,
        "max_length":
            args.max_length,
        "lr":
            args.lr,
        "grad_accum":
            args.grad_accum,
        "margin":
            args.margin,
        "bce_weight":
            args.bce_weight,
        "worst_weight":
            args.worst_weight,
        "seed":
            args.seed,
    }

    atomic_json(
        out / "config.json",
        config,
    )

    history = []

    model.train()
    opt.zero_grad(
        set_to_none=True
    )

    running = {
        "loss": 0.0,
        "rank": 0.0,
        "bce": 0.0,
        "worst": 0.0,
        "rows": 0,
    }

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        epoch_ids = balanced_epoch_indices(
            buckets,
            args.train_rows_per_epoch,
            args.seed + 1000 * epoch,
        )

        pos0 = (
            start_sample_pos
            if epoch == start_epoch
            else 0
        )

        print(
            "\n"
            + "=" * 90,
            flush=True,
        )
        print(
            f"EPOCH {epoch}/{args.epochs} "
            f"starting at sample {pos0}",
            flush=True,
        )

        for sample_pos in range(
            pos0,
            len(epoch_ids),
        ):
            (
                row_idx,
                focus_role,
            ) = epoch_ids[
                sample_pos
            ]

            row = train_rows[
                row_idx
            ]

            ex = make_example(
                row,
                args.negatives_per_row,
                args.seed
                + epoch * 10_000_000
                + sample_pos,
                focus_role=focus_role,
            )

            if ex is None:
                continue

            enc = tok(
                ex["texts"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            )

            enc = {
                k: v.to(device)
                for k, v in enc.items()
            }

            logits = model(
                **enc
            ).logits.float().view(-1)

            (
                loss,
                rank_loss,
                bce_loss,
                worst_loss,
            ) = compute_loss(
                logits,
                ex,
                args.margin,
                args.bce_weight,
                args.worst_weight,
            )

            (
                loss / args.grad_accum
            ).backward()

            running["loss"] += float(
                loss.detach()
            )
            running["rank"] += float(
                rank_loss
            )
            running["bce"] += float(
                bce_loss
            )
            running["worst"] += float(
                worst_loss
            )
            running["rows"] += 1

            do_step = (
                running["rows"]
                % args.grad_accum
                == 0
                or sample_pos
                == len(epoch_ids) - 1
            )

            if do_step:
                torch.nn.utils.clip_grad_norm_(
                    params,
                    1.0,
                )

                opt.step()
                scheduler.step()

                opt.zero_grad(
                    set_to_none=True
                )

                optimizer_steps += 1

                if (
                    optimizer_steps
                    % args.log_every
                    == 0
                ):
                    n = max(
                        running["rows"],
                        1,
                    )

                    print(
                        f"[train] epoch={epoch} "
                        f"sample={sample_pos+1}/{len(epoch_ids)} "
                        f"opt={optimizer_steps}/{total_opt_steps} "
                        f"loss={running['loss']/n:.5f} "
                        f"rank={running['rank']/n:.5f} "
                        f"bce={running['bce']/n:.5f} "
                        f"worst={running['worst']/n:.5f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )

                    running = {
                        "loss": 0.0,
                        "rank": 0.0,
                        "bce": 0.0,
                        "worst": 0.0,
                        "rows": 0,
                    }

                if (
                    optimizer_steps
                    % args.checkpoint_every
                    == 0
                ):
                    save_checkpoint(
                        model,
                        tok,
                        opt,
                        scheduler,
                        out,
                        epoch,
                        sample_pos + 1,
                        optimizer_steps,
                        best_dev,
                    )

                if STOP_REQUESTED:
                    save_checkpoint(
                        model,
                        tok,
                        opt,
                        scheduler,
                        out,
                        epoch,
                        sample_pos + 1,
                        optimizer_steps,
                        best_dev,
                    )

                    print(
                        "[signal] emergency checkpoint complete; exiting",
                        flush=True,
                    )

                    return

        start_sample_pos = 0

        # Always persist epoch end before long dev evaluation.
        save_checkpoint(
            model,
            tok,
            opt,
            scheduler,
            out,
            epoch + 1,
            0,
            optimizer_steps,
            best_dev,
        )

        dev_metrics = eval_dev(
            model,
            tok,
            dev_rows,
            device,
            args,
        )

        dev_score = float(
            dev_metrics[
                "within_output_auc"
            ]
        )

        record = {
            "epoch":
                epoch,
            "optimizer_steps":
                optimizer_steps,
            "dev_metrics":
                dev_metrics,
        }

        history.append(
            record
        )

        atomic_json(
            out / "history.json",
            history,
        )

        if dev_score > best_dev:
            best_dev = dev_score

            print(
                f"[best] new dev within-output AUC={best_dev:.6f}",
                flush=True,
            )

            save_adapter_atomic(
                model,
                tok,
                out / "best_adapter",
            )

            atomic_json(
                out / "best.json",
                {
                    "epoch":
                        epoch,
                    "optimizer_steps":
                        optimizer_steps,
                    "dev_metrics":
                        dev_metrics,
                },
            )

        save_adapter_atomic(
            model,
            tok,
            out / "last_adapter",
        )

        model.train()

    atomic_json(
        out / "summary.json",
        {
            "experiment":
                "v7_hardrank_night",
            "best_dev_within_output_auc":
                best_dev,
            "history":
                history,
            "best_adapter":
                str(
                    out / "best_adapter"
                ),
            "last_adapter":
                str(
                    out / "last_adapter"
                ),
            "test_split_used_for_training_or_selection":
                False,
        },
    )

    print(
        "\nTRAINING COMPLETE",
        flush=True,
    )
    print(
        f"best dev within-output AUC = {best_dev:.6f}",
        flush=True,
    )


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--initial-adapter",
        default=str(DEFAULT_ADAPTER),
    )
    ap.add_argument(
        "--out",
        default="artifacts/night_20260926/v7_hardrank",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--train-rows-per-epoch",
        type=int,
        default=8000,
    )
    ap.add_argument(
        "--negatives-per-row",
        type=int,
        default=6,
    )
    ap.add_argument(
        "--max-length",
        type=int,
        default=768,
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1.0e-5,
    )
    ap.add_argument(
        "--grad-accum",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--margin",
        type=float,
        default=1.0,
    )
    ap.add_argument(
        "--bce-weight",
        type=float,
        default=0.15,
    )
    ap.add_argument(
        "--worst-weight",
        type=float,
        default=0.50,
    )
    ap.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
    )
    ap.add_argument(
        "--dev-limit",
        type=int,
        default=256,
    )
    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=20,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260926,
    )
    ap.add_argument(
        "--resume-dir",
        default=None,
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
    )

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.smoke:
        run_smoke(args)
    else:
        train(args)
