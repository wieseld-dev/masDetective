#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from v11.role_universe import TRAIN_ROLES


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    path = Path(path)

    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def render_input(row):
    rp = row["role_probs_before"]
    kp = row["k_probs_before"]

    role_lines = []

    for role in TRAIN_ROLES:
        p = float(rp.get(role, 0.0))

        role_lines.append(
            f"{role}: {p:.4f}"
        )

    k_lines = []

    for k in sorted(
        kp,
        key=lambda z: int(z)
    ):
        k_lines.append(
            f"K={k}: {float(kp[k]):.4f}"
        )

    return (
        "You are estimating the expected information utility "
        "of ONE candidate benign black-box probe BEFORE it is executed.\n\n"

        "The hidden application contains an unknown subset of "
        "specialist capabilities. Only public observations from "
        "previous probes are represented in the current beliefs below.\n\n"

        "CURRENT ROLE BELIEFS\n"
        + "\n".join(role_lines)
        + "\n\nCURRENT K BELIEF\n"
        + "\n".join(k_lines)
        + "\n\nROUND\n"
        + str(int(row.get("round", 1)))
        + "\n\nCANDIDATE QUERY\n"
        + str(row["query"]).strip()
        + "\n\nEstimate the candidate's information utility."
    )


def group_rows(rows):
    g = defaultdict(list)

    for x in rows:
        g[str(x["state_id"])].append(x)

    out = []

    for sid, xs in sorted(g.items()):
        xs = sorted(
            xs,
            key=lambda z:
                int(z["candidate_index"]),
        )

        if len(xs) >= 2:
            out.append(
                (sid, xs)
            )

    return out


def split_train_by_system(
    rows,
    val_fraction,
    seed,
):
    systems = sorted({
        str(x["system_id"])
        for x in rows
    })

    rng = random.Random(seed)
    rng.shuffle(systems)

    n_val = max(
        1,
        int(round(
            len(systems)
            * val_fraction
        )),
    )

    val_systems = set(
        systems[:n_val]
    )

    tr = [
        x for x in rows
        if str(x["system_id"])
        not in val_systems
    ]

    va = [
        x for x in rows
        if str(x["system_id"])
        in val_systems
    ]

    return tr, va, sorted(val_systems)


def stats_target(rows, key):
    a = np.asarray(
        [float(x[key]) for x in rows],
        dtype=np.float64,
    )

    return (
        float(a.mean()),
        max(
            float(a.std()),
            1e-8,
        ),
    )


def standardized_targets(
    rows,
    role_mu,
    role_sd,
    k_mu,
    k_sd,
    device,
):
    yr = torch.tensor(
        [
            (
                float(x[
                    "role_brier_gain_training_only"
                ])
                - role_mu
            ) / role_sd
            for x in rows
        ],
        dtype=torch.float32,
        device=device,
    )

    yk = torch.tensor(
        [
            (
                float(x[
                    "k_brier_gain_training_only"
                ])
                - k_mu
            ) / k_sd
            for x in rows
        ],
        dtype=torch.float32,
        device=device,
    )

    return yr, yk


def listwise_loss(
    pred,
    target,
    target_tau,
    pred_tau,
):
    p_star = torch.softmax(
        target / target_tau,
        dim=0,
    )

    log_p = torch.log_softmax(
        pred / pred_tau,
        dim=0,
    )

    return -(
        p_star * log_p
    ).sum()


def pairwise_loss(pred, target):
    vals = []

    n = len(pred)

    for i in range(n):
        for j in range(i + 1, n):
            d = target[i] - target[j]

            if abs(float(d.detach())) < 1e-6:
                continue

            sign = torch.sign(d)

            vals.append(
                F.softplus(
                    -sign
                    * (pred[i] - pred[j])
                )
            )

    if not vals:
        return pred.sum() * 0.0

    return torch.stack(vals).mean()


def head_loss(
    pred,
    target,
    target_tau,
    pred_tau,
):
    lw = listwise_loss(
        pred,
        target,
        target_tau,
        pred_tau,
    )

    pw = pairwise_loss(
        pred,
        target,
    )

    hub = F.smooth_l1_loss(
        pred,
        target,
        beta=0.5,
    )

    total = (
        lw
        + 0.5 * pw
        + 0.2 * hub
    )

    return total, {
        "listwise": lw,
        "pairwise": pw,
        "huber": hub,
    }


def rankdata(a):
    order = np.argsort(a)
    ranks = np.empty(
        len(a),
        dtype=np.float64,
    )

    ranks[order] = np.arange(
        len(a),
        dtype=np.float64,
    )

    return ranks


def safe_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        len(a) < 2
        or np.std(a) < 1e-12
        or np.std(b) < 1e-12
    ):
        return None

    ra = rankdata(a)
    rb = rankdata(b)

    v = np.corrcoef(
        ra,
        rb,
    )[0, 1]

    if np.isnan(v):
        return None

    return float(v)


def pair_accuracy(pred, truth):
    c = 0.0
    n = 0

    for i in range(len(pred)):
        for j in range(i + 1, len(pred)):
            dy = truth[i] - truth[j]

            if abs(dy) < 1e-12:
                continue

            dp = pred[i] - pred[j]

            if dp * dy > 0:
                c += 1.0
            elif abs(dp) < 1e-12:
                c += 0.5

            n += 1

    return (
        c / n
        if n
        else None
    )


@torch.no_grad()
def score_rows(
    model,
    tok,
    rows,
    max_length,
    batch_size,
):
    model.eval()

    out = []

    texts = [
        render_input(x)
        for x in rows
    ]

    for s in range(
        0,
        len(rows),
        batch_size,
    ):
        batch = texts[
            s:s + batch_size
        ]

        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        enc = {
            k: v.to("cuda")
            for k, v in enc.items()
        }

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            logits = model(
                **enc
            ).logits.float()

        out.append(
            logits.cpu().numpy()
        )

    return np.concatenate(
        out,
        axis=0,
    )


def evaluate_head(
    rows,
    pred,
    target_key,
):
    by_state = defaultdict(list)

    for i, x in enumerate(rows):
        by_state[
            str(x["state_id"])
        ].append(i)

    selected = []
    random_gain = []
    oracle = []
    regret = []
    hit = []
    top2 = []
    rhos = []
    pair_acc = []

    for sid, idx in by_state.items():
        yy = np.asarray(
            [
                float(rows[i][target_key])
                for i in idx
            ],
            dtype=float,
        )

        pp = np.asarray(
            [float(pred[i]) for i in idx],
            dtype=float,
        )

        im = int(
            np.argmax(pp)
        )

        io = int(
            np.argmax(yy)
        )

        order = np.argsort(
            -pp
        )

        mg = float(yy[im])
        og = float(yy[io])

        selected.append(mg)
        random_gain.append(
            float(np.mean(yy))
        )
        oracle.append(og)
        regret.append(
            og - mg
        )
        hit.append(
            float(
                mg >= og - 1e-12
            )
        )
        top2.append(
            float(
                io in set(
                    order[:min(2, len(order))]
                )
            )
        )

        rho = safe_spearman(
            pp,
            yy,
        )

        if rho is not None:
            rhos.append(rho)

        pa = pair_accuracy(
            pp,
            yy,
        )

        if pa is not None:
            pair_acc.append(pa)

    def mean(xs):
        return (
            float(np.mean(xs))
            if xs
            else None
        )

    om = mean(oracle)
    rm = mean(random_gain)
    sm = mean(selected)

    denom = (
        om - rm
        if om is not None
        and rm is not None
        else 0.0
    )

    captured = (
        (sm - rm) / denom
        if abs(denom) > 1e-12
        else None
    )

    return {
        "states":
            len(by_state),

        "mean_selected_gain":
            sm,

        "mean_random_gain":
            rm,

        "mean_oracle_gain":
            om,

        "mean_regret":
            mean(regret),

        "oracle_top1_hit":
            mean(hit),

        "oracle_top2_hit":
            mean(top2),

        "mean_spearman":
            mean(rhos),

        "mean_pair_accuracy":
            mean(pair_acc),

        "oracle_signal_captured":
            captured,
    }


def evaluate(
    model,
    tok,
    rows,
    max_length,
    eval_batch_size,
    role_mu,
    role_sd,
    k_mu,
    k_sd,
):
    z = score_rows(
        model,
        tok,
        rows,
        max_length,
        eval_batch_size,
    )

    role_pred = (
        z[:, 0]
        * role_sd
        + role_mu
    )

    k_pred = (
        z[:, 1]
        * k_sd
        + k_mu
    )

    return {
        "role":
            evaluate_head(
                rows,
                role_pred,
                "role_brier_gain_training_only",
            ),

        "K":
            evaluate_head(
                rows,
                k_pred,
                "k_brier_gain_training_only",
            ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--train",
        default=(
            "artifacts/"
            "visible_realized_gain_v3/"
            "train.jsonl"
        ),
    )

    ap.add_argument(
        "--challenge",
        default=(
            "artifacts/"
            "visible_realized_gain_v3/"
            "val.jsonl"
        ),
    )

    ap.add_argument(
        "--output-dir",
        default=(
            "artifacts/"
            "llm_selector_v1"
        ),
    )

    ap.add_argument(
        "--model-name",
        default=(
            "Qwen/"
            "Qwen2.5-3B-Instruct"
        ),
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=6,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    ap.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    ap.add_argument(
        "--grad-acc",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.20,
    )

    ap.add_argument(
        "--lambda-k",
        type=float,
        default=0.50,
    )

    ap.add_argument(
        "--target-tau",
        type=float,
        default=0.50,
    )

    ap.add_argument(
        "--pred-tau",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--patience",
        type=int,
        default=2,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=1397,
    )

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA required"
        )

    seed_all(
        args.seed
    )

    out = Path(
        args.output_dir
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = read_jsonl(
        args.train
    )

    challenge = read_jsonl(
        args.challenge
    )

    tr, va, val_systems = \
        split_train_by_system(
            source,
            args.val_fraction,
            args.seed,
        )

    tr_groups = group_rows(tr)
    va_groups = group_rows(va)

    role_mu, role_sd = \
        stats_target(
            tr,
            "role_brier_gain_training_only",
        )

    k_mu, k_sd = \
        stats_target(
            tr,
            "k_brier_gain_training_only",
        )

    print(
        "=== LLM SELECTOR V1 ===",
        flush=True,
    )

    print(
        f"TRAIN rows={len(tr)} "
        f"groups={len(tr_groups)} "
        f"systems="
        f"{len(set(x['system_id'] for x in tr))}",
        flush=True,
    )

    print(
        f"VAL rows={len(va)} "
        f"groups={len(va_groups)} "
        f"systems="
        f"{len(set(x['system_id'] for x in va))}",
        flush=True,
    )

    print(
        f"HISTORICAL_CHALLENGE "
        f"rows={len(challenge)} "
        f"groups={len(group_rows(challenge))}",
        flush=True,
    )

    print(
        "role target "
        f"mu={role_mu:+.6f} "
        f"sd={role_sd:.6f}",
        flush=True,
    )

    print(
        "K target "
        f"mu={k_mu:+.6f} "
        f"sd={k_sd:.6f}",
        flush=True,
    )

    write_json(
        out / "split_manifest.json",
        {
            "seed":
                args.seed,

            "split_unit":
                "system_id",

            "val_fraction":
                args.val_fraction,

            "val_systems":
                val_systems,

            "train_rows":
                len(tr),

            "val_rows":
                len(va),

            "challenge_rows":
                len(challenge),

            "internal_test_used":
                False,
        },
    )

    tok = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    if tok.pad_token_id is None:
        tok.pad_token = \
            tok.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=
            torch.bfloat16,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.model_name,
            num_labels=2,
            quantization_config=quant,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )
    )

    model.config.pad_token_id = \
        tok.pad_token_id

    model.config.use_cache = False

    model = \
        prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        modules_to_save=[
            "score",
        ],
    )

    model = get_peft_model(
        model,
        lora,
    )

    model.print_trainable_parameters()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=
            args.weight_decay,
    )

    best_state = None
    best_epoch = None
    best_role_regret = float("inf")
    bad = 0
    history = []

    t0 = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        shuffled = list(
            tr_groups
        )

        random.Random(
            args.seed
            + epoch * 1009
        ).shuffle(shuffled)

        opt.zero_grad(
            set_to_none=True
        )

        running = []
        steps = 0

        for gi, (
            sid,
            rows,
        ) in enumerate(
            shuffled,
            1,
        ):
            texts = [
                render_input(x)
                for x in rows
            ]

            enc = tok(
                texts,
                padding=True,
                truncation=True,
                max_length=
                    args.max_length,
                return_tensors="pt",
            )

            enc = {
                k: v.to("cuda")
                for k, v in enc.items()
            }

            yr, yk = \
                standardized_targets(
                    rows,
                    role_mu,
                    role_sd,
                    k_mu,
                    k_sd,
                    "cuda",
                )

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(
                    **enc
                ).logits.float()

                pr = logits[:, 0]
                pk = logits[:, 1]

                lr, _ = head_loss(
                    pr,
                    yr,
                    args.target_tau,
                    args.pred_tau,
                )

                lk, _ = head_loss(
                    pk,
                    yk,
                    args.target_tau,
                    args.pred_tau,
                )

                loss = (
                    lr
                    + args.lambda_k * lk
                )

            (
                loss
                / args.grad_acc
            ).backward()

            running.append(
                float(
                    loss.detach().cpu()
                )
            )

            if (
                gi % args.grad_acc == 0
                or gi == len(shuffled)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                )

                opt.step()

                opt.zero_grad(
                    set_to_none=True
                )

                steps += 1

        val_metrics = evaluate(
            model,
            tok,
            va,
            args.max_length,
            args.eval_batch_size,
            role_mu,
            role_sd,
            k_mu,
            k_sd,
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                float(
                    np.mean(running)
                ),

            "optimizer_steps":
                steps,

            "val":
                val_metrics,

            "elapsed_hours":
                (
                    time.time() - t0
                ) / 3600.0,
        }

        history.append(row)

        rr = (
            val_metrics[
                "role"
            ][
                "mean_regret"
            ]
        )

        print(
            f"epoch={epoch} "
            f"train={row['train_loss']:.4f} "
            f"VAL role_regret={rr:.6f} "
            f"role_hit="
            f"{val_metrics['role']['oracle_top1_hit']:.3f} "
            f"role_rho="
            f"{val_metrics['role']['mean_spearman']:.3f} "
            f"K_regret="
            f"{val_metrics['K']['mean_regret']:.6f}",
            flush=True,
        )

        if (
            rr
            < best_role_regret
            - 1e-8
        ):
            best_role_regret = rr
            best_epoch = epoch

            state = \
                get_peft_model_state_dict(
                    model
                )

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v
                in state.items()
            }

            bad = 0

            print(
                f"[BEST] epoch={epoch} "
                f"role_regret={rr:.6f}",
                flush=True,
            )
        else:
            bad += 1

            if (
                bad
                >= args.patience
            ):
                print(
                    "EARLY STOP",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError(
            "No best checkpoint"
        )

    set_peft_model_state_dict(
        model,
        best_state,
    )

    best_dir = (
        out / "best_adapter"
    )

    best_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        best_dir
    )

    tok.save_pretrained(
        best_dir
    )

    final_val = evaluate(
        model,
        tok,
        va,
        args.max_length,
        args.eval_batch_size,
        role_mu,
        role_sd,
        k_mu,
        k_sd,
    )

    challenge_metrics = evaluate(
        model,
        tok,
        challenge,
        args.max_length,
        args.eval_batch_size,
        role_mu,
        role_sd,
        k_mu,
        k_sd,
    )

    summary = {
        "version":
            "llm_selector_v1",

        "model":
            args.model_name,

        "best_epoch":
            best_epoch,

        "selection_metric":
            "VAL role mean_regret",

        "role_target":
            "role_brier_gain_training_only",

        "k_target":
            "k_brier_gain_training_only",

        "loss":
            (
                "role(listwise + "
                "0.5 pairwise + "
                "0.2 Huber) + "
                "lambda_k * "
                "K(same)"
            ),

        "lambda_k":
            args.lambda_k,

        "target_stats": {
            "role_mu":
                role_mu,
            "role_sd":
                role_sd,
            "k_mu":
                k_mu,
            "k_sd":
                k_sd,
        },

        "train": {
            "rows":
                len(tr),
            "states":
                len(tr_groups),
            "systems":
                len(set(
                    x["system_id"]
                    for x in tr
                )),
        },

        "val": {
            "rows":
                len(va),
            "states":
                len(va_groups),
            "systems":
                len(set(
                    x["system_id"]
                    for x in va
                )),
            "metrics":
                final_val,
        },

        "historical_challenge": {
            "note":
                (
                    "Existing visible_realized_gain_v3/"
                    "val split. Not used for "
                    "training or model selection. "
                    "Not treated as untouched final TEST."
                ),
            "rows":
                len(challenge),
            "states":
                len(group_rows(
                    challenge
                )),
            "metrics":
                challenge_metrics,
        },

        "internal_test_used":
            False,

        "runtime_hours":
            (
                time.time() - t0
            ) / 3600.0,
    }

    write_json(
        out / "summary.json",
        summary,
    )

    write_json(
        out / "training_history.json",
        history,
    )

    print()
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
