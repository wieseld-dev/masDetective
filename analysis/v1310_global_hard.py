#!/usr/bin/env python3

import argparse
import gc
import json
import math
import random
import shutil
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import analysis.train_v13_6b_activation_judge as tr
import analysis.train_v13_8b_balanced_repair as old
import analysis.build_visible_realized_gain_v3 as vg
import analysis.v139_overnight as v139


BASE = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/best_adapter"
)

MINE_OUT = Path(
    "artifacts/v13_10_global_hard_v1/"
    "hardneg_train.json"
)

ROLES = list(tr.ROLE_DESCRIPTIONS)
RD = dict(tr.ROLE_DESCRIPTIONS)

SEED = 1310001

assert len(ROLES) == 20, len(ROLES)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
    )


def worker_examples(split):
    rows = [
        x for x in v139.load_worker_records()
        if x["split"] == split
    ]

    out = []

    for x in rows:
        sim = SequenceMatcher(
            None,
            x["output_a"],
            x["output_b"],
        ).ratio()

        # Same observability weighting used in V13.9:
        # never throw weak examples away completely.
        obs_weight = (
            0.25
            + 0.75
            * min(
                1.0,
                max(
                    0.0,
                    (1.0 - sim) / 0.50,
                ),
            )
        )

        out.append({
            "query_id": x["query_id"],
            "query": x["query"],
            "answer": x["output_a"],
            "true_role": x["role_a"],
            "paired_other": x["role_b"],
            "stratum": x["semantic_stratum"],
            "obs_weight": obs_weight,
        })

        out.append({
            "query_id": x["query_id"],
            "query": x["query"],
            "answer": x["output_b"],
            "true_role": x["role_b"],
            "paired_other": x["role_a"],
            "stratum": x["semantic_stratum"],
            "obs_weight": obs_weight,
        })

    return out


def cmd_mine():
    train = worker_examples("train")

    assert len(train) == 1440, len(train)

    tok, model, dtype = vg.load_sensor(BASE)

    mined = []

    top1_ok = 0
    pair_ok = 0
    top3_ok = 0

    hard_counts = Counter()
    source_counts = Counter()

    for i, x in enumerate(train, 1):
        scores = vg.sensor_score_row(
            model=model,
            tok=tok,
            dtype=dtype,
            roles=ROLES,
            role_desc=RD,
            query=x["query"],
            answer=x["answer"],
            max_length=4096,
        )

        score_map = {
            r: float(scores[j])
            for j, r in enumerate(ROLES)
        }

        order = sorted(
            ROLES,
            key=lambda r: score_map[r],
            reverse=True,
        )

        true = x["true_role"]
        paired = x["paired_other"]

        false_order = [
            r for r in order
            if r != true
        ]

        # Static hard pool saved for diagnostics.
        top8 = false_order[:8]

        # Training set:
        # strongest 3 global distractors
        # + original paired competitor if absent.
        selected = list(false_order[:3])

        if paired not in selected:
            selected.append(paired)

        # Should be 3 or 4 unique negatives.
        selected = list(dict.fromkeys(selected))

        true_rank = order.index(true) + 1

        top1_ok += int(true_rank == 1)
        top3_ok += int(true_rank <= 3)
        pair_ok += int(
            score_map[true] > score_map[paired]
        )

        source_counts[true] += 1

        for r in selected:
            hard_counts[r] += 1

        mined.append({
            **x,
            "true_score": score_map[true],
            "paired_score": score_map[paired],
            "true_rank_baseline": true_rank,
            "baseline_top5": order[:5],
            "top8_false_roles": top8,
            "selected_negatives": selected,
            "selected_negative_scores": {
                r: score_map[r]
                for r in selected
            },
        })

        if i % 50 == 0 or i == len(train):
            print(
                f"MINED {i}/{len(train)}",
                flush=True,
            )

    report = {
        "protocol":
            "v1310_train_only_global_hardneg_v1",
        "source_adapter": str(BASE),
        "split": "TRAIN_ONLY",
        "n_outputs": len(mined),
        "candidate_roles": ROLES,
        "baseline_train_top1":
            top1_ok / len(mined),
        "baseline_train_top3":
            top3_ok / len(mined),
        "baseline_train_pairwise":
            pair_ok / len(mined),
        "source_counts": dict(source_counts),
        "missing_true_source_roles": [
            r for r in ROLES
            if source_counts[r] == 0
        ],
        "selected_negative_counts":
            dict(hard_counts),
        "records": mined,
    }

    dump(MINE_OUT, report)

    print("\n" + "=" * 100)
    print("MINING COMPLETE")
    print("TRAIN outputs:", len(mined))
    print(
        "baseline global Top1:",
        report["baseline_train_top1"],
    )
    print(
        "baseline global Top3:",
        report["baseline_train_top3"],
    )
    print(
        "baseline pairwise:",
        report["baseline_train_pairwise"],
    )
    print(
        "missing true-source roles:",
        report["missing_true_source_roles"],
    )

    print("\nTOP NEGATIVE ROLES:")
    for r, n in hard_counts.most_common():
        print(f"{r:<28} {n}")

    del model, tok
    gc.collect()


def eval_global(
    model,
    tok,
    dtype,
    examples,
    max_length=4096,
):
    import torch

    model.eval()

    rows = []

    with torch.no_grad():
        for i, x in enumerate(examples, 1):
            scores = vg.sensor_score_row(
                model=model,
                tok=tok,
                dtype=dtype,
                roles=ROLES,
                role_desc=RD,
                query=x["query"],
                answer=x["answer"],
                max_length=max_length,
            )

            score_map = {
                r: float(scores[j])
                for j, r in enumerate(ROLES)
            }

            ranked = sorted(
                ROLES,
                key=lambda r: score_map[r],
                reverse=True,
            )

            rank = ranked.index(
                x["true_role"]
            ) + 1

            rows.append({
                "true_role": x["true_role"],
                "paired_other":
                    x["paired_other"],
                "rank": rank,
                "top1": ranked[0],
                "pairwise_correct":
                    score_map[x["true_role"]]
                    > score_map[x["paired_other"]],
            })

    ranks = np.asarray(
        [x["rank"] for x in rows],
        dtype=float,
    )

    result = {
        "n": len(rows),
        "top1": float(
            np.mean(ranks == 1)
        ),
        "top3": float(
            np.mean(ranks <= 3)
        ),
        "mrr": float(
            np.mean(1.0 / ranks)
        ),
        "mean_rank": float(
            np.mean(ranks)
        ),
        "pairwise_accuracy": float(
            np.mean([
                x["pairwise_correct"]
                for x in rows
            ])
        ),
        "by_true_role": {},
    }

    for role in ROLES:
        rr = [
            x for x in rows
            if x["true_role"] == role
        ]

        if not rr:
            continue

        rranks = np.asarray(
            [x["rank"] for x in rr],
            dtype=float,
        )

        result["by_true_role"][role] = {
            "n": len(rr),
            "top1": float(
                np.mean(rranks == 1)
            ),
            "top3": float(
                np.mean(rranks <= 3)
            ),
            "mean_rank": float(
                np.mean(rranks)
            ),
        }

    return result


def old_pair_item(x):
    return {
        "kind": "old",
        "pos_text": x["pair"].pos_text,
        "neg_text": x["pair"].neg_text,
    }


def hard_item(x):
    return {
        "kind": "hard",
        "query": x["query"],
        "answer": x["answer"],
        "true_role": x["true_role"],
        "neg_roles": x["selected_negatives"],
    }


def cmd_train(args):
    import torch
    import torch.nn.functional as F
    from transformers import (
        get_cosine_schedule_with_warmup,
    )

    if not MINE_OUT.exists():
        raise FileNotFoundError(MINE_OUT)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    mined_obj = json.loads(
        MINE_OUT.read_text()
    )

    mined = mined_obj["records"]

    if len(mined) != 1440:
        raise RuntimeError(
            f"expected 1440 mined train outputs, "
            f"got {len(mined)}"
        )

    val = worker_examples("val")
    assert len(val) == 216, len(val)

    # Old rehearsal / regression data.
    (
        rd_old,
        train_cases,
        cal_cases,
        _,
        _,
    ) = old.split_data()

    _, old_train, _ = old.build_meta(
        train_cases,
        rd_old,
    )

    _, old_cal, _ = old.build_meta(
        cal_cases,
        rd_old,
    )

    old_weights, _, _ = old.sampler_weights(
        old_train
    )

    # Equal source-role mass among roles represented
    # in the causal corpus, while retaining the
    # observability soft weight.
    source_counts = Counter(
        x["true_role"] for x in mined
    )

    hard_weights = [
        float(x["obs_weight"])
        / source_counts[x["true_role"]]
        for x in mined
    ]

    out = Path(args.outdir)

    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"Refusing non-empty output dir: {out}"
        )

    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")

    tok, model, dtype = (
        v139.load_model_trainable(BASE)
    )

    eval_args = SimpleNamespace(
        eval_batch_size=1,
        max_length=2048,
    )

    print("\n===== BASELINE GLOBAL VAL =====")
    base_global = eval_global(
        model,
        tok,
        dtype,
        val,
    )
    print(
        json.dumps(
            base_global,
            indent=2,
        )
    )

    print("\n===== BASELINE OLD CAL =====")
    base_old = old.eval_meta(
        model,
        tok,
        old_cal,
        eval_args,
        device,
        dtype,
    )

    print(
        "old selection:",
        base_old["selection_score"],
    )
    print(
        "same-role:",
        base_old["same_role_cf_role_macro"],
    )
    print(
        "target-abs:",
        base_old["target_absent_role_macro"],
    )
    print(
        "hard-neighbor:",
        base_old["hard_neighbor_role_macro"],
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

    hard_n = int(
        round(
            args.samples_per_epoch
            * args.hard_mass
        )
    )
    old_n = (
        args.samples_per_epoch
        - hard_n
    )

    steps_per_epoch = math.ceil(
        args.samples_per_epoch
        / args.grad_accum
    )

    total_steps = (
        steps_per_epoch
        * args.epochs
    )

    sched = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=max(
            10,
            int(0.05 * total_steps),
        ),
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(dtype == torch.float16),
    )

    history = []

    best_top1 = base_global["top1"]
    best_mrr = base_global["mrr"]
    best = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        rng = random.Random(
            SEED + epoch
        )

        hard_epoch = [
            hard_item(x)
            for x in rng.choices(
                mined,
                weights=hard_weights,
                k=hard_n,
            )
        ]

        old_epoch = [
            old_pair_item(x)
            for x in rng.choices(
                old_train,
                weights=old_weights,
                k=old_n,
            )
        ]

        items = hard_epoch + old_epoch
        rng.shuffle(items)

        model.train()
        opt.zero_grad(set_to_none=True)

        loss_sum = 0.0
        hard_seen = 0
        old_seen = 0

        for step, x in enumerate(
            items,
            1,
        ):
            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                if x["kind"] == "hard":
                    true = x["true_role"]
                    negs = x["neg_roles"]

                    texts = [
                        tr.presence_text(
                            true,
                            RD[true],
                            true,
                            x["query"],
                            x["answer"],
                        )
                    ]

                    texts += [
                        tr.presence_text(
                            r,
                            RD[r],
                            r,
                            x["query"],
                            x["answer"],
                        )
                        for r in negs
                    ]

                    z = old.forward_logits(
                        model,
                        tok,
                        texts,
                        2048,
                        device,
                    )

                    pos = z[0]
                    nz = z[1:]

                    diff = pos - nz

                    rank_mean = F.softplus(
                        args.margin - diff
                    ).mean()

                    hardest = nz.max()

                    rank_max = F.softplus(
                        args.margin
                        - (pos - hardest)
                    )

                    bce_pos = (
                        F.binary_cross_entropy_with_logits(
                            pos.view(1),
                            torch.ones(
                                1,
                                device=device,
                            ),
                        )
                    )

                    bce_neg = (
                        F.binary_cross_entropy_with_logits(
                            nz,
                            torch.zeros_like(nz),
                        )
                    )

                    # Preserve independent activation semantics;
                    # ranking only augments BCE.
                    loss = (
                        0.60 * rank_max
                        + 0.40 * rank_mean
                        + 0.15 * 0.5
                        * (bce_pos + bce_neg)
                    )

                    hard_seen += 1

                else:
                    z = old.forward_logits(
                        model,
                        tok,
                        [
                            x["pos_text"],
                            x["neg_text"],
                        ],
                        2048,
                        device,
                    )

                    pos = z[0]
                    neg = z[1]

                    rank = F.softplus(
                        args.margin
                        - (pos - neg)
                    )

                    bce = 0.5 * (
                        F.binary_cross_entropy_with_logits(
                            pos.view(1),
                            torch.ones(
                                1,
                                device=device,
                            ),
                        )
                        +
                        F.binary_cross_entropy_with_logits(
                            neg.view(1),
                            torch.zeros(
                                1,
                                device=device,
                            ),
                        )
                    )

                    loss = (
                        rank
                        + 0.15 * bce
                    )

                    old_seen += 1

                scaled = (
                    loss
                    / args.grad_accum
                )

            scaler.scale(
                scaled
            ).backward()

            loss_sum += float(
                loss.detach().cpu()
            )

            if (
                step % args.grad_accum == 0
                or step == len(items)
            ):
                scaler.unscale_(opt)

                torch.nn.utils.clip_grad_norm_(
                    params,
                    1.0,
                )

                scaler.step(opt)
                scaler.update()

                opt.zero_grad(
                    set_to_none=True
                )

                sched.step()

            if step % 250 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={step}/{len(items)} "
                    f"mean_loss="
                    f"{loss_sum/step:.4f}",
                    flush=True,
                )

        print(
            f"\n===== EPOCH {epoch} GLOBAL VAL =====",
            flush=True,
        )

        global_val = eval_global(
            model,
            tok,
            dtype,
            val,
        )

        old_val = old.eval_meta(
            model,
            tok,
            old_cal,
            eval_args,
            device,
            dtype,
        )

        old_ok = (
            old_val["selection_score"]
            >= base_old["selection_score"]
            - 0.020
        )

        for key, tol in (
            (
                "same_role_cf_role_macro",
                0.030,
            ),
            (
                "target_absent_role_macro",
                0.030,
            ),
            (
                "hard_neighbor_role_macro",
                0.030,
            ),
        ):
            if (
                old_val[key]
                < base_old[key] - tol
            ):
                old_ok = False

        ranking_ok = (
            global_val["pairwise_accuracy"]
            >= base_global["pairwise_accuracy"]
            - 0.020
            and
            global_val["top3"]
            >= base_global["top3"]
            - 0.020
            and
            global_val["mrr"]
            >= base_global["mrr"]
            - 0.010
        )

        eligible = (
            old_ok
            and ranking_ok
        )

        rec = {
            "epoch": epoch,
            "train_mean_loss":
                loss_sum / len(items),
            "hard_seen": hard_seen,
            "old_seen": old_seen,
            "global_val": global_val,
            "old_val": {
                "selection_score":
                    old_val[
                        "selection_score"
                    ],
                "same_role_cf_role_macro":
                    old_val[
                        "same_role_cf_role_macro"
                    ],
                "target_absent_role_macro":
                    old_val[
                        "target_absent_role_macro"
                    ],
                "hard_neighbor_role_macro":
                    old_val[
                        "hard_neighbor_role_macro"
                    ],
            },
            "eligible": eligible,
        }

        history.append(rec)

        print(
            json.dumps(
                rec,
                indent=2,
            ),
            flush=True,
        )

        top1 = global_val["top1"]
        mrr = global_val["mrr"]

        improved = (
            top1 > best_top1 + 1e-12
            or (
                abs(
                    top1 - best_top1
                ) <= 1e-12
                and mrr > best_mrr
            )
        )

        if eligible and improved:
            best_top1 = top1
            best_mrr = mrr

            best = {
                "epoch": epoch,
                "top1": top1,
                "top3":
                    global_val["top3"],
                "mrr": mrr,
                "pairwise_accuracy":
                    global_val[
                        "pairwise_accuracy"
                    ],
                "old_selection_score":
                    old_val[
                        "selection_score"
                    ],
            }

            bp = out / "best_adapter"
            shutil.rmtree(
                bp,
                ignore_errors=True,
            )

            model.save_pretrained(bp)
            tok.save_pretrained(bp)

            print(
                "NEW BEST:",
                json.dumps(
                    best,
                    indent=2,
                ),
                flush=True,
            )

    summary = {
        "protocol":
            "v13_10_global_hard_multineg_v1",
        "initial_adapter": str(BASE),
        "hardneg_artifact":
            str(MINE_OUT),
        "hard_mass": args.hard_mass,
        "samples_per_epoch":
            args.samples_per_epoch,
        "epochs": args.epochs,
        "lr": args.lr,
        "baseline_global_val":
            base_global,
        "baseline_old": {
            "selection_score":
                base_old[
                    "selection_score"
                ],
            "same_role_cf_role_macro":
                base_old[
                    "same_role_cf_role_macro"
                ],
            "target_absent_role_macro":
                base_old[
                    "target_absent_role_macro"
                ],
            "hard_neighbor_role_macro":
                base_old[
                    "hard_neighbor_role_macro"
                ],
        },
        "history": history,
        "best": best,
        "test_used": False,
        "api_calls": 0,
        "mas_calls": 0,
    }

    dump(
        out / "summary.json",
        summary,
    )

    print("\n" + "=" * 100)
    print("V13.10 COMPLETE")
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(
        dest="cmd",
        required=True,
    )

    sub.add_parser("mine")

    p = sub.add_parser("train")

    p.add_argument(
        "--outdir",
        default=
        "artifacts/v13_10_global_hard_v1/run1",
    )
    p.add_argument(
        "--hard-mass",
        type=float,
        default=0.40,
    )
    p.add_argument(
        "--samples-per-epoch",
        type=int,
        default=2048,
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=2,
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=8,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=2e-6,
    )
    p.add_argument(
        "--margin",
        type=float,
        default=1.0,
    )

    args = ap.parse_args()

    if args.cmd == "mine":
        cmd_mine()
    else:
        cmd_train(args)


if __name__ == "__main__":
    main()
