from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOTDIR))

import numpy as np

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_curriculum.hashguard import tree_sha256
from adaptive_v11_curriculum.policy import CurriculumAdaptiveProbePolicyV11


EXPECTED_PREF_SHA = (
    "11829d804c9f7519cec76f50aab173d4"
    "b73615cccac411a81425ae3cb4b59498"
)


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def balanced_positive_order(n_pos, n_slots, seed):
    """
    Produce exactly n_slots positive indices while keeping
    exposures as balanced across the 17 Golds as possible.

    With 122 slots and 17 Golds:
      14 Golds appear 7 times
       3 Golds appear 8 times.

    Which Golds get the extra exposure rotates deterministically
    with the epoch seed.
    """
    base = n_slots // n_pos
    rem = n_slots % n_pos

    order = []

    for i in range(n_pos):
        order.extend([i] * base)

    extra_candidates = list(range(n_pos))
    random.Random(seed + 100000).shuffle(extra_candidates)

    order.extend(extra_candidates[:rem])

    assert len(order) == n_slots

    random.Random(seed + 200000).shuffle(order)

    counts = Counter(order)

    assert max(counts.values()) - min(counts.values()) <= 1

    return order


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--positives",
        default="artifacts/v4_training_ready/positives.jsonl",
    )

    ap.add_argument(
        "--preferences",
        default=(
            "artifacts/v6_pref_final_training/"
            "preferences_weighted.jsonl"
        ),
    )

    ap.add_argument(
        "--input-adapter",
        default=(
            "artifacts/"
            "robust_visible_generator_v1_epoch2/policy"
        ),
    )

    ap.add_argument(
        "--output-dir",
        required=True,
    )

    ap.add_argument(
        "--mode",
        choices=["sft", "combined"],
        required=True,
    )

    ap.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )

    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
    )

    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=4)

    ap.add_argument("--sft-weight", type=float, default=1.0)
    ap.add_argument("--pref-weight", type=float, default=0.5)
    ap.add_argument("--dpo-beta", type=float, default=1.0)

    ap.add_argument(
        "--anchor-weight",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260924,
    )

    ap.add_argument(
        "--expected-pref-sha",
        default=EXPECTED_PREF_SHA,
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()

    positive_path = Path(args.positives)
    pref_path = Path(args.preferences)
    input_adapter = Path(args.input_adapter)
    output_dir = Path(args.output_dir)

    positives = read_jsonl(positive_path)
    prefs = read_jsonl(pref_path)

    # ------------------------------------------------------------
    # DATA INTEGRITY
    # ------------------------------------------------------------

    assert len(positives) == 17, len(positives)
    assert len(prefs) == 122, len(prefs)

    assert (
        sum(x["family"] == "DIRECTION" for x in prefs)
        == 58
    )

    assert (
        sum(x["family"] == "META" for x in prefs)
        == 64
    )

    pref_sha = file_sha256(pref_path)

    if pref_sha != args.expected_pref_sha:
        raise RuntimeError(
            "V6 preference dataset SHA mismatch:\n"
            f"expected={args.expected_pref_sha}\n"
            f"actual  ={pref_sha}"
        )

    pos_by_gold = {}

    for x in positives:
        gi = int(x["gold_index"])

        assert gi not in pos_by_gold

        assert x["prompt"].strip()
        assert x["completion"].strip()

        pos_by_gold[gi] = x

    assert len(pos_by_gold) == 17

    weights = []

    for x in prefs:
        gi = int(x["gold_index"])

        assert gi in pos_by_gold

        assert x["prompt"].strip()
        assert x["chosen"].strip()
        assert x["rejected"].strip()

        assert x["chosen"] != x["rejected"]

        # Critical: each preference must use the exact original Gold
        # conditioning prompt and the exact original Gold completion.
        assert x["prompt"] == pos_by_gold[gi]["prompt"]
        assert x["chosen"] == pos_by_gold[gi]["completion"]

        w = float(x["weight"])

        assert w > 0
        weights.append(w)

    assert abs(float(np.mean(weights)) - 1.0) < 1e-8

    gold_mass = defaultdict(float)

    for x in prefs:
        gold_mass[int(x["gold_index"])] += float(x["weight"])

    masses = list(gold_mass.values())

    assert len(masses) == 17

    if max(masses) - min(masses) > 1e-6:
        raise RuntimeError(
            "per-Gold preference mass is not equal"
        )

    # ------------------------------------------------------------
    # MATCHED SCHEDULE PREFLIGHT
    # ------------------------------------------------------------

    slots_per_epoch = len(prefs)
    total_micro = slots_per_epoch * args.epochs

    expected_opt_steps = (
        total_micro + args.grad_accum - 1
    ) // args.grad_accum

    schedule_audit = []

    for ep in range(args.epochs):
        pord = list(range(len(prefs)))

        random.Random(
            args.seed + ep
        ).shuffle(pord)

        sord = balanced_positive_order(
            len(positives),
            len(prefs),
            args.seed + ep,
        )

        pc = Counter(sord)

        schedule_audit.append({
            "epoch": ep + 1,
            "slots": len(pord),
            "positive_exposure_min": min(pc.values()),
            "positive_exposure_max": max(pc.values()),
            "preference_unique":
                len(set(pord)),
        })

        assert len(pord) == len(sord) == 122
        assert len(set(pord)) == 122

    print("=" * 76)
    print(" V6 MATCHED PREFERENCE TRAIN")
    print("=" * 76)
    print("mode              :", args.mode)
    print("positives         :", len(positives))
    print("preferences       :", len(prefs))
    print("DIRECTION         :", 58)
    print("META              :", 64)
    print("pref SHA          :", pref_sha)
    print("weight min/max    :", min(weights), max(weights))
    print("weight mean       :", float(np.mean(weights)))
    print("slots / epoch     :", slots_per_epoch)
    print("epochs            :", args.epochs)
    print("total microbatches:", total_micro)
    print("grad accum        :", args.grad_accum)
    print("expected opt steps:", expected_opt_steps)
    print("input adapter     :", input_adapter)
    print("output            :", output_dir)
    print("lr                :", args.lr)
    print("pref weight       :", args.pref_weight)
    print("dpo beta          :", args.dpo_beta)
    print("anchor weight     :", args.anchor_weight)
    print()

    for x in schedule_audit:
        print("schedule:", x)

    if args.dry_run:
        print()
        print("DRY RUN: PASS")
        return

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for V6 training"
        )

    if output_dir.exists():
        raise RuntimeError(
            f"output already exists: {output_dir}"
        )

    before_hash = tree_sha256(input_adapter)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ============================================================
    # 1. FROZEN EPOCH2 REFERENCE
    # ============================================================

    print(
        "=== REFERENCE PRECOMPUTE ===",
        flush=True,
    )

    ref = CurriculumAdaptiveProbePolicyV11(
        args.model_name,
        adapter_path=str(input_adapter),
        trainable=False,
        use_4bit=False,
        max_seq_length=args.max_seq_length,
    )

    ref.model.eval()

    pos_ref = {}
    pref_ref = {}

    with torch.no_grad():

        for x in positives:
            gi = int(x["gold_index"])

            lp = ref.mean_logprob(
                x["prompt"],
                x["completion"],
            )

            pos_ref[gi] = float(
                lp.detach().cpu()
            )

        for i, x in enumerate(prefs):

            c = ref.mean_logprob(
                x["prompt"],
                x["chosen"],
            )

            r = ref.mean_logprob(
                x["prompt"],
                x["rejected"],
            )

            pref_ref[i] = (
                float(c.detach().cpu()),
                float(r.detach().cpu()),
            )

    del ref
    gc.collect()
    torch.cuda.empty_cache()

    print(
        "reference precompute: PASS",
        flush=True,
    )

    # ============================================================
    # 2. TRAINABLE POLICY
    # ============================================================

    pol = CurriculumAdaptiveProbePolicyV11(
        args.model_name,
        adapter_path=str(input_adapter),
        trainable=True,
        use_4bit=False,
        max_seq_length=args.max_seq_length,
    )

    pol.model.train()

    params = [
        p for p in pol.model.parameters()
        if p.requires_grad
    ]

    if not params:
        raise RuntimeError(
            "no trainable parameters"
        )

    print(
        "trainable parameters:",
        sum(p.numel() for p in params),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=0.01,
    )

    # ============================================================
    # 3. MATCHED TRAINING
    #
    # Both modes receive EXACTLY:
    #   122 slots/epoch × 3 epochs = 366 microbatches.
    #
    # Both modes use the same balanced SFT positive in each slot.
    #
    # COMBINED differs only by adding weighted DPO in that slot.
    # ============================================================

    stats = []

    optimizer.zero_grad(
        set_to_none=True
    )

    micro = 0
    opt_steps = 0

    for ep in range(args.epochs):

        pref_order = list(
            range(len(prefs))
        )

        random.Random(
            args.seed + ep
        ).shuffle(pref_order)

        pos_order = balanced_positive_order(
            len(positives),
            len(prefs),
            args.seed + ep,
        )

        assert (
            len(pref_order)
            == len(pos_order)
            == 122
        )

        for slot, (
            pref_idx,
            pos_idx,
        ) in enumerate(
            zip(pref_order, pos_order)
        ):

            # ----------------------------------------------------
            # Matched SFT term: identical in BOTH modes.
            # ----------------------------------------------------

            px = positives[pos_idx]
            pgi = int(px["gold_index"])

            new_lp = pol.mean_logprob(
                px["prompt"],
                px["completion"],
            )

            ref_lp = new_lp.new_tensor(
                pos_ref[pgi]
            )

            sft = -new_lp

            anchor = 0.5 * (
                new_lp - ref_lp
            ).pow(2)

            sft_loss = (
                args.sft_weight * sft
                + args.anchor_weight * anchor
            )

            dpo = new_lp.new_tensor(0.0)
            pref_delta = new_lp.new_tensor(0.0)
            weighted_dpo = new_lp.new_tensor(0.0)

            x = prefs[pref_idx]
            pair_weight = float(x["weight"])

            if args.mode == "combined":

                chosen_lp = pol.mean_logprob(
                    x["prompt"],
                    x["chosen"],
                )

                rejected_lp = pol.mean_logprob(
                    x["prompt"],
                    x["rejected"],
                )

                rc, rr = pref_ref[pref_idx]

                ref_margin = (
                    chosen_lp.new_tensor(rc)
                    - chosen_lp.new_tensor(rr)
                )

                current_margin = (
                    chosen_lp - rejected_lp
                )

                pref_delta = (
                    current_margin
                    - ref_margin
                )

                dpo = -F.logsigmoid(
                    args.dpo_beta
                    * pref_delta
                )

                weighted_dpo = (
                    args.pref_weight
                    * pair_weight
                    * dpo
                )

            loss = (
                sft_loss
                + weighted_dpo
            )

            (
                loss
                / args.grad_accum
            ).backward()

            micro += 1

            if micro % args.grad_accum == 0:

                torch.nn.utils.clip_grad_norm_(
                    params,
                    1.0,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                opt_steps += 1

            stats.append({
                "epoch": ep + 1,
                "slot": slot,
                "positive_gold_index": pgi,
                "preference_index": pref_idx,
                "preference_gold_index":
                    int(x["gold_index"]),
                "family": x["family"],
                "pair_weight": pair_weight,
                "loss":
                    float(loss.detach().cpu()),
                "sft":
                    float(sft.detach().cpu()),
                "anchor":
                    float(anchor.detach().cpu()),
                "dpo":
                    float(dpo.detach().cpu()),
                "weighted_dpo":
                    float(weighted_dpo.detach().cpu()),
                "pref_delta":
                    float(pref_delta.detach().cpu()),
            })

        print(
            f"epoch {ep+1}/{args.epochs} "
            f"complete; "
            f"micro={micro} "
            f"opt_steps={opt_steps}",
            flush=True,
        )

    # Residual accumulation.
    if micro % args.grad_accum != 0:

        torch.nn.utils.clip_grad_norm_(
            params,
            1.0,
        )

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        opt_steps += 1

    if micro != total_micro:
        raise RuntimeError(
            f"micro mismatch: "
            f"{micro} != {total_micro}"
        )

    if opt_steps != expected_opt_steps:
        raise RuntimeError(
            f"optimizer step mismatch: "
            f"{opt_steps} != "
            f"{expected_opt_steps}"
        )

    # ============================================================
    # 4. DETERMINISTIC POST-TRAIN DIAGNOSTICS
    # ============================================================

    pol.model.eval()

    post_pos = []
    post_pref = []

    with torch.no_grad():

        for x in positives:

            gi = int(x["gold_index"])

            lp = float(
                pol.mean_logprob(
                    x["prompt"],
                    x["completion"],
                ).detach().cpu()
            )

            post_pos.append({
                "gold_index": gi,
                "ref_lp":
                    pos_ref[gi],
                "post_lp":
                    lp,
                "delta_lp":
                    lp - pos_ref[gi],
            })

        for i, x in enumerate(prefs):

            c = float(
                pol.mean_logprob(
                    x["prompt"],
                    x["chosen"],
                ).detach().cpu()
            )

            r = float(
                pol.mean_logprob(
                    x["prompt"],
                    x["rejected"],
                ).detach().cpu()
            )

            rc, rr = pref_ref[i]

            ref_margin = rc - rr
            post_margin = c - r

            post_pref.append({
                "index": i,
                "family":
                    x["family"],
                "gold_index":
                    int(x["gold_index"]),
                "weight":
                    float(x["weight"]),
                "ref_chosen_lp": rc,
                "ref_rejected_lp": rr,
                "ref_margin":
                    ref_margin,
                "post_chosen_lp": c,
                "post_rejected_lp": r,
                "post_margin":
                    post_margin,
                "margin_gain":
                    post_margin
                    - ref_margin,
            })

    # ============================================================
    # 5. SAVE ATOMICALLY
    # ============================================================

    tmp = output_dir.parent / (
        output_dir.name + ".tmp"
    )

    if tmp.exists():
        shutil.rmtree(tmp)

    tmp.mkdir(
        parents=True,
        exist_ok=True,
    )

    pol.model.save_pretrained(tmp)
    pol.tokenizer.save_pretrained(tmp)

    # Fail instead of deleting an existing run.
    if output_dir.exists():
        raise RuntimeError(
            f"output appeared during run: "
            f"{output_dir}"
        )

    tmp.rename(output_dir)

    after_hash = tree_sha256(
        input_adapter
    )

    if after_hash != before_hash:
        raise RuntimeError(
            "INPUT EPOCH2 ADAPTER "
            "CHANGED DURING TRAINING"
        )

    # ============================================================
    # 6. SUMMARIES
    # ============================================================

    weighted_margin_num = sum(
        x["weight"] * x["margin_gain"]
        for x in post_pref
    )

    weighted_margin_den = sum(
        x["weight"]
        for x in post_pref
    )

    def family_mean(family):
        vals = [
            x["margin_gain"]
            for x in post_pref
            if x["family"] == family
        ]
        return (
            float(np.mean(vals))
            if vals else 0.0
        )

    def family_weighted_mean(family):
        vals = [
            x for x in post_pref
            if x["family"] == family
        ]

        if not vals:
            return 0.0

        return float(
            sum(
                x["weight"]
                * x["margin_gain"]
                for x in vals
            )
            / sum(
                x["weight"]
                for x in vals
            )
        )

    summary = {
        "mode": args.mode,
        "input_adapter":
            str(input_adapter),
        "output_adapter":
            str(output_dir),

        "positives":
            len(positives),

        "preferences":
            len(prefs),

        "preference_families": {
            "DIRECTION": 58,
            "META": 64,
        },

        "pref_dataset_sha256":
            pref_sha,

        "epochs":
            args.epochs,

        "slots_per_epoch":
            slots_per_epoch,

        "microbatches":
            micro,

        "optimizer_steps":
            opt_steps,

        "expected_optimizer_steps":
            expected_opt_steps,

        "grad_accum":
            args.grad_accum,

        "lr":
            args.lr,

        "sft_weight":
            args.sft_weight,

        "pref_weight":
            args.pref_weight,

        "dpo_beta":
            args.dpo_beta,

        "anchor_weight":
            args.anchor_weight,

        "mean_train_sft":
            float(
                np.mean([
                    x["sft"]
                    for x in stats
                ])
            ),

        "mean_train_anchor":
            float(
                np.mean([
                    x["anchor"]
                    for x in stats
                ])
            ),

        "mean_train_dpo": (
            float(
                np.mean([
                    x["dpo"]
                    for x in stats
                ])
            )
            if args.mode == "combined"
            else 0.0
        ),

        "mean_train_weighted_dpo": (
            float(
                np.mean([
                    x["weighted_dpo"]
                    for x in stats
                ])
            )
            if args.mode == "combined"
            else 0.0
        ),

        "mean_positive_lp_gain":
            float(
                np.mean([
                    x["delta_lp"]
                    for x in post_pos
                ])
            ),

        "min_positive_lp_gain":
            float(
                min(
                    x["delta_lp"]
                    for x in post_pos
                )
            ),

        "mean_preference_margin_gain":
            float(
                np.mean([
                    x["margin_gain"]
                    for x in post_pref
                ])
            ),

        "weighted_preference_margin_gain":
            float(
                weighted_margin_num
                / weighted_margin_den
            ),

        "direction_margin_gain":
            family_mean(
                "DIRECTION"
            ),

        "meta_margin_gain":
            family_mean(
                "META"
            ),

        "weighted_direction_margin_gain":
            family_weighted_mean(
                "DIRECTION"
            ),

        "weighted_meta_margin_gain":
            family_weighted_mean(
                "META"
            ),

        "input_adapter_sha256":
            after_hash,

        "schedule_audit":
            schedule_audit,
    }

    write_json(
        output_dir.parent
        / "train_summary.json",
        summary,
    )

    write_json(
        output_dir.parent
        / "positive_diagnostics.json",
        post_pos,
    )

    write_json(
        output_dir.parent
        / "preference_diagnostics.json",
        post_pref,
    )

    write_json(
        output_dir.parent
        / "training_stats.json",
        stats,
    )

    print()
    print("=" * 76)
    print(" TRAINING SUMMARY")
    print("=" * 76)
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    del pol
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
