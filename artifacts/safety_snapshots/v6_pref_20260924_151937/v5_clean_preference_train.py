from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
from pathlib import Path

ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOTDIR))

import numpy as np

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_curriculum.hashguard import tree_sha256
from adaptive_v11_curriculum.policy import CurriculumAdaptiveProbePolicyV11


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-root",
        default="artifacts/v4_training_ready",
    )
    ap.add_argument(
        "--input-adapter",
        default="artifacts/robust_visible_generator_v1_epoch2/policy",
    )
    ap.add_argument("--output-dir", required=True)

    ap.add_argument(
        "--mode",
        choices=["sft", "combined"],
        required=True,
    )

    ap.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    ap.add_argument("--max-seq-length", type=int, default=4096)

    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=4)

    ap.add_argument("--sft-weight", type=float, default=1.0)
    ap.add_argument("--pref-weight", type=float, default=0.5)
    ap.add_argument("--dpo-beta", type=float, default=1.0)

    # Keeps positive log-probs near Epoch2 while still allowing improvement.
    ap.add_argument("--anchor-weight", type=float, default=0.05)

    ap.add_argument("--seed", type=int, default=20260924)

    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    data_root = Path(args.data_root)
    input_adapter = Path(args.input_adapter)
    output_dir = Path(args.output_dir)

    positives = read_jsonl(data_root / "positives.jsonl")
    prefs = read_jsonl(data_root / "preferences.jsonl")

    assert len(positives) == 17, len(positives)
    assert len(prefs) == 8, len(prefs)

    assert sum(x["pair_type"] == "DIRECTION" for x in prefs) == 6
    assert sum(x["pair_type"] == "META" for x in prefs) == 2

    for x in positives:
        assert x["prompt"].strip()
        assert x["completion"].strip()

    for x in prefs:
        assert x["prompt"].strip()
        assert x["chosen"].strip()
        assert x["rejected"].strip()
        assert x["chosen"] != x["rejected"]

    before_hash = tree_sha256(input_adapter)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("======================================")
    print(" V5 CLEAN PREFERENCE TRAIN")
    print("======================================")
    print("mode          :", args.mode)
    print("positives     :", len(positives))
    print("preferences   :", len(prefs))
    print("input adapter :", input_adapter)
    print("output        :", output_dir)
    print("lr            :", args.lr)
    print("epochs        :", args.epochs)
    print("pref weight   :", args.pref_weight)
    print("dpo beta      :", args.dpo_beta)
    print("anchor weight :", args.anchor_weight)
    print()

    # ============================================================
    # 1. Frozen Epoch2 reference.
    # Precompute all reference scores, then unload it.
    # This avoids keeping two 1.5B models in GPU memory.
    # ============================================================

    print("=== REFERENCE PRECOMPUTE ===", flush=True)

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

            pos_ref[gi] = float(lp.detach().cpu())

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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("reference precompute: PASS", flush=True)

    # ============================================================
    # 2. Trainable policy starts from EXACT SAME Epoch2 adapter.
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
        raise RuntimeError("no trainable parameters")

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
    # Training schedule
    # ============================================================

    stats = []

    optimizer.zero_grad(set_to_none=True)

    micro = 0
    opt_steps = 0

    for ep in range(args.epochs):

        items = [
            ("SFT", i)
            for i in range(len(positives))
        ]

        if args.mode == "combined":
            items += [
                ("PREF", i)
                for i in range(len(prefs))
            ]

        random.Random(args.seed + ep).shuffle(items)

        for kind, idx in items:

            if kind == "SFT":

                x = positives[idx]
                gi = int(x["gold_index"])

                new_lp = pol.mean_logprob(
                    x["prompt"],
                    x["completion"],
                )

                ref_lp = new_lp.new_tensor(
                    pos_ref[gi]
                )

                sft = -new_lp

                anchor = 0.5 * (
                    new_lp - ref_lp
                ).pow(2)

                loss = (
                    args.sft_weight * sft
                    + args.anchor_weight * anchor
                )

                dpo = new_lp.new_tensor(0.0)
                pref_delta = new_lp.new_tensor(0.0)

            else:

                x = prefs[idx]

                chosen_lp = pol.mean_logprob(
                    x["prompt"],
                    x["chosen"],
                )

                rejected_lp = pol.mean_logprob(
                    x["prompt"],
                    x["rejected"],
                )

                rc, rr = pref_ref[idx]

                ref_margin = (
                    chosen_lp.new_tensor(rc)
                    - chosen_lp.new_tensor(rr)
                )

                current_margin = (
                    chosen_lp - rejected_lp
                )

                # Reference-relative preference improvement.
                pref_delta = (
                    current_margin - ref_margin
                )

                dpo = -F.logsigmoid(
                    args.dpo_beta * pref_delta
                )

                loss = args.pref_weight * dpo

                sft = chosen_lp.new_tensor(0.0)
                anchor = chosen_lp.new_tensor(0.0)

            (loss / args.grad_accum).backward()

            micro += 1

            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    params,
                    1.0,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

            stats.append({
                "epoch": ep + 1,
                "kind": kind,
                "loss": float(loss.detach().cpu()),
                "sft": float(sft.detach().cpu()),
                "anchor": float(anchor.detach().cpu()),
                "dpo": float(dpo.detach().cpu()),
                "pref_delta": float(
                    pref_delta.detach().cpu()
                ),
            })

        print(
            f"epoch {ep+1}/{args.epochs} complete",
            flush=True,
        )

    # Flush any residual gradient accumulation.
    if micro % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(
            params,
            1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        opt_steps += 1

    # ============================================================
    # 3. Deterministic post-training diagnostics.
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
                "ref_lp": pos_ref[gi],
                "post_lp": lp,
                "delta_lp": lp - pos_ref[gi],
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
                "pair_type": x["pair_type"],
                "gold_index": int(x["gold_index"]),
                "ref_chosen_lp": rc,
                "ref_rejected_lp": rr,
                "ref_margin": ref_margin,
                "post_chosen_lp": c,
                "post_rejected_lp": r,
                "post_margin": post_margin,
                "margin_gain": post_margin - ref_margin,
            })

    # ============================================================
    # Save atomically.
    # ============================================================

    tmp = output_dir.parent / (
        output_dir.name + ".tmp"
    )

    if tmp.exists():
        shutil.rmtree(tmp)

    tmp.mkdir(parents=True, exist_ok=True)

    pol.model.save_pretrained(tmp)
    pol.tokenizer.save_pretrained(tmp)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    tmp.rename(output_dir)

    after_hash = tree_sha256(input_adapter)

    if after_hash != before_hash:
        raise RuntimeError(
            "INPUT EPOCH2 ADAPTER CHANGED DURING TRAINING"
        )

    sft_stats = [
        x for x in stats
        if x["kind"] == "SFT"
    ]

    pref_stats = [
        x for x in stats
        if x["kind"] == "PREF"
    ]

    summary = {
        "mode": args.mode,
        "input_adapter": str(input_adapter),
        "output_adapter": str(output_dir),

        "positives": len(positives),
        "preferences": (
            len(prefs)
            if args.mode == "combined"
            else 0
        ),

        "epochs": args.epochs,
        "optimizer_steps": opt_steps,
        "lr": args.lr,
        "sft_weight": args.sft_weight,
        "pref_weight": args.pref_weight,
        "dpo_beta": args.dpo_beta,
        "anchor_weight": args.anchor_weight,

        "mean_train_sft": (
            float(np.mean([x["sft"] for x in sft_stats]))
            if sft_stats else 0.0
        ),

        "mean_train_anchor": (
            float(np.mean([x["anchor"] for x in sft_stats]))
            if sft_stats else 0.0
        ),

        "mean_train_dpo": (
            float(np.mean([x["dpo"] for x in pref_stats]))
            if pref_stats else 0.0
        ),

        "mean_positive_lp_gain": float(
            np.mean([x["delta_lp"] for x in post_pos])
        ),

        "min_positive_lp_gain": float(
            min(x["delta_lp"] for x in post_pos)
        ),

        "mean_preference_margin_gain": float(
            np.mean(
                [x["margin_gain"] for x in post_pref]
            )
        ),

        "direction_margin_gain": float(
            np.mean([
                x["margin_gain"]
                for x in post_pref
                if x["pair_type"] == "DIRECTION"
            ])
        ),

        "meta_margin_gain": float(
            np.mean([
                x["margin_gain"]
                for x in post_pref
                if x["pair_type"] == "META"
            ])
        ),

        "input_adapter_sha256": after_hash,
    }

    write_json(
        output_dir.parent / "train_summary.json",
        summary,
    )

    write_json(
        output_dir.parent / "positive_diagnostics.json",
        post_pos,
    )

    write_json(
        output_dir.parent / "preference_diagnostics.json",
        post_pref,
    )

    print()
    print("======================================")
    print(" TRAINING SUMMARY")
    print("======================================")
    print(json.dumps(summary, indent=2))

    print()
    print("=== PREFERENCE MARGIN GAINS ===")

    for x in post_pref:
        print(
            f"{x['pair_type']:9s} "
            f"G{x['gold_index']:02d} "
            f"ref={x['ref_margin']:+.4f} "
            f"post={x['post_margin']:+.4f} "
            f"gain={x['margin_gain']:+.4f}"
        )

    del pol
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
