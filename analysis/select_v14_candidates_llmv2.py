#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch

from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from analysis.train_llm_selector_v1 import render_input


def read_jsonl(p):
    p = Path(p)
    return [
        json.loads(x)
        for x in p.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def write_jsonl(p, rows):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument(
        "--adapter",
        default="artifacts/llm_selector_v2_role_only/best_adapter",
    )
    ap.add_argument("--max-length", type=int, default=1024)

    args = ap.parse_args()

    root = Path(args.root)
    adapter = Path(args.adapter)

    cand_path = (
        root /
        f"candidates_{args.split}_round_{args.round:02d}.jsonl"
    )

    pool_path = (
        root /
        f"candidate_pool_{args.split}_round_{args.round:02d}.jsonl"
    )

    state_path = (
        root /
        f"states_{args.split}_round_{args.round-1:02d}.jsonl"
    )

    diag_path = (
        root /
        f"selector_diag_{args.split}_round_{args.round:02d}.jsonl"
    )

    candidates = read_jsonl(cand_path)
    states = {
        str(x["session_id"]): x
        for x in read_jsonl(state_path)
    }

    # Preserve all 8 candidates before overwriting candidates_* with
    # the single selected query per system.
    shutil.copyfile(cand_path, pool_path)

    cfg = PeftConfig.from_pretrained(str(adapter))
    base_name = cfg.base_model_name_or_path

    print("SELECTOR ADAPTER =", adapter, flush=True)
    print("SELECTOR BASE    =", base_name, flush=True)

    tok = AutoTokenizer.from_pretrained(
        str(adapter),
        use_fast=True,
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=2,
        quantization_config=qconf,
        device_map={"": 0},
        dtype=dtype,
    )

    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        str(adapter),
        is_trainable=False,
    )

    model.eval()

    grouped = defaultdict(list)

    for c in candidates:
        grouped[str(c["session_id"])].append(c)

    chosen_rows = []
    diag_rows = []

    for sid, cs in sorted(grouped.items()):

        if sid not in states:
            raise RuntimeError(f"missing state for {sid}")

        st = states[sid]

        viable = [
            c for c in cs
            if bool(c.get("valid", False))
            and str(c.get("query", "")).strip()
        ]

        if not viable:
            raise RuntimeError(
                f"{sid}: Generator produced zero valid candidates"
            )

        texts = []

        for c in viable:
            row = {
                "role_probs_before": st["role_probs"],
                "k_probs_before": st["k_probs"],
                "round": args.round,
                "query": c["query"],
            }

            texts.append(render_input(row))

        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        ).to("cuda:0")

        with torch.inference_mode():
            with torch.autocast(
                "cuda",
                dtype=dtype,
                enabled=True,
            ):
                logits = model(**enc).logits.float().cpu()

        # Selector V2 is role-only (lambda_k=0).
        # Head 0 is the role-utility prediction used for ranking.
        role_scores = logits[:, 0].numpy()

        best_local = int(role_scores.argmax())
        chosen = dict(viable[best_local])

        chosen["selector_score"] = float(
            role_scores[best_local]
        )
        chosen["selector_model"] = (
            "llm_selector_v2_role_only"
        )

        chosen_rows.append(chosen)

        ranking = sorted(
            zip(viable, role_scores),
            key=lambda z: -float(z[1]),
        )

        print()
        print("=" * 100)
        print(
            f"ROUND={args.round} SID={sid} "
            f"valid={len(viable)}/{len(cs)}"
        )

        for rank, (c, score) in enumerate(ranking, 1):
            selected = (
                int(c["candidate_index"])
                == int(chosen["candidate_index"])
            )

            print(
                f"{rank:02d} "
                f"score={float(score):+.5f} "
                f"cand={int(c['candidate_index']):02d} "
                f"{'* SELECTED' if selected else ''}\n"
                f"    {c['query']}"
            )

            d = dict(c)
            d["selector_score"] = float(score)
            d["selector_rank"] = rank
            d["selector_selected"] = selected
            diag_rows.append(d)

    if len(chosen_rows) != len(states):
        raise RuntimeError(
            f"selected {len(chosen_rows)} rows "
            f"for {len(states)} systems"
        )

    write_jsonl(diag_path, diag_rows)
    write_jsonl(cand_path, chosen_rows)

    print()
    print(
        f"[done] pool={len(candidates)} "
        f"selected={len(chosen_rows)}"
    )
    print("POOL     =", pool_path)
    print("SELECTED =", cand_path)
    print("DIAG     =", diag_path)


if __name__ == "__main__":
    main()
