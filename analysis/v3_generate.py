#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("analysis").resolve()))

from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_active.io import stable_int
from v11.role_universe import TRAIN_ROLES

from v3_common import load_matrix_npz
from v3_coverage_policy import CoverageEliminationProbePolicyV3


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--root", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--v11-reference", default="artifacts/v11_policy")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--temperature", type=float, default=.85)
    ap.add_argument("--top-p", type=float, default=.95)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--local-retries", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for generation")

    root = Path(args.root)
    states = read_jsonl(root / f"states_{args.split}_round_{args.round-1:02d}.jsonl")
    cov_ids, cov_mat = load_matrix_npz(
        root / f"coverage_{args.split}_round_{args.round-1:02d}.npz",
        "coverage",
    )
    cov_by = {sid: cov_mat[i] for i, sid in enumerate(cov_ids)}

    current = CoverageEliminationProbePolicyV3(
        args.model_name,
        adapter_path=args.adapter,
        trainable=False,
        use_4bit=False,
        max_seq_length=args.max_seq_length,
    )
    reference = CoverageEliminationProbePolicyV3(
        args.model_name,
        adapter_path=args.v11_reference,
        trainable=False,
        use_4bit=False,
        max_seq_length=args.max_seq_length,
    )

    rows = []
    G = args.group_size if args.split == "train" else args.group_size

    for si, raw in enumerate(states):
        st = PublicBeliefState.from_public_dict(raw)
        st.validate(TRAIN_ROLES, sorted(st.k_probs))
        coverage = {
            r: float(cov_by[st.session_id][i])
            for i, r in enumerate(TRAIN_ROLES)
        }
        seen = [h.query for h in st.history]
        group = []

        for j in range(G):
            last = None
            for retry in range(max(1, args.local_retries)):
                seed = stable_int(args.seed, args.split, args.round, st.session_id, j, retry)
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                obs = {
                    "belief_state": st.to_public_dict(),
                    "coverage": coverage,
                    "candidate_index": j,
                    "max_history_turns": 6,
                }
                sample = current.sample_probe(
                    obs,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                q = str(sample.query or "").strip()
                ok, why = surface_valid(q)
                valid = (
                    bool(sample.valid)
                    and ok
                    and q not in seen
                    and all(q != x.get("query") for x in group)
                )
                last = (sample, q, valid, why, seed, retry)
                if valid:
                    break

            sample, q, valid, why, seed, retry = last
            prompt = str(sample.prompt)
            completion = str(sample.completion)
            old_lp = getattr(sample, "old_mean_logprob", None)
            if old_lp is None and completion:
                old_lp = float(current.mean_logprob(prompt, completion).detach().cpu())
            if old_lp is None:
                old_lp = -20.0
            ref_lp = (
                float(reference.mean_logprob(prompt, completion).detach().cpu())
                if completion else -20.0
            )
            rec = {
                "session_id": st.session_id,
                "split": args.split,
                "round": args.round,
                "candidate_index": j,
                "group_id": f"{st.session_id}::r{args.round:02d}",
                "query": q,
                "valid": bool(valid),
                "invalid_reason": "" if valid else (
                    str(getattr(sample, "invalid_reason", "")) or why or "duplicate"
                ),
                "prompt": prompt,
                "completion": completion,
                "old_logp": float(old_lp),
                "ref_logp": float(ref_lp),
                "raw_completion": str(sample.raw_completion),
                "seed": int(seed),
                "local_attempts": int(retry + 1),
                "adapter": str(args.adapter),
                "coverage_before": coverage,
            }
            group.append(rec)
            if valid:
                seen.append(q)

        valid_idx = [x["candidate_index"] for x in group if x["valid"]]
        # Fixed before execution/reward: no best-of-G trajectory leakage.
        if valid_idx:
            rr = random.Random(stable_int(
                args.seed, "continuation", args.split, args.round, st.session_id
            ))
            chosen = rr.choice(valid_idx)
        else:
            chosen = -1

        for x in group:
            x["chosen_for_continuation"] = bool(x["candidate_index"] == chosen)
            rows.append(x)

        print(
            f"[gen] {si+1}/{len(states)} r={args.round} sid={st.session_id} "
            f"valid={len(valid_idx)}/{G} continuation={chosen}",
            flush=True,
        )

    out = root / f"candidates_{args.split}_round_{args.round:02d}.jsonl"
    write_jsonl(out, rows)
    print(f"[done] {out} rows={len(rows)} valid={sum(int(x['valid']) for x in rows)}")

    del current, reference
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
