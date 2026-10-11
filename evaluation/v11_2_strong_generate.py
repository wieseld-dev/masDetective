from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

from v10.policy import SeededProbePolicyV10
from v11.role_universe import TRAIN_ROLES, DEV_UNSEEN_ROLES, ROLE_DESCRIPTIONS, anchor_probe
from v11_2.counterfactual import PREFERRED_CONFUSERS
from v11_2.hashcheck import assert_exact_v11_start
from v11_2.policy import HardConfuserProbePolicyV112
from v11_2.split_common import _load_seed_elites


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def generate_method(method: str, adapter: str, args, seed_elites):
    import torch

    if method == "v11":
        fp = assert_exact_v11_start(adapter, args.v11_frozen_summary, strict=True)
        print(f"[verify] baseline exact V11 match={fp['policy_match'] and fp['summary_match']}", flush=True)
        policy = SeededProbePolicyV10(
            args.model_name,
            adapter_path=adapter,
            trainable=False,
            use_4bit=not args.no_4bit,
            max_seq_length=args.max_seq_length,
        )
    elif method == "v11_2":
        policy = HardConfuserProbePolicyV112(
            args.model_name,
            adapter_path=adapter,
            trainable=False,
            use_4bit=not args.no_4bit,
            max_seq_length=args.max_seq_length,
        )
    else:
        raise ValueError(method)

    rows = []
    roles = list(TRAIN_ROLES) + list(DEV_UNSEEN_ROLES)
    for ri, role in enumerate(roles):
        seen = []
        for j in range(args.probes_per_role):
            # Same seed schedule for both methods. Different prompt/policy is the intervention.
            seed = args.seed + 10000 * ri + j
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            avoid = list(seed_elites.get(role, [])) + list(seen)
            obs = {
                "candidate_role": role,
                "candidate_description": ROLE_DESCRIPTIONS[role],
                "anchor_probe": anchor_probe(role),
                "avoid_probes": avoid[-8:],
            }
            if method == "v11_2":
                obs["hard_confusers"] = [
                    {"role": c, "description": ROLE_DESCRIPTIONS[c]}
                    for c in PREFERRED_CONFUSERS.get(role, ())
                    if c in ROLE_DESCRIPTIONS
                ]
            sample = policy.sample_probe(
                obs,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            q = str(sample.query or "").strip()
            if q:
                seen.append(q)
            row = {
                "method": method,
                "role_set": "train_known" if role in TRAIN_ROLES else "dev_unseen",
                "target_role": role,
                "probe_index": j + 1,
                "seed": seed,
                "query": q,
                "valid": bool(sample.valid),
                "invalid_reason": str(sample.invalid_reason),
                "raw_completion": str(sample.raw_completion),
                "prompt": str(sample.prompt),
                "completion": str(sample.completion),
                "anchor_probe": anchor_probe(role),
                "hard_confusers_public": list(PREFERRED_CONFUSERS.get(role, ())) if method == "v11_2" else [],
            }
            rows.append(row)
            print(
                f"[generate] method={method:<5} role={role:<24} p={j+1}/{args.probes_per_role} "
                f"valid={sample.valid} q={q[:120]!r}",
                flush=True,
            )
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description="Locked V11 vs V11.2 probe generation for strong evaluation")
    ap.add_argument("--baseline-adapter", default="artifacts/v11_policy")
    ap.add_argument("--candidate-adapter", default="artifacts/v11_2_split/checkpoints/round_02/policy")
    ap.add_argument("--v11-frozen-summary", default="artifacts/v11_summary.json")
    ap.add_argument("--initial-elite-summary", default="artifacts/v11_summary.json")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output", default="artifacts/v11_2_strong_eval/generated.jsonl")
    ap.add_argument("--probes-per-role", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.90)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-seq-length", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=260914)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for generation comparison")
    if not Path(args.baseline_adapter).is_dir():
        raise SystemExit(f"missing baseline adapter: {args.baseline_adapter}")
    if not Path(args.candidate_adapter).is_dir():
        raise SystemExit(f"missing candidate adapter: {args.candidate_adapter}")

    seed_elites = _load_seed_elites(
        args.initial_elite_summary,
        list(TRAIN_ROLES) + list(DEV_UNSEEN_ROLES),
        max_per_role=3,
    )
    rows = []
    rows.extend(generate_method("v11", args.baseline_adapter, args, seed_elites))
    rows.extend(generate_method("v11_2", args.candidate_adapter, args, seed_elites))
    write_jsonl(Path(args.output), rows)
    print(f"[done] generated={len(rows)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
