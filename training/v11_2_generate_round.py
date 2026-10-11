from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from v11.role_universe import ROLE_DESCRIPTIONS, anchor_probe
from v11_2.hashcheck import assert_exact_v11_start
from v11_2.policy import HardConfuserProbePolicyV112
from v11_2.split_common import build_context, load_previous_elites, pair_meta, role_order


def main():
    import torch

    ap = argparse.ArgumentParser(description="V11.2 split pipeline: GPU candidate generation only")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--profile", default="config/application_profile_v11.json")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--initial-elite-summary", default="artifacts/v11_summary.json")
    ap.add_argument("--previous-eval", default="")
    ap.add_argument("--v11-frozen-summary", default="artifacts/v11_summary.json")
    ap.add_argument("--rounds-per-role", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--bundle-size", type=int, default=4)
    ap.add_argument("--confirmation-size", type=int, default=2)
    ap.add_argument("--max-avoid-probes", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--syntax-retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1010)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for candidate generation")
    if args.round < 1 or args.round > args.rounds_per_role:
        raise SystemExit("round out of range")
    if not Path(args.adapter).is_dir():
        raise SystemExit(f"adapter missing: {args.adapter}")

    if args.round == 1:
        fp = assert_exact_v11_start(args.adapter, args.v11_frozen_summary, strict=True)
        print("[start] exact trained V11 start match:", fp["policy_match"] and fp["summary_match"], flush=True)
    else:
        print("[start] continuation adapter:", args.adapter, flush=True)

    ctx = build_context(
        profile_path=args.profile,
        rounds=args.rounds_per_role,
        bundle_size=args.bundle_size,
        confirmation_size=args.confirmation_size,
        seed=args.seed,
    )
    usable_roles = ctx["usable_roles"]
    schedule = ctx["schedule"]
    conf_schedule = ctx["confirmation_schedule"]
    novelty_refs = load_previous_elites(
        args.initial_elite_summary,
        args.previous_eval or None,
        usable_roles,
    )

    random.seed(args.seed + args.round)
    np.random.seed(args.seed + args.round)
    torch.manual_seed(args.seed + args.round)

    policy = HardConfuserProbePolicyV112(
        args.model_name,
        adapter_path=args.adapter,
        trainable=False,
        use_4bit=not args.no_4bit,
        max_seq_length=1536,
    )
    policy.model.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    role_seq = role_order(usable_roles, args.round, args.seed)
    total = len(role_seq) * args.group_size
    written = 0

    print(f"[generate] round={args.round} roles={len(role_seq)} candidates={total} adapter={args.adapter}", flush=True)
    with tmp.open("w", encoding="utf-8") as f:
        for role_pos, role in enumerate(role_seq, 1):
            bundle = schedule[(args.round, role)]
            confirmation_bundle = conf_schedule[(args.round, role)]
            anchor = anchor_probe(role)
            avoid = novelty_refs.get(role, [])[-args.max_avoid_probes:]
            hard_confuser_names = []
            for p in bundle:
                h = str(getattr(p, "negative_replacement", "")).strip()
                if h and h != role and h not in hard_confuser_names:
                    hard_confuser_names.append(h)
            obs = {
                "candidate_role": role,
                "candidate_description": ROLE_DESCRIPTIONS.get(role, role),
                "anchor_probe": anchor,
                "avoid_probes": avoid,
                "hard_confusers": [
                    {"role": h, "description": ROLE_DESCRIPTIONS.get(h, h)}
                    for h in hard_confuser_names[:4]
                ],
            }
            torch.manual_seed(args.seed + 100003 * args.round + 997 * role_pos)
            group_id = f"v112r{args.round:02d}_{role}_{((args.round - 1) * len(usable_roles) + role_pos):03d}"
            for ci in range(args.group_size):
                sample = None
                attempts = []
                for _ in range(max(1, args.syntax_retries)):
                    trial = policy.sample_probe(obs, temperature=args.temperature, top_p=args.top_p)
                    attempts.append({"raw": trial.raw_completion, "valid": trial.valid, "reason": trial.invalid_reason})
                    sample = trial
                    if trial.valid:
                        break
                assert sample is not None
                row = {
                    "phase": "generated",
                    "round": args.round,
                    "group_id": group_id,
                    "role_pos": role_pos,
                    "candidate_index": ci + 1,
                    "target_role": role,
                    "anchor_probe": anchor,
                    "avoid_probes_public": avoid,
                    "hard_confusers_training_only": hard_confuser_names[:4],
                    "query": sample.query,
                    "prompt": sample.prompt,
                    "completion": sample.completion,
                    "raw_completion": sample.raw_completion,
                    "valid": bool(sample.valid),
                    "invalid_reason": sample.invalid_reason,
                    "old_mean_logprob": float(sample.old_mean_logprob),
                    "ref_mean_logprob": float(sample.ref_mean_logprob),
                    "syntax_attempts": attempts,
                    "bundle_training_only": [pair_meta(p) for p in bundle],
                    "confirmation_bundle_training_only": [pair_meta(p) for p in confirmation_bundle],
                    "source_adapter": args.adapter,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                print(
                    f"[generate] round={args.round} {written}/{total} role={role:<24} cand={ci+1}/{args.group_size} valid={sample.valid} q={sample.query[:100]!r}",
                    flush=True,
                )
    os.replace(tmp, out)
    print(f"[generate-done] round={args.round} rows={written} output={out}", flush=True)


if __name__ == "__main__":
    main()
