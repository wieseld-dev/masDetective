from __future__ import annotations

import argparse
import gc
import json
import os
import random
from collections import Counter
from pathlib import Path

from v10.policy import SeededProbePolicyV10
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS, anchor_probe
from v11_2.hashcheck import assert_exact_v11_start
from v11_2.split_common import _load_seed_elites


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a frozen benign adaptive action library from ORIGINAL V11"
    )
    ap.add_argument("--adapter", default="artifacts/v11_policy")
    ap.add_argument("--v11-frozen-summary", default="artifacts/v11_summary.json")
    ap.add_argument("--initial-elite-summary", default="artifacts/v11_summary.json")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output", default="artifacts/adaptive_v1/action_library.jsonl")

    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--max-attempts-per-variant", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.75)
    ap.add_argument("--top-p", type=float, default=0.90)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-seq-length", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    if args.variants < 1:
        raise SystemExit("--variants must be >= 1")
    if args.max_attempts_per_variant < 1:
        raise SystemExit("--max-attempts-per-variant must be >= 1")
    if not Path(args.adapter).is_dir():
        raise SystemExit(f"missing ORIGINAL V11 adapter: {args.adapter}")
    if not Path(args.v11_frozen_summary).is_file():
        raise SystemExit(f"missing frozen V11 summary: {args.v11_frozen_summary}")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for V11 generation")

    # Scientific guard: this must be the exact frozen V11 checkpoint.
    fp = assert_exact_v11_start(
        args.adapter,
        args.v11_frozen_summary,
        strict=True,
    )
    exact = bool(fp["policy_match"] and fp["summary_match"])
    print(f"[verify] exact frozen V11 match={exact}", flush=True)
    if not exact:
        raise SystemExit("V11 checkpoint verification failed")

    seed_elites = _load_seed_elites(
        args.initial_elite_summary,
        list(TRAIN_ROLES),
        max_per_role=3,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    policy = SeededProbePolicyV10(
        args.model_name,
        adapter_path=args.adapter,
        trainable=False,
        use_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
    )

    rows: list[dict] = []
    per_role_valid = Counter()

    for ri, role in enumerate(TRAIN_ROLES):
        seen: list[str] = []

        for variant in range(args.variants):
            chosen = None

            # These retries are local generation only. No black-box MAS call is made.
            for attempt in range(args.max_attempts_per_variant):
                seed = args.seed + 10000 * ri + 100 * variant + attempt
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                avoid = list(seed_elites.get(role, [])) + list(seen)
                obs = {
                    "candidate_role": role,
                    "candidate_description": ROLE_DESCRIPTIONS[role],
                    "anchor_probe": anchor_probe(role),
                    "avoid_probes": avoid[-8:],
                }

                sample = policy.sample_probe(
                    obs,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )

                q = str(sample.query or "").strip()
                valid = bool(sample.valid and q and q not in seen)

                print(
                    f"[try] role={role:<24} variant={variant} "
                    f"attempt={attempt + 1}/{args.max_attempts_per_variant} "
                    f"valid={valid} q={q[:120]!r}",
                    flush=True,
                )

                if valid:
                    chosen = {
                        "method": "v11",
                        "target": role,
                        "competitor": None,
                        "hard_confusers": [],
                        "variant": int(variant),
                        "query": q,
                        "valid": True,
                        "invalid_reason": "",
                        "seed": int(seed),
                        "generation_attempt": int(attempt + 1),
                        "anchor_probe": anchor_probe(role),
                        "raw_completion": str(sample.raw_completion),
                        "prompt": str(sample.prompt),
                        "completion": str(sample.completion),
                    }
                    break

            if chosen is None:
                chosen = {
                    "method": "v11",
                    "target": role,
                    "competitor": None,
                    "hard_confusers": [],
                    "variant": int(variant),
                    "query": "",
                    "valid": False,
                    "invalid_reason": "all_generation_attempts_failed",
                    "seed": None,
                    "generation_attempt": int(args.max_attempts_per_variant),
                    "anchor_probe": anchor_probe(role),
                    "raw_completion": "",
                    "prompt": "",
                    "completion": "",
                }
            else:
                seen.append(chosen["query"])
                per_role_valid[role] += 1

            rows.append(chosen)

    del policy
    gc.collect()
    torch.cuda.empty_cache()

    expected = len(TRAIN_ROLES) * args.variants
    valid_n = sum(1 for row in rows if row["valid"])
    missing_roles = [r for r in TRAIN_ROLES if per_role_valid[r] == 0]
    duplicate_queries = len({r["query"] for r in rows if r["valid"]}) != valid_n

    print(
        f"[summary] rows={len(rows)}/{expected} valid={valid_n}/{expected} "
        f"valid_rate={valid_n / max(1, expected):.3f} "
        f"duplicate_queries={duplicate_queries}",
        flush=True,
    )

    if len(rows) != expected:
        raise SystemExit(f"unexpected action count: {len(rows)} != {expected}")
    if missing_roles:
        raise SystemExit(f"no valid V11 probe for roles: {missing_roles}")
    if duplicate_queries:
        raise SystemExit("duplicate valid queries detected in action library")

    _write_jsonl_atomic(Path(args.output), rows)
    print(
        f"[done] ORIGINAL V11 action_library={args.output} rows={len(rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
