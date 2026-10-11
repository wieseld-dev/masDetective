from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path

from adaptive_v11_active.io import read_jsonl, stable_int, write_jsonl
from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe
from v10.policy import SeededProbePolicyV10
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES, anchor_probe


def binary_entropy(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def static_role_order(session_id: str, seed: int) -> list[str]:
    # Fixed per-system permutation. It depends on no answers/evidence.
    return sorted(
        TRAIN_ROLES,
        key=lambda r: stable_int(seed, "static-role-order", session_id, r),
    )


def previous_targets(root: Path, split: str, round_index: int, session_id: str) -> list[str]:
    out: list[str] = []
    for rr in range(1, round_index):
        p = root / f"candidates_{split}_round_{rr:02d}.jsonl"
        if not p.exists():
            continue
        for row in read_jsonl(p):
            if str(row.get("session_id", "")) == session_id:
                role = str(row.get("target_role", "")).strip()
                if role:
                    out.append(role)
                break
    return out


def choose_target(
    state: PublicBeliefState,
    *,
    controller: str,
    seed: int,
    root: Path,
    split: str,
    round_index: int,
) -> tuple[str, float, float, list[str]]:
    order = static_role_order(state.session_id, seed)
    tested = previous_targets(root, split, round_index, state.session_id)

    # Shared round 1: both arms begin with exactly the same target role.
    if controller == "common":
        if round_index != 1:
            raise ValueError("controller=common is only valid for round 1")
        role = order[0]
    elif controller == "static":
        role = order[(round_index - 1) % len(order)]
    elif controller == "adaptive":
        if round_index == 1:
            role = order[0]
        else:
            untested = [r for r in TRAIN_ROLES if r not in set(tested)]
            pool = untested if untested else list(TRAIN_ROLES)
            # Highest posterior uncertainty. Stable per-system fixed-order rank only breaks ties.
            rank = {r: i for i, r in enumerate(order)}
            role = max(
                pool,
                key=lambda r: (
                    binary_entropy(state.role_probs[r]),
                    -rank[r],
                ),
            )
    else:
        raise ValueError(controller)

    p = float(state.role_probs[role])
    return role, p, binary_entropy(p), tested


def main():
    ap = argparse.ArgumentParser(
        description="Generate one exact-V11 role-conditioned benign probe; controller chooses only the target role"
    )
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--controller", choices=["common", "static", "adaptive"], required=True)
    ap.add_argument("--adapter", default="artifacts/v11_policy")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--local-retries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=917411)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    root = Path(args.root)
    states_path = root / f"states_{args.split}_round_{args.round-1:02d}.jsonl"
    if not states_path.exists():
        raise SystemExit(f"missing state file: {states_path}")
    if not Path(args.adapter).is_dir():
        raise SystemExit(f"missing V11 adapter: {args.adapter}")

    # IMPORTANT: this is the original V11 generator API/prompt. The intervention is
    # only which candidate_role the external controller selects.
    policy = SeededProbePolicyV10(
        args.model_name,
        adapter_path=args.adapter,
        trainable=False,
        use_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
    )

    states = read_jsonl(states_path)
    rows: list[dict] = []

    for i, raw in enumerate(states):
        assert_policy_safe(raw)
        state = PublicBeliefState.from_public_dict(raw)
        state.validate(TRAIN_ROLES, sorted(state.k_probs))

        target, target_p, target_h, tested = choose_target(
            state,
            controller=args.controller,
            seed=args.seed,
            root=root,
            split=args.split,
            round_index=args.round,
        )

        previous_queries = [str(h.query).strip() for h in state.history if str(h.query).strip()]
        avoid = previous_queries[-6:]
        best = None

        for retry in range(max(1, args.local_retries)):
            # Controller name is deliberately NOT part of the seed. If two arms
            # select the same target with the same history, generation is matched.
            seed = stable_int(args.seed, args.split, args.round, state.session_id, target, retry)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            obs = {
                "candidate_role": target,
                "candidate_description": ROLE_DESCRIPTIONS[target],
                "anchor_probe": anchor_probe(target),
                "avoid_probes": avoid,
            }
            sample = policy.sample_probe(
                obs,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            q = str(getattr(sample, "query", "") or "").strip()
            ok, why = surface_valid(q)
            generator_valid = bool(getattr(sample, "valid", bool(q)))
            valid = generator_valid and ok and q not in set(previous_queries)

            ranked_uncertain = sorted(
                TRAIN_ROLES,
                key=lambda r: (-binary_entropy(state.role_probs[r]), r),
            )[:5]

            rec = {
                "session_id": state.session_id,
                "split": args.split,
                "round": args.round,
                "candidate_index": 0,
                "group_id": f"{state.session_id}::r{args.round:02d}",
                "controller": args.controller,
                "target_role": target,
                "target_role_probability_before": target_p,
                "target_role_entropy_before": target_h,
                "previous_target_roles": tested,
                "top_uncertain_roles_before": [
                    {"role": r, "p": float(state.role_probs[r]), "entropy": binary_entropy(state.role_probs[r])}
                    for r in ranked_uncertain
                ],
                "query": q,
                "valid": bool(valid),
                "invalid_reason": "" if valid else (
                    str(getattr(sample, "invalid_reason", "")) or why or "duplicate"
                ),
                "prompt": str(getattr(sample, "prompt", "")),
                "completion": str(getattr(sample, "completion", "")),
                "raw_completion": str(getattr(sample, "raw_completion", getattr(sample, "completion", ""))),
                "seed": int(seed),
                "adapter": str(args.adapter),
                "local_retry": int(retry),
                "phase": "simple_role_controller_v1",
            }
            best = rec
            if valid:
                break

        assert best is not None
        rows.append(best)
        print(
            f"[ctrl-gen] {i+1}/{len(states)} mode={args.controller} split={args.split} "
            f"r={args.round} sid={state.session_id} target={target} p={target_p:.3f} "
            f"valid={best['valid']} retry={best['local_retry']} q={best['query'][:140]!r}",
            flush=True,
        )

    out = root / f"candidates_{args.split}_round_{args.round:02d}.jsonl"
    write_jsonl(out, rows)
    print(
        f"[done] {out} rows={len(rows)} valid={sum(int(x['valid']) for x in rows)} "
        f"controller={args.controller}",
        flush=True,
    )

    del policy
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
