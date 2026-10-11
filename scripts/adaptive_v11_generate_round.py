from __future__ import annotations

import argparse
import gc
from pathlib import Path

from adaptive_v11_active.io import read_jsonl, stable_int, write_jsonl
from adaptive_v11_active.policy import AdaptiveProbePolicyV11
from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe
from v10.policy import SeededProbePolicyV10


def adapter_for_round(round_index: int, root: Path, v11: Path) -> Path:
    return v11 if round_index == 1 else root / "checkpoints" / f"round_{round_index-1:02d}" / "policy"


def main():
    ap = argparse.ArgumentParser(description="Generate fresh belief-conditioned Adaptive-V11 probe candidates")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--root", default="artifacts/adaptive_v11_active_v1")
    ap.add_argument("--v11-adapter", default="artifacts/v11_policy")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.90)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=722031)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for Adaptive-V11 generation")

    root = Path(args.root)
    states_path = root / f"states_{args.split}_round_{args.round-1:02d}.jsonl"
    if not states_path.exists():
        raise SystemExit(f"missing prior public state file: {states_path}")
    current_adapter = Path(args.adapter) if args.adapter else adapter_for_round(args.round, root, Path(args.v11_adapter))
    if not current_adapter.is_dir():
        raise SystemExit(f"missing current adapter: {current_adapter}")
    if not Path(args.v11_adapter).is_dir():
        raise SystemExit(f"missing frozen V11 reference: {args.v11_adapter}")

    current = AdaptiveProbePolicyV11(
        args.model_name, adapter_path=str(current_adapter), trainable=False,
        use_4bit=not args.no_4bit, max_seq_length=args.max_seq_length,
    )
    reference = SeededProbePolicyV10(
        args.model_name, adapter_path=args.v11_adapter, trainable=False,
        use_4bit=not args.no_4bit, max_seq_length=args.max_seq_length,
    )

    rows = []
    states = read_jsonl(states_path)
    G = args.group_size if args.split == "train" else max(1, args.group_size)
    for si, raw_state in enumerate(states):
        assert_policy_safe(raw_state)
        state = PublicBeliefState.from_public_dict(raw_state)
        seen = [h.query for h in state.history]
        for j in range(G):
            seed = stable_int(args.seed, args.split, args.round, state.session_id, j)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            obs = {
                "belief_state": state.to_public_dict(),
                "top_hypotheses": 6,
                "max_history_turns": 3,
                "max_answer_chars": 900,
            }
            sample = current.sample_probe(
                obs, temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_new_tokens
            )
            q = str(sample.query or "").strip()
            surface_ok, surface_reason = surface_valid(q)
            gen_valid = bool(sample.valid) and surface_ok and q not in seen
            prompt = str(sample.prompt)
            completion = str(sample.completion)
            # old_mean_logprob/ref_mean_logprob may be exposed by the sample; recompute if absent.
            old_lp = getattr(sample, "old_mean_logprob", None)
            if old_lp is None:
                old_lp = float(current.mean_logprob(prompt, completion).detach().cpu())
            ref_lp = float(reference.mean_logprob(prompt, completion).detach().cpu())
            rows.append({
                "session_id": state.session_id,
                "split": args.split,
                "round": args.round,
                "candidate_index": j,
                "group_id": f"{state.session_id}::r{args.round:02d}",
                "query": q,
                "valid": bool(gen_valid),
                "invalid_reason": "" if gen_valid else (str(getattr(sample, "invalid_reason", "")) or surface_reason or "duplicate"),
                "prompt": prompt,
                "completion": completion,
                "old_logp": float(old_lp),
                "ref_logp": float(ref_lp),
                "raw_completion": str(sample.raw_completion),
                "seed": int(seed),
                "adapter": str(current_adapter),
            })
            if gen_valid:
                seen.append(q)
            print(f"[generate] {si+1}/{len(states)} r={args.round} {state.session_id} cand={j+1}/{G} valid={gen_valid} q={q[:120]!r}", flush=True)

    out = root / f"candidates_{args.split}_round_{args.round:02d}.jsonl"
    write_jsonl(out, rows)
    print(f"[done] candidates={out} rows={len(rows)}", flush=True)

    del current, reference
    gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
