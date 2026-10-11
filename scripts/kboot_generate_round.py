from __future__ import annotations

import argparse
import gc
from pathlib import Path

from adaptive_v11_active.io import read_jsonl, stable_int, write_jsonl
from adaptive_v11_active.policy import AdaptiveProbePolicyV11
from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe
from v10.policy import SeededProbePolicyV10


def main():
    ap = argparse.ArgumentParser(description="Generate one fresh full K-aware-prompt benign probe per session, with local retries")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--adapter", default="artifacts/v11_policy")
    ap.add_argument("--v11-adapter", default="artifacts/v11_policy")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--local-retries", type=int, default=6)
    ap.add_argument("--seed", type=int, default=917031)
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
        raise SystemExit(f"missing adapter: {args.adapter}")
    if not Path(args.v11_adapter).is_dir():
        raise SystemExit(f"missing frozen V11 reference: {args.v11_adapter}")

    current = AdaptiveProbePolicyV11(
        args.model_name,
        adapter_path=args.adapter,
        trainable=False,
        use_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
    )
    reference = SeededProbePolicyV10(
        args.model_name,
        adapter_path=args.v11_adapter,
        trainable=False,
        use_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
    )

    rows = []
    states = read_jsonl(states_path)
    for si, raw in enumerate(states):
        assert_policy_safe(raw)
        state = PublicBeliefState.from_public_dict(raw)
        seen = {h.query.strip() for h in state.history}
        best = None

        for retry in range(max(1, args.local_retries)):
            seed = stable_int(args.seed, args.split, args.round, state.session_id, retry)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            obs = {
                "belief_state": state.to_public_dict(),
                "top_hypotheses": 6,
                "max_history_turns": 3,
                "max_answer_chars": 900,
            }
            sample = current.sample_probe(
                obs,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            q = str(sample.query or "").strip()
            ok, why = surface_valid(q)
            valid = bool(sample.valid) and ok and q not in seen
            prompt = str(sample.prompt)
            completion = str(sample.completion)

            old_lp = getattr(sample, "old_mean_logprob", None)
            if old_lp is None:
                old_lp = float(current.mean_logprob(prompt, completion).detach().cpu())
            ref_lp = float(reference.mean_logprob(prompt, completion).detach().cpu())

            rec = {
                "session_id": state.session_id,
                "split": args.split,
                "round": args.round,
                "candidate_index": 0,
                "group_id": f"{state.session_id}::r{args.round:02d}",
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
                "adapter": str(args.adapter),
                "local_retry": retry,
                "phase": "k_bootstrap_uniform_k_role_adaptive",
            }
            best = rec
            if valid:
                break

        assert best is not None
        rows.append(best)
        print(
            f"[kboot-gen] {si+1}/{len(states)} split={args.split} r={args.round} "
            f"sid={state.session_id} valid={best['valid']} retry={best['local_retry']} "
            f"q={best['query'][:150]!r}",
            flush=True,
        )

    out = root / f"candidates_{args.split}_round_{args.round:02d}.jsonl"
    write_jsonl(out, rows)
    print(f"[done] {out} rows={len(rows)} valid={sum(int(x['valid']) for x in rows)}", flush=True)

    del current, reference
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
