from __future__ import annotations

import os
import argparse
import gc
from pathlib import Path

from adaptive_v11_active.io import (
    read_jsonl,
    stable_int,
    write_jsonl,
)
from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import (
    PublicBeliefState,
    assert_policy_safe,
)

from q38.adaptive_belief_policy import (
    Q38AdaptiveBeliefPolicy,
)


def adapter_for_round(
    round_index: int,
    root: Path,
    initial: Path,
) -> Path:
    if round_index == 1:
        return initial

    return (
        root
        / "checkpoints"
        / f"round_{round_index-1:02d}"
        / "policy"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--round",
        type=int,
        required=True,
    )
    ap.add_argument(
        "--split",
        choices=["train", "dev"],
        default="train",
    )
    ap.add_argument(
        "--group-size",
        type=int,
        default=6,
    )
    ap.add_argument(
        "--root",
        required=True,
    )
    ap.add_argument(
        "--v11-adapter",
        default="artifacts/q38_utility_lora_v1/policy",
    )
    ap.add_argument(
        "--reference-adapter",
        default="artifacts/q38_utility_lora_v1/policy",
    )
    ap.add_argument(
        "--adapter",
        default=None,
    )
    ap.add_argument(
        "--model-name",
        default="Qwen/Qwen3.8-27B",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.85,
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=3072,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=722031,
    )
    ap.add_argument(
        "--no-4bit",
        action="store_true",
    )

    args = ap.parse_args()

    args.seed = int(
        os.environ.get(
            "AV11_SEED",
            args.seed,
        )
    )

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    root = Path(args.root)

    states_path = (
        root
        / f"states_{args.split}_round_{args.round-1:02d}.jsonl"
    )

    if not states_path.exists():
        raise SystemExit(
            f"missing {states_path}"
        )

    initial = Path(args.v11_adapter)

    current_adapter = (
        Path(args.adapter)
        if args.adapter
        else adapter_for_round(
            args.round,
            root,
            initial,
        )
    )

    if not current_adapter.is_dir():
        raise SystemExit(
            f"missing adapter {current_adapter}"
        )

    reference_adapter = Path(
        args.reference_adapter
    )

    if not reference_adapter.is_dir():
        raise SystemExit(
            f"missing reference {reference_adapter}"
        )

    policy = Q38AdaptiveBeliefPolicy(
        args.model_name,
        adapter_path=str(current_adapter),
        reference_adapter_path=str(
            reference_adapter
        ),
        trainable=False,
        use_4bit=False,
        max_seq_length=args.max_seq_length,
    )

    states = read_jsonl(
        states_path
    )

    G = max(
        1,
        int(args.group_size),
    )

    rows = []

    for si, raw in enumerate(states):
        assert_policy_safe(raw)

        state = (
            PublicBeliefState
            .from_public_dict(raw)
        )

        seen = [
            h.query
            for h in state.history
        ]

        for j in range(G):
            seed = stable_int(
                args.seed,
                args.split,
                args.round,
                state.session_id,
                j,
            )

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            obs = {
                "belief_state":
                    state.to_public_dict(),

                "candidate_index":
                    j
                    + (
                        stable_int(
                            "v14a_focus",
                            state.session_id,
                        )
                        % max(
                            1,
                            len(state.role_probs),
                        )
                    ),

                "max_history_turns": 2,
                "max_answer_chars": 0,
            }

            sample = policy.sample_probe(
                obs,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=min(
                    int(args.max_new_tokens),
                    64,
                ),
            )

            q = str(
                sample.query or ""
            ).strip()

            ok, why = surface_valid(q)

            valid = (
                bool(sample.valid)
                and ok
                and q not in seen
            )

            prompt = str(
                sample.prompt
            )

            completion = str(
                sample.completion
            )

            old_lp = getattr(
                sample,
                "old_mean_logprob",
                None,
            )

            if old_lp is None:
                with torch.no_grad():
                    old_lp = float(
                        policy.mean_logprob(
                            prompt,
                            completion,
                            adapter="current",
                        )
                        .detach()
                        .cpu()
                    )

            with torch.no_grad():
                ref_lp = float(
                    policy.mean_logprob(
                        prompt,
                        completion,
                        adapter="reference",
                    )
                    .detach()
                    .cpu()
                )

            rows.append(
                {
                    "session_id":
                        state.session_id,
                    "split":
                        args.split,
                    "round":
                        args.round,
                    "candidate_index":
                        j,
                    "group_id":
                        f"{state.session_id}::r{args.round:02d}",
                    "query":
                        q,
                    "valid":
                        bool(valid),
                    "invalid_reason":
                        ""
                        if valid
                        else (
                            str(
                                getattr(
                                    sample,
                                    "invalid_reason",
                                    "",
                                )
                            )
                            or why
                            or "duplicate"
                        ),
                    "prompt":
                        prompt,
                    "completion":
                        completion,
                    "old_logp":
                        float(old_lp),
                    "ref_logp":
                        float(ref_lp),
                    "raw_completion":
                        str(
                            sample.raw_completion
                        ),
                    "local_attempts":
                        int(
                            getattr(
                                sample,
                                "local_attempts",
                                1,
                            )
                        ),
                    "seed":
                        int(seed),
                    "adapter":
                        str(current_adapter),
                    "reference_adapter":
                        str(reference_adapter),
                    "phase":
                        "role_only",
                }
            )

            if valid:
                seen.append(q)

            print(
                f"[Q38-role-gen] "
                f"{si+1}/{len(states)} "
                f"split={args.split} "
                f"r={args.round} "
                f"sid={state.session_id} "
                f"cand={j+1}/{G} "
                f"valid={valid} "
                f"q={q[:150]!r}",
                flush=True,
            )

    out = (
        root
        / f"candidates_{args.split}_round_{args.round:02d}.jsonl"
    )

    write_jsonl(
        out,
        rows,
    )

    nvalid = sum(
        int(r["valid"])
        for r in rows
    )

    print(
        f"[done] {out} "
        f"rows={len(rows)} "
        f"valid={nvalid}",
        flush=True,
    )

    del policy
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
