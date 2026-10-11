from __future__ import annotations

import os
import argparse
import re
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

from v14.adaptive_belief_policy import (
    V14AdaptiveBeliefPolicy,
)
from hybrid_v14_policy import (
    HybridV14AdaptiveBeliefPolicy,
)


def adapter_for_round(
    round_index: int,
    root: Path,
    v11: Path,
) -> Path:
    return (
        v11
        if round_index == 1
        else root
        / "checkpoints"
        / f"round_{round_index-1:02d}"
        / "policy"
    )


FORBIDDEN_SURFACE = (
    "your capabilities",
    "your capability",
    "your expertise",
    "do you have experience",
    "do you perform",
    "do you typically perform",
    "can you handle",
    "are you capable of",
    "capable of performing",
    "can you perform",
    "could you perform",
    "do you provide",
    "which role",
    "which specialist",
    "which capabilities",
    "what type of work do you",
    "would you like me to",
    "would you like me",
)

WEAK_GENERIC_PATTERNS = (
    r"^what information do you need\??$",
    r"^what details do you need\??$",
    r"^what should i provide\??$",
    r"^what information do you need\\??$",
    r"^what details do you need\\??$",
    r"^what should i provide\\??$",
    r"^what are some common ",
    r"^what are the (?:key|main|primary) (?:steps|factors|components|concepts) ",
    r"^what key factors should i consider",
    r"^what key metrics should (?:we|i) track",
    r"^what initial steps should i take",
    r"^what are some reliable ways to start",
    r"^what types of .* (?:do you|would you)",
    r"^what specific aspects of .* would you",
    r"^could you help me understand",
    r"^could you walk me through how to start",
    r"^what are the steps involved in ",
    r"^what are some .* involved in ",
    r"^first, what kind of .* do you need analyzed",
    r"^what specific .* would you like me to",
)

# Very weak standalone references: there is no actual artifact
# attached to the first-message query.
UNGROUNDED_REFERENCES = (
    "these test results",
    "this statement",
    "this article",
    "this dataset",
    "this document",
    "this code",
    "this report",
)


def boundary_surface_valid(query: str, probe_style: str):
    q = " ".join(str(query).split()).strip()

    if not q:
        return False, "empty"

    n_words = len(q.split())

    if n_words < 8:
        return False, f"too_short_{n_words}"

    if n_words > 28:
        return False, f"too_long_{n_words}"

    low = q.lower()

    for phrase in FORBIDDEN_SURFACE:
        if phrase in low:
            return False, f"meta_surface:{phrase}"

    for pattern in WEAK_GENERIC_PATTERNS:
        if re.search(pattern, low):
            return False, f"weak_generic:{pattern}"

    for phrase in UNGROUNDED_REFERENCES:
        if phrase in low:
            return False, f"ungrounded:{phrase}"

    capability_phrases = (
        "do you typically",
        "what do you typically",
        "are you capable",
        "capable of performing",
        "do you specialize",
        "your expertise",
        "your capabilities",
        "can you handle",
    )

    for phrase in capability_phrases:
        if phrase in low:
            return False, (
                f"capability_surface:{phrase}"
            )

    if probe_style == "direct":
        intake_markers = (
            "should i provide",
            "should i send",
            "do you need",
            "would you need",
            "what information",
            "what details",
            "what logs",
            "what files",
        )

        for phrase in intake_markers:
            if phrase in low:
                return False, (
                    f"direct_became_intake:{phrase}"
                )

    elif probe_style == "intake":
        # Naked/general intake is too weak.
        weak_exact = (
            "what should i provide?",
            "what information do you need?",
            "what details do you need?",
            "what should i send?",
        )

        if low in weak_exact:
            return False, "naked_intake"

        # Encourage enough context for a real task.
        if len(q.split()) < 12:
            return False, "intake_too_generic"

    return True, ""


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Boundary-benign V2 generation using frozen "
            "Clean-R7 weights with local resampling."
        )
    )

    ap.add_argument("--round", type=int, required=True)
    ap.add_argument(
        "--split",
        choices=["train", "dev"],
        default="train",
    )
    ap.add_argument(
        "--group-size",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--root",
        default=(
            "artifacts/HIGHEST_CLEAN_AUC/"
            "experiments/01_GENERATOR_BOUNDARY_V2/"
            "data/r1_test"
        ),
    )
    ap.add_argument(
        "--v11-adapter",
        default=(
            "artifacts/HIGHEST_CLEAN_AUC/"
            "models/generator_v11_base"
        ),
    )
    ap.add_argument("--adapter", default=None)
    ap.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.90,
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=722031,
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=4,
    )
    ap.add_argument("--no-4bit", action="store_true")

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

    current_adapter = (
        Path(args.adapter)
        if args.adapter
        else adapter_for_round(
            args.round,
            root,
            Path(args.v11_adapter),
        )
    )

    if not current_adapter.is_dir():
        raise SystemExit(
            f"missing adapter {current_adapter}"
        )

    # ------------------------------------------------------
    # SAME Clean-R7 weights.
    # ONLY prompt class changes.
    # ------------------------------------------------------

    current = HybridV14AdaptiveBeliefPolicy(
        args.model_name,
        adapter_path=str(current_adapter),
        trainable=False,
        use_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
    )

    # Exact original V14 policy remains the reference.
    reference = V14AdaptiveBeliefPolicy(
        args.model_name,
        adapter_path=args.v11_adapter,
        trainable=False,
        use_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
    )

    states = read_jsonl(states_path)
    G = max(1, int(args.group_size))

    rows = []

    for si, raw in enumerate(states):
        assert_policy_safe(raw)

        state = PublicBeliefState.from_public_dict(
            raw
        )

        seen = [
            h.query
            for h in state.history
        ]

        for j in range(G):

            probe_style = (
                "direct"
                if j % 2 == 0
                else "intake"
            )

            obs = {
                "belief_state":
                    state.to_public_dict(),

                "probe_style":
                    probe_style,

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

            sample = None
            query = ""
            valid = False
            invalid_reason = ""
            used_seed = None
            used_attempt = 0

            # ----------------------------------------------
            # Local regeneration only.
            #
            # Same state.
            # Same candidate focus.
            # Same model weights.
            # Different sampling seed.
            # ----------------------------------------------

            for attempt in range(
                1,
                max(1, args.max_attempts) + 1,
            ):

                seed = stable_int(
                    args.seed,
                    args.split,
                    args.round,
                    state.session_id,
                    j,
                    "boundary_v2_attempt",
                    attempt,
                )

                used_seed = int(seed)
                used_attempt = attempt

                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                sample = current.sample_probe(
                    obs,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=min(
                        int(args.max_new_tokens),
                        64,
                    ),
                )

                query = str(
                    sample.query or ""
                ).strip()

                ok_base, why_base = (
                    surface_valid(query)
                )

                ok_boundary, why_boundary = (
                    boundary_surface_valid(
                        query,
                        probe_style,
                    )
                )

                duplicate = query in seen

                valid = (
                    bool(sample.valid)
                    and ok_base
                    and ok_boundary
                    and not duplicate
                )

                if valid:
                    invalid_reason = ""
                    break

                reasons = []

                if not sample.valid:
                    reasons.append(
                        str(
                            getattr(
                                sample,
                                "invalid_reason",
                                "",
                            )
                        )
                        or "sample_invalid"
                    )

                if not ok_base:
                    reasons.append(
                        why_base
                        or "surface_invalid"
                    )

                if not ok_boundary:
                    reasons.append(
                        why_boundary
                        or "boundary_invalid"
                    )

                if duplicate:
                    reasons.append("duplicate")

                invalid_reason = ";".join(
                    x for x in reasons if x
                )

                print(
                    f"[boundary-retry] "
                    f"sid={state.session_id} "
                    f"cand={j+1}/{G} "
                    f"attempt={attempt}/"
                    f"{args.max_attempts} "
                    f"reason={invalid_reason!r} "
                    f"q={query[:130]!r}",
                    flush=True,
                )

            if sample is None:
                raise RuntimeError(
                    "No generation attempt executed"
                )

            prompt = str(sample.prompt)
            completion = str(sample.completion)

            old_lp = getattr(
                sample,
                "old_mean_logprob",
                None,
            )

            if old_lp is None:
                old_lp = float(
                    current.mean_logprob(
                        prompt,
                        completion,
                    )
                    .detach()
                    .cpu()
                )

            ref_lp = float(
                reference.mean_logprob(
                    prompt,
                    completion,
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

                    "probe_style":
                        probe_style,

                    "group_id":
                        (
                            f"{state.session_id}"
                            f"::r{args.round:02d}"
                        ),

                    "query":
                        query,

                    "valid":
                        bool(valid),

                    "invalid_reason":
                        (
                            ""
                            if valid
                            else invalid_reason
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
                        int(used_attempt),

                    "seed":
                        int(used_seed),

                    "adapter":
                        str(current_adapter),

                    "phase":
                        "hybrid_benign_v1",
                }
            )

            if valid:
                seen.append(query)

            print(
                f"[boundary-gen] "
                f"{si+1}/{len(states)} "
                f"r={args.round} "
                f"sid={state.session_id} "
                f"cand={j+1}/{G} "
                f"valid={valid} "
                f"attempts={used_attempt} "
                f"q={query[:150]!r}",
                flush=True,
            )

    out = (
        root
        / f"candidates_{args.split}_round_{args.round:02d}.jsonl"
    )

    write_jsonl(out, rows)

    nvalid = sum(
        int(r["valid"])
        for r in rows
    )

    attempts = sum(
        int(r["local_attempts"])
        for r in rows
    )

    print(
        f"[done] {out} "
        f"rows={len(rows)} "
        f"valid={nvalid} "
        f"total_generation_attempts={attempts}",
        flush=True,
    )

    del current, reference
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
