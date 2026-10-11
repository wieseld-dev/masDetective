from __future__ import annotations

import gc
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from rl.generative_probe_env_v7 import ResponseCacheV7
from v8.counterfactual import get_k, get_mode, get_roles
from v8.embedding import FrozenOutputEmbedderV8
from v8.execution import execute_cached
from v10.gate import BenignGateV10
from v10.reward import PairwiseRobustCounterfactualGraderV10
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS, anchor_probe
from training.train_cross_role_probe_discovery_v11 import (
    _build_stratified_schedule,
    _norm_text,
    _word_jaccard,
)
from v13.hard_negative import (
    build_hard_pairs,
    build_neighbor_map,
    composition_key,
    excluded_v12_compositions,
    make_fresh_fanout_specs,
)
from v13.policy import DisambiguationProbePolicyV13


ROOT = Path("artifacts/v14_generator_ab_pilot_v1")
ROOT.mkdir(parents=True, exist_ok=True)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ARMS = {
    "BASE_V11": "artifacts/v11_policy",
    "NEW_V14": "artifacts/v14_benign_policy",
}

SEED = 20260929

# Deterministic, not hand-picked from V14 results:
# indices 0,2,4,...,18 of the frozen 20-role universe.
PILOT_ROLES = tuple(list(TRAIN_ROLES)[::2])

CANDIDATES_PER_ROLE = 2
TRAIN_SYSTEMS_PER_CELL = 4
EVAL_SYSTEMS_PER_CELL = 4

TEMPERATURE = 0.90
TOP_P = 0.92


def dump_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pair_compositions(p) -> set[tuple[str, ...]]:
    out = set()
    for spec in (p.plus_spec, p.minus_spec, p.sham_spec):
        if spec is not None:
            out.add(composition_key(get_roles(spec)))
    return out


def reward_to_dict(rb) -> dict:
    return {
        "total": float(rb.total),
        "eligible": bool(rb.eligible),
        "activation_rate": float(rb.activation_rate),
        "plus_activation_rate": float(rb.plus_activation_rate),
        "sham_activation_rate": float(rb.sham_activation_rate),
        "pair_count": int(rb.pair_count),
        "specific_pairwise": [float(x) for x in rb.specific_pairwise],
        "specific_mean": float(rb.specific_mean),
        "specific_median": float(rb.specific_median),
        "specific_q25": float(rb.specific_q25),
        "robust_specific": float(rb.robust_specific),
        "positive_specific_pair_rate": float(rb.positive_specific_pair_rate),
        "strict_specific_pass": bool(rb.strict_specific_pass),
        "strict_delta_pass": bool(rb.strict_delta_pass),
    }


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    for name, path in ARMS.items():
        if not Path(path).exists():
            raise SystemExit(f"Missing {name} adapter: {path}")

    print("=" * 88, flush=True)
    print("V14 GENERATOR A/B PILOT", flush=True)
    print("=" * 88, flush=True)
    print("BASE :", ARMS["BASE_V11"], flush=True)
    print("NEW  :", ARMS["NEW_V14"], flush=True)
    print("roles:", PILOT_ROLES, flush=True)
    print("candidates/role:", CANDIDATES_PER_ROLE, flush=True)
    print(
        "IMPORTANT: same V13 policy class/prompt for both arms; only adapter weights differ.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 1. Build public semantic neighbor map.
    # ------------------------------------------------------------------
    embedder = FrozenOutputEmbedderV8(device="cpu")
    neighbor_map = build_neighbor_map(
        embedder,
        TRAIN_ROLES,
        top_k=5,
    )

    # ------------------------------------------------------------------
    # 2. Reconstruct V14/V13 training compositions and EXCLUDE them.
    #    This mirrors the known training seed/setup.
    # ------------------------------------------------------------------
    excluded = set(excluded_v12_compositions(TRAIN_ROLES))

    train_specs = make_fresh_fanout_specs(
        TRAIN_ROLES,
        prefix="v13_train",
        systems_per_cell=TRAIN_SYSTEMS_PER_CELL,
        seed=1313,
        excluded_compositions=excluded,
    )

    train_pairs = build_hard_pairs(
        train_specs,
        TRAIN_ROLES,
        neighbor_map,
    )

    training_compositions = set(excluded)

    for s in train_specs:
        training_compositions.add(
            composition_key(s.worker_roles)
        )

    for p in train_pairs:
        training_compositions |= pair_compositions(p)

    print(
        "excluded training/previous compositions:",
        len(training_compositions),
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Fresh evaluation compositions.
    # ------------------------------------------------------------------
    eval_specs = make_fresh_fanout_specs(
        TRAIN_ROLES,
        prefix="v14_ab_fresh",
        systems_per_cell=EVAL_SYSTEMS_PER_CELL,
        seed=SEED,
        excluded_compositions=training_compositions,
    )

    raw_eval_pairs = build_hard_pairs(
        eval_specs,
        TRAIN_ROLES,
        neighbor_map,
    )

    # Extra protection: drop any counterfactual pair whose PLUS/MINUS/SHAM
    # composition accidentally equals a training composition.
    eval_pairs = [
        p
        for p in raw_eval_pairs
        if pair_compositions(p).isdisjoint(training_compositions)
    ]

    by_role = defaultdict(list)
    for p in eval_pairs:
        by_role[p.target_role].append(p)

    missing = {
        r: len(by_role[r])
        for r in PILOT_ROLES
        if len(by_role[r]) < 2
    }
    if missing:
        raise RuntimeError(
            f"Not enough fresh disjoint pairs for pilot roles: {missing}"
        )

    # One primary + one disjoint confirmation pair per role.
    schedule, confirmation_schedule = _build_stratified_schedule(
        by_role,
        list(PILOT_ROLES),
        rounds=1,
        bundle_size=1,
        confirmation_size=1,
        seed=SEED,
    )

    manifest = []
    for role in PILOT_ROLES:
        p = schedule[(1, role)][0]
        c = confirmation_schedule[(1, role)][0]

        manifest.append(
            {
                "role": role,
                "primary_pair_id": p.pair_id,
                "primary_mode": get_mode(p.plus_spec),
                "primary_k": get_k(p.plus_spec),
                "primary_plus_roles": list(get_roles(p.plus_spec)),
                "primary_minus_roles": list(get_roles(p.minus_spec)),
                "primary_sham_roles": list(get_roles(p.sham_spec)),
                "confirmation_pair_id": c.pair_id,
                "confirmation_mode": get_mode(c.plus_spec),
                "confirmation_k": get_k(c.plus_spec),
                "confirmation_plus_roles": list(get_roles(c.plus_spec)),
                "confirmation_minus_roles": list(get_roles(c.minus_spec)),
                "confirmation_sham_roles": list(get_roles(c.sham_spec)),
            }
        )

    dump_json(ROOT / "manifest_eval_only.json", manifest)

    # ------------------------------------------------------------------
    # 4. Generate queries.
    #
    # SAME prompt class.
    # SAME observation.
    # SAME random seeds.
    # ONLY adapter weights differ.
    # ------------------------------------------------------------------
    gate = BenignGateV10(
        mode="hard",
        guard_model=None,
    )

    generated = []

    for arm, adapter in ARMS.items():
        print("\n" + "=" * 88, flush=True)
        print(f"GENERATING ARM={arm}", flush=True)
        print("=" * 88, flush=True)

        policy = DisambiguationProbePolicyV13(
            MODEL,
            adapter_path=adapter,
            trainable=False,
            use_4bit=True,
            max_seq_length=1536,
        )
        policy.model.eval()

        for ri, role in enumerate(PILOT_ROLES):
            anchor = anchor_probe(role)
            competitors = neighbor_map[role][:4]

            obs = {
                "candidate_role": role,
                "candidate_description": ROLE_DESCRIPTIONS[role],
                "anchor_probe": anchor,
                "avoid_probes": [],
                "confusable_roles": [
                    {
                        "role": r,
                        "description": ROLE_DESCRIPTIONS[r],
                    }
                    for r in competitors
                ],
                "public_history": [],
            }

            group_seen = []

            for ci in range(CANDIDATES_PER_ROLE):
                final = None

                for retry in range(8):
                    # IMPORTANT: identical seed schedule across arms.
                    seed = (
                        SEED
                        + 10000 * ri
                        + 100 * ci
                        + retry
                    )
                    set_seed(seed)

                    sample = policy.sample_probe(
                        obs,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )

                    q = str(sample.query or "").strip()

                    status = "ok"

                    if not sample.valid or not q:
                        status = "invalid"
                    elif _norm_text(q) == _norm_text(anchor):
                        status = "anchor_duplicate"
                    elif any(
                        _word_jaccard(q, old) >= 0.95
                        for old in group_seen
                    ):
                        status = "group_duplicate"
                    else:
                        gd = gate.check(q)
                        if not gd.allowed:
                            status = (
                                f"blocked:"
                                f"{getattr(gd, 'stage', '')}:"
                                f"{getattr(gd, 'category', '')}"
                            )

                    final = {
                        "arm": arm,
                        "adapter": adapter,
                        "role": role,
                        "candidate_index": ci + 1,
                        "seed": seed,
                        "query": q,
                        "generation_status": status,
                        "valid": status == "ok",
                    }

                    if status == "ok":
                        group_seen.append(q)
                        break

                generated.append(final)
                print(
                    f"{arm:<9} "
                    f"{role:<26} "
                    f"cand={ci+1} "
                    f"{final['generation_status']:<18} | "
                    f"{final['query']}",
                    flush=True,
                )

        del policy
        gc.collect()
        torch.cuda.empty_cache()

    generated_path = ROOT / "generated_queries.jsonl"
    if generated_path.exists():
        generated_path.unlink()

    for row in generated:
        append_jsonl(generated_path, row)

    # ------------------------------------------------------------------
    # 5. Evaluate with the SAME counterfactual grader.
    # ------------------------------------------------------------------
    grader = PairwiseRobustCounterfactualGraderV10(
        min_activation_rate=0.75,
        novelty_free_similarity=0.90,
        novelty_weight=0.25,
    )

    cache = ResponseCacheV7(
        ROOT / "response_public.jsonl",
        ROOT / "response_truth_training_only.jsonl",
    )

    def emb(text: str):
        return embedder.encode(text)

    def eval_query(
        role: str,
        query: str,
        bundle,
        anchor: str,
    ):
        plus_q = []
        minus_q = []
        sham_q = []

        plus_a = []
        minus_a = []
        sham_a = []

        selected = []
        sham_selected = []

        branches = []

        for p in bundle:
            # Anchor executions are cached and therefore shared across
            # candidates and both A/B arms.
            pa = execute_cached(p.plus_spec, anchor, cache)
            ma = execute_cached(p.minus_spec, anchor, cache)
            sa = execute_cached(p.sham_spec, anchor, cache)

            pq = execute_cached(p.plus_spec, query, cache)
            mq = execute_cached(p.minus_spec, query, cache)
            sq = execute_cached(p.sham_spec, query, cache)

            plus_a.append(emb(pa.output))
            minus_a.append(emb(ma.output))
            sham_a.append(emb(sa.output))

            plus_q.append(emb(pq.output))
            minus_q.append(emb(mq.output))
            sham_q.append(emb(sq.output))

            selected.append(
                role in set(pq.selected_roles_training_only)
            )
            sham_selected.append(
                role in set(sq.selected_roles_training_only)
            )

            branches.append(
                {
                    "pair_id": p.pair_id,
                    "plus_selected": list(
                        pq.selected_roles_training_only
                    ),
                    "minus_selected": list(
                        mq.selected_roles_training_only
                    ),
                    "sham_selected": list(
                        sq.selected_roles_training_only
                    ),
                    "plus_output": pq.output,
                    "minus_output": mq.output,
                    "sham_output": sq.output,
                }
            )

        rb = grader.score(
            target_selected_flags=selected,
            sham_target_selected_flags=sham_selected,
            plus_query_embs=plus_q,
            minus_query_embs=minus_q,
            plus_anchor_embs=plus_a,
            minus_anchor_embs=minus_a,
            sham_query_embs=sham_q,
            sham_anchor_embs=sham_a,
            # We compare fingerprint separation here, not novelty.
            semantic_max_similarity=0.0,
        )

        return rb, branches

    result_path = ROOT / "results.jsonl"
    if result_path.exists():
        result_path.unlink()

    results = []

    for i, row in enumerate(generated, 1):
        rec = dict(row)

        if not rec["valid"]:
            rec["primary"] = None
            rec["confirmation"] = None
            rec["confirmed"] = False
            results.append(rec)
            append_jsonl(result_path, rec)
            continue

        role = rec["role"]
        anchor = anchor_probe(role)

        primary_bundle = schedule[(1, role)]
        confirm_bundle = confirmation_schedule[(1, role)]

        print(
            f"\nEVAL {i}/{len(generated)} "
            f"arm={rec['arm']} role={role} "
            f"cand={rec['candidate_index']}",
            flush=True,
        )

        rb, branches = eval_query(
            role,
            rec["query"],
            primary_bundle,
            anchor,
        )

        rec["primary"] = reward_to_dict(rb)
        rec["primary_branches_training_only"] = branches

        confirmed = False

        if rb.strict_specific_pass:
            crb, cbranches = eval_query(
                role,
                rec["query"],
                confirm_bundle,
                anchor,
            )

            rec["confirmation"] = reward_to_dict(crb)
            rec["confirmation_branches_training_only"] = (
                cbranches
            )

            confirmed = bool(
                crb.strict_specific_pass
            )
        else:
            rec["confirmation"] = None
            rec["confirmation_branches_training_only"] = []

        rec["confirmed"] = confirmed

        print(
            f"  ROB={rb.robust_specific:+.4f} "
            f"ACT={rb.activation_rate:.2f} "
            f"strict={rb.strict_specific_pass} "
            f"confirmed={confirmed}",
            flush=True,
        )

        results.append(rec)
        append_jsonl(result_path, rec)

    # ------------------------------------------------------------------
    # 6. Summaries.
    # ------------------------------------------------------------------
    def summarize(arm: str) -> dict:
        rr = [x for x in results if x["arm"] == arm]

        valid = [x for x in rr if x["valid"]]
        evaluated = [
            x for x in valid
            if x.get("primary") is not None
        ]

        strict = [
            x for x in evaluated
            if x["primary"]["strict_specific_pass"]
        ]

        confirmed = [
            x for x in evaluated
            if x.get("confirmed")
        ]

        robs = np.asarray(
            [
                x["primary"]["robust_specific"]
                for x in evaluated
            ],
            dtype=float,
        )

        role_best = {}
        for role in PILOT_ROLES:
            xs = [
                x["primary"]["robust_specific"]
                for x in evaluated
                if x["role"] == role
            ]
            role_best[role] = (
                float(max(xs))
                if xs
                else None
            )

        best_vals = [
            x
            for x in role_best.values()
            if x is not None
        ]

        confirm_robs = [
            x["confirmation"]["robust_specific"]
            for x in confirmed
            if x.get("confirmation")
        ]

        return {
            "arm": arm,
            "n_generated": len(rr),
            "n_valid": len(valid),
            "valid_rate": (
                len(valid) / len(rr)
                if rr else None
            ),
            "n_evaluated": len(evaluated),
            "strict_count": len(strict),
            "strict_rate": (
                len(strict) / len(evaluated)
                if evaluated else None
            ),
            "confirmed_count": len(confirmed),
            "confirmation_rate_given_strict": (
                len(confirmed) / len(strict)
                if strict else None
            ),
            "confirmed_rate_all": (
                len(confirmed) / len(evaluated)
                if evaluated else None
            ),
            "roles_with_confirmed_probe": sorted(
                {
                    x["role"]
                    for x in confirmed
                }
            ),
            "role_confirmed_coverage": (
                len(
                    {
                        x["role"]
                        for x in confirmed
                    }
                )
                / len(PILOT_ROLES)
            ),
            "mean_primary_ROB": (
                float(robs.mean())
                if len(robs) else None
            ),
            "median_primary_ROB": (
                float(np.median(robs))
                if len(robs) else None
            ),
            "positive_ROB_rate": (
                float(np.mean(robs > 0))
                if len(robs) else None
            ),
            "mean_best_of_2_ROB_per_role": (
                float(np.mean(best_vals))
                if best_vals else None
            ),
            "median_best_of_2_ROB_per_role": (
                float(np.median(best_vals))
                if best_vals else None
            ),
            "mean_confirmation_ROB_confirmed_only": (
                float(np.mean(confirm_robs))
                if confirm_robs else None
            ),
            "best_ROB_by_role": role_best,
        }

    base = summarize("BASE_V11")
    new = summarize("NEW_V14")

    # Pair candidate positions by role/index, since generation used
    # exactly the same seed schedule.
    base_lookup = {
        (x["role"], x["candidate_index"]): x
        for x in results
        if x["arm"] == "BASE_V11"
        and x.get("primary")
    }

    new_lookup = {
        (x["role"], x["candidate_index"]): x
        for x in results
        if x["arm"] == "NEW_V14"
        and x.get("primary")
    }

    deltas = []
    wins = 0
    ties = 0
    pairs_compared = 0

    for key in sorted(
        set(base_lookup) & set(new_lookup)
    ):
        b = base_lookup[key]["primary"]["robust_specific"]
        n = new_lookup[key]["primary"]["robust_specific"]

        d = float(n - b)
        deltas.append(d)
        pairs_compared += 1

        if d > 1e-12:
            wins += 1
        elif abs(d) <= 1e-12:
            ties += 1

    comparison = {
        "paired_positions_compared": pairs_compared,
        "mean_delta_ROB_NEW_minus_BASE": (
            float(np.mean(deltas))
            if deltas else None
        ),
        "median_delta_ROB_NEW_minus_BASE": (
            float(np.median(deltas))
            if deltas else None
        ),
        "NEW_win_rate_on_ROB": (
            wins / pairs_compared
            if pairs_compared else None
        ),
        "tie_rate": (
            ties / pairs_compared
            if pairs_compared else None
        ),
        "strict_rate_delta": (
            new["strict_rate"] - base["strict_rate"]
            if (
                new["strict_rate"] is not None
                and base["strict_rate"] is not None
            )
            else None
        ),
        "confirmed_rate_all_delta": (
            new["confirmed_rate_all"]
            - base["confirmed_rate_all"]
            if (
                new["confirmed_rate_all"] is not None
                and base["confirmed_rate_all"] is not None
            )
            else None
        ),
        "best_of_2_ROB_delta": (
            new["mean_best_of_2_ROB_per_role"]
            - base["mean_best_of_2_ROB_per_role"]
            if (
                new["mean_best_of_2_ROB_per_role"] is not None
                and base["mean_best_of_2_ROB_per_role"] is not None
            )
            else None
        ),
    }

    summary = {
        "protocol": (
            "v14_generator_ab_pilot_same_prompt_"
            "same_seed_fresh_counterfactuals"
        ),
        "pilot_roles": list(PILOT_ROLES),
        "candidates_per_role": CANDIDATES_PER_ROLE,
        "primary_pairs_per_candidate": 1,
        "confirmation_pairs_if_strict": 1,
        "base_adapter": ARMS["BASE_V11"],
        "new_adapter": ARMS["NEW_V14"],
        "same_policy_class": "DisambiguationProbePolicyV13",
        "fresh_eval_seed": SEED,
        "v14_training_seed_reconstructed": 1313,
        "old_internal_test_used": False,
        "BASE_V11": base,
        "NEW_V14": new,
        "comparison": comparison,
    }

    dump_json(ROOT / "summary.json", summary)

    print("\n" + "=" * 88, flush=True)
    print("FINAL A/B SUMMARY", flush=True)
    print("=" * 88, flush=True)

    for arm, s in (
        ("BASE_V11", base),
        ("NEW_V14", new),
    ):
        print(
            f"{arm:<9} "
            f"strict={s['strict_count']}/{s['n_evaluated']} "
            f"({s['strict_rate']:.3f}) | "
            f"confirmed={s['confirmed_count']}/{s['n_evaluated']} "
            f"({s['confirmed_rate_all']:.3f}) | "
            f"meanROB={s['mean_primary_ROB']:+.4f} | "
            f"medianROB={s['median_primary_ROB']:+.4f} | "
            f"best2ROB={s['mean_best_of_2_ROB_per_role']:+.4f} | "
            f"coverage={s['role_confirmed_coverage']:.3f}",
            flush=True,
        )

    print("\nNEW - BASE:", flush=True)
    print(
        json.dumps(
            comparison,
            indent=2,
        ),
        flush=True,
    )

    print(
        "\nsummary:",
        ROOT / "summary.json",
        flush=True,
    )
    print(
        "queries:",
        ROOT / "generated_queries.jsonl",
        flush=True,
    )
    print(
        "results:",
        ROOT / "results.jsonl",
        flush=True,
    )


if __name__ == "__main__":
    main()
