from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from estimator.candidate_presence_contrastive_v64 import load_profile
from probes.role_probe_bank import primary_probe
from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_zoo_v10_final import get_v10_final_specs, validate_final_design
from v8.counterfactual import build_counterfactual_pairs, get_k, get_mode
from v8.embedding import FrozenOutputEmbedderV8
from v8.execution import execute_cached
from v10.reward import PairwiseRobustCounterfactualGraderV10


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: str | Path, obj: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=p.parent, delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, p)


def _role_pair_table(pairs, roles):
    out = {r: [] for r in roles}
    for p in pairs:
        out[p.target_role].append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "POST-HOC diagnostic on the already-consumed V10 final set. "
            "Evaluates the exact V10 confirmed training elites; NO TRAINING."
        )
    )
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--v10-summary", default="artifacts/v10_summary.json")
    ap.add_argument("--final-summary", default="artifacts/v10_final_summary.json")
    ap.add_argument("--rows", default="artifacts/v10_final_elite_diagnostic_rows.jsonl")
    ap.add_argument("--summary", default="artifacts/v10_final_elite_diagnostic_summary.json")
    ap.add_argument("--response-public-cache", default="artifacts/v10_final_response_public.jsonl")
    ap.add_argument("--response-truth-cache", default="artifacts/v10_final_response_truth.jsonl")
    ap.add_argument("--min-activation-rate", type=float, default=0.75)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    validate_final_design()
    if not Path(args.v10_summary).exists():
        raise SystemExit(f"missing V10 summary: {args.v10_summary}")
    if not Path(args.final_summary).exists():
        raise SystemExit(
            f"missing consumed final summary: {args.final_summary}. "
            "This diagnostic must only run after the original final reveal."
        )

    v10 = _load_json(args.v10_summary)
    final = _load_json(args.final_summary)
    if not bool(final.get("test_consumed")):
        raise SystemExit("refusing: final set is not marked consumed")

    profile = load_profile(args.profile)
    roles = tuple(profile["candidate_roles"])
    specs = get_v10_final_specs()
    pairs = build_counterfactual_pairs(specs, roles, include_sham=True)
    by_role = _role_pair_table(pairs, roles)
    if len(specs) != 16 or len(pairs) != 72 or any(len(by_role[r]) != 9 for r in roles):
        raise RuntimeError("unexpected final-set structure")

    elites_by_role = {r: [] for r in roles}
    raw_elites = v10.get("elite_probes", {})
    for role in roles:
        seen = set()
        for row in raw_elites.get(role, []):
            q = str(row.get("query", "")).strip()
            if q and q not in seen:
                seen.add(q)
                elites_by_role[role].append({
                    "query": q,
                    "train_reward": row.get("reward"),
                    "train_breakdown": row.get("breakdown"),
                    "confirmation_breakdown": row.get("confirmation_breakdown"),
                })

    n_elites = sum(len(v) for v in elites_by_role.values())
    print("=== V10 POST-HOC ELITE DIAGNOSTIC ON CONSUMED FINAL SET ===", flush=True)
    print("NOT A FINAL TEST / NOT UNTOUCHED / NO TRAINING / NO POLICY UPDATE", flush=True)
    print(f"confirmed V10 elites to evaluate: {n_elites}", flush=True)
    print("per role:", {r: len(elites_by_role[r]) for r in roles}, flush=True)
    print("purpose: separate generator-sampling failure from learned-probe generalization", flush=True)

    if not args.execute:
        print("PLAN ONLY. Add --execute to run.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    cache = ResponseCacheV7(args.response_public_cache, args.response_truth_cache)
    embedder = FrozenOutputEmbedderV8(device="cpu")
    text_emb: dict[str, np.ndarray] = {}

    def emb(text: str) -> np.ndarray:
        t = str(text)
        if t not in text_emb:
            text_emb[t] = embedder.encode(t)
        return text_emb[t]

    grader = PairwiseRobustCounterfactualGraderV10(min_activation_rate=args.min_activation_rate)
    all_rows = []

    def eval_query(role: str, query: str, label: str):
        hpairs = by_role[role]
        anchor = primary_probe(role)
        plus_q=[]; minus_q=[]; sham_q=[]; plus_a=[]; minus_a=[]; sham_a=[]
        plus_selected=[]; sham_selected=[]; branches=[]
        for pi, p in enumerate(hpairs, 1):
            pa = execute_cached(p.plus_spec, anchor, cache)
            ma = execute_cached(p.minus_spec, anchor, cache)
            sa = execute_cached(p.sham_spec, anchor, cache)
            pq = execute_cached(p.plus_spec, query, cache)
            mq = execute_cached(p.minus_spec, query, cache)
            sq = execute_cached(p.sham_spec, query, cache)

            plus_a.append(emb(pa.output)); minus_a.append(emb(ma.output)); sham_a.append(emb(sa.output))
            plus_q.append(emb(pq.output)); minus_q.append(emb(mq.output)); sham_q.append(emb(sq.output))
            plus_selected.append(role in pq.selected_roles_training_only)
            sham_selected.append(role in sq.selected_roles_training_only)
            branches.append({
                "pair_id": p.pair_id,
                "mode": get_mode(p.plus_spec),
                "k": get_k(p.plus_spec),
                "negative_replacement": p.negative_replacement,
                "plus_selected_target_diagnostic_only": bool(plus_selected[-1]),
                "sham_selected_target_diagnostic_only": bool(sham_selected[-1]),
                "cache_hits": {
                    "plus_anchor": pa.cache_hit, "minus_anchor": ma.cache_hit, "sham_anchor": sa.cache_hit,
                    "plus_query": pq.cache_hit, "minus_query": mq.cache_hit, "sham_query": sq.cache_hit,
                },
            })
            print(f"  progress role={role:<10} {label:<7} pair={pi}/9", flush=True)

        rb = grader.score(
            target_selected_flags=plus_selected,
            sham_target_selected_flags=sham_selected,
            plus_query_embs=plus_q,
            minus_query_embs=minus_q,
            plus_anchor_embs=plus_a,
            minus_anchor_embs=minus_a,
            sham_query_embs=sham_q,
            sham_anchor_embs=sham_a,
            semantic_max_similarity=0.0,
        )
        return rb.to_dict(), branches

    for role in roles:
        elites = elites_by_role[role]
        if not elites:
            print(f"ROLE {role}: no confirmed V10 elites; skipped", flush=True)
            continue
        print(f"ROLE {role}: evaluating {len(elites)} confirmed elites over 9 matched pairs", flush=True)
        for idx, e in enumerate(elites, 1):
            rb, branches = eval_query(role, e["query"], f"elite{idx}")
            all_rows.append({
                "phase": "posthoc_consumed_final_elite_diagnostic",
                "target_role": role,
                "elite_index": idx,
                "query": e["query"],
                "train_reward": e["train_reward"],
                "train_breakdown": e["train_breakdown"],
                "confirmation_breakdown": e["confirmation_breakdown"],
                "final_consumed_breakdown_diagnostic_only": rb,
                "branches_ground_truth_diagnostic_only": branches,
            })
        print(f"ROLE {role}: complete", flush=True)

    # Compare to the already-revealed frozen-generator result. This is diagnostic only.
    role_stats = {}
    print("=== POST-HOC ELITE DIAGNOSTIC RESULTS ===", flush=True)
    final_role_stats = final.get("role_stats", {})
    for role in roles:
        rr = [x for x in all_rows if x["target_role"] == role]
        if not rr:
            role_stats[role] = {
                "confirmed_elites": 0,
                "any_strict_specific": False,
                "strict_specific_count": 0,
                "best_robust_specific": None,
                "best_activation": None,
                "original_final_generator_any_strict": bool(final_role_stats.get(role, {}).get("any_strict_specific", False)),
            }
            print(
                f"{role:<10} elites=0 | generatorStrict={role_stats[role]['original_final_generator_any_strict']}",
                flush=True,
            )
            continue
        br = [x["final_consumed_breakdown_diagnostic_only"] for x in rr]
        strict = [x for x in br if x["strict_specific_pass"]]
        best_rob = max(float(x["robust_specific"]) for x in br)
        best_act = max(float(x["activation_rate"]) for x in br)
        role_stats[role] = {
            "confirmed_elites": len(rr),
            "strict_specific_count": len(strict),
            "any_strict_specific": bool(strict),
            "best_robust_specific": best_rob,
            "best_activation": best_act,
            "original_final_generator_any_strict": bool(final_role_stats.get(role, {}).get("any_strict_specific", False)),
        }
        print(
            f"{role:<10} eliteStrict={len(strict)}/{len(rr)} ACT={best_act:.2f} bestROB={best_rob:+.3f} "
            f"| generatorStrict={role_stats[role]['original_final_generator_any_strict']}",
            flush=True,
        )

    roles_with_elites = [r for r in roles if role_stats[r]["confirmed_elites"] > 0]
    elite_success = sum(role_stats[r]["any_strict_specific"] for r in roles_with_elites)
    rescued = [
        r for r in roles_with_elites
        if role_stats[r]["any_strict_specific"] and not role_stats[r]["original_final_generator_any_strict"]
    ]
    summary = {
        "version": "V10.1-posthoc-elite-diagnostic",
        "diagnostic_only": True,
        "final_set_was_already_consumed_before_this_run": True,
        "no_training": True,
        "policy_updated": False,
        "purpose": "separate stochastic frozen-policy generation quality from exact confirmed-elite probe generalization",
        "confirmed_elites_total": n_elites,
        "roles_with_confirmed_elites": roles_with_elites,
        "roles_with_elite_strict_specific": elite_success,
        "roles_rescued_vs_original_generator_final": rescued,
        "role_stats": role_stats,
        "warning": "These post-hoc numbers are development diagnostics and MUST NOT be reported as untouched final-test results.",
    }

    Path(args.rows).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.rows) + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, args.rows)
    _atomic_json(args.summary, summary)
    print("---", flush=True)
    print(f"roles with confirmed elites that pass strict SPEC: {elite_success}/{len(roles_with_elites)}", flush=True)
    print("rescued vs original generator final:", rescued, flush=True)
    print("wrote:", args.rows, flush=True)
    print("wrote:", args.summary, flush=True)
    print("DIAGNOSTIC ONLY: the original final result remains 3/8 weak replication.", flush=True)


if __name__ == "__main__":
    main()
