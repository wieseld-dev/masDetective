from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from estimator.candidate_presence_contrastive_v64 import load_profile
from probes.role_probe_bank import primary_probe
from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_zoo_v65 import get_v65_split
from v8.counterfactual import build_counterfactual_pairs, get_k, get_mode, get_roles, get_system_id
from v8.embedding import FrozenOutputEmbedderV8
from v8.execution import execute_cached
from v9.reward import RobustCounterfactualGraderV9


def _read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalized(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n <= 1e-12:
        return x
    return x / n


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    x, y = _normalized(a), _normalized(b)
    return float(1.0 - np.clip(float(np.dot(x, y)), -1.0, 1.0))


def _q25(xs: Iterable[float]) -> float:
    a = np.asarray(list(xs), dtype=np.float32)
    if len(a) == 0:
        return float("nan")
    try:
        return float(np.quantile(a, 0.25, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(a, 0.25, interpolation="linear"))


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "median": None, "q25": None, "min": None, "max": None}
    a = np.asarray(xs, dtype=np.float32)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "q25": _q25(xs),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _branch_specs(pair):
    out = [("plus", pair.plus_spec), ("minus", pair.minus_spec)]
    if pair.sham_spec is not None:
        out.append(("sham", pair.sham_spec))
    return out


def _cache_key(cache: ResponseCacheV7, spec, query: str) -> bool:
    return cache.get(get_system_id(spec), query) is not None


def _extract_original_holdout(log_rows: list[dict], roles: list[str]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    holdout = [r for r in log_rows if r.get("phase") == "holdout"]
    by_role: dict[str, list[dict]] = {r: [] for r in roles}
    for row in holdout:
        role = str(row.get("target_role", ""))
        if role in by_role:
            by_role[role].append(row)

    # Recover anchors from training rows. Fall back to the canonical probe bank.
    anchors: dict[str, str] = {}
    for row in log_rows:
        if row.get("phase") != "train":
            continue
        role = str(row.get("target_role", ""))
        anchor = str(row.get("anchor_probe", "")).strip()
        if role in by_role and anchor and role not in anchors:
            anchors[role] = anchor
    for role in roles:
        anchors.setdefault(role, primary_probe(role))
    return by_role, anchors


def _plan(cache, by_role, anchors, holdout_by_role, roles) -> dict:
    all_requests: list[tuple[str, str, str, str, str]] = []
    # (role, kind, pair_id, system_id, query)
    for role in roles:
        valid_queries = [
            str(r.get("query", "")).strip()
            for r in by_role.get(role, [])
            if r.get("status") == "ok" and str(r.get("query", "")).strip()
        ]
        queries = [("anchor", anchors[role])] + [("probe", q) for q in valid_queries]
        for pair in holdout_by_role.get(role, []):
            for branch, spec in _branch_specs(pair):
                sid = get_system_id(spec)
                for kind, query in queries:
                    all_requests.append((role, kind, pair.pair_id, sid, query))

    # Duplicate generated probes (notably writing) should not create duplicate API work.
    unique = {(sid, q) for _, _, _, sid, q in all_requests}
    hits = sum(cache.get(sid, q) is not None for sid, q in unique)
    missing = len(unique) - hits

    role_plan = {}
    for role in roles:
        pairs = holdout_by_role.get(role, [])
        valid = [r for r in by_role.get(role, []) if r.get("status") == "ok"]
        blocked = [r for r in by_role.get(role, []) if r.get("status") != "ok"]
        req = {
            (get_system_id(spec), query)
            for pair in pairs
            for _, spec in _branch_specs(pair)
            for query in [anchors[role]] + [str(r["query"]) for r in valid]
        }
        rhits = sum(cache.get(sid, q) is not None for sid, q in req)
        role_plan[role] = {
            "pairs": len(pairs),
            "valid_original_holdout_probes": len(valid),
            "blocked_or_invalid_original_holdout_probes": len(blocked),
            "unique_interactions": len(req),
            "cache_hits": rhits,
            "missing_interactions": len(req) - rhits,
            "pair_modes": [get_mode(p.plus_spec) for p in pairs],
            "pair_k": [get_k(p.plus_spec) for p in pairs],
        }

    return {
        "all_request_rows_including_duplicate_queries": len(all_requests),
        "unique_interactions": len(unique),
        "cache_hits": hits,
        "missing_interactions": missing,
        "role_plan": role_plan,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Exhaustively re-evaluate the ORIGINAL V9-FULL holdout probes on every "
            "composition-disjoint V6.5 validation counterfactual pair. No new probe generation."
        )
    )
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--source-log", default="artifacts/v9_full_probe_discovery.jsonl")
    ap.add_argument("--response-public-cache", default="artifacts/v9_full_response_public.jsonl")
    ap.add_argument("--response-truth-cache", default="artifacts/v9_full_response_truth.jsonl")
    ap.add_argument("--rows-out", default="artifacts/v9_full_exhaustive_holdout_rows.jsonl")
    ap.add_argument("--summary-out", default="artifacts/v9_full_exhaustive_holdout_summary.json")
    ap.add_argument("--execute", action="store_true", help="Run missing MAS interactions. Default is plan-only.")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    roles = list(profile["candidate_roles"])
    log_rows = _read_jsonl(args.source_log)
    by_role, anchors = _extract_original_holdout(log_rows, roles)

    holdout_specs = list(get_v65_split("val"))
    holdout_pairs = build_counterfactual_pairs(holdout_specs, roles, include_sham=True)
    holdout_by_role: dict[str, list[object]] = {r: [] for r in roles}
    for p in holdout_pairs:
        holdout_by_role.setdefault(p.target_role, []).append(p)

    cache = ResponseCacheV7(args.response_public_cache, args.response_truth_cache)
    plan = _plan(cache, by_role, anchors, holdout_by_role, roles)

    print("=== V9-FULL EXHAUSTIVE HOLDOUT PLAN ===")
    print(f"validation base systems: {len(holdout_specs)}")
    print(f"matched holdout pairs:   {len(holdout_pairs)}")
    print(f"original holdout probes: {sum(len(v) for v in by_role.values())}")
    print(f"unique MAS interactions required (anchors + previously-valid probes): {plan['unique_interactions']}")
    print(f"already cached: {plan['cache_hits']}")
    print(f"missing:        {plan['missing_interactions']}")
    for role in roles:
        rp = plan["role_plan"][role]
        print(
            f"  {role:<10} pairs={rp['pairs']} valid_probes={rp['valid_original_holdout_probes']} "
            f"blocked={rp['blocked_or_invalid_original_holdout_probes']} "
            f"cache={rp['cache_hits']}/{rp['unique_interactions']} missing={rp['missing_interactions']} "
            f"modes={rp['pair_modes']} K={rp['pair_k']}"
        )

    if not args.execute:
        print("PLAN ONLY. No MAS/API calls were made. Add --execute inside a SLURM job to fill the missing cache entries and score all pairs.")
        return

    embedder = FrozenOutputEmbedderV8(device="cpu")
    legacy_grader = RobustCounterfactualGraderV9()
    rows_out: list[dict] = []
    role_summary: dict[str, dict] = {}

    for role in roles:
        pair_list = holdout_by_role.get(role, [])
        anchor = anchors[role]
        original_rows = by_role.get(role, [])
        valid_probes = [r for r in original_rows if r.get("status") == "ok"]
        blocked = [
            {"query": r.get("query", ""), "status": r.get("status", "")}
            for r in original_rows
            if r.get("status") != "ok"
        ]

        # Materialize/cache the anchor on every pair once, then reuse embeddings.
        anchor_material = {}
        for pair in pair_list:
            pa = execute_cached(pair.plus_spec, anchor, cache)
            ma = execute_cached(pair.minus_spec, anchor, cache)
            sa = execute_cached(pair.sham_spec, anchor, cache) if pair.sham_spec is not None else None
            anchor_material[pair.pair_id] = {
                "plus": pa,
                "minus": ma,
                "sham": sa,
                "plus_emb": embedder.encode(pa.output),
                "minus_emb": embedder.encode(ma.output),
                "sham_emb": embedder.encode(sa.output) if sa is not None else None,
            }

        anchor_pair_specific = []
        for pair in pair_list:
            a = anchor_material[pair.pair_id]
            at = _cosine_distance(a["plus_emb"], a["minus_emb"])
            ash = _cosine_distance(a["plus_emb"], a["sham_emb"]) if a["sham_emb"] is not None else 0.0
            anchor_pair_specific.append(at - ash)

        probe_summaries = []
        for probe_index, source_row in enumerate(valid_probes, 1):
            query = str(source_row["query"])
            pair_metrics = []
            plus_q_embs=[]; minus_q_embs=[]; sham_q_embs=[]
            plus_a_embs=[]; minus_a_embs=[]; sham_a_embs=[]
            selected_flags=[]

            for pair in pair_list:
                a = anchor_material[pair.pair_id]
                pq = execute_cached(pair.plus_spec, query, cache)
                mq = execute_cached(pair.minus_spec, query, cache)
                sq = execute_cached(pair.sham_spec, query, cache) if pair.sham_spec is not None else None
                pe = embedder.encode(pq.output)
                me = embedder.encode(mq.output)
                se = embedder.encode(sq.output) if sq is not None else None

                target_d = _cosine_distance(pe, me)
                sham_d = _cosine_distance(pe, se) if se is not None else 0.0
                specific = target_d - sham_d
                anchor_target_d = _cosine_distance(a["plus_emb"], a["minus_emb"])
                anchor_sham_d = _cosine_distance(a["plus_emb"], a["sham_emb"]) if a["sham_emb"] is not None else 0.0
                anchor_specific = anchor_target_d - anchor_sham_d
                delta = specific - anchor_specific
                activated = role in set(pq.selected_roles_training_only)

                pm = {
                    "phase": "exhaustive_holdout_pair",
                    "target_role": role,
                    "probe_index": probe_index,
                    "query": query,
                    "pair_id": pair.pair_id,
                    "plus_system_id": get_system_id(pair.plus_spec),
                    "base_mode": get_mode(pair.plus_spec),
                    "base_k": get_k(pair.plus_spec),
                    "plus_roles_training_only": list(get_roles(pair.plus_spec)),
                    "minus_roles_training_only": list(get_roles(pair.minus_spec)),
                    "sham_roles_training_only": list(get_roles(pair.sham_spec)) if pair.sham_spec is not None else [],
                    "plus_selected_training_only": list(pq.selected_roles_training_only),
                    "minus_selected_training_only": list(mq.selected_roles_training_only),
                    "target_activated_training_only": bool(activated),
                    "target_distance": float(target_d),
                    "sham_distance": float(sham_d),
                    "specific_pairwise": float(specific),
                    "anchor_target_distance": float(anchor_target_d),
                    "anchor_sham_distance": float(anchor_sham_d),
                    "anchor_specific_pairwise": float(anchor_specific),
                    "delta_vs_anchor_pairwise": float(delta),
                    "query_cache_hit_plus": bool(pq.cache_hit),
                    "query_cache_hit_minus": bool(mq.cache_hit),
                    "query_cache_hit_sham": bool(sq.cache_hit) if sq is not None else None,
                }
                pair_metrics.append(pm)
                rows_out.append(pm)

                plus_q_embs.append(pe); minus_q_embs.append(me)
                plus_a_embs.append(a["plus_emb"]); minus_a_embs.append(a["minus_emb"])
                selected_flags.append(activated)
                if se is not None and a["sham_emb"] is not None:
                    sham_q_embs.append(se); sham_a_embs.append(a["sham_emb"])

            legacy = legacy_grader.score(
                target_selected_flags=selected_flags,
                plus_query_embs=plus_q_embs,
                minus_query_embs=minus_q_embs,
                plus_anchor_embs=plus_a_embs,
                minus_anchor_embs=minus_a_embs,
                sham_query_embs=sham_q_embs if len(sham_q_embs) == len(pair_list) else None,
                sham_anchor_embs=sham_a_embs if len(sham_a_embs) == len(pair_list) else None,
            )

            specifics = [x["specific_pairwise"] for x in pair_metrics]
            deltas = [x["delta_vs_anchor_pairwise"] for x in pair_metrics]
            activation_rate = float(np.mean([x["target_activated_training_only"] for x in pair_metrics])) if pair_metrics else 0.0
            ss = _stats(specifics)
            ds = _stats(deltas)
            robust_specific = None if ss["median"] is None else 0.5 * ss["median"] + 0.5 * ss["q25"]
            robust_delta = None if ds["median"] is None else 0.5 * ds["median"] + 0.5 * ds["q25"]
            strict_specific = bool(
                activation_rate >= 0.75
                and ss["median"] is not None and ss["median"] > 0.0
                and ss["q25"] is not None and ss["q25"] >= 0.0
            )
            strict_delta = bool(
                activation_rate >= 0.75
                and ds["median"] is not None and ds["median"] > 0.0
                and ds["q25"] is not None and ds["q25"] >= 0.0
            )
            ps = {
                "probe_index": probe_index,
                "query": query,
                "pair_count": len(pair_metrics),
                "activation_rate": activation_rate,
                "specific_pairwise": ss,
                "delta_vs_anchor_pairwise": ds,
                "positive_specific_pair_rate": float(np.mean(np.asarray(specifics) > 0.0)) if specifics else None,
                "beat_anchor_pair_rate": float(np.mean(np.asarray(deltas) > 0.0)) if deltas else None,
                "robust_specific_median_q25": robust_specific,
                "robust_delta_median_q25": robust_delta,
                "strict_specific_pass": strict_specific,
                "strict_delta_pass": strict_delta,
                "legacy_all_pairs": legacy.to_dict(),
            }
            probe_summaries.append(ps)

        anchor_stats = _stats(anchor_pair_specific)
        role_summary[role] = {
            "pair_count": len(pair_list),
            "pair_modes": [get_mode(p.plus_spec) for p in pair_list],
            "pair_k": [get_k(p.plus_spec) for p in pair_list],
            "anchor_specific_pairwise": anchor_stats,
            "original_holdout_probe_count": len(original_rows),
            "valid_probe_count": len(valid_probes),
            "blocked_or_invalid": blocked,
            "any_legacy_specific_positive": any(
                p["legacy_all_pairs"]["specific_separation"] > 0.0 for p in probe_summaries
            ),
            "any_strict_specific_pass": any(p["strict_specific_pass"] for p in probe_summaries),
            "any_strict_delta_pass": any(p["strict_delta_pass"] for p in probe_summaries),
            "probes": probe_summaries,
        }

    Path(args.rows_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.rows_out).open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    role_any_legacy = sum(bool(v["any_legacy_specific_positive"]) for v in role_summary.values())
    role_any_strict_specific = sum(bool(v["any_strict_specific_pass"]) for v in role_summary.values())
    role_any_strict_delta = sum(bool(v["any_strict_delta_pass"]) for v in role_summary.values())
    summary = {
        "evaluation": "V9-FULL original holdout probes, exhaustive over every V6.5 validation matched pair",
        "new_probe_generation": False,
        "policy_loaded": False,
        "validation_base_systems": len(holdout_specs),
        "matched_holdout_pairs": len(holdout_pairs),
        "source_holdout_probe_rows": sum(len(v) for v in by_role.values()),
        "plan_before_execution": plan,
        "role_any_legacy_positive": role_any_legacy,
        "role_any_legacy_positive_rate": role_any_legacy / max(1, len(roles)),
        "role_any_strict_specific": role_any_strict_specific,
        "role_any_strict_specific_rate": role_any_strict_specific / max(1, len(roles)),
        "role_any_strict_delta": role_any_strict_delta,
        "role_any_strict_delta_rate": role_any_strict_delta / max(1, len(roles)),
        "role_summary": role_summary,
        "rows_out": args.rows_out,
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== EXHAUSTIVE HOLDOUT RESULTS ===")
    for role in roles:
        rs = role_summary[role]
        print(f"\n{role.upper()} pairs={rs['pair_count']} valid_probes={rs['valid_probe_count']}")
        a = rs["anchor_specific_pairwise"]
        print(f"  anchor pairwise SPEC: mean={a['mean']:+.4f} med={a['median']:+.4f} q25={a['q25']:+.4f}")
        for p in rs["probes"]:
            s = p["specific_pairwise"]; d = p["delta_vs_anchor_pairwise"]; l = p["legacy_all_pairs"]
            print(
                f"  p{p['probe_index']} ACT={p['activation_rate']:.2f} "
                f"pairSPEC(mean/med/q25)={s['mean']:+.4f}/{s['median']:+.4f}/{s['q25']:+.4f} "
                f"DELTA(mean/med/q25)={d['mean']:+.4f}/{d['median']:+.4f}/{d['q25']:+.4f} "
                f"legacySPEC={l['specific_separation']:+.4f} "
                f"strictSPEC={p['strict_specific_pass']} strictDELTA={p['strict_delta_pass']}"
            )
            print(f"     {p['query']}")

    print("\nROLE-LEVEL")
    print(f"  any legacy all-pairs positive SPEC: {role_any_legacy}/{len(roles)}")
    print(f"  any strict pairwise-SPEC pass:      {role_any_strict_specific}/{len(roles)}")
    print(f"  any strict delta-vs-anchor pass:    {role_any_strict_delta}/{len(roles)}")
    print(f"wrote: {args.rows_out}")
    print(f"wrote: {args.summary_out}")


if __name__ == "__main__":
    main()
