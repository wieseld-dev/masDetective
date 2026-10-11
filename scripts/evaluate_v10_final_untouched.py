from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from estimator.candidate_presence_contrastive_v64 import load_profile
from probes.role_probe_bank import primary_probe
from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_zoo_v10_final import DESIGN_SEED, FINAL_TEST, get_v10_final_specs, validate_final_design
from v8.counterfactual import build_counterfactual_pairs, get_k, get_mode
from v8.embedding import FrozenOutputEmbedderV8
from v8.execution import execute_cached
from v10.gate import BenignGateV10
from v10.policy import SeededProbePolicyV10
from v10.reward import PairwiseRobustCounterfactualGraderV10


def _norm_text(s: str) -> str:
    import re
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def _atomic_json(path: str | Path, obj: object) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=p.parent, delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
        tmp = f.name
    os.replace(tmp, p)


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_tree(path: str | Path) -> str:
    root = Path(path)
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        with p.open("rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
        h.update(b"\0")
    return h.hexdigest()


def _test_set_hash() -> str:
    payload = json.dumps(FINAL_TEST, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    aa = aa / max(float(np.linalg.norm(aa)), 1e-8)
    bb = bb / max(float(np.linalg.norm(bb)), 1e-8)
    return float(np.clip(float(np.dot(aa, bb)), -1.0, 1.0))


def _merge_unique(*groups: Sequence[str], limit: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for x in group:
            q = str(x).strip()
            k = _norm_text(q)
            if q and k not in seen:
                seen.add(k)
                out.append(q)
    return out[-limit:]


def _elite_queries(summary_path: str | Path, roles: Sequence[str], max_per_role: int = 4) -> dict[str, list[str]]:
    out = {r: [] for r in roles}
    p = Path(summary_path)
    if not p.exists():
        return out
    obj = _load_json(p)
    raw = obj.get("elite_probes", {})
    for role in roles:
        rows = list(raw.get(role, []))
        rows.sort(key=lambda x: float(x.get("reward", -1e9)), reverse=True)
        for row in rows:
            q = str(row.get("query", "")).strip()
            if q and _norm_text(q) not in {_norm_text(x) for x in out[role]}:
                out[role].append(q)
            if len(out[role]) >= max_per_role:
                break
    return out


def _role_pair_table(pairs, roles: Sequence[str]) -> dict[str, list[object]]:
    out = {r: [] for r in roles}
    for p in pairs:
        out[p.target_role].append(p)
    return out


def _generate_manifest(args, roles, profile, policy_hash: str, v10_summary_hash: str, test_hash: str):
    import torch

    print("=== PHASE 1: FREEZE PROBES BEFORE ANY FINAL-TEST MAS EXECUTION ===", flush=True)
    gate = BenignGateV10(mode=args.gate_mode, guard_model=args.guard_model)
    embedder = FrozenOutputEmbedderV8(device="cpu")
    emb_cache: dict[str, np.ndarray] = {}

    def emb(text: str) -> np.ndarray:
        t = str(text)
        if t not in emb_cache:
            emb_cache[t] = embedder.encode(t)
        return emb_cache[t]

    def sim(q: str, refs: Sequence[str]) -> float:
        refs = [str(x).strip() for x in refs if str(x).strip()]
        if not refs:
            return 0.0
        qv = emb(q)
        return max(_cosine_similarity(qv, emb(r)) for r in refs)

    # Training-derived references only: V9/V10 elites. Development-validation probes
    # are deliberately not loaded into the final-test generation context.
    v9_elites = _elite_queries(args.v9_summary, roles, max_per_role=3)
    v10_elites = _elite_queries(args.v10_summary, roles, max_per_role=4)
    novelty_refs = {r: _merge_unique(v9_elites[r], v10_elites[r], limit=8) for r in roles}

    policy = SeededProbePolicyV10(
        args.model_name,
        adapter_path=args.policy_dir,
        trainable=False,
        use_4bit=not args.no_4bit,
        max_seq_length=1536,
    )
    for p in policy.model.parameters():
        p.requires_grad_(False)
    if any(bool(p.requires_grad) for p in policy.model.parameters()):
        raise RuntimeError("final-test policy is not fully frozen")
    policy.model.eval()

    probe_rows: list[dict] = []
    for ri, role in enumerate(roles):
        anchor = primary_probe(role)
        accepted: list[str] = []
        for j in range(args.probes_per_role):
            attempts: list[dict] = []
            final = None
            for retry in range(max(1, args.generation_retries)):
                # Fixed deterministic seed per role/probe/retry. Once accepted, the exact
                # question is written to the manifest and never regenerated on resume.
                seed = args.seed + 100000 * ri + 1000 * j + retry
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                avoid = _merge_unique(novelty_refs[role], accepted, limit=max(6, args.max_avoid_probes))
                obs = {
                    "candidate_role": role,
                    "candidate_description": profile.get("role_descriptions", {}).get(role, role),
                    "anchor_probe": anchor,
                    "avoid_probes": avoid[-max(6, args.max_avoid_probes):],
                }
                sample = policy.sample_probe(obs, temperature=0.75, top_p=0.90)
                att = {
                    "retry": retry + 1,
                    "seed": seed,
                    "raw": sample.raw_completion,
                    "valid": bool(sample.valid),
                    "invalid_reason": sample.invalid_reason,
                }
                if not sample.valid:
                    att["decision"] = "retry_invalid"
                    attempts.append(att)
                    continue

                refs = [anchor] + novelty_refs[role] + accepted
                semantic_sim = sim(sample.query, refs)
                att["semantic_max_similarity"] = semantic_sim
                if semantic_sim >= args.max_similarity:
                    att["decision"] = "retry_semantic_duplicate"
                    attempts.append(att)
                    continue

                gd = gate.check(sample.query)
                att.update({"gate_allowed": gd.allowed, "gate_stage": gd.stage, "gate_category": gd.category})
                if not gd.allowed:
                    att["decision"] = "retry_gate_blocked"
                    attempts.append(att)
                    continue

                att["decision"] = "accept"
                attempts.append(att)
                final = {
                    "role": role,
                    "probe_index": j + 1,
                    "query": sample.query,
                    "semantic_max_similarity": semantic_sim,
                    "attempts": attempts,
                }
                accepted.append(sample.query)
                break

            if final is None:
                final = {
                    "role": role,
                    "probe_index": j + 1,
                    "query": None,
                    "semantic_max_similarity": None,
                    "attempts": attempts,
                    "status": "generation_failed",
                }
            probe_rows.append(final)
            print(f"freeze role={role:<10} p{j+1} status={'ok' if final.get('query') else 'FAILED'} | {final.get('query')}", flush=True)

    manifest = {
        "protocol": "V10.1 frozen-policy final untouched probe-generalization test",
        "created_before_any_final_test_mas_execution": True,
        "no_training": True,
        "policy_dir": args.policy_dir,
        "policy_sha256": policy_hash,
        "v10_summary_sha256": v10_summary_hash,
        "test_set_sha256": test_hash,
        "test_design_seed": DESIGN_SEED,
        "generation_seed": args.seed,
        "probes_per_role": args.probes_per_role,
        "pre_registered_success_criterion": {
            "primary_metric": "role_any_strict_specific",
            "strong_replication": ">=6/8 strict roles AND >=7/8 roles with activation >=0.75",
            "partial_replication": "5/8 strict roles",
            "weak_replication": "<=4/8 strict roles",
            "secondary_metric": "role_any_strict_delta (learned probe robustly beats human anchor)",
        },
        "temperature": 0.75,
        "top_p": 0.90,
        "max_similarity": args.max_similarity,
        "gate_mode": args.gate_mode,
        "roles": list(roles),
        "probes": probe_rows,
    }
    _atomic_json(args.probe_manifest, manifest)
    print("FROZEN PROBE MANIFEST:", args.probe_manifest, flush=True)
    print("manifest sha256:", _sha256_file(args.probe_manifest), flush=True)
    return manifest


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="Frozen V10.1 final untouched probe-generalization test; NO TRAINING")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--policy-dir", default="artifacts/v10_policy")
    ap.add_argument("--v9-summary", default="artifacts/v9_full_summary.json")
    ap.add_argument("--v10-summary", default="artifacts/v10_summary.json")
    ap.add_argument("--probe-manifest", default="artifacts/v10_final_probe_manifest.json")
    ap.add_argument("--rows", default="artifacts/v10_final_rows.jsonl")
    ap.add_argument("--summary", default="artifacts/v10_final_summary.json")
    ap.add_argument("--response-public-cache", default="artifacts/v10_final_response_public.jsonl")
    ap.add_argument("--response-truth-cache", default="artifacts/v10_final_response_truth.jsonl")
    ap.add_argument("--probes-per-role", type=int, default=3)
    ap.add_argument("--generation-retries", type=int, default=8)
    ap.add_argument("--max-avoid-probes", type=int, default=6)
    ap.add_argument("--max-similarity", type=float, default=0.94)
    ap.add_argument("--gate-mode", choices=["hard", "hybrid"], default="hybrid")
    ap.add_argument("--guard-model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--min-activation-rate", type=float, default=0.75)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    validate_final_design()
    if not Path(args.v10_summary).exists():
        raise SystemExit(f"missing completed V10 summary: {args.v10_summary}")
    v10_summary = _load_json(args.v10_summary)
    if not bool(v10_summary.get("development_validation_inspected")):
        print("WARNING: V10 summary does not carry the expected development-validation marker", flush=True)
    if not Path(args.policy_dir).is_dir():
        raise SystemExit(f"missing frozen V10 policy: {args.policy_dir}")

    profile = load_profile(args.profile)
    roles = tuple(profile["candidate_roles"])
    specs = get_v10_final_specs()
    pairs = build_counterfactual_pairs(specs, roles, include_sham=True)
    by_role = _role_pair_table(pairs, roles)

    # Structural assertions happen before any API/test execution.
    if len(specs) != 16 or len(pairs) != 72:
        raise RuntimeError(f"unexpected final design size: specs={len(specs)} pairs={len(pairs)}")
    if any(len(by_role[r]) != 9 for r in roles):
        raise RuntimeError(f"final pairs are not exactly 9/role: {{r: len(by_role[r]) for r in roles}}")

    policy_hash = _sha256_tree(args.policy_dir)
    v10_summary_hash = _sha256_file(args.v10_summary)
    test_hash = _test_set_hash()

    print("=== V10.1 FINAL UNTOUCHED TEST PLAN ===", flush=True)
    print("NO TRAINING / NO OPTIMIZER / NO POLICY UPDATE", flush=True)
    print("frozen policy sha256:", policy_hash, flush=True)
    print("test set sha256:     ", test_hash, flush=True)
    print("base systems:", len(specs), "matched pairs:", len(pairs), "pairs/role: 9", flush=True)
    print("mode/K cells: 16 = 4 modes x K{3,4,5,6}", flush=True)
    print("role prevalence: exactly 9 each; every role covers every mode and every K", flush=True)
    print("probes/role:", args.probes_per_role, "max MAS interactions:", len(roles) * 9 * 3 * (1 + args.probes_per_role), flush=True)
    print("IMPORTANT: once these results are inspected, this set is consumed as a final test for V10.1.", flush=True)

    if not args.execute:
        print("PLAN ONLY. Add --execute to run.")
        return
    if Path(args.summary).exists():
        raise SystemExit(
            f"FINAL TEST ALREADY CONSUMED: {args.summary} exists. "
            "Do not rerun/regenerate after seeing results."
        )
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to load the frozen V10 policy for probe generation")

    # Freeze all probes BEFORE the first final-test system is queried. On a crash,
    # the existing manifest is reused exactly; probes are never regenerated.
    if Path(args.probe_manifest).exists():
        manifest = _load_json(args.probe_manifest)
        if manifest.get("policy_sha256") != policy_hash or manifest.get("test_set_sha256") != test_hash:
            raise SystemExit("existing final probe manifest does not match current frozen policy/test set")
        print("REUSING EXISTING FROZEN PROBE MANIFEST:", args.probe_manifest, flush=True)
    else:
        manifest = _generate_manifest(args, roles, profile, policy_hash, v10_summary_hash, test_hash)

    probes_by_role: dict[str, list[dict]] = {r: [] for r in roles}
    for row in manifest.get("probes", []):
        if row.get("role") in probes_by_role and row.get("query"):
            probes_by_role[row["role"]].append(row)
    for role in roles:
        if len(probes_by_role[role]) != args.probes_per_role:
            raise SystemExit(f"frozen manifest does not contain {args.probes_per_role} accepted probes for {role}")

    # Policy generation is over. No final-test output has been observed by policy.
    # From here onward we only execute and score the precommitted questions.
    print("=== PHASE 2: EXECUTE FROZEN PROBES ON FINAL TEST ===", flush=True)
    cache = ResponseCacheV7(args.response_public_cache, args.response_truth_cache)
    embedder = FrozenOutputEmbedderV8(device="cpu")
    text_emb: dict[str, np.ndarray] = {}
    def emb(text: str) -> np.ndarray:
        t = str(text)
        if t not in text_emb:
            text_emb[t] = embedder.encode(t)
        return text_emb[t]

    grader = PairwiseRobustCounterfactualGraderV10(min_activation_rate=args.min_activation_rate)
    all_rows: list[dict] = []

    def eval_query(role: str, query: str, semantic_sim: float, label: str):
        hpairs = by_role[role]
        anchor = primary_probe(role)
        plus_q=[]; minus_q=[]; sham_q=[]; plus_a=[]; minus_a=[]; sham_a=[]
        plus_selected=[]; sham_selected=[]; anchor_plus_selected=[]; anchor_sham_selected=[]; branches=[]
        for pi, p in enumerate(hpairs, 1):
            # Anchors are cached after the first probe for a role.
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
            anchor_plus_selected.append(role in pa.selected_roles_training_only)
            anchor_sham_selected.append(role in sa.selected_roles_training_only)
            branches.append({
                "pair_id": p.pair_id,
                "mode": get_mode(p.plus_spec),
                "k": get_k(p.plus_spec),
                "negative_replacement": p.negative_replacement,
                "sham_removed_role": p.sham_removed_role,
                "sham_replacement": p.sham_replacement,
                "plus_selected_target_training_only": bool(plus_selected[-1]),
                "sham_selected_target_training_only": bool(sham_selected[-1]),
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
            semantic_max_similarity=float(semantic_sim),
        )
        anchor_rb = grader.score(
            target_selected_flags=anchor_plus_selected,
            sham_target_selected_flags=anchor_sham_selected,
            plus_query_embs=plus_a,
            minus_query_embs=minus_a,
            plus_anchor_embs=plus_a,
            minus_anchor_embs=minus_a,
            sham_query_embs=sham_a,
            sham_anchor_embs=sham_a,
            semantic_max_similarity=0.0,
        )
        return rb, anchor_rb, branches

    anchor_by_role: dict[str, dict] = {}
    for role in roles:
        print(f"FINAL TEST ROLE {role}: starting 3 frozen probes over 9 matched pairs", flush=True)
        for pr in probes_by_role[role]:
            rb, anchor_rb, branches = eval_query(role, pr["query"], float(pr.get("semantic_max_similarity") or 0.0), f"p{pr['probe_index']}")
            anchor_by_role.setdefault(role, anchor_rb.to_dict())
            all_rows.append({
                "phase": "final_untouched_test",
                "target_role": role,
                "probe_index": pr["probe_index"],
                "query": pr["query"],
                "semantic_max_similarity": pr.get("semantic_max_similarity"),
                "pair_count": 9,
                "reward_breakdown_evaluation_only": rb.to_dict(),
                "branches_ground_truth_evaluation_only": branches,
            })
        print(f"FINAL TEST ROLE {role}: complete", flush=True)

    # Only now, after every role/probe is executed, expose aggregate outcomes.
    role_stats: dict[str, dict] = {}
    print("=== FINAL UNTOUCHED RESULTS (FIRST REVEAL) ===", flush=True)
    for role in roles:
        rr = [r for r in all_rows if r["target_role"] == role]
        rbs = [r["reward_breakdown_evaluation_only"] for r in rr]
        strict = [x for x in rbs if x["strict_specific_pass"]]
        delta = [x for x in rbs if x["strict_delta_pass"]]
        act = [x for x in rbs if float(x["activation_rate"]) >= args.min_activation_rate]
        best_rob = max(float(x["robust_specific"]) for x in rbs)
        ar = anchor_by_role[role]
        role_stats[role] = {
            "probes": len(rr),
            "anchor_activation": float(ar["activation_rate"]),
            "anchor_robust_specific": float(ar["robust_specific"]),
            "anchor_strict_specific": bool(ar["strict_specific_pass"]),
            "any_activation_ge_075": bool(act),
            "strict_specific_count": len(strict),
            "any_strict_specific": bool(strict),
            "strict_delta_count": len(delta),
            "any_strict_delta": bool(delta),
            "best_robust_specific": best_rob,
            "mean_robust_specific": float(np.mean([x["robust_specific"] for x in rbs])),
            "best_activation": max(float(x["activation_rate"]) for x in rbs),
        }
        print(
            f"{role:<10} ACT={role_stats[role]['best_activation']:.2f} "
            f"strict={len(strict)}/3 delta={len(delta)}/3 bestROB={best_rob:+.3f} "
            f"| anchorACT={role_stats[role]['anchor_activation']:.2f} "
            f"anchorROB={role_stats[role]['anchor_robust_specific']:+.3f} "
            f"anchorStrict={role_stats[role]['anchor_strict_specific']}",
            flush=True,
        )

    any_act = sum(v["any_activation_ge_075"] for v in role_stats.values())
    any_strict = sum(v["any_strict_specific"] for v in role_stats.values())
    any_delta = sum(v["any_strict_delta"] for v in role_stats.values())

    summary = {
        "version": "V10.1",
        "evaluation": "FINAL UNTOUCHED frozen-policy probe-generalization test",
        "no_training": True,
        "policy_updated": False,
        "final_untouched_test": True,
        "test_consumed": True,
        "policy_dir": args.policy_dir,
        "policy_sha256": policy_hash,
        "v10_summary_sha256": v10_summary_hash,
        "probe_manifest": args.probe_manifest,
        "probe_manifest_sha256": _sha256_file(args.probe_manifest),
        "test_set_sha256": test_hash,
        "test_design_seed": DESIGN_SEED,
        "base_systems": len(specs),
        "matched_pairs": len(pairs),
        "pairs_per_role": {r: len(by_role[r]) for r in roles},
        "probes_per_role": args.probes_per_role,
        "role_any_activation_ge_075": any_act,
        "role_any_activation_ge_075_rate": any_act / len(roles),
        "role_any_strict_specific": any_strict,
        "role_any_strict_specific_rate": any_strict / len(roles),
        "role_any_strict_delta": any_delta,
        "role_any_strict_delta_rate": any_delta / len(roles),
        "role_stats": role_stats,
        "pre_registered_success_criterion": manifest.get("pre_registered_success_criterion"),
        "pre_registered_interpretation": (
            "strong_replication" if any_strict >= 6 and any_act >= 7
            else "partial_replication" if any_strict == 5
            else "weak_replication"
        ),
        "development_reference_only": {
            "v10_1_dev_role_any_activation_ge_075": v10_summary.get("holdout_role_any_activation_ge_075"),
            "v10_1_dev_role_any_strict_specific": v10_summary.get("holdout_role_any_strict_specific"),
            "v10_1_dev_role_any_strict_delta": v10_summary.get("holdout_role_any_strict_delta"),
        },
        "important_scope": "This evaluates the learned benign probe policy, not an end-to-end PRESENT/ABSENT detector or K estimator.",
    }

    Path(args.rows).parent.mkdir(parents=True, exist_ok=True)
    tmp_rows = Path(str(args.rows) + ".tmp")
    with tmp_rows.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_rows, args.rows)
    _atomic_json(args.summary, summary)

    print("---", flush=True)
    print(f"ROLE ANY ACTIVATION >= .75: {any_act}/8 = {any_act/8:.1%}", flush=True)
    print(f"ROLE ANY STRICT SPEC:      {any_strict}/8 = {any_strict/8:.1%}", flush=True)
    print(f"ROLE ANY STRICT DELTA:     {any_delta}/8 = {any_delta/8:.1%}", flush=True)
    print("wrote:", args.rows, flush=True)
    print("wrote:", args.summary, flush=True)
    print("FINAL TEST IS NOW CONSUMED. If the method is changed after inspecting this result, use a NEW final test set.", flush=True)


if __name__ == "__main__":
    main()
