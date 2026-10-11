from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from rl.generative_probe_env_v7 import ResponseCacheV7
from v8.embedding import FrozenOutputEmbedderV8
from v8.execution import execute_cached
from v8.counterfactual import get_roles
from v10.gate import BenignGateV10
from v10.reward import PairwiseRobustCounterfactualGraderV10, group_relative_advantages
from v11_2.split_common import (
    _copy_cache_seed,
    _cosine_similarity,
    _norm_text,
    _word_jaccard,
    build_context,
    load_previous_elites,
    role_order,
)


def load_rows(path: str):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description="V11.2 split pipeline: CPU/API evaluation only")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--generated", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--profile", default="config/application_profile_v11.json")
    ap.add_argument("--initial-elite-summary", default="artifacts/v11_summary.json")
    ap.add_argument("--previous-eval", default="")
    ap.add_argument("--rounds-per-role", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--bundle-size", type=int, default=4)
    ap.add_argument("--confirmation-size", type=int, default=2)
    ap.add_argument("--confirm-top-k", type=int, default=2)
    ap.add_argument("--gate-mode", choices=["hard", "hybrid"], default="hybrid")
    ap.add_argument("--guard-model", default="gpt-4o-mini")
    ap.add_argument("--min-activation-rate", type=float, default=0.75)
    ap.add_argument("--novelty-free-similarity", type=float, default=0.90)
    ap.add_argument("--novelty-weight", type=float, default=0.25)
    ap.add_argument("--semantic-hard-duplicate", type=float, default=0.97)
    ap.add_argument("--holdout-max-similarity", type=float, default=0.94)
    ap.add_argument("--seed", type=int, default=1010)
    ap.add_argument("--response-public-cache", default="artifacts/v11_2_split/response_public.jsonl")
    ap.add_argument("--response-truth-cache", default="artifacts/v11_2_split/response_truth.jsonl")
    ap.add_argument("--seed-response-public-cache", default="artifacts/v11_seed_none_public.jsonl")
    ap.add_argument("--seed-response-truth-cache", default="artifacts/v11_seed_none_truth.jsonl")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if not Path(args.generated).is_file():
        raise SystemExit(f"generated candidate file missing: {args.generated}")

    ctx = build_context(
        profile_path=args.profile,
        rounds=args.rounds_per_role,
        bundle_size=args.bundle_size,
        confirmation_size=args.confirmation_size,
        seed=args.seed,
    )
    usable_roles = ctx["usable_roles"]
    schedule = ctx["schedule"]
    conf_schedule = ctx["confirmation_schedule"]
    role_seq = role_order(usable_roles, args.round, args.seed)
    novelty_refs = load_previous_elites(args.initial_elite_summary, args.previous_eval or None, usable_roles)

    raw_rows = load_rows(args.generated)
    by_role = {r: [] for r in usable_roles}
    for row in raw_rows:
        if int(row.get("round", -1)) != args.round:
            raise SystemExit("generated file contains a different round")
        by_role[str(row["target_role"])].append(row)
    for r in usable_roles:
        by_role[r].sort(key=lambda x: int(x["candidate_index"]))
        if len(by_role[r]) != args.group_size:
            raise SystemExit(f"role {r} has {len(by_role[r])} generated candidates, expected {args.group_size}")

    _copy_cache_seed(args.seed_response_public_cache, args.response_public_cache)
    _copy_cache_seed(args.seed_response_truth_cache, args.response_truth_cache)
    cache = ResponseCacheV7(args.response_public_cache, args.response_truth_cache)
    gate = BenignGateV10(mode=args.gate_mode, guard_model=args.guard_model)
    embedder = FrozenOutputEmbedderV8(device="cpu")
    emb_cache = {}

    def embed_text(text: str):
        key = str(text)
        if key not in emb_cache:
            emb_cache[key] = embedder.encode(key)
        return emb_cache[key]

    def semantic_similarity(query: str, refs: Sequence[str]) -> float:
        clean = [str(x).strip() for x in refs if str(x).strip()]
        if not clean:
            return 0.0
        qv = embed_text(query)
        return max(_cosine_similarity(qv, embed_text(r)) for r in clean)

    grader = PairwiseRobustCounterfactualGraderV10(
        min_activation_rate=args.min_activation_rate,
        novelty_free_similarity=args.novelty_free_similarity,
        novelty_weight=args.novelty_weight,
    )

    def eval_query(role: str, query: str, bundle, anchor: str, semantic_sim: float):
        plus_q_embs=[]; minus_q_embs=[]; sham_q_embs=[]
        plus_a_embs=[]; minus_a_embs=[]; sham_a_embs=[]
        selected=[]; sham_selected=[]; branch_rows=[]
        for p in bundle:
            pa = execute_cached(p.plus_spec, anchor, cache)
            ma = execute_cached(p.minus_spec, anchor, cache)
            sa = execute_cached(p.sham_spec, anchor, cache) if p.sham_spec is not None else None
            pq = execute_cached(p.plus_spec, query, cache)
            mq = execute_cached(p.minus_spec, query, cache)
            sq = execute_cached(p.sham_spec, query, cache) if p.sham_spec is not None else None
            plus_a_embs.append(embed_text(pa.output)); minus_a_embs.append(embed_text(ma.output))
            plus_q_embs.append(embed_text(pq.output)); minus_q_embs.append(embed_text(mq.output))
            if sa is not None and sq is not None:
                sham_a_embs.append(embed_text(sa.output)); sham_q_embs.append(embed_text(sq.output))
            selected.append(role in set(pq.selected_roles_training_only))
            sham_selected.append(role in set(sq.selected_roles_training_only) if sq is not None else role in set(pq.selected_roles_training_only))
            branch_rows.append({
                "pair_id": p.pair_id,
                "plus_selected": list(pq.selected_roles_training_only),
                "minus_selected": list(mq.selected_roles_training_only),
                "sham_selected": list(sq.selected_roles_training_only) if sq is not None else [],
                "plus_roles_training_only": list(get_roles(p.plus_spec)),
                "minus_roles_training_only": list(get_roles(p.minus_spec)),
                "sham_roles_training_only": list(get_roles(p.sham_spec)) if p.sham_spec is not None else [],
                "hard_negative_replacement_training_only": str(getattr(p, "negative_replacement", "")),
                "plus_output": pq.output,
                "minus_output": mq.output,
                "sham_output": sq.output if sq is not None else "",
                "plus_cache_hit": pq.cache_hit,
                "minus_cache_hit": mq.cache_hit,
                "sham_cache_hit": sq.cache_hit if sq is not None else None,
            })
        rb = grader.score(
            target_selected_flags=selected,
            sham_target_selected_flags=sham_selected,
            plus_query_embs=plus_q_embs,
            minus_query_embs=minus_q_embs,
            plus_anchor_embs=plus_a_embs,
            minus_anchor_embs=minus_a_embs,
            sham_query_embs=sham_q_embs if len(sham_q_embs)==len(bundle) else None,
            sham_anchor_embs=sham_a_embs if len(sham_a_embs)==len(bundle) else None,
            semantic_max_similarity=semantic_sim,
        )
        for i,row in enumerate(branch_rows[:rb.pair_count]):
            row["target_distance"] = rb.target_pairwise[i]
            row["sham_distance"] = rb.sham_pairwise[i]
            row["specific_pairwise"] = rb.specific_pairwise[i]
            row["anchor_specific_pairwise"] = rb.anchor_specific_pairwise[i]
            row["delta_vs_anchor_pairwise"] = rb.delta_vs_anchor_pairwise[i]
        return rb, branch_rows

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    print(f"[evaluate] round={args.round} roles={len(role_seq)} candidates={len(raw_rows)} GPU_REQUIRED=False", flush=True)
    with tmp.open("w", encoding="utf-8") as f:
        for role_pos, role in enumerate(role_seq, 1):
            bundle = schedule[(args.round, role)]
            confirmation_bundle = conf_schedule[(args.round, role)]
            anchor = by_role[role][0]["anchor_probe"]
            group_seen=[]; candidates=[]; rewards=[]
            for src in by_role[role]:
                status="ok"; rb=None; branches=[]; semantic_sim=0.0
                q=str(src.get("query", ""))
                if not src.get("valid", False):
                    status=f"invalid:{src.get('invalid_reason','')}"; reward=-0.80
                elif _norm_text(q)==_norm_text(anchor):
                    status="exact_anchor_duplicate"; reward=-0.70
                elif any(_norm_text(q)==_norm_text(x) for x in group_seen):
                    status="exact_group_duplicate"; reward=-0.65
                elif _word_jaccard(q, anchor)>=0.92:
                    status="near_anchor_duplicate"; reward=-0.60
                elif any(_word_jaccard(q,x)>=0.95 for x in group_seen):
                    status="near_group_duplicate"; reward=-0.55
                else:
                    refs=[anchor]+novelty_refs.get(role,[])+group_seen
                    semantic_sim=semantic_similarity(q, refs)
                    group_seen.append(q)
                    if semantic_sim>=args.semantic_hard_duplicate:
                        status="semantic_duplicate"; reward=-0.60
                    else:
                        g=gate.check(q)
                        if not g.allowed:
                            status=f"blocked:{g.stage}:{g.category}"; reward=-1.50
                        else:
                            rb,branches=eval_query(role,q,bundle,anchor,semantic_sim)
                            reward=float(rb.total)
                c={"src":src,"status":status,"rb":rb,"branches":branches,"reward":float(reward),"semantic_sim":semantic_sim,"confirm_rb":None,"confirm_branches":[],"confirmation_pass":False}
                candidates.append(c); rewards.append(float(reward))
                ci=int(src["candidate_index"])
                if rb is None:
                    print(f"[eval] round={args.round} role={role:<24} cand={ci}/{args.group_size} {status} R={reward:+.3f}", flush=True)
                else:
                    print(f"[eval] round={args.round} role={role:<24} cand={ci}/{args.group_size} R={reward:+.3f} ACT={rb.activation_rate:.2f} ROB={rb.robust_specific:+.3f} pairs+={rb.positive_specific_pair_rate:.2f}", flush=True)

            adv=group_relative_advantages(rewards)
            primary_strict=[i for i,x in enumerate(candidates) if x["status"]=="ok" and x["rb"] is not None and x["rb"].strict_specific_pass and x["semantic_sim"]<=args.holdout_max_similarity]
            primary_strict.sort(key=lambda i:candidates[i]["reward"], reverse=True)
            for i in primary_strict[:max(0,args.confirm_top_k)]:
                if not confirmation_bundle: continue
                c=candidates[i]
                crb,cbranches=eval_query(role,c["src"]["query"],confirmation_bundle,anchor,c["semantic_sim"])
                c["confirm_rb"]=crb; c["confirm_branches"]=cbranches; c["confirmation_pass"]=bool(crb.strict_specific_pass)
                print(f"[confirm] role={role:<24} cand={c['src']['candidate_index']} pass={c['confirmation_pass']} ACT={crb.activation_rate:.2f} ROB={crb.robust_specific:+.3f}", flush=True)
            confirmed=[i for i in primary_strict if candidates[i]["confirmation_pass"]]
            elite_i=max(confirmed,key=lambda i:candidates[i]["reward"]) if confirmed else None
            for i,(c,a) in enumerate(zip(candidates,adv)):
                src=c["src"]; elite=bool(elite_i is not None and i==elite_i)
                rec=dict(src)
                rec.update({
                    "phase":"train",
                    "status":c["status"],
                    "reward":c["reward"],
                    "advantage":float(a),
                    "elite":elite,
                    "semantic_max_similarity":c["semantic_sim"],
                    "reward_breakdown_training_only":c["rb"].to_dict() if c["rb"] is not None else None,
                    "branches_training_only":c["branches"],
                    "confirmation_pass_training_only":c["confirmation_pass"],
                    "confirmation_reward_breakdown_training_only":c["confirm_rb"].to_dict() if c["confirm_rb"] is not None else None,
                    "confirmation_branches_training_only":c["confirm_branches"],
                })
                f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush()
                if elite:
                    novelty_refs[role] = list(dict.fromkeys(novelty_refs.get(role,[]) + [src["query"]]))[-8:]
            print(f"[group-done] round={args.round} role={role:<24} mean={float(np.mean(rewards)):+.3f} best={max(rewards):+.3f} strict={len(primary_strict)}/{args.group_size} confirmed={len(confirmed)}", flush=True)
    os.replace(tmp,out)
    print(f"[evaluate-done] round={args.round} output={out}", flush=True)


if __name__ == "__main__":
    main()
