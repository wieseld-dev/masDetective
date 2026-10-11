from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_zoo_v11 import _make_train_specs, get_v11_split, MODES
from v8.counterfactual import clone_spec, get_k, get_mode, get_roles
from v8.embedding import FrozenOutputEmbedderV8
from v8.execution import execute_cached
from v10.gate import BenignGateV10
from v10.reward import PairwiseRobustCounterfactualGraderV10
from v11.role_universe import TRAIN_ROLES, DEV_UNSEEN_ROLES, ALL_ROLES, anchor_probe
from v11_2.counterfactual import build_hard_counterfactual_pairs
from v11_2.split_common import _cosine_similarity, _norm_text, _word_jaccard, _load_seed_elites


def load_jsonl(path: str):
    rows=[]
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp,path)


def locked_known_specs(seed: int, per_mode: int):
    original = {tuple(sorted(map(str, s.worker_roles))) for s in get_v11_split("train")}
    candidates = _make_train_specs(per_mode=max(per_mode+2, 8), seed=seed)
    fresh = [s for s in candidates if tuple(sorted(map(str, s.worker_roles))) not in original]
    selected=[]
    for mode in MODES:
        mm=[s for s in fresh if s.execution_mode==mode]
        if len(mm)<per_mode:
            raise RuntimeError(f"not enough fresh {mode} specs: {len(mm)}")
        selected.extend(mm[:per_mode])
    out=[]
    for i,s in enumerate(selected):
        out.append(clone_spec(
            s,
            system_id=f"v112eval_known_{seed}_{i:03d}_{get_mode(s)}_k{get_k(s)}",
            worker_roles=list(get_roles(s)),
        ))
    train_comps={tuple(sorted(map(str,s.worker_roles))) for s in get_v11_split("train")}
    if any(tuple(sorted(map(str,s.worker_roles))) in train_comps for s in out):
        raise RuntimeError("locked known benchmark overlaps V11 training composition")
    return out


def locked_unseen_specs():
    out=[]
    for i,s in enumerate(get_v11_split("dev_unseen_roles")):
        out.append(clone_spec(
            s,
            system_id=f"v112eval_unseen_{i:03d}_{get_mode(s)}_k{get_k(s)}",
            worker_roles=list(get_roles(s)),
        ))
    return out


def manifest_for(known_specs, unseen_specs, known_pairs, unseen_pairs, args):
    def spec_row(s):
        return {"system_id":str(s.system_id),"mode":get_mode(s),"k":get_k(s),"roles":list(get_roles(s))}
    def pair_row(p):
        return {"pair_id":str(p.pair_id),"target":str(p.target_role),"negative_replacement":str(p.negative_replacement),"plus_roles":list(get_roles(p.plus_spec)),"minus_roles":list(get_roles(p.minus_spec)),"sham_roles":list(get_roles(p.sham_spec)) if p.sham_spec else []}
    obj={
        "version":"v11_2_strong_eval_v1",
        "known_seed":args.known_seed,
        "known_per_mode":args.known_per_mode,
        "known_specs":[spec_row(x) for x in known_specs],
        "unseen_specs":[spec_row(x) for x in unseen_specs],
        "known_pairs":[pair_row(x) for x in known_pairs],
        "unseen_pairs":[pair_row(x) for x in unseen_pairs],
        "notes":{
            "known":"fresh synthetic compositions generated after V11/V11.2 training design; composition-disjoint from V11 train",
            "unseen":"the six V11 unseen role types; this is development regression, not a pristine final-paper test",
        },
    }
    canonical=json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    obj["sha256"]=hashlib.sha256(canonical).hexdigest()
    return obj


def main():
    ap=argparse.ArgumentParser(description="CPU/API strong matched evaluation of V11 vs V11.2")
    ap.add_argument("--generated",default="artifacts/v11_2_strong_eval/generated.jsonl")
    ap.add_argument("--output",default="artifacts/v11_2_strong_eval/probe_eval.jsonl")
    ap.add_argument("--manifest",default="artifacts/v11_2_strong_eval/benchmark_manifest.json")
    ap.add_argument("--initial-elite-summary",default="artifacts/v11_summary.json")
    ap.add_argument("--known-seed",type=int,default=24681357)
    ap.add_argument("--known-per-mode",type=int,default=6)
    ap.add_argument("--gate-mode",choices=["hard","hybrid"],default="hybrid")
    ap.add_argument("--guard-model",default="gpt-4o-mini")
    ap.add_argument("--min-activation-rate",type=float,default=.75)
    ap.add_argument("--novelty-free-similarity",type=float,default=.90)
    ap.add_argument("--novelty-weight",type=float,default=.25)
    ap.add_argument("--semantic-hard-duplicate",type=float,default=.97)
    ap.add_argument("--response-public-cache",default="artifacts/v11_2_strong_eval/response_public.jsonl")
    ap.add_argument("--response-truth-cache",default="artifacts/v11_2_strong_eval/response_truth.jsonl")
    ap.add_argument("--seed-response-public-cache",default="artifacts/v11_seed_none_public.jsonl")
    ap.add_argument("--seed-response-truth-cache",default="artifacts/v11_seed_none_truth.jsonl")
    args=ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    rows=load_jsonl(args.generated)
    if not rows: raise SystemExit("no generated probes")

    known_specs=locked_known_specs(args.known_seed,args.known_per_mode)
    unseen_specs=locked_unseen_specs()
    known_pairs=build_hard_counterfactual_pairs(known_specs,TRAIN_ROLES,include_sham=True)
    unseen_pairs_all=build_hard_counterfactual_pairs(unseen_specs,ALL_ROLES,include_sham=True)
    unseen_pairs=[p for p in unseen_pairs_all if p.target_role in set(DEV_UNSEEN_ROLES)]
    by_split_role={"known_hard_locked":defaultdict(list),"unseen_hard_dev":defaultdict(list)}
    for p in known_pairs: by_split_role["known_hard_locked"][p.target_role].append(p)
    for p in unseen_pairs: by_split_role["unseen_hard_dev"][p.target_role].append(p)

    manifest=manifest_for(known_specs,unseen_specs,known_pairs,unseen_pairs,args)
    atomic_json(Path(args.manifest),manifest)
    print(f"[benchmark] sha256={manifest['sha256']} known_specs={len(known_specs)} known_pairs={len(known_pairs)} unseen_specs={len(unseen_specs)} unseen_pairs={len(unseen_pairs)}",flush=True)
    kc=Counter(p.target_role for p in known_pairs); uc=Counter(p.target_role for p in unseen_pairs)
    print("[coverage-known] "+json.dumps(dict(kc),sort_keys=True),flush=True)
    print("[coverage-unseen] "+json.dumps(dict(uc),sort_keys=True),flush=True)

    for src,dst in ((args.seed_response_public_cache,args.response_public_cache),(args.seed_response_truth_cache,args.response_truth_cache)):
        if Path(src).exists() and not Path(dst).exists():
            Path(dst).parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    cache=ResponseCacheV7(args.response_public_cache,args.response_truth_cache)
    gate=BenignGateV10(mode=args.gate_mode,guard_model=args.guard_model)
    embedder=FrozenOutputEmbedderV8(device="cpu")
    emb_cache={}
    def emb(t):
        t=str(t)
        if t not in emb_cache: emb_cache[t]=embedder.encode(t)
        return emb_cache[t]
    seed_elites=_load_seed_elites(args.initial_elite_summary,list(TRAIN_ROLES)+list(DEV_UNSEEN_ROLES),max_per_role=3)
    grader=PairwiseRobustCounterfactualGraderV10(min_activation_rate=args.min_activation_rate,novelty_free_similarity=args.novelty_free_similarity,novelty_weight=args.novelty_weight)

    def semantic_similarity(q,refs:Sequence[str]):
        refs=[str(x).strip() for x in refs if str(x).strip()]
        if not refs: return 0.0
        qv=emb(q); return max(_cosine_similarity(qv,emb(x)) for x in refs)

    def eval_query(role,query,bundle,anchor,semantic_sim):
        pqe=[]; mqe=[]; sqe=[]; pae=[]; mae=[]; sae=[]; selected=[]; sham_selected=[]; br=[]
        for p in bundle:
            pa=execute_cached(p.plus_spec,anchor,cache); ma=execute_cached(p.minus_spec,anchor,cache); sa=execute_cached(p.sham_spec,anchor,cache) if p.sham_spec else None
            pq=execute_cached(p.plus_spec,query,cache); mq=execute_cached(p.minus_spec,query,cache); sq=execute_cached(p.sham_spec,query,cache) if p.sham_spec else None
            pae.append(emb(pa.output)); mae.append(emb(ma.output)); pqe.append(emb(pq.output)); mqe.append(emb(mq.output))
            if sa is not None and sq is not None: sae.append(emb(sa.output)); sqe.append(emb(sq.output))
            selected.append(role in set(pq.selected_roles_training_only)); sham_selected.append(role in set(sq.selected_roles_training_only) if sq else role in set(pq.selected_roles_training_only))
            br.append({
                "pair_id":str(p.pair_id),"target_role":role,"hard_negative_replacement":str(getattr(p,"negative_replacement","")),
                "plus_roles":list(get_roles(p.plus_spec)),"minus_roles":list(get_roles(p.minus_spec)),"sham_roles":list(get_roles(p.sham_spec)) if p.sham_spec else [],
                "plus_selected":list(pq.selected_roles_training_only),"minus_selected":list(mq.selected_roles_training_only),"sham_selected":list(sq.selected_roles_training_only) if sq else [],
                "plus_output":pq.output,"minus_output":mq.output,"sham_output":sq.output if sq else "",
                "plus_cache_hit":pq.cache_hit,"minus_cache_hit":mq.cache_hit,"sham_cache_hit":sq.cache_hit if sq else None,
            })
        rb=grader.score(target_selected_flags=selected,sham_target_selected_flags=sham_selected,plus_query_embs=pqe,minus_query_embs=mqe,plus_anchor_embs=pae,minus_anchor_embs=mae,sham_query_embs=sqe if len(sqe)==len(bundle) else None,sham_anchor_embs=sae if len(sae)==len(bundle) else None,semantic_max_similarity=semantic_sim)
        for i,row in enumerate(br[:rb.pair_count]):
            row["target_distance"]=rb.target_pairwise[i]; row["sham_distance"]=rb.sham_pairwise[i]; row["specific_pairwise"]=rb.specific_pairwise[i]; row["anchor_specific_pairwise"]=rb.anchor_specific_pairwise[i]; row["delta_vs_anchor_pairwise"]=rb.delta_vs_anchor_pairwise[i]
        return rb,br

    grouped=defaultdict(list)
    for r in rows: grouped[(r["method"],r["target_role"])].append(r)
    for xs in grouped.values(): xs.sort(key=lambda x:int(x["probe_index"]))

    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix(out.suffix+".tmp")
    total=sum(len(v) for v in grouped.values()); done=0
    with tmp.open("w",encoding="utf-8") as f:
        for (method,role),xs in sorted(grouped.items()):
            split="known_hard_locked" if role in TRAIN_ROLES else "unseen_hard_dev"
            bundle=by_split_role[split].get(role,[])
            anchor=anchor_probe(role); seen=[]
            for src in xs:
                done+=1; q=str(src.get("query","")).strip(); status="ok"; rb=None; branches=[]
                refs=[anchor]+list(seed_elites.get(role,[]))+list(seen)
                sem=semantic_similarity(q,refs) if q else 1.0
                if not src.get("valid",False): status=f"invalid:{src.get('invalid_reason','')}"
                elif not bundle: status="no_benchmark_pairs"
                elif _norm_text(q)==_norm_text(anchor): status="exact_anchor_duplicate"
                elif any(_norm_text(q)==_norm_text(x) for x in seen): status="exact_group_duplicate"
                elif _word_jaccard(q,anchor)>=.92: status="near_anchor_duplicate"
                elif any(_word_jaccard(q,x)>=.95 for x in seen): status="near_group_duplicate"
                elif sem>=args.semantic_hard_duplicate: status="semantic_duplicate"
                else:
                    g=gate.check(q)
                    if not g.allowed: status=f"blocked:{g.stage}:{g.category}"
                    else: rb,branches=eval_query(role,q,bundle,anchor,sem)
                if q: seen.append(q)
                rec=dict(src)
                rec.update({"eval_split":split,"benchmark_sha256":manifest["sha256"],"status":status,"semantic_max_similarity":sem,"pair_count":len(bundle),"reward":float(rb.total) if rb else None,"reward_breakdown":rb.to_dict() if rb else None,"branches":branches})
                f.write(json.dumps(rec,ensure_ascii=False)+"\n"); f.flush()
                if rb:
                    print(f"[eval] {done}/{total} split={split:<17} method={method:<5} role={role:<24} p={src['probe_index']} R={rb.total:+.3f} ACT={rb.activation_rate:.2f} ROB={rb.robust_specific:+.3f} Q25={rb.specific_q25:+.3f} pair+={rb.positive_specific_pair_rate:.2f} strict={rb.strict_specific_pass}",flush=True)
                else:
                    print(f"[eval] {done}/{total} split={split:<17} method={method:<5} role={role:<24} p={src['probe_index']} status={status}",flush=True)
    os.replace(tmp,out)
    print(f"[done] output={out}",flush=True)


if __name__=="__main__": main()
