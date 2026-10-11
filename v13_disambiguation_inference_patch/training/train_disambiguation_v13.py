from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.adaptive_fanout_count_v12 import tree_sha256
from rl.generative_probe_env_v7 import ResponseCacheV7
from v8.counterfactual import get_k, get_mode, get_roles
from v8.embedding import FrozenOutputEmbedderV8
from v8.execution import execute_cached
from v10.checkpoint import commit_round_snapshot, load_committed_state
from v10.gate import BenignGateV10
from v10.reward import PairwiseRobustCounterfactualGraderV10, group_relative_advantages
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS, anchor_probe
from training.train_cross_role_probe_discovery_v11 import (
    TrainSample,
    _balanced_minibatches,
    _build_stratified_schedule,
    _cosine_similarity,
    _kl_k3,
    _load_seed_elites,
    _merge_unique,
    _norm_text,
    _word_jaccard,
)
from v13.hard_negative import (
    build_hard_pairs,
    build_neighbor_map,
    excluded_v12_compositions,
    make_fresh_fanout_specs,
)
from v13.policy import DisambiguationProbePolicyV13


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="V13 hard-negative behavioral-disambiguation continuation from V11")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--resume-adapter", default="artifacts/v11_policy")
    ap.add_argument("--v11-summary", default="artifacts/v11_summary.json")
    ap.add_argument("--rounds-per-role", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--bundle-size", type=int, default=3)
    ap.add_argument("--confirmation-size", type=int, default=1)
    ap.add_argument("--confirm-top-k", type=int, default=2)
    ap.add_argument("--systems-per-cell", type=int, default=4)
    ap.add_argument("--neighbor-k", type=int, default=5)
    ap.add_argument("--balanced-minibatches", type=int, default=3)
    ap.add_argument("--grpo-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--clip", type=float, default=0.20)
    ap.add_argument("--kl-beta", type=float, default=0.0)
    ap.add_argument("--elite-nll-weight", type=float, default=0.06)
    ap.add_argument("--temperature", type=float, default=0.90)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--gate-mode", choices=["hard", "hybrid"], default="hard")
    ap.add_argument("--guard-model", default="gpt-4o-mini")
    ap.add_argument("--min-activation-rate", type=float, default=0.75)
    ap.add_argument("--novelty-free-similarity", type=float, default=0.90)
    ap.add_argument("--novelty-weight", type=float, default=0.25)
    ap.add_argument("--semantic-hard-duplicate", type=float, default=0.97)
    ap.add_argument("--seed", type=int, default=1313)
    ap.add_argument("--syntax-retries", type=int, default=3)
    ap.add_argument("--resume-from-checkpoint", action="store_true")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--output-dir", default="artifacts/v13_policy")
    ap.add_argument("--log", default="artifacts/v13_probe_disambiguation.jsonl")
    ap.add_argument("--summary", default="artifacts/v13_summary.json")
    ap.add_argument("--response-public-cache", default="artifacts/v13_train_response_public.jsonl")
    ap.add_argument("--response-truth-cache", default="artifacts/v13_train_response_truth.jsonl")
    ap.add_argument("--checkpoint-state", default="artifacts/v13_checkpoint.json")
    ap.add_argument("--checkpoint-root", default="artifacts/v13_checkpoints")
    ap.add_argument("--optimizer-mirror", default="artifacts/v13_optimizer.pt")
    ap.add_argument("--optimizer-state", default="artifacts/v13_optimizer.pt")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    v11_adapter = str(args.resume_adapter)
    if not Path(v11_adapter).exists():
        raise SystemExit(f"V11 adapter missing: {v11_adapter}")
    v11_hash = tree_sha256(v11_adapter)
    start_round = 1
    resume_optimizer_path = None
    if args.resume_from_checkpoint:
        state = load_committed_state(args.checkpoint_state)
        start_round = int(state.get("next_round", 1))
        if start_round > args.rounds_per_role:
            print("V13 training already completed according to", args.checkpoint_state)
            return
        args.resume_adapter = str(state["adapter_dir"])
        resume_optimizer_path = str(state["optimizer_state"])
        # Drop any partial rows from an interrupted uncommitted round.
        lp = Path(args.log)
        if lp.exists():
            kept=[]
            for line in lp.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                try: r=json.loads(line)
                except Exception: continue
                if int(r.get("round",0)) < start_round: kept.append(r)
            lp.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in kept),encoding="utf-8")

    # Neighbor graph is derived from public role descriptions, not worker prompts/truth.
    embedder = FrozenOutputEmbedderV8(device="cpu")
    neighbor_map = build_neighbor_map(embedder, TRAIN_ROLES, top_k=args.neighbor_k)
    excluded = excluded_v12_compositions(TRAIN_ROLES)
    specs = make_fresh_fanout_specs(
        TRAIN_ROLES,
        prefix="v13_train",
        systems_per_cell=args.systems_per_cell,
        seed=args.seed,
        excluded_compositions=excluded,
    )
    pairs = build_hard_pairs(specs, TRAIN_ROLES, neighbor_map)
    by_role: dict[str, list[object]] = {r: [] for r in TRAIN_ROLES}
    for p in pairs:
        by_role[p.target_role].append(p)
    usable = [r for r in TRAIN_ROLES if len(by_role[r]) >= args.bundle_size + args.confirmation_size]
    if len(usable) != len(TRAIN_ROLES):
        missing = {r: len(by_role[r]) for r in TRAIN_ROLES if r not in usable}
        raise SystemExit(f"insufficient hard-pair coverage; increase --systems-per-cell: {missing}")

    schedule, confirmation_schedule = _build_stratified_schedule(
        by_role,
        usable,
        rounds=args.rounds_per_role,
        bundle_size=args.bundle_size,
        confirmation_size=args.confirmation_size,
        seed=args.seed,
    )

    print("=== V13 HARD-NEGATIVE BEHAVIORAL DISAMBIGUATION ===")
    print("continuation adapter:", args.resume_adapter)
    print("V11 policy sha256:", v11_hash)
    print("roles:", len(usable))
    print("fresh FANOUT train systems:", len(specs))
    print("hard matched pairs:", len(pairs))
    print("rounds/role:", args.rounds_per_role, "group:", args.group_size, "bundle:", args.bundle_size, "confirm:", args.confirmation_size)
    print("V12 inspected compositions excluded:", len(excluded))
    print("automatic hard neighbors (public descriptions only):")
    for r in usable:
        print(f"  {r:<26} -> {neighbor_map[r]}")
    print("IMPORTANT: V11 is read-only. V13 writes only artifacts/v13_*.")
    print("Policy observation never receives true roles, K, routing, traces, or branch identity.")
    if not args.execute:
        print("PLAN ONLY. Add --execute to train.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if Path(args.checkpoint_state).exists() and not args.resume_from_checkpoint:
        raise SystemExit(f"Existing V13 checkpoint found: {args.checkpoint_state}. Use the resume sbatch or archive/remove V13 artifacts.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    gate = BenignGateV10(mode=args.gate_mode, guard_model=args.guard_model)
    grader = PairwiseRobustCounterfactualGraderV10(
        min_activation_rate=args.min_activation_rate,
        novelty_free_similarity=args.novelty_free_similarity,
        novelty_weight=args.novelty_weight,
    )
    cache = ResponseCacheV7(args.response_public_cache, args.response_truth_cache)
    policy = DisambiguationProbePolicyV13(
        args.model_name,
        adapter_path=args.resume_adapter,
        trainable=True,
        use_4bit=not args.no_4bit,
        max_seq_length=1536,
    )
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    if resume_optimizer_path:
        optimizer.load_state_dict(torch.load(resume_optimizer_path, map_location="cpu"))
        print("resumed optimizer:", resume_optimizer_path, flush=True)

    outdir = Path(args.output_dir)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logf = Path(args.log).open("a" if args.resume_from_checkpoint else "w", encoding="utf-8")

    seed_elites = _load_seed_elites(args.v11_summary, usable, max_per_role=3)
    if args.resume_from_checkpoint and Path(args.log).exists():
        for line in Path(args.log).read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try: rr=json.loads(line)
            except Exception: continue
            if rr.get("elite") and rr.get("target_role") in seed_elites:
                seed_elites[rr["target_role"]] = _merge_unique(seed_elites[rr["target_role"]],[rr.get("query","")],limit=6)
    novelty_refs = {r: list(seed_elites.get(r, [])) for r in usable}

    def emb(text: str):
        return embedder.encode(text)

    def semantic_similarity(query: str, refs: list[str]) -> float:
        refs = [x for x in refs if str(x).strip()]
        if not refs:
            return 0.0
        qv = emb(query)
        return max(_cosine_similarity(qv, emb(x)) for x in refs)

    def eval_query(role: str, query: str, bundle: list[object], anchor: str, sem: float):
        p_q=[]; m_q=[]; s_q=[]; p_a=[]; m_a=[]; s_a=[]; sel=[]; sham_sel=[]; rows=[]
        for p in bundle:
            pa=execute_cached(p.plus_spec, anchor, cache); ma=execute_cached(p.minus_spec, anchor, cache); sa=execute_cached(p.sham_spec, anchor, cache)
            pq=execute_cached(p.plus_spec, query, cache); mq=execute_cached(p.minus_spec, query, cache); sq=execute_cached(p.sham_spec, query, cache)
            p_a.append(emb(pa.output)); m_a.append(emb(ma.output)); s_a.append(emb(sa.output))
            p_q.append(emb(pq.output)); m_q.append(emb(mq.output)); s_q.append(emb(sq.output))
            sel.append(role in set(pq.selected_roles_training_only))
            sham_sel.append(role in set(sq.selected_roles_training_only))
            rows.append({
                "pair_id": p.pair_id,
                "mode": get_mode(p.plus_spec), "k": get_k(p.plus_spec),
                "plus_roles_training_only": list(get_roles(p.plus_spec)),
                "minus_roles_training_only": list(get_roles(p.minus_spec)),
                "sham_roles_training_only": list(get_roles(p.sham_spec)),
                "hard_negative_replacement_training_only": p.negative_replacement,
                "plus_selected": list(pq.selected_roles_training_only),
                "minus_selected": list(mq.selected_roles_training_only),
                "sham_selected": list(sq.selected_roles_training_only),
                "plus_output": pq.output, "minus_output": mq.output, "sham_output": sq.output,
            })
        rb=grader.score(
            target_selected_flags=sel, sham_target_selected_flags=sham_sel,
            plus_query_embs=p_q, minus_query_embs=m_q,
            plus_anchor_embs=p_a, minus_anchor_embs=m_a,
            sham_query_embs=s_q, sham_anchor_embs=s_a,
            semantic_max_similarity=sem,
        )
        for i,row in enumerate(rows[:rb.pair_count]):
            row["specific_pairwise_training_only"] = rb.specific_pairwise[i]
            row["delta_vs_anchor_pairwise_training_only"] = rb.delta_vs_anchor_pairwise[i]
        return rb, rows

    update_idx = 0
    def update(buf: list[TrainSample]):
        nonlocal update_idx
        if not buf: return
        update_idx += 1
        policy.model.train()
        for epoch in range(args.grpo_epochs):
            optimizer.zero_grad(set_to_none=True)
            n=max(len(buf),1); loss_sum=0.0
            for tr in buf:
                new_lp=policy.mean_logprob(tr.prompt,tr.completion)
                old_lp=torch.tensor(tr.old_logp,dtype=new_lp.dtype,device=new_lp.device)
                ratio=torch.exp(torch.clamp(new_lp-old_lp,-4.0,4.0))
                adv=torch.tensor(tr.advantage,dtype=new_lp.dtype,device=new_lp.device)
                pg=-torch.minimum(ratio*adv, torch.clamp(ratio,1-args.clip,1+args.clip)*adv)
                if math.isfinite(tr.ref_logp) and args.kl_beta>0:
                    ref_lp=torch.tensor(tr.ref_logp,dtype=new_lp.dtype,device=new_lp.device)
                    kl=_kl_k3(new_lp,ref_lp,torch)
                else: kl=new_lp*0.0
                elite_nll=(-new_lp) if tr.elite else new_lp*0.0
                loss=pg+args.kl_beta*kl+args.elite_nll_weight*elite_nll
                (loss/n).backward(); loss_sum += float(loss.detach().cpu())
            torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(),1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            print(f"update={update_idx} epoch={epoch+1} loss={loss_sum/n:+.4f}", flush=True)
        policy.model.eval()

    all_elites: dict[str,list[dict]]={r:[] for r in usable}
    total_groups=len(usable)*args.rounds_per_role
    group_num=0
    for rd in range(start_round,args.rounds_per_role+1):
        print(f"=== V13 TRAIN ROUND {rd}/{args.rounds_per_role} ===", flush=True)
        round_by_role={r:[] for r in usable}
        order=list(usable); random.Random(args.seed+rd*7919).shuffle(order)
        for role in order:
            group_num += 1
            bundle=schedule[(rd,role)]; conf_bundle=confirmation_schedule[(rd,role)]
            anchor=anchor_probe(role)
            competitors=neighbor_map[role][:4]
            obs={
                "candidate_role":role,
                "candidate_description":ROLE_DESCRIPTIONS[role],
                "anchor_probe":anchor,
                "avoid_probes":novelty_refs[role][-6:],
                "confusable_roles":[{"role":r,"description":ROLE_DESCRIPTIONS[r]} for r in competitors],
                "public_history":[],
            }
            candidates=[]; rewards=[]; group_seen=[]
            for ci in range(args.group_size):
                sample=None
                for _try in range(max(1,args.syntax_retries)):
                    trial=policy.sample_probe(obs,temperature=args.temperature,top_p=args.top_p)
                    sample=trial
                    if trial.valid: break
                assert sample is not None
                status="ok"; rb=None; branches=[]; sem=0.0
                if not sample.valid:
                    status=f"invalid:{sample.invalid_reason}"; reward=-0.8
                elif _norm_text(sample.query)==_norm_text(anchor):
                    status="anchor_duplicate"; reward=-0.7
                elif any(_word_jaccard(sample.query,q)>=0.95 for q in group_seen):
                    status="group_duplicate"; reward=-0.6
                else:
                    refs=[anchor]+novelty_refs[role]+group_seen
                    sem=semantic_similarity(sample.query,refs)
                    group_seen.append(sample.query)
                    if sem>=args.semantic_hard_duplicate:
                        status="semantic_duplicate"; reward=-0.6
                    else:
                        g=gate.check(sample.query)
                        if not g.allowed:
                            status=f"blocked:{g.stage}:{g.category}"; reward=-1.5
                        else:
                            rb,branches=eval_query(role,sample.query,bundle,anchor,sem)
                            reward=float(rb.total)
                rewards.append(float(reward))
                candidates.append({"sample":sample,"status":status,"rb":rb,"branches":branches,"reward":float(reward),"sem":sem,"confirm":None,"confirm_branches":[],"confirmed":False})
                if rb:
                    print(f"  cand={ci+1}/{args.group_size} role={role:<26} R={reward:+.3f} ACT={rb.activation_rate:.2f} ROB={rb.robust_specific:+.3f} strict={rb.strict_specific_pass}",flush=True)
                else:
                    print(f"  cand={ci+1}/{args.group_size} role={role:<26} {status} R={reward:+.3f}",flush=True)

            adv=group_relative_advantages(rewards)
            strict=[i for i,c in enumerate(candidates) if c["rb"] is not None and c["rb"].strict_specific_pass]
            strict.sort(key=lambda i:candidates[i]["reward"],reverse=True)
            for i in strict[:args.confirm_top_k]:
                if not conf_bundle: continue
                c=candidates[i]
                crb,cbr=eval_query(role,c["sample"].query,conf_bundle,anchor,c["sem"])
                c["confirm"]=crb; c["confirm_branches"]=cbr; c["confirmed"]=bool(crb.strict_specific_pass)
                print(f"    confirm cand={i+1} role={role:<26} pass={c['confirmed']} ROB={crb.robust_specific:+.3f}",flush=True)
            confirmed=[i for i in strict if candidates[i]["confirmed"]]
            elite_i=max(confirmed,key=lambda i:candidates[i]["reward"]) if confirmed else None
            gid=f"v13r{rd:02d}_{role}_{group_num:03d}"
            for i,(c,a) in enumerate(zip(candidates,adv)):
                s=c["sample"]; elite=(i==elite_i)
                if s.completion.strip():
                    round_by_role[role].append(TrainSample(role,s.prompt,s.completion,s.old_mean_logprob,s.ref_mean_logprob,float(a),c["reward"],elite,gid))
                rec={
                    "phase":"train","round":rd,"group_id":gid,"candidate_index":i+1,
                    "target_role":role,"candidate_description_public":ROLE_DESCRIPTIONS[role],
                    "confusable_roles_public":competitors,
                    "query":s.query,"raw_completion":s.raw_completion,"status":c["status"],
                    "reward":c["reward"],"advantage":float(a),"elite":elite,"semantic_max_similarity":c["sem"],
                    "reward_breakdown_training_only":c["rb"].to_dict() if c["rb"] else None,
                    "branches_training_only":c["branches"],
                    "confirmation_pass_training_only":c["confirmed"],
                    "confirmation_reward_breakdown_training_only":c["confirm"].to_dict() if c["confirm"] else None,
                    "confirmation_branches_training_only":c["confirm_branches"],
                }
                logf.write(json.dumps(rec,ensure_ascii=False)+"\n"); logf.flush()
                if elite:
                    all_elites[role].append(rec); novelty_refs[role]=_merge_unique(novelty_refs[role],[s.query],limit=10)
            best=max(rewards)
            print(f"group={group_num}/{total_groups} role={role:<26} Rmean={np.mean(rewards):+.3f} Rbest={best:+.3f} strict={len(strict)}/{args.group_size} confirmed={len(confirmed)}",flush=True)

        minibatches=_balanced_minibatches(round_by_role,usable,batches=args.balanced_minibatches,seed=args.seed+123457*rd)
        for bi,mb in enumerate(minibatches,1):
            print(f"=== OPTIMIZE round={rd} minibatch={bi}/{len(minibatches)} n={len(mb)} ===",flush=True)
            update(mb)
        state=commit_round_snapshot(policy=policy,optimizer=optimizer,round_i=rd,total_rounds=args.rounds_per_role,checkpoint_root=args.checkpoint_root,checkpoint_state=args.checkpoint_state,public_output_dir=outdir,log_path=args.log,torch_module=torch)
        mirror=Path(args.optimizer_mirror); mirror.parent.mkdir(parents=True,exist_ok=True)
        if Path(state["optimizer_state"]).resolve()!=mirror.resolve(): shutil.copy2(state["optimizer_state"],mirror)
        print(f"COMMITTED V13 round={rd} snapshot={state['round_snapshot']}",flush=True)

    logf.close()
    rows=[json.loads(x) for x in Path(args.log).read_text(encoding="utf-8").splitlines() if x.strip()]
    evalrows=[r for r in rows if r.get("reward_breakdown_training_only")]
    summary={
        "protocol":"v13_continuation_from_v11_hard_semantic_neighbor_counterfactuals",
        "v11_start_policy_sha256":v11_hash,
        "v13_policy_sha256":tree_sha256(args.output_dir),
        "train_roles":list(usable),
        "neighbor_map_public":neighbor_map,
        "fresh_train_systems":len(specs),"hard_pairs":len(pairs),
        "candidate_count":len(rows),"evaluated_count":len(evalrows),
        "strict_specific_count":sum(bool(r["reward_breakdown_training_only"].get("strict_specific_pass")) for r in evalrows),
        "confirmed_elite_count":sum(bool(r.get("elite")) for r in rows),
        "roles_with_confirmed_elite":sorted({r["target_role"] for r in rows if r.get("elite")}),
        "elite_probes":{role:[{"query":r["query"],"reward":r["reward"]} for r in rows if r.get("elite") and r.get("target_role")==role] for role in usable},
        "development_training_only":True,
    }
    Path(args.summary).write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("V13 training summary:",args.summary)
    print("V13 policy:",args.output_dir)


if __name__ == "__main__":
    main()
