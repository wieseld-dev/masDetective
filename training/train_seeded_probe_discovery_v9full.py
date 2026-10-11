from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from estimator.candidate_presence_contrastive_v64 import load_profile
from probes.role_probe_bank import primary_probe
from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_zoo_v65 import get_v65_split
from v8.counterfactual import build_counterfactual_pairs, split_base_specs
from v8.embedding import FrozenOutputEmbedderV8
from v8.execution import execute_cached
from v8.gate import BenignGateV8
from v9.policy import SeededProbePolicyV9
from v9.reward import RobustCounterfactualGraderV9, group_relative_advantages


@dataclass
class TrainSample:
    prompt: str
    completion: str
    old_logp: float
    ref_logp: float
    advantage: float
    reward: float
    elite: bool
    group_id: str


def _word_jaccard(a: str, b: str) -> float:
    import re
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(1, len(wa | wb))


def _kl_k3(new_lp, ref_lp, torch):
    delta = ref_lp - new_lp
    return torch.exp(torch.clamp(delta, -6.0, 6.0)) - delta - 1.0


def _choose_bundle(rng: random.Random, pairs: Sequence[object], n: int) -> list[object]:
    if not pairs:
        return []
    if len(pairs) <= n:
        out = list(pairs)
        rng.shuffle(out)
        return out
    return rng.sample(list(pairs), n)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="V9-FULL seeded robust counterfactual benign probe discovery")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--rounds-per-role", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--bundle-size", type=int, default=2)
    ap.add_argument("--groups-per-update", type=int, default=2)
    ap.add_argument("--grpo-epochs", type=int, default=2)
    ap.add_argument("--elite-nll-weight", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--clip", type=float, default=0.20)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gate-mode", choices=["hard", "hybrid"], default="hybrid")
    ap.add_argument("--guard-model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=909)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--resume-adapter", default="")
    ap.add_argument("--output-dir", default="artifacts/v9_probe_policy")
    ap.add_argument("--log", default="artifacts/v9_probe_discovery.jsonl")
    ap.add_argument("--summary", default="artifacts/v9_probe_summary.json")
    ap.add_argument("--response-public-cache", default="artifacts/v9_response_public.jsonl")
    ap.add_argument("--response-truth-cache", default="artifacts/v9_response_truth.jsonl")
    ap.add_argument("--holdout-eval", action="store_true")
    ap.add_argument("--holdout-probes-per-role", type=int, default=3)
    ap.add_argument("--syntax-retries", type=int, default=3, help="local generations per slot before giving up; no API cost")
    ap.add_argument("--start-group", type=int, default=1, help="resume from this 1-indexed group")
    ap.add_argument("--checkpoint-state", default="artifacts/v9_full_checkpoint.json")
    ap.add_argument("--optimizer-state", default="artifacts/v9_full_optimizer.pt")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    roles = tuple(profile["candidate_roles"])
    all_specs = list(get_v65_split("train"))
    # Use ~5/6 of V6.5 train for RL, reserve ~1/6 for the later detector, and use
    # the composition-disjoint V6.5 validation split as the actual policy holdout.
    rl_specs, detector_reserved_specs = split_base_specs(
        all_specs, detector_fraction=1/6, seed=args.seed
    )
    holdout_specs = list(get_v65_split("val"))
    train_pairs = build_counterfactual_pairs(rl_specs, roles, include_sham=True)
    holdout_pairs = build_counterfactual_pairs(holdout_specs, roles, include_sham=True)
    by_role: dict[str, list[object]] = {r: [] for r in roles}
    holdout_by_role: dict[str, list[object]] = {r: [] for r in roles}
    for p in train_pairs:
        by_role.setdefault(p.target_role, []).append(p)
    for p in holdout_pairs:
        holdout_by_role.setdefault(p.target_role, []).append(p)
    usable_roles = [r for r in roles if len(by_role.get(r, [])) >= 1]
    groups = len(usable_roles) * args.rounds_per_role
    candidate_count = groups * args.group_size
    # Each valid candidate is evaluated on plus/minus/sham for bundle_size pairs.
    upper_interactions = candidate_count * args.bundle_size * 3
    # Anchor baselines are cached per bundle/role and are much smaller than candidate work.
    print("=== V9-FULL SEEDED COUNTERFACTUAL PROBE DISCOVERY ===")
    print("human anchor per role: existing primary_probe(role)")
    print("policy generates: NEW standalone free-text benign probes, not bank selection")
    print("usable roles:", len(usable_roles), usable_roles)
    print("RL base systems:", len(rl_specs), "detector-reserved:", len(detector_reserved_specs), "composition-disjoint holdout:", len(holdout_specs))
    print("train matched pairs:", len(train_pairs), "holdout matched pairs:", len(holdout_pairs))
    print("rounds/role:", args.rounds_per_role, "group size:", args.group_size, "bundle size:", args.bundle_size)
    print("candidate probes:", candidate_count)
    print("syntax retries/slot (local only):", args.syntax_retries)
    print("upper-bound candidate MAS interactions:", upper_interactions)
    print("reward: activation is eligibility; robust target-specific external separation across matched bundles")
    print("reward model / V6.5 classifier in loop: NO")
    print("optimization: GRPO + small elite-imitation term on group winner")
    print("holdout evaluation in same job:", bool(args.holdout_eval))
    if not args.execute:
        print("PLAN ONLY. Add --execute to run.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gate = BenignGateV8(mode=args.gate_mode, guard_model=args.guard_model)
    embedder = FrozenOutputEmbedderV8(device="cpu")
    grader = RobustCounterfactualGraderV9()
    cache = ResponseCacheV7(args.response_public_cache, args.response_truth_cache)
    policy = SeededProbePolicyV9(
        args.model_name,
        adapter_path=(args.resume_adapter or None),
        trainable=True,
        use_4bit=not args.no_4bit,
        max_seq_length=1408,
    )
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    if args.start_group > 1 and Path(args.optimizer_state).exists():
        try:
            optimizer.load_state_dict(torch.load(args.optimizer_state, map_location="cpu"))
            print("resumed optimizer state:", args.optimizer_state, flush=True)
        except Exception as exc:
            print("WARNING: optimizer state could not be restored:", type(exc).__name__, flush=True)

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if args.start_group > 1 and Path(args.log).exists() else "w"
    logf = Path(args.log).open(log_mode, encoding="utf-8")

    update_buffer: list[TrainSample] = []
    update_idx = 0
    total_valid = 0
    total_candidates = 0
    positive_specific = 0
    elite_records: dict[str, list[dict]] = {r: [] for r in usable_roles}
    group_rewards: list[float] = []

    def update(buf: list[TrainSample]):
        nonlocal update_idx
        if not buf:
            return
        update_idx += 1
        policy.model.train()
        for epoch in range(args.grpo_epochs):
            optimizer.zero_grad(set_to_none=True)
            n = max(len(buf), 1)
            loss_sum = ratio_sum = kl_sum = 0.0
            for tr in buf:
                new_lp = policy.mean_logprob(tr.prompt, tr.completion)
                old_lp = torch.tensor(tr.old_logp, dtype=new_lp.dtype, device=new_lp.device)
                ratio = torch.exp(torch.clamp(new_lp - old_lp, -4.0, 4.0))
                adv = torch.tensor(tr.advantage, dtype=new_lp.dtype, device=new_lp.device)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv
                pg = -torch.minimum(unclipped, clipped)
                if math.isfinite(tr.ref_logp) and args.kl_beta > 0:
                    ref_lp = torch.tensor(tr.ref_logp, dtype=new_lp.dtype, device=new_lp.device)
                    kl = _kl_k3(new_lp, ref_lp, torch)
                else:
                    kl = new_lp * 0.0
                # Elite imitation makes the tiny 1.5B policy actually absorb successful
                # discoveries instead of relying only on a near-zero mean GRPO scalar.
                elite_nll = (-new_lp) if tr.elite else new_lp * 0.0
                loss = pg + args.kl_beta * kl + args.elite_nll_weight * elite_nll
                (loss / n).backward()
                loss_sum += float(loss.detach().cpu())
                ratio_sum += float(ratio.detach().cpu())
                kl_sum += float(kl.detach().cpu())
                del new_lp, old_lp, ratio, adv, unclipped, clipped, pg, kl, elite_nll, loss
            torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"update={update_idx} epoch={epoch+1} loss={loss_sum/n:+.4f} ratio={ratio_sum/n:.3f} kl={kl_sum/n:.4f}", flush=True)
        policy.model.eval()
        policy.save_adapter(str(outdir))
        Path(args.optimizer_state).parent.mkdir(parents=True, exist_ok=True)
        torch.save(optimizer.state_dict(), args.optimizer_state)

    def eval_query(role: str, query: str, bundle: Sequence[object], anchor: str):
        plus_q_embs=[]; minus_q_embs=[]; sham_q_embs=[]
        plus_a_embs=[]; minus_a_embs=[]; sham_a_embs=[]
        selected=[]
        branch_rows=[]
        for p in bundle:
            pa = execute_cached(p.plus_spec, anchor, cache)
            ma = execute_cached(p.minus_spec, anchor, cache)
            sa = execute_cached(p.sham_spec, anchor, cache) if p.sham_spec is not None else None
            pq = execute_cached(p.plus_spec, query, cache)
            mq = execute_cached(p.minus_spec, query, cache)
            sq = execute_cached(p.sham_spec, query, cache) if p.sham_spec is not None else None
            plus_a_embs.append(embedder.encode(pa.output)); minus_a_embs.append(embedder.encode(ma.output))
            plus_q_embs.append(embedder.encode(pq.output)); minus_q_embs.append(embedder.encode(mq.output))
            if sa is not None and sq is not None:
                sham_a_embs.append(embedder.encode(sa.output)); sham_q_embs.append(embedder.encode(sq.output))
            selected.append(role in set(pq.selected_roles_training_only))
            branch_rows.append({
                "pair_id": p.pair_id,
                "plus_selected": list(pq.selected_roles_training_only),
                "minus_selected": list(mq.selected_roles_training_only),
                "plus_output": pq.output,
                "minus_output": mq.output,
                "sham_output": sq.output if sq is not None else "",
            })
        rb = grader.score(
            target_selected_flags=selected,
            plus_query_embs=plus_q_embs,
            minus_query_embs=minus_q_embs,
            plus_anchor_embs=plus_a_embs,
            minus_anchor_embs=minus_a_embs,
            sham_query_embs=sham_q_embs if len(sham_q_embs)==len(bundle) else None,
            sham_anchor_embs=sham_a_embs if len(sham_a_embs)==len(bundle) else None,
        )
        return rb, branch_rows

    group_num = 0
    for round_i in range(args.rounds_per_role):
        role_order = list(usable_roles); rng.shuffle(role_order)
        for role in role_order:
            group_num += 1
            bundle = _choose_bundle(rng, by_role[role], args.bundle_size)
            if not bundle:
                continue
            if group_num < max(1, args.start_group):
                continue
            # Make generation randomness group-local so resume does not depend on consuming
            # all earlier torch RNG draws.
            torch.manual_seed(args.seed + 100003 * group_num)
            anchor = primary_probe(role)
            obs = {
                "candidate_role": role,
                "candidate_description": profile.get("role_descriptions", {}).get(role, role),
                "anchor_probe": anchor,
            }
            candidates=[]; rewards=[]; seen=[]
            for ci in range(args.group_size):
                total_candidates += 1
                sample = None
                syntax_attempts = []
                for _try in range(max(1, args.syntax_retries)):
                    trial = policy.sample_probe(obs, temperature=args.temperature, top_p=args.top_p)
                    syntax_attempts.append({"raw": trial.raw_completion, "valid": trial.valid, "reason": trial.invalid_reason})
                    sample = trial
                    if trial.valid:
                        break
                assert sample is not None
                status="ok"; rb=None; branches=[]
                if not sample.valid:
                    status=f"invalid:{sample.invalid_reason}"; reward=-0.80
                elif _word_jaccard(sample.query, anchor) >= 0.80:
                    status="anchor_paraphrase"; reward=-0.55
                elif any(_word_jaccard(sample.query, q) >= 0.90 for q in seen):
                    status="duplicate"; reward=-0.45
                else:
                    seen.append(sample.query)
                    g = gate.check(sample.query)
                    if not g.allowed:
                        status=f"blocked:{g.stage}:{g.category}"; reward=-1.50
                    else:
                        total_valid += 1
                        rb, branches = eval_query(role, sample.query, bundle, anchor)
                        reward=float(rb.total)
                        if rb.specific_separation > 0:
                            positive_specific += 1
                rewards.append(float(reward))
                candidates.append((sample,status,rb,branches,float(reward)))

            adv = group_relative_advantages(rewards)
            best_i = int(np.argmax(rewards))
            best_positive = rewards[best_i] > 0
            group_id=f"r{round_i+1:02d}_{role}_{group_num:03d}"
            for i, ((sample,status,rb,branches,reward), a) in enumerate(zip(candidates,adv)):
                elite = bool(i==best_i and best_positive and status=="ok")
                if sample.completion.strip():
                    update_buffer.append(TrainSample(
                        prompt=sample.prompt, completion=sample.completion,
                        old_logp=sample.old_mean_logprob, ref_logp=sample.ref_mean_logprob,
                        advantage=float(a), reward=reward, elite=elite, group_id=group_id,
                    ))
                rec={
                    "phase":"train","round":round_i+1,"group_id":group_id,"candidate_index":i+1,
                    "target_role":role,"anchor_probe":anchor,"query":sample.query,
                    "raw_completion":sample.raw_completion,"syntax_attempts":syntax_attempts,"status":status,"reward":reward,
                    "advantage":float(a),"elite":elite,
                    "reward_breakdown_training_only": rb.to_dict() if rb is not None else None,
                    "branches_training_only": branches,
                }
                logf.write(json.dumps(rec,ensure_ascii=False)+"\n"); logf.flush()
                if elite:
                    elite_records[role].append(rec)
            mean_r=float(np.mean(rewards)); group_rewards.append(mean_r)
            rb_best=candidates[best_i][2]
            spec=(rb_best.specific_separation if rb_best is not None else float("nan"))
            ar=(rb_best.activation_rate if rb_best is not None else float("nan"))
            print(f"group={group_num}/{groups} round={round_i+1} role={role:<10} Rmean={mean_r:+.3f} Rbest={max(rewards):+.3f} bestSpec={spec:+.3f} bestAct={ar:.2f} valid={sum(1 for x in candidates if x[1]=='ok')}/{args.group_size}",flush=True)
            if group_num % args.groups_per_update==0 or group_num==groups:
                update(update_buffer); update_buffer.clear()
                Path(args.checkpoint_state).parent.mkdir(parents=True, exist_ok=True)
                Path(args.checkpoint_state).write_text(json.dumps({
                    "next_group": group_num + 1,
                    "completed_group": group_num,
                    "total_groups": groups,
                    "adapter_dir": str(outdir),
                    "optimizer_state": args.optimizer_state,
                    "log": args.log,
                }, indent=2), encoding="utf-8")

    policy.save_adapter(str(outdir))

    # One integrated held-out check so the user does not need another smoke cycle.
    holdout_rows=[]
    if args.holdout_eval:
        print("=== HELD-OUT BASE-SYSTEM EVALUATION ===", flush=True)
        policy.model.eval()
        for role in usable_roles:
            hpairs=holdout_by_role.get(role,[])
            if not hpairs:
                continue
            bundle=_choose_bundle(rng,hpairs,args.bundle_size)
            anchor=primary_probe(role)
            obs={"candidate_role":role,"candidate_description":profile.get("role_descriptions",{}).get(role,role),"anchor_probe":anchor}
            for j in range(args.holdout_probes_per_role):
                sample=None
                for _try in range(max(1, args.syntax_retries)):
                    trial=policy.sample_probe(obs,temperature=0.75,top_p=0.90)
                    sample=trial
                    if trial.valid:
                        break
                assert sample is not None
                status="ok"; rb=None
                if not sample.valid:
                    status=f"invalid:{sample.invalid_reason}"
                else:
                    g=gate.check(sample.query)
                    if not g.allowed:
                        status=f"blocked:{g.stage}:{g.category}"
                    else:
                        rb,_=eval_query(role,sample.query,bundle,anchor)
                rec={"phase":"holdout","target_role":role,"query":sample.query,"status":status,"reward":float(rb.total) if rb else None,"reward_breakdown_training_only":rb.to_dict() if rb else None}
                holdout_rows.append(rec); logf.write(json.dumps(rec,ensure_ascii=False)+"\n"); logf.flush()
                if rb:
                    print(f"holdout role={role:<10} R={rb.total:+.3f} spec={rb.specific_separation:+.3f} act={rb.activation_rate:.2f} | {sample.query}",flush=True)
                else:
                    print(f"holdout role={role:<10} {status} | {sample.query}",flush=True)

    logf.close()
    train_valid_rate=total_valid/max(1,total_candidates)
    pos_spec_rate=positive_specific/max(1,total_valid)
    hgood=[r for r in holdout_rows if r["reward_breakdown_training_only"] is not None]
    hpos=[r for r in hgood if r["reward_breakdown_training_only"]["specific_separation"]>0]
    holdout_role_stats={}
    for role in usable_roles:
        rr=[r for r in hgood if r["target_role"]==role]
        specs=[r["reward_breakdown_training_only"]["specific_separation"] for r in rr]
        holdout_role_stats[role]={
            "valid":len(rr),
            "total":sum(1 for r in holdout_rows if r["target_role"]==role),
            "positive":sum(float(x)>0 for x in specs),
            "mean_specific":float(np.mean(specs)) if specs else None,
            "best_specific":float(np.max(specs)) if specs else None,
        }
    role_any_positive=sum(1 for v in holdout_role_stats.values() if v["positive"]>0)
    role_mean_positive=sum(1 for v in holdout_role_stats.values() if v["mean_specific"] is not None and v["mean_specific"]>0)
    summary={
        "train_groups":group_num,"roles":usable_roles,"rounds_per_role":args.rounds_per_role,
        "group_size":args.group_size,"bundle_size":args.bundle_size,"updates":update_idx,
        "train_valid_rate":train_valid_rate,"train_positive_specific_rate":pos_spec_rate,
        "mean_group_reward":float(np.mean(group_rewards)) if group_rewards else 0.0,
        "holdout_valid":len(hgood),"holdout_total":len(holdout_rows),
        "holdout_positive_specific_rate":len(hpos)/max(1,len(hgood)),
        "holdout_role_any_positive_rate":role_any_positive/max(1,len(usable_roles)),
        "holdout_role_mean_positive_rate":role_mean_positive/max(1,len(usable_roles)),
        "holdout_role_stats":holdout_role_stats,
        "detector_reserved_base_systems":len(detector_reserved_specs),
        "elite_probes":{r:[{"query":x["query"],"reward":x["reward"],"breakdown":x["reward_breakdown_training_only"]} for x in xs] for r,xs in elite_records.items()},
        "policy_dir":str(outdir),
        "reward_model_used":False,
        "v65_detector_used_in_reward":False,
    }
    Path(args.summary).write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    print("=== V9-FULL COMPLETE ===")
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
