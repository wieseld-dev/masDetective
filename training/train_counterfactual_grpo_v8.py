from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from estimator.candidate_presence_contrastive_v64 import load_profile
from probes.role_probe_bank import primary_probe
from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_zoo_v65 import get_v65_split
from v8.counterfactual import build_counterfactual_pairs, get_k, get_mode, get_system_id, split_base_specs
from v8.embedding import FrozenOutputEmbedderV8
from v8.execution import execute_cached
from v8.gate import BenignGateV8
from v8.policy import GroupProbePolicyV8
from v8.reward import CounterfactualRewardV8, group_relative_advantages


@dataclass
class GRPOSampleV8:
    prompt: str
    completion: str
    query: str
    old_logp: float
    ref_logp: float
    advantage: float
    reward: float
    group_id: str


def _kl_k3(new_lp, ref_lp, torch):
    # Non-negative sample KL estimator: exp(log r_ref/pol) - log r_ref/pol - 1.
    delta = ref_lp - new_lp
    return torch.exp(torch.clamp(delta, -6.0, 6.0)) - delta - 1.0


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="V8 counterfactual matched-pair benign-probing GRPO")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--states", type=int, default=64, help="number of black-box seed states")
    ap.add_argument("--group-size", type=int, default=4, help="candidate questions generated from the same state")
    ap.add_argument("--groups-per-update", type=int, default=4)
    ap.add_argument("--grpo-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--clip", type=float, default=0.20)
    ap.add_argument("--kl-beta", type=float, default=0.015)
    ap.add_argument("--temperature", type=float, default=0.90)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--include-sham", action="store_true")
    ap.add_argument("--gate-mode", choices=["hard", "hybrid"], default="hybrid")
    ap.add_argument("--guard-model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=808)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--resume-adapter", default="")
    ap.add_argument("--output-dir", default="artifacts/v8_grpo_policy")
    ap.add_argument("--log", default="artifacts/v8_grpo_train.jsonl")
    ap.add_argument("--response-public-cache", default="artifacts/v8_response_public.jsonl")
    ap.add_argument("--response-truth-cache", default="artifacts/v8_response_truth.jsonl")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    roles = tuple(profile["candidate_roles"])
    all_train_specs = list(get_v65_split("train"))
    rl_specs, detector_specs = split_base_specs(all_train_specs, seed=args.seed)
    pairs = build_counterfactual_pairs(rl_specs, roles, include_sham=args.include_sham)
    if not pairs:
        raise SystemExit("No eligible counterfactual pairs could be constructed")

    branches = 3 if args.include_sham else 2
    candidate_questions = args.states * args.group_size
    max_live_interactions = candidate_questions * branches + args.states * branches
    max_calls_per_interaction = max(int(getattr(s, "estimated_calls_per_query", 5)) for s in all_train_specs)
    print("=== V8 COUNTERFACTUAL GRPO PLAN ===")
    print("RL train base systems:", len(rl_specs), "detector-reserved base systems:", len(detector_specs))
    print("counterfactual pairs:", len(pairs), "constant K/mode via role SWAP (not deletion)")
    print("states:", args.states, "group size:", args.group_size, "groups/update:", args.groups_per_update)
    print("branches/query:", branches, "sham control:", args.include_sham)
    print("generated candidate questions:", candidate_questions)
    print("upper-bound live MAS interactions incl. seed baselines:", max_live_interactions)
    print("rough upper-bound target-model API calls:", max_live_interactions * max_calls_per_interaction)
    print("rough upper-bound semantic-guard calls:", candidate_questions if args.gate_mode == "hybrid" else 0)
    print("reward uses TRAINING-ONLY selected_roles + matched external-output separability; no V6.5 detector in reward")
    print("policy observation: candidate description + ONE fixed seed query/output only")
    print("policy forbidden: truth, K, mode, system id, selected roles, routing, traces, counterfactual branch")
    print("Stage 1: exactly one generated follow-up; STOP disabled")
    if not args.execute:
        print("PLAN ONLY. Re-run with --execute after reviewing cost.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gate = BenignGateV8(mode=args.gate_mode, guard_model=args.guard_model)
    embedder = FrozenOutputEmbedderV8(device="cpu")
    grader = CounterfactualRewardV8()
    cache = ResponseCacheV7(args.response_public_cache, args.response_truth_cache)
    policy = GroupProbePolicyV8(
        args.model_name,
        adapter_path=(args.resume_adapter or None),
        trainable=True,
        use_4bit=not args.no_4bit,
        max_seq_length=1408,
    )
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logf = Path(args.log).open("a", encoding="utf-8")

    # Balanced black-box source states: the detective sees either the PRESENT or ABSENT seed output.
    states = []
    for p in pairs:
        states.append((p, "plus")); states.append((p, "minus"))
    rng.shuffle(states)

    update_buffer: list[GRPOSampleV8] = []
    update_idx = 0
    rolling_rewards: list[float] = []
    valid_rates: list[float] = []

    def grpo_update(buf: list[GRPOSampleV8]):
        nonlocal update_idx
        if not buf:
            return
        update_idx += 1
        policy.model.train()
        for epoch in range(args.grpo_epochs):
            optimizer.zero_grad(set_to_none=True)
            n = max(len(buf), 1)
            loss_sum = ratio_sum = kl_sum = 0.0
            # Memory-safe gradient accumulation: one LM graph at a time, one optimizer step at end.
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
                loss_i = pg + args.kl_beta * kl
                (loss_i / n).backward()
                loss_sum += float(loss_i.detach().cpu())
                ratio_sum += float(ratio.detach().cpu())
                kl_sum += float(kl.detach().cpu())
                del new_lp, old_lp, ratio, adv, unclipped, clipped, pg, kl, loss_i
            torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"update={update_idx} epoch={epoch+1} loss={loss_sum/n:+.4f} ratio={ratio_sum/n:.3f} kl={kl_sum/n:.4f}", flush=True)
        policy.model.eval()
        policy.save_adapter(str(outdir))

    for state_i in range(args.states):
        pair, source_branch = states[state_i % len(states)]
        if state_i and state_i % len(states) == 0:
            rng.shuffle(states)
        seed_q = primary_probe(pair.target_role)
        plus_seed = execute_cached(pair.plus_spec, seed_q, cache)
        minus_seed = execute_cached(pair.minus_spec, seed_q, cache)
        sham_seed = execute_cached(pair.sham_spec, seed_q, cache) if pair.sham_spec is not None and args.include_sham else None
        source_seed = plus_seed if source_branch == "plus" else minus_seed
        obs = {
            "candidate_role": pair.target_role,
            "candidate_description": profile.get("role_descriptions", {}).get(pair.target_role, pair.target_role),
            "seed_query": seed_q,
            "seed_output": source_seed.output,
        }
        plus_seed_emb = embedder.encode(plus_seed.output)
        minus_seed_emb = embedder.encode(minus_seed.output)
        sham_seed_emb = embedder.encode(sham_seed.output) if sham_seed is not None else None

        candidates = []
        seen_queries: set[str] = set()
        rewards = []
        for gi in range(args.group_size):
            sample = policy.sample_question(obs, temperature=args.temperature, top_p=args.top_p, deterministic=False)
            if not sample.valid:
                rb = grader.invalid()
                rewards.append(rb.total)
                candidates.append((sample, rb, None, None, None, "invalid"))
                continue
            qkey = sample.query.lower().strip()
            if qkey in seen_queries:
                rb = grader.duplicate()
                rewards.append(rb.total)
                candidates.append((sample, rb, None, None, None, "duplicate"))
                continue
            seen_queries.add(qkey)
            gate_result = gate.check(sample.query)
            if not gate_result.allowed:
                rb = grader.unsafe()
                rewards.append(rb.total)
                candidates.append((sample, rb, None, None, None, f"blocked:{gate_result.stage}:{gate_result.category}"))
                continue

            plus_res = execute_cached(pair.plus_spec, sample.query, cache)
            minus_res = execute_cached(pair.minus_spec, sample.query, cache)
            sham_res = execute_cached(pair.sham_spec, sample.query, cache) if pair.sham_spec is not None and args.include_sham else None
            rb = grader.score(
                target_selected_in_plus=pair.target_role in set(plus_res.selected_roles_training_only),
                plus_seed_emb=plus_seed_emb,
                minus_seed_emb=minus_seed_emb,
                plus_query_emb=embedder.encode(plus_res.output),
                minus_query_emb=embedder.encode(minus_res.output),
                sham_seed_emb=sham_seed_emb,
                sham_query_emb=embedder.encode(sham_res.output) if sham_res is not None else None,
            )
            rewards.append(rb.total)
            candidates.append((sample, rb, plus_res, minus_res, sham_res, "ok"))

        adv = group_relative_advantages(rewards)
        group_id = f"s{state_i+1:04d}_{pair.pair_id}_{source_branch}"
        valid_count = 0
        for gi, ((sample, rb, plus_res, minus_res, sham_res, status), a) in enumerate(zip(candidates, adv), 1):
            if status == "ok": valid_count += 1
            if sample.completion.strip():
                update_buffer.append(GRPOSampleV8(
                    prompt=sample.prompt, completion=sample.completion, query=sample.query,
                    old_logp=sample.old_mean_logprob, ref_logp=sample.ref_mean_logprob,
                    advantage=float(a), reward=float(rb.total), group_id=group_id,
                ))
            logf.write(json.dumps({
                "state": state_i + 1, "group_id": group_id, "candidate_index": gi,
                "pair_id_training_only": pair.pair_id,
                "source_branch_training_only": source_branch,
                "target_role": pair.target_role,
                "plus_roles_training_only": list(getattr(pair.plus_spec, "worker_roles", [])),
                "minus_roles_training_only": list(getattr(pair.minus_spec, "worker_roles", [])),
                "mode_training_only": get_mode(pair.plus_spec), "K_training_only": get_k(pair.plus_spec),
                "seed_query": seed_q, "seed_output_policy_visible": source_seed.output,
                "raw_completion": sample.raw_completion, "train_completion": sample.completion, "query": sample.query,
                "status": status, "reward": float(rb.total), "advantage": float(a),
                "reward_breakdown_training_only": rb.to_dict(),
                "plus_selected_training_only": list(plus_res.selected_roles_training_only) if plus_res else [],
                "minus_selected_training_only": list(minus_res.selected_roles_training_only) if minus_res else [],
                "plus_output_training_only": plus_res.output if plus_res else "",
                "minus_output_training_only": minus_res.output if minus_res else "",
                "sham_output_training_only": sham_res.output if sham_res else "",
            }, ensure_ascii=False) + "\n"); logf.flush()

        mean_r = float(np.mean(rewards)); rolling_rewards.append(mean_r); valid_rates.append(valid_count / args.group_size)
        print(
            f"state={state_i+1}/{args.states} role={pair.target_role:<10} source={source_branch:<5} "
            f"Rmean={mean_r:+.3f} Rbest={max(rewards):+.3f} valid={valid_count}/{args.group_size} "
            f"recentR={np.mean(rolling_rewards[-16:]):+.3f}", flush=True,
        )
        if (state_i + 1) % args.groups_per_update == 0 or state_i + 1 == args.states:
            grpo_update(update_buffer)
            update_buffer.clear()

    logf.close(); policy.save_adapter(str(outdir))
    summary = {
        "states": args.states, "group_size": args.group_size, "updates": update_idx,
        "mean_group_reward": float(np.mean(rolling_rewards)),
        "mean_valid_rate": float(np.mean(valid_rates)),
        "include_sham": bool(args.include_sham),
        "rl_train_base_systems": len(rl_specs), "detector_reserved_base_systems": len(detector_specs),
        "policy_dir": str(outdir),
        "reward": "training-only target activation + incremental matched external separability + optional sham specificity",
        "v65_detector_used_in_reward": False,
    }
    (outdir / "v8_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TRAINING COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
