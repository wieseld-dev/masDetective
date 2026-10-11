from __future__ import annotations

import argparse
import json
import os
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib

from detective.benign_gate_v7 import CombinedBenignGateV7, SemanticBenignGuardV7
from detective.generative_probe_policy_v7 import GenerativeProbePolicyV7
from estimator.candidate_presence_contrastive_v64 import load_profile
from rl.generative_probe_env_v7 import GenerativeProbeEnvV7, ResponseCacheV7, SeedStoreV7
from targets.single_router_fanout_mas import SingleRouterFanoutMAS
from targets.single_router_zoo_v65 import get_v65_split


@dataclass
class TransitionV7:
    prompt: str
    completion: str
    old_logp: float
    reward: float
    episode_id: int


def discounted_returns(transitions: Sequence[TransitionV7], gamma: float) -> List[float]:
    out = [0.0] * len(transitions)
    by_ep: Dict[int, List[int]] = {}
    for i, t in enumerate(transitions):
        by_ep.setdefault(int(t.episode_id), []).append(i)
    for idxs in by_ep.values():
        running = 0.0
        for i in reversed(idxs):
            running = float(transitions[i].reward) + float(gamma) * running
            out[i] = running
    return out


def build_balanced_pairs(specs, roles, seed: int) -> List[Tuple[object, str]]:
    rng = random.Random(seed)
    present, absent = [], []
    for spec in specs:
        sroles = set(spec.worker_roles)
        for role in roles:
            (present if role in sroles else absent).append((spec, role))
    rng.shuffle(present); rng.shuffle(absent)
    pairs = []
    i = j = 0
    while i < len(present) or j < len(absent):
        if i < len(present): pairs.append(present[i]); i += 1
        if j < len(absent): pairs.append(absent[j]); j += 1
    return pairs


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="V7 generative benign follow-up PPO")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--estimator", default="artifacts/candidate_presence_v65.joblib")
    ap.add_argument("--seed-public", default="artifacts/v65_train_public.jsonl")
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--max-followups", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=6)
    ap.add_argument("--rollout-episodes", type=int, default=8)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.20)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--temperature-start", type=float, default=0.95)
    ap.add_argument("--temperature-end", type=float, default=0.65)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1701)
    ap.add_argument("--gate-mode", choices=["hard", "hybrid"], default="hybrid")
    ap.add_argument("--guard-model", default="gpt-4o-mini")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--resume-adapter", default="", help="optional LoRA adapter directory to continue from")
    ap.add_argument("--start-episode", type=int, default=1, help="episode number label when resuming")
    ap.add_argument("--output-dir", default="artifacts/v7_generative_policy")
    ap.add_argument("--log", default="artifacts/v7_train.jsonl")
    ap.add_argument("--response-public-cache", default="artifacts/v7_train_response_public.jsonl")
    ap.add_argument("--response-truth-cache", default="artifacts/v7_train_response_truth.jsonl")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    roles = tuple(profile["candidate_roles"])
    specs = get_v65_split("train")
    max_external = int(args.episodes) * int(args.max_followups)
    max_target_calls = max_external * max(s.estimated_calls_per_query for s in specs)
    max_guard_calls = max_external if args.gate_mode == "hybrid" else 0
    print("=== V7 GENERATIVE BENIGN-PROBING PPO PLAN ===")
    print("fixed seed: V6.5 primary_probe(role), replayed from already-paid V6.5 train data")
    print("policy action: FREE-TEXT benign follow-up question OR STOP")
    print("training systems:", len(specs), "unique compositions:", len({tuple(sorted(s.worker_roles)) for s in specs}))
    print("episodes:", args.episodes, "max followups/episode:", args.max_followups, "max attempts:", args.max_attempts)
    print("gate mode:", args.gate_mode, "guard model:", args.guard_model)
    print("upper-bound NEW external follow-up interactions:", max_external)
    print("rough upper-bound target-model API calls:", max_target_calls)
    print("rough upper-bound semantic-guard calls:", max_guard_calls)
    print("policy observation excludes: truth, K, mode, router choice, selected roles, traces, system id")
    print("truth/selected roles are scalar-reward/evaluation only")
    if not args.execute:
        print("PLAN ONLY. Re-run with --execute to train.")
        return

    if not Path(args.estimator).exists():
        raise SystemExit(f"missing {args.estimator}; run V6.5 evaluation first")
    if not Path(args.seed_public).exists():
        raise SystemExit(f"missing {args.seed_public}; V7 reuses the V6.5 seed outputs")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    random.seed(args.seed); torch.manual_seed(args.seed)
    estimator = joblib.load(args.estimator)
    if tuple(estimator.candidate_roles) != roles:
        raise SystemExit("V6.5 estimator candidate roles do not match profile")
    # Keep the black-box detector on CPU to leave the 11GB GPU to Qwen + LoRA.
    if hasattr(estimator, "embedder"):
        estimator.embedder.device = "cpu"
    seed_store = SeedStoreV7.from_jsonl(args.seed_public)
    semantic = SemanticBenignGuardV7(model=args.guard_model) if args.gate_mode == "hybrid" else None
    gate = CombinedBenignGateV7(mode=args.gate_mode, semantic_guard=semantic)
    cache = ResponseCacheV7(args.response_public_cache, args.response_truth_cache)
    policy = GenerativeProbePolicyV7(
        args.model_name, adapter_path=(args.resume_adapter or None),
        use_4bit=not args.no_4bit, trainable=True,
    )
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logf = Path(args.log).open("a", encoding="utf-8")
    pairs = build_balanced_pairs(specs, roles, args.seed)
    rng = random.Random(args.seed)
    transitions: List[TransitionV7] = []
    recent_rewards = deque(maxlen=32)
    update_idx = 0

    def ppo_update(buffer: List[TransitionV7]) -> None:
        nonlocal update_idx
        if not buffer:
            return

        update_idx += 1
        returns = discounted_returns(buffer, args.gamma)
        adv = torch.tensor(returns, dtype=torch.float32, device=policy.device)

        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
        else:
            adv = adv * 0.0

        policy.model.train()

        for epoch in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)

            ratio_sum = 0.0
            loss_sum = 0.0
            n = max(len(buffer), 1)

            # IMPORTANT: backward one transition at a time.
            # This avoids retaining every LM computation graph on the 11GB GPU.
            for tr, a in zip(buffer, adv):
                new_lp = policy.mean_logprob(tr.prompt, tr.completion)

                old_lp = torch.tensor(
                    tr.old_logp,
                    dtype=new_lp.dtype,
                    device=new_lp.device,
                )

                ratio = torch.exp(torch.clamp(new_lp - old_lp, -4.0, 4.0))
                clipped = torch.clamp(
                    ratio,
                    1.0 - args.clip,
                    1.0 + args.clip,
                )

                loss_i = -torch.minimum(ratio * a, clipped * a)
                (loss_i / n).backward()

                ratio_sum += float(ratio.detach().cpu())
                loss_sum += float(loss_i.detach().cpu())

                del new_lp, old_lp, ratio, clipped, loss_i

            torch.nn.utils.clip_grad_norm_(
                policy.trainable_parameters(), 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                f"update={update_idx} epoch={epoch+1} "
                f"loss={loss_sum/n:+.4f} ratio={ratio_sum/n:.3f}",
                flush=True,
            )

        policy.model.eval()
        policy.save_adapter(args.output_dir)

    for local_ep in range(1, args.episodes + 1):
        ep = int(args.start_episode) + local_ep - 1
        if (local_ep - 1) % len(pairs) == 0 and local_ep > 1:
            rng.shuffle(pairs)
        spec, role = pairs[(local_ep - 1) % len(pairs)]
        target = SingleRouterFanoutMAS(spec)
        env = GenerativeProbeEnvV7(
            target=target, estimator=estimator, profile=profile, candidate_role=role,
            seed_store=seed_store, gate=gate, response_cache=cache,
            max_followups=args.max_followups, max_attempts=args.max_attempts,
        )
        obs = env.observation()
        ep_reward = 0.0
        decision = 0
        while not env.done and decision < args.max_attempts:
            decision += 1
            frac = (local_ep - 1) / max(args.episodes - 1, 1)
            temp = args.temperature_start + frac * (args.temperature_end - args.temperature_start)
            sample = policy.sample(obs, temperature=temp, top_p=args.top_p, deterministic=False)
            parsed = sample.parsed
            next_obs, reward, done, info = env.step(parsed.action, parsed.query)
            transitions.append(TransitionV7(sample.prompt, sample.completion, sample.old_mean_logprob, float(reward), ep))
            ep_reward += float(reward)
            row = {
                "episode": ep, "decision": decision,
                # Training-only metadata below is logged for analysis but never appears in policy observation.
                "system_id_training_only": spec.system_id,
                "candidate_role": role,
                "truth_present_training_only": bool(role in set(spec.worker_roles)),
                "mode_training_only": spec.execution_mode,
                "completion": sample.completion,
                "parsed_action": parsed.action,
                "query": parsed.query,
                "allowed": info.allowed,
                "gate_reason": info.gate_reason,
                "gate_stage": info.gate_stage,
                "external_query_sent": info.external_query_sent,
                "cache_hit": info.cache_hit,
                "belief_before": info.belief_before,
                "belief_after": info.belief_after,
                "raw_response_probability": info.raw_response_probability,
                "selected_target_training_only": info.selected_target_training_only,
                "reward": float(reward),
                "reward_breakdown": asdict(info.reward),
            }
            logf.write(json.dumps(row, ensure_ascii=False) + "\n"); logf.flush()
            obs = next_obs
            if done:
                break
        recent_rewards.append(ep_reward)
        print(
            f"ep={local_ep}/{args.episodes} (global={ep}) role={role:<10} present={int(env.truth_present)} "
            f"reward={ep_reward:+.3f} p={env.belief:.3f} q={env.external_followups} attempts={env.attempts} "
            f"recentR={sum(recent_rewards)/len(recent_rewards):+.3f}", flush=True,
        )
        if local_ep % args.rollout_episodes == 0 or local_ep == args.episodes:
            ppo_update(transitions)
            transitions.clear()

    logf.close()
    policy.save_adapter(args.output_dir)
    summary = {
        "episodes": args.episodes, "max_followups": args.max_followups, "gate_mode": args.gate_mode,
        "final_recent_reward_mean": float(sum(recent_rewards) / max(len(recent_rewards), 1)),
        "policy_dir": args.output_dir,
        "public_response_cache": args.response_public_cache,
        "truth_response_cache_training_only": args.response_truth_cache,
    }
    Path(args.output_dir, "v7_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("training complete; adapter:", args.output_dir)


if __name__ == "__main__":
    main()
