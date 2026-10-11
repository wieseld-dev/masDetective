from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from detective.role_policy_model import RoleDetectivePolicy
from estimator.role_presence_estimator import RolePresenceEstimator, load_profile
from rl.role_live_env import RoleAdaptiveLiveEnv
from targets.hierarchical_mas import HierarchicalWhiteBoxMAS
from targets.topology_zoo import get_split


@dataclass
class Transition:
    prompt: str
    action: str
    old_logp: float
    reward: float
    episode_id: int


def discounted_returns(transitions: List[Transition], gamma: float) -> List[float]:
    out = [0.0] * len(transitions)
    by_episode: Dict[int, List[int]] = {}
    for i, t in enumerate(transitions):
        by_episode.setdefault(t.episode_id, []).append(i)
    for idxs in by_episode.values():
        running = 0.0
        for i in reversed(idxs):
            running = transitions[i].reward + gamma * running
            out[i] = running
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--estimator", default="artifacts/role_presence_estimator.joblib")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--budget", type=int, default=3, help="adaptive probes after the mandatory anchor")
    ap.add_argument("--rollout-batch", type=int, default=2)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.20)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--output-dir", default="outputs/detective_v6_role_run20")
    ap.add_argument("--log", default="artifacts/train_v6_role_run20.jsonl")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    if not Path(args.estimator).exists():
        raise SystemExit(f"missing estimator: {args.estimator}; run bootstrap_role_estimator first")
    profile = load_profile(args.profile)
    if not args.execute:
        print("=== V6 ROLE-ADAPTIVE PPO PLAN ===")
        print("application:", profile.get("app_name"))
        print("candidate roles:", profile["candidate_roles"])
        print("episodes:", args.episodes, "adaptive budget:", args.budget, "+ 1 mandatory anchor")
        print("max live MAS interactions:", args.episodes * (args.budget + 1))
        print("PLAN ONLY. Re-run with --execute to train.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    estimator = RolePresenceEstimator.load(args.estimator)
    if tuple(estimator.candidate_roles) != tuple(profile["candidate_roles"]):
        raise SystemExit("estimator candidate roles do not match profile")

    policy = RoleDetectivePolicy(args.model_name, use_4bit=not args.no_4bit)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr)
    train_specs = get_split("train")
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logf = Path(args.log).open("w", encoding="utf-8")
    batch: List[Transition] = []

    def update_policy(buffer: List[Transition]) -> None:
        if not buffer:
            return
        returns = discounted_returns(buffer, args.gamma)
        vals = torch.tensor(returns, dtype=torch.float32, device=policy.device)
        adv = vals - vals.mean()
        adv = adv / (adv.std(unbiased=False) + 1e-6)
        policy.model.train()
        for epoch in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            ratios = []
            for t, a in zip(buffer, adv):
                new_logp = policy.mean_logprob(t.prompt, t.action)
                old = torch.tensor(t.old_logp, dtype=new_logp.dtype, device=new_logp.device)
                log_ratio = torch.clamp(new_logp - old, -5.0, 5.0)
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                losses.append(-torch.minimum(ratio * a, clipped * a))
                ratios.append(float(ratio.detach().cpu()))
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1.0)
            optimizer.step()
            print(f"ppo_epoch={epoch+1} loss={float(loss.detach().cpu()):+.4f} ratio_mean={sum(ratios)/len(ratios):.3f}")
        policy.model.eval()

    for ep in range(1, args.episodes + 1):
        spec = random.choice(train_specs)
        target = HierarchicalWhiteBoxMAS(spec)
        env = RoleAdaptiveLiveEnv(target, estimator, profile, max_steps=args.budget, use_anchor=True)
        obs = env.reset()
        ep_reward = 0.0

        for step in range(args.budget):
            before_belief = dict(obs["role_belief"])
            sample = policy.sample(obs)
            next_obs, reward, done, info = env.step(sample.focus_role, sample.query)
            batch.append(Transition(sample.prompt, sample.action_text, sample.old_mean_logprob, reward, ep))
            ep_reward += reward
            row = {
                "episode": ep,
                "step": step + 1,
                "system_id": spec.system_id,
                "family": spec.family,
                "depth": spec.depth,
                "focus_role": sample.focus_role,
                "query": sample.query,
                "action_text": sample.action_text,
                "reward": reward,
                "blocked": info.get("blocked", 0.0),
                "block_reason": info.get("block_reason", "ok"),
                "r_role_raw": info.get("r_role_raw"),
                "r_role": info.get("r_role"),
                "r_k_raw": info.get("r_k_raw"),
                "r_k": info.get("r_k"),
                "worker_coverage": info.get("worker_coverage"),
                "role_belief_before": before_belief,
                "role_belief_after": next_obs["role_belief"],
                "expected_k_after": next_obs["expected_k"],
                "true_roles_training_only": info.get("true_roles"),
                "k_true_training_only": info.get("k_true"),
            }
            logf.write(json.dumps(row, ensure_ascii=False) + "\n")
            logf.flush()
            obs = next_obs
            if done:
                break

        print(f"episode={ep}/{args.episodes} system={spec.system_id} reward={ep_reward:+.3f} expectedK={obs['expected_k']:.2f}")
        if ep % args.rollout_batch == 0 or ep == args.episodes:
            update_policy(batch)
            batch.clear()
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            policy.save_adapter(args.output_dir)
            print("checkpoint:", args.output_dir)

    logf.close()
    print("training complete")


if __name__ == "__main__":
    main()
