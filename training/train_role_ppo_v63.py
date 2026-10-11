from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from detective.role_choice_policy_v63 import RoleChoicePolicyV63
from estimator.role_presence_estimator_v63 import RolePresenceEstimatorV63, load_profile
from rl.role_live_env_v63 import RoleAdaptiveLiveEnvV63
from targets.single_router_fanout_mas import SingleRouterFanoutMAS
from targets.single_router_zoo import get_split


@dataclass
class Transition:
    prompt: str
    choices: Tuple[str, ...]
    action_index: int
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


def _episode_metrics(belief, true_roles, candidates):
    truth = set(true_roles)
    brier = sum((belief[r] - (1.0 if r in truth else 0.0)) ** 2 for r in candidates) / len(candidates)
    acc = sum((belief[r] >= 0.5) == (r in truth) for r in candidates) / len(candidates)
    k_mae = abs(sum(belief.values()) - len(truth))
    return brier, acc, k_mae


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--estimator", default="artifacts/role_presence_estimator_v63.joblib")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--rollout-batch", type=int, default=8, help="episodes per PPO update")
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.20)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--output-dir", default="outputs/detective_v6_3_fanout48")
    ap.add_argument("--log", default="artifacts/train_v6_3_fanout48.jsonl")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    specs = get_split("train")
    if not args.execute:
        print("=== V6.3 SINGLE-ROUTER FANOUT PPO PLAN ===")
        print("modes:", sorted({s.execution_mode for s in specs}))
        print("systems:", len(specs), "balanced across K=3..6 and 4 execution modes")
        print("policy action: choose one UNTESTED ROLE; fixed benign probe")
        print("episodes:", args.episodes, "budget:", args.budget, "+ anchor")
        print("rollout batch:", args.rollout_batch, "PPO epochs:", args.ppo_epochs)
        print("reward: 1.0*focus_Brier_improvement + 1.5*global_Brier_improvement + 0.15*K_improvement - 0.01")
        print("PPO: clip=", args.clip, "gamma=", args.gamma, "entropy=", args.entropy_coef)
        print("PLAN ONLY. Re-run with --execute to train.")
        return
    if not Path(args.estimator).exists():
        raise SystemExit(f"missing estimator: {args.estimator}; run bootstrap_role_estimator_v63 first")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    estimator = RolePresenceEstimatorV63.load(args.estimator)
    candidates = tuple(profile["candidate_roles"])
    if tuple(estimator.candidate_roles) != candidates:
        raise SystemExit("estimator candidate roles do not match profile")

    policy = RoleChoicePolicyV63(args.model_name, use_4bit=not args.no_4bit)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logf = Path(args.log).open("w", encoding="utf-8")
    buffer: List[Transition] = []

    def update_policy(transitions: List[Transition]) -> None:
        if not transitions:
            return
        returns = discounted_returns(transitions, args.gamma)
        vals = torch.tensor(returns, dtype=torch.float32, device=policy.device)
        adv = (vals - vals.mean()) / (vals.std(unbiased=False) + 1e-6)
        n = len(transitions)
        policy.model.train()
        for epoch in range(args.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            ratio_vals = []
            loss_vals = []
            entropy_vals = []
            # Memory-safe gradient accumulation: backward one transition at a time, one optimizer
            # step after the full rollout. This avoids retaining N Qwen graphs on the 1080 Ti.
            for t, a in zip(transitions, adv):
                new_logp, entropy = policy.action_logprob_and_entropy(t.prompt, len(t.choices), t.action_index)
                old = torch.tensor(t.old_logp, dtype=new_logp.dtype, device=new_logp.device)
                ratio = torch.exp(torch.clamp(new_logp - old, -5.0, 5.0))
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                policy_loss = -torch.minimum(ratio * a, clipped * a)
                loss_i = policy_loss - args.entropy_coef * entropy
                (loss_i / n).backward()
                ratio_vals.append(float(ratio.detach().cpu()))
                loss_vals.append(float(loss_i.detach().cpu()))
                entropy_vals.append(float(entropy.detach().cpu()))
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1.0)
            optimizer.step()
            print(
                f"ppo_epoch={epoch+1} loss={sum(loss_vals)/n:+.4f} "
                f"ratio_mean={sum(ratio_vals)/n:.3f} entropy={sum(entropy_vals)/n:.3f} "
                f"grad_norm={float(grad_norm):.3f}"
            )
        policy.model.eval()

    # Balanced schedule: every 16 episodes sees every topology/mode exactly once.
    schedule: List = []
    rng = random.Random(args.seed)
    while len(schedule) < args.episodes:
        cycle = list(specs)
        rng.shuffle(cycle)
        schedule.extend(cycle)
    schedule = schedule[: args.episodes]

    for ep, spec in enumerate(schedule, 1):
        target = SingleRouterFanoutMAS(spec)
        env = RoleAdaptiveLiveEnvV63(target, estimator, profile, max_steps=args.budget, use_anchor=True)
        obs = env.reset()
        ep_reward = 0.0

        for step in range(args.budget):
            before = dict(obs["role_belief"])
            sample = policy.sample(obs, temperature=args.temperature)
            next_obs, reward, done, info = env.step(sample.focus_role, sample.query)
            buffer.append(Transition(sample.prompt, sample.choices, sample.action_index, sample.old_logprob, reward, ep))
            ep_reward += reward
            row = {
                "episode": ep,
                "step": step + 1,
                "system_id": spec.system_id,
                "execution_mode": spec.execution_mode,
                "k_true_training_only": spec.k_true,
                "focus_role": sample.focus_role,
                "query": sample.query,
                "available_roles_before": list(sample.choices),
                "reward": reward,
                "r_focus": info["r_focus"],
                "r_global": info["r_global"],
                "r_k": info["r_k"],
                "focus_true_training_only": info["focus_true"],
                "focus_activated_training_only": info["focus_activated"],
                "selected_roles_training_only": info["selected_roles"],
                "role_belief_before": before,
                "role_belief_after": next_obs["role_belief"],
                "expected_k_after": next_obs["expected_k"],
                "true_roles_training_only": info["true_roles"],
            }
            logf.write(json.dumps(row, ensure_ascii=False) + "\n")
            logf.flush()
            obs = next_obs
            if done:
                break

        brier, acc, k_mae = _episode_metrics(obs["role_belief"], spec.worker_roles, candidates)
        print(
            f"episode={ep}/{args.episodes} system={spec.system_id} mode={spec.execution_mode} "
            f"reward={ep_reward:+.3f} brier={brier:.3f} roleAcc={acc:.3f} KMAE={k_mae:.2f} "
            f"tested={obs['tested_roles']}"
        )
        if ep % args.rollout_batch == 0 or ep == args.episodes:
            update_policy(buffer)
            buffer.clear()
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            policy.save_adapter(args.output_dir)
            print("checkpoint:", args.output_dir)

    logf.close()
    print("training complete")


if __name__ == "__main__":
    main()
