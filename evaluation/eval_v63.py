from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from detective.role_choice_policy_v63 import RoleChoicePolicyV63, available_roles, fixed_query_for_role
from estimator.role_presence_estimator_v63 import RolePresenceEstimatorV63, load_profile
from rl.role_live_env_v63 import RoleAdaptiveLiveEnvV63
from targets.single_router_fanout_mas import SingleRouterFanoutMAS
from targets.single_router_zoo import get_split


def metrics(belief, true_roles, roles):
    truth = set(true_roles)
    brier = sum((belief[r] - (1.0 if r in truth else 0.0)) ** 2 for r in roles) / len(roles)
    acc = sum((belief[r] >= 0.5) == (r in truth) for r in roles) / len(roles)
    k_mae = abs(sum(belief.values()) - len(truth))
    return brier, acc, k_mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--estimator", default="artifacts/role_presence_estimator_v63.joblib")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--policy", choices=("qwen", "uncertainty", "random"), default="qwen")
    ap.add_argument("--split", choices=("val", "test"), default="val")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--output", default="artifacts/eval_v63.jsonl")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    specs = get_split(args.split)
    print("=== V6.3 HELD-OUT EVALUATION PLAN ===")
    print("policy:", args.policy, "adapter:", args.adapter)
    print("systems:", len(specs), "split:", args.split, "budget:", args.budget, "+ anchor")
    if not args.execute:
        print("PLAN ONLY")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    profile = load_profile(args.profile)
    roles = tuple(profile["candidate_roles"])
    est = RolePresenceEstimatorV63.load(args.estimator)
    rng = random.Random(args.seed)
    policy = None
    if args.policy == "qwen":
        policy = RoleChoicePolicyV63(adapter_path=args.adapter, train_adapter=False) if args.adapter else RoleChoicePolicyV63(train_adapter=False)

    rows = []
    totals = []
    for spec in specs:
        env = RoleAdaptiveLiveEnvV63(SingleRouterFanoutMAS(spec), est, profile, max_steps=args.budget, use_anchor=True)
        obs = env.reset()
        for _ in range(args.budget):
            choices = available_roles(obs)
            if args.policy == "qwen":
                sample = policy.sample(obs, temperature=0.0)
                role, query = sample.focus_role, sample.query
            elif args.policy == "uncertainty":
                role = max(choices, key=lambda r: 1.0 - 2.0 * abs(float(obs["role_belief"][r]) - 0.5))
                query = fixed_query_for_role(role)
            else:
                role = rng.choice(list(choices))
                query = fixed_query_for_role(role)
            obs, reward, done, info = env.step(role, query)
            if done:
                break
        brier, acc, k_mae = metrics(obs["role_belief"], spec.worker_roles, roles)
        totals.append((brier, acc, k_mae))
        rows.append({"system_id": spec.system_id, "mode": spec.execution_mode, "brier": brier, "role_accuracy": acc, "k_mae": k_mae, "final_belief": obs["role_belief"], "true_roles_training_only": list(spec.worker_roles)})
        print(f"{spec.system_id}: brier={brier:.3f} roleAcc={acc:.3f} KMAE={k_mae:.2f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = max(len(totals), 1)
    print("MEAN brier=%.3f roleAcc=%.3f KMAE=%.3f" % (
        sum(x[0] for x in totals)/n, sum(x[1] for x in totals)/n, sum(x[2] for x in totals)/n
    ))


if __name__ == "__main__":
    main()
