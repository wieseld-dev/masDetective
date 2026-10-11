from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import requests


ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
API = "https://api.typesafe.ai/v1/systemone"


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def is_chosen(v):
    if v is True or v == 1:
        return True
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return False


def load_round1_groups():
    groups = []

    for resp_path in sorted(ROOT.rglob("responses_train_round_01.jsonl")):
        parent = resp_path.parent
        state_path = parent / "states_train_round_01.jsonl"

        if not state_path.exists():
            continue

        states = {}

        for x in read_jsonl(state_path):
            sid = x.get("session_id") or x.get("system_id")
            if sid:
                states[sid] = x

        by_sid = defaultdict(list)

        for x in read_jsonl(resp_path):
            sid = x.get("system_id") or x.get("session_id")
            if sid:
                by_sid[sid].append(x)

        for sid, rows in by_sid.items():
            if sid not in states:
                continue

            # Keep normal K=6/8 candidate pools only.
            if len(rows) not in {6, 8}:
                continue

            chosen = [r for r in rows if is_chosen(
                r.get("chosen_for_continuation")
            )]

            if len(chosen) != 1:
                continue

            # Must have oracle outcome for every candidate.
            if any(
                r.get("selected_roles_training_only") is None
                for r in rows
            ):
                continue

            groups.append({
                "parent": str(parent),
                "sid": sid,
                "state": states[sid],
                "rows": sorted(
                    rows,
                    key=lambda r: int(r.get("candidate_index", 0))
                ),
            })

    return groups


def compact_belief(state):
    role_probs = state.get("role_probs") or {}
    k_probs = state.get("k_probs") or {}

    # Show JEV the uncertainty, but absolutely NO hidden truth.
    roles = sorted(
        role_probs.items(),
        key=lambda kv: abs(float(kv[1]) - 0.5)
    )

    return {
        "role_probabilities": {
            k: round(float(v), 4)
            for k, v in roles
        },
        "k_probabilities": {
            str(k): round(float(v), 4)
            for k, v in k_probs.items()
        },
    }


def call_jev(state, rows):
    criteria = {}

    for r in rows:
        idx = int(r["candidate_index"])
        criteria[f"p{idx}"] = r.get("query", "").strip()

    payload = {
        "model": "jev-latest",
        "state": {
            "task": (
                "We are performing benign black-box active diagnosis of a "
                "multi-agent application. We need to choose ONE ordinary-user "
                "probe to execute next."
            ),
            "current_public_belief": compact_belief(state),
            "candidate_probes": criteria,
        },
        "questions": {
            "best_probe": {
                "type": "choice",
                "instructions": (
                    "Choose the candidate probe expected to provide the most "
                    "useful NEW evidence about which specialist capabilities "
                    "are active. Prefer a probe that discriminates between "
                    "currently plausible role configurations, targets uncertain "
                    "roles, and is non-redundant. Prefer concrete diagnostic "
                    "tasks over vague or generic questions."
                ),
                "criteria": criteria,
            }
        },
    }

    r = requests.post(
        API,
        headers={
            "Authorization":
                f"Bearer {os.environ['TYPESAFE_API_KEY']}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )

    if not r.ok:
        raise RuntimeError(
            f"JEV HTTP {r.status_code}: {r.text[:1000]}"
        )

    data = r.json()
    ans = data["answers"]["best_probe"]

    return {
        "choice": ans["choice"],
        "confidence": ans.get("confidence"),
        "probabilities": ans.get("probabilities", {}),
        "usage": data.get("usage", {}),
    }


def oracle_reward(row):
    # Round 1 => nothing has been seen yet.
    return len(set(
        row.get("selected_roles_training_only") or []
    ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if "TYPESAFE_API_KEY" not in os.environ:
        raise RuntimeError(
            "TYPESAFE_API_KEY missing. Run: "
            "source ~/.config/typesafe/env"
        )

    groups = load_round1_groups()

    def reward_spread(g):
        rewards = [
            oracle_reward(r)
            for r in g["rows"]
        ]
        return max(rewards) - min(rewards)

    all_groups = groups

    # Keep only real selector decision points:
    # at least one candidate must be better than another
    # according to the held-out oracle outcome.
    groups = [
        g for g in groups
        if reward_spread(g) > 0
    ]

    # Test the strongest decision points first.
    groups.sort(
        key=reward_spread,
        reverse=True
    )

    print(
        f"all round1 groups={len(all_groups)} | "
        f"discriminative={len(groups)}"
    )

    groups = groups[:args.limit]

    print(f"groups tested={len(groups)}")

    results = []

    for gi, g in enumerate(groups, 1):
        rows = g["rows"]

        reward = {
            int(r["candidate_index"]): oracle_reward(r)
            for r in rows
        }

        max_reward = max(reward.values())
        oracle_best = {
            idx for idx, v in reward.items()
            if v == max_reward
        }

        actual_row = next(
            r for r in rows
            if is_chosen(r.get("chosen_for_continuation"))
        )
        actual_idx = int(actual_row["candidate_index"])

        jev = call_jev(g["state"], rows)
        jev_idx = int(jev["choice"].lstrip("p"))

        random_expected = (
            sum(reward.values()) / len(reward)
        )

        out = {
            "group": gi,
            "sid": g["sid"],
            "jev_idx": jev_idx,
            "actual_idx": actual_idx,
            "oracle_best": sorted(oracle_best),

            "jev_reward": reward[jev_idx],
            "actual_reward": reward[actual_idx],
            "random_expected_reward": random_expected,
            "oracle_reward": max_reward,

            "jev_regret": max_reward - reward[jev_idx],
            "actual_regret": max_reward - reward[actual_idx],

            "jev_oracle_match": jev_idx in oracle_best,
            "actual_oracle_match": actual_idx in oracle_best,

            "jev_confidence": jev["confidence"],
            "jev_probs": jev["probabilities"],

            "queries": {
                int(r["candidate_index"]): r.get("query")
                for r in rows
            },

            # Evaluation only. NEVER sent to JEV.
            "oracle_selected_roles": {
                int(r["candidate_index"]):
                    r.get("selected_roles_training_only") or []
                for r in rows
            },
        }

        results.append(out)

        print(
            f"[{gi:02d}] "
            f"JEV={jev_idx} r={out['jev_reward']} | "
            f"current={actual_idx} r={out['actual_reward']} | "
            f"oracle={sorted(oracle_best)} r={max_reward} | "
            f"conf={jev['confidence']}"
        )

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "n": len(results),

        "mean_reward": {
            "jev": mean([x["jev_reward"] for x in results]),
            "current": mean([x["actual_reward"] for x in results]),
            "random_expected": mean([
                x["random_expected_reward"]
                for x in results
            ]),
            "oracle": mean([x["oracle_reward"] for x in results]),
        },

        "mean_regret": {
            "jev": mean([x["jev_regret"] for x in results]),
            "current": mean([
                x["actual_regret"] for x in results
            ]),
        },

        "oracle_top1_match": {
            "jev": mean([
                float(x["jev_oracle_match"])
                for x in results
            ]),
            "current": mean([
                float(x["actual_oracle_match"])
                for x in results
            ]),
        },
    }

    outdir = Path("artifacts/jev_selector_poc")
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "results.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    with (outdir / "summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print()
    print("saved:", outdir / "summary.json")
    print("saved:", outdir / "results.jsonl")


if __name__ == "__main__":
    main()
