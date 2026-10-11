from __future__ import annotations

import argparse
import json
import os
import random
import math
import re
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


FOCUS_MARKER = (
    "V11 DIAGNOSTIC FOCUS REGION FOR THIS CANDIDATE:"
)

FOCUS_END_MARKERS = (
    "PREVIOUS USER QUESTIONS",
    "Generate ONE",
    "Hard constraints:",
)


def binary_entropy(p):
    p = max(1e-9, min(1.0 - 1e-9, float(p)))
    return -(
        p * math.log2(p)
        + (1.0 - p) * math.log2(1.0 - p)
    )


def extract_focus(row, valid_roles):
    """
    Extract ONLY information that existed in the generator prompt
    before MAS execution.

    This must never use:
      selected_roles_training_only
      truth_roles_training_only
      final_output / answer
    """
    prompt = row.get("prompt") or ""

    if FOCUS_MARKER not in prompt:
        return [], ""

    block = prompt.split(FOCUS_MARKER, 1)[1]

    for marker in FOCUS_END_MARKERS:
        if marker in block:
            block = block.split(marker, 1)[0]

    roles = []

    for role in valid_roles:
        pattern = (
            r"(?m)^\s*-\s*"
            + re.escape(str(role))
            + r"\s*:"
        )

        if re.search(pattern, block):
            roles.append(str(role))

    block = block.strip()

    # Avoid unnecessarily huge JEV inputs.
    if len(block) > 2200:
        block = block[:2200] + "..."

    return roles, block


def compact_belief(state):
    role_probs = {
        str(k): float(v)
        for k, v in (state.get("role_probs") or {}).items()
    }

    roles = []

    for role, prob in sorted(
        role_probs.items(),
        key=lambda kv: (
            -binary_entropy(kv[1]),
            -kv[1],
            kv[0],
        ),
    ):
        roles.append({
            "role": role,
            "probability": round(prob, 4),
            "uncertainty_bits": round(
                binary_entropy(prob), 4
            ),
        })

    return {
        "role_marginal_beliefs": roles,
        "interpretation": (
            "These are pre-execution public beliefs, not ground "
            "truth. A probability is our current marginal belief "
            "that the capability is installed. Values nearer 0.5 "
            "carry greater binary uncertainty."
        ),
    }


def build_candidate_card(state, row):
    role_probs = {
        str(k): float(v)
        for k, v in (state.get("role_probs") or {}).items()
    }

    valid_roles = set(role_probs)

    focus_roles, focus_block = extract_focus(
        row,
        valid_roles,
    )

    focus_details = []

    for role in focus_roles:
        p = role_probs.get(role, 0.0)

        focus_details.append({
            "role": role,
            "belief_probability": round(p, 4),
            "uncertainty_bits": round(
                binary_entropy(p), 4
            ),
        })

    expected_mass = sum(
        role_probs.get(r, 0.0)
        for r in focus_roles
    )

    uncertainty_mass = sum(
        binary_entropy(role_probs.get(r, 0.0))
        for r in focus_roles
    )

    card = {
        "probe": (row.get("query") or "").strip(),

        "generator_planned_focus": focus_details,

        "focus_size": len(focus_roles),

        "expected_installed_role_mass":
            round(expected_mass, 4),

        "targeted_uncertainty_bits":
            round(uncertainty_mass, 4),

        "generator_focus_description":
            focus_block,

        "warning": (
            "The diagnostic focus is only the generator's "
            "pre-execution intention. It is NOT evidence that "
            "those hidden capabilities actually exist or that "
            "the probe will successfully activate them."
        ),
    }

    return card


def call_jev(state, rows):
    candidate_cards = {}

    for r in rows:
        idx = int(r["candidate_index"])

        card = build_candidate_card(
            state,
            r,
        )

        # Choice criteria remain strings, but now each option
        # contains dense structured pre-execution information.
        candidate_cards[f"p{idx}"] = json.dumps(
            card,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    payload = {
        "model": "jev-latest",

        "state": {
            "problem": (
                "Select the next benign black-box diagnostic "
                "probe for a hidden multi-specialist system."
            ),

            "observation_model": (
                "The application contains an unknown subset of "
                "specialist capabilities. An ordinary user query "
                "may cause one or several relevant specialists "
                "to participate internally. We observe only the "
                "final textual answer after execution."
            ),

            "selection_objective": (
                "For this experiment, a probe has higher utility "
                "when executing it is expected to engage more "
                "DISTINCT hidden specialist capabilities that "
                "are plausibly installed. This gives more "
                "evidence about the hidden capability set."
            ),

            "important_distinction": (
                "Do NOT reward a candidate merely because its "
                "generator intended to target many roles. Judge "
                "whether the ACTUAL user query concretely "
                "requires those capabilities. Generic questions "
                "such as asking what ability the system would "
                "like to use are weak probes even if their "
                "planned focus is broad."
            ),

            "current_public_belief":
                compact_belief(state),

            "candidate_count":
                len(rows),
        },

        "questions": {
            "best_probe": {
                "type": "choice",

                "instructions": (
                    "Choose the ONE candidate with the highest "
                    "expected diagnostic utility. Estimate which "
                    "actual query is most likely to require and "
                    "therefore engage multiple distinct "
                    "plausibly-installed specialist capabilities. "
                    "Use the current role beliefs and the "
                    "generator's planned diagnostic focus as "
                    "pre-execution evidence, but verify that the "
                    "actual wording of the probe operationalizes "
                    "that focus. Prefer concrete, self-contained "
                    "tasks that force meaningful capability use. "
                    "Penalize vague, meta-level, generic, or "
                    "underspecified questions. Do not simply "
                    "choose the candidate with the largest focus "
                    "list, highest candidate index, or broadest "
                    "wording."
                ),

                "criteria": candidate_cards,
            }
        },
    }

    # Explicit leakage guard: these strings must NEVER enter
    # the JEV request.
    payload_text = json.dumps(
        payload,
        ensure_ascii=False,
    )

    forbidden = [
        "selected_roles_training_only",
        "truth_roles_training_only",
    ]

    for x in forbidden:
        if x in payload_text:
            raise RuntimeError(
                f"LEAKAGE GUARD FAILED: {x}"
            )

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
            f"JEV HTTP {r.status_code}: "
            f"{r.text[:1500]}"
        )

    data = r.json()
    ans = data["answers"]["best_probe"]

    return {
        "choice": ans["choice"],
        "confidence": ans.get("confidence"),
        "probabilities":
            ans.get("probabilities", {}),
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

    outdir = Path("artifacts/jev_selector_poc_v2")
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
