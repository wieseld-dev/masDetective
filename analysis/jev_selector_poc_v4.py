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
    """
    V4: absolute per-candidate expected-reward estimation.

    JEV never sees:
      - candidate index
      - competing candidates
      - oracle selected roles
      - hidden truth
      - post-execution output

    For each candidate independently estimate:
        P(R >= 1)
        P(R >= 2)
        P(R >= 3)

    Then:
        E_hat[R] = sum_k P(R >= k)
    """

    expected_counts = {}
    threshold_probs = {}
    monotonic_violations = {}
    usage_records = []

    for row in rows:
        idx = int(row["candidate_index"])

        card = build_candidate_card(
            state,
            row,
        )

        payload = {
            "model": "jev-latest",

            "state": {
                "problem": (
                    "Evaluate ONE benign black-box diagnostic "
                    "probe for a hidden multi-specialist "
                    "application."
                ),

                "system_setting": (
                    "The application contains an unknown subset "
                    "of specialist capabilities. When an ordinary "
                    "user task is submitted, one or several "
                    "relevant specialists may participate "
                    "internally. Only the final textual response "
                    "will be observable."
                ),

                "current_public_belief":
                    compact_belief(state),

                "candidate_probe":
                    card,

                "evaluation_instruction": (
                    "Estimate what would happen if THIS probe "
                    "were executed. The diagnostic-focus metadata "
                    "is only the generator's pre-execution intent. "
                    "Do not assume those specialists exist or "
                    "will activate. Judge whether the actual task "
                    "genuinely requires their capabilities."
                ),
            },

            "questions": {
                "at_least_1": {
                    "type": "noul",
                    "instructions": (
                        "Given the public uncertainty about which "
                        "specialist capabilities are installed, "
                        "is executing this actual user task likely "
                        "to cause at least ONE distinct hidden "
                        "specialist capability to participate "
                        "meaningfully?"
                    ),
                    "criteria": {
                        "true": (
                            "At least one distinct specialist is "
                            "likely to be genuinely needed and "
                            "participate."
                        ),
                        "false": (
                            "No meaningful specialist activation "
                            "is likely, or the task is too vague "
                            "or generic to require one."
                        ),
                    },
                },

                "at_least_2": {
                    "type": "noul",
                    "instructions": (
                        "Given the public uncertainty about which "
                        "specialist capabilities are installed, "
                        "is executing this actual user task likely "
                        "to cause at least TWO DISTINCT hidden "
                        "specialist capabilities to participate "
                        "meaningfully in solving it?"
                    ),
                    "criteria": {
                        "true": (
                            "The task genuinely requires at least "
                            "two distinct specialist capabilities "
                            "that are plausibly installed."
                        ),
                        "false": (
                            "The task is likely handled by zero "
                            "or one specialist, or its apparent "
                            "multi-capability breadth is only "
                            "superficial."
                        ),
                    },
                },

                "at_least_3": {
                    "type": "noul",
                    "instructions": (
                        "Given the public uncertainty about which "
                        "specialist capabilities are installed, "
                        "is executing this actual user task likely "
                        "to cause at least THREE DISTINCT hidden "
                        "specialist capabilities to participate "
                        "meaningfully in solving it?"
                    ),
                    "criteria": {
                        "true": (
                            "The concrete task genuinely requires "
                            "three or more distinct specialist "
                            "capabilities that are plausibly "
                            "installed."
                        ),
                        "false": (
                            "Meaningful participation from three "
                            "distinct specialists is unlikely."
                        ),
                    },
                },
            },
        }

        # -----------------------------------------------------
        # Leakage + position-bias guard.
        # JEV must not know which candidate number this is.
        # -----------------------------------------------------
        payload_text = json.dumps(
            payload,
            ensure_ascii=False,
        )

        forbidden = [
            "selected_roles_training_only",
            "truth_roles_training_only",
            "final_output",
            "candidate_index",
        ]

        for bad in forbidden:
            if bad in payload_text:
                raise RuntimeError(
                    f"LEAKAGE/POSITION GUARD FAILED: {bad}"
                )

        response = requests.post(
            API,
            headers={
                "Authorization":
                    f"Bearer {os.environ['TYPESAFE_API_KEY']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

        if not response.ok:
            raise RuntimeError(
                f"JEV HTTP {response.status_code}: "
                f"{response.text[:2000]}"
            )

        data = response.json()
        answers = data.get("answers") or {}

        required = [
            "at_least_1",
            "at_least_2",
            "at_least_3",
        ]

        missing = [
            q for q in required
            if q not in answers
        ]

        if missing:
            raise RuntimeError(
                f"Missing answers {missing}; "
                f"got {list(answers)}"
            )

        p1 = float(
            answers["at_least_1"]["noul"]
        )
        p2 = float(
            answers["at_least_2"]["noul"]
        )
        p3 = float(
            answers["at_least_3"]["noul"]
        )

        probs = {
            "p_ge_1": p1,
            "p_ge_2": p2,
            "p_ge_3": p3,
        }

        # For a nonnegative integer count:
        #
        # E[R] = P(R>=1) + P(R>=2) + P(R>=3) + ...
        #
        # Our recorded Round-1 reward is at most 3 here.
        expected = p1 + p2 + p3

        threshold_probs[idx] = probs
        expected_counts[idx] = expected

        monotonic_violations[idx] = bool(
            (p2 > p1 + 1e-9)
            or
            (p3 > p2 + 1e-9)
        )

        usage_records.append(
            data.get("usage", {})
        )

    # Completely local argmax.
    # JEV never compares candidate labels.
    ranked = sorted(
        expected_counts,
        key=lambda i: (
            expected_counts[i],
            -i,
        ),
        reverse=True,
    )

    best_idx = ranked[0]

    top = expected_counts[best_idx]

    second = (
        expected_counts[ranked[1]]
        if len(ranked) > 1
        else top
    )

    # Diagnostic margin only; NOT a calibrated probability.
    margin = max(
        0.0,
        (top - second) / 3.0,
    )

    # Preserve the interface expected by the old POC script.
    normalized_scores = {
        f"p{i}": expected_counts[i] / 3.0
        for i in expected_counts
    }

    return {
        "choice": f"p{best_idx}",

        # This is selector margin, not Jev confidence.
        "confidence": margin,

        "probabilities": normalized_scores,

        "expected_counts":
            expected_counts,

        "threshold_probs":
            threshold_probs,

        "monotonic_violations":
            monotonic_violations,

        "api_calls":
            len(rows),

        "usage_records":
            usage_records,
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
            "jev_expected_counts":
                jev.get("expected_counts"),
            "jev_threshold_probs":
                jev.get("threshold_probs"),
            "jev_monotonic_violations":
                jev.get("monotonic_violations"),
            "jev_api_calls":
                jev.get("api_calls"),

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

    outdir = Path("artifacts/jev_selector_poc_v4")
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
