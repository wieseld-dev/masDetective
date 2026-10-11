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


def _normalized_probs(answer, labels):
    """
    Return a proper distribution over candidate labels.

    Prefer Jev's probability distribution.
    Fall back to a one-hot distribution if the API returns only choice.
    """
    raw = answer.get("probabilities") or {}

    out = {
        label: max(0.0, float(raw.get(label, 0.0)))
        for label in labels
    }

    z = sum(out.values())

    if z > 0:
        return {
            k: v / z
            for k, v in out.items()
        }

    choice = answer.get("choice")

    return {
        label: 1.0 if label == choice else 0.0
        for label in labels
    }


def call_jev(state, rows):
    candidate_cards = {}

    for r in rows:
        idx = int(r["candidate_index"])

        card = build_candidate_card(
            state,
            r,
        )

        candidate_cards[f"p{idx}"] = json.dumps(
            card,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    common_state = {
        "problem": (
            "Select the next benign black-box diagnostic probe "
            "for a hidden multi-specialist application."
        ),

        "observation_model": (
            "The application contains an unknown subset of "
            "specialist capabilities. A normal user query can "
            "cause one or several relevant specialists to "
            "participate internally. Only the final textual "
            "answer is visible after execution."
        ),

        "true_selection_goal": (
            "We want a probe whose execution is expected to "
            "engage useful DISTINCT hidden specialist "
            "capabilities and therefore reveal information "
            "about the installed capability set."
        ),

        "important_warning": (
            "The generator's diagnostic focus is only a "
            "pre-execution intention. It is not evidence that "
            "the roles exist, and it is not evidence that the "
            "actual query truly requires them. Judge the actual "
            "query wording."
        ),

        "current_public_belief":
            compact_belief(state),

        "candidate_count":
            len(rows),
    }

    questions = {
        "operationalization": {
            "type": "choice",
            "instructions": (
                "Which ONE candidate most strongly operationalizes "
                "its claimed diagnostic focus in the actual user "
                "query? Choose the query whose task genuinely "
                "REQUIRES the relevant specialist capabilities "
                "rather than merely mentioning them or asking a "
                "generic/meta question. A good candidate makes "
                "those capabilities necessary to produce a strong "
                "answer."
            ),
            "criteria": candidate_cards,
        },

        "breadth": {
            "type": "choice",
            "instructions": (
                "Which ONE candidate is most likely, if executed "
                "on the hidden multi-specialist application, to "
                "naturally require participation from the largest "
                "number of DISTINCT relevant specialist "
                "capabilities? Favor substantive multi-faceted "
                "tasks. Do not reward vague breadth: several "
                "capabilities should actually be needed to solve "
                "the stated task."
            ),
            "criteria": candidate_cards,
        },

        "uncertainty_targeting": {
            "type": "choice",
            "instructions": (
                "Which ONE candidate best targets capabilities "
                "whose presence is currently uncertain while also "
                "remaining plausible under the public belief "
                "state? Prefer useful uncertainty reduction over "
                "re-testing capabilities that are already nearly "
                "certain or implausible. Judge using only the "
                "pre-execution beliefs supplied here."
            ),
            "criteria": candidate_cards,
        },

        "concreteness": {
            "type": "choice",
            "instructions": (
                "Which ONE candidate is the strongest concrete "
                "diagnostic task? Prefer self-contained tasks with "
                "specific work to perform and enough information "
                "to force meaningful behavior. Penalize questions "
                "such as 'what ability would you use?', 'what can "
                "you help with?', generic preference questions, "
                "underspecified requests, or other meta-level "
                "prompts that may produce generic answers."
            ),
            "criteria": candidate_cards,
        },
    }

    payload = {
        "model": "jev-latest",
        "state": common_state,
        "questions": questions,
    }

    # ---------------------------------------------------------
    # Hard leakage guard.
    # ---------------------------------------------------------
    payload_text = json.dumps(
        payload,
        ensure_ascii=False,
    )

    forbidden = [
        "selected_roles_training_only",
        "truth_roles_training_only",
        "final_output",
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
            f"{r.text[:2000]}"
        )

    data = r.json()

    answers = data.get("answers") or {}

    labels = [
        f"p{int(row['candidate_index'])}"
        for row in rows
    ]

    required = [
        "operationalization",
        "breadth",
        "uncertainty_targeting",
        "concreteness",
    ]

    missing = [
        q for q in required
        if q not in answers
    ]

    if missing:
        raise RuntimeError(
            f"Missing JEV answers: {missing}; "
            f"got keys={list(answers)}"
        )

    components = {
        q: _normalized_probs(
            answers[q],
            labels,
        )
        for q in required
    }

    # Fixed BEFORE looking at V3 labels/results.
    #
    # Breadth gets highest weight because the Round-1 oracle
    # utility is number of distinct hidden roles exposed.
    weights = {
        "operationalization": 0.30,
        "breadth": 0.35,
        "uncertainty_targeting": 0.15,
        "concreteness": 0.20,
    }

    combined = {}

    for label in labels:
        combined[label] = sum(
            weights[q] * components[q][label]
            for q in required
        )

    # Normalize only for readability; argmax is unchanged.
    z = sum(combined.values())

    if z > 0:
        combined = {
            k: v / z
            for k, v in combined.items()
        }

    choice = max(
        labels,
        key=lambda label: (
            combined[label],
            -int(label[1:]),
        ),
    )

    # Save enough information to diagnose WHY V3 chose it.
    component_choices = {
        q: answers[q].get("choice")
        for q in required
    }

    component_confidences = {
        q: answers[q].get("confidence")
        for q in required
    }

    return {
        "choice": choice,
        "confidence": combined[choice],
        "probabilities": combined,
        "components": components,
        "component_choices": component_choices,
        "component_confidences": component_confidences,
        "weights": weights,
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
            "jev_components": jev.get("components"),
            "jev_component_choices":
                jev.get("component_choices"),
            "jev_component_confidences":
                jev.get("component_confidences"),
            "jev_weights": jev.get("weights"),

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

    outdir = Path("artifacts/jev_selector_poc_v3")
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
