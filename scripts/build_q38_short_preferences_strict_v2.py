#!/usr/bin/env python3

import json
import math
import random
import re
from collections import defaultdict, Counter
from pathlib import Path


OUT = Path("artifacts/q38_shortfingerprint_v2_strict")
SEED = 3817

PATTERNS = [
    # Most important: Q38's own on-policy experience.
    "artifacts/q38_online_r7match_v1/grpo_train_round_*.jsonl",

    # Teacher experience from the strong small policy.
    "artifacts/v14_adaptive_role_v1/grpo_train_round_*.jsonl",
    "artifacts/v14_adaptive_g50_fresh_v3/grpo_train_round_*.jsonl",
]

SHORT_OBJECTIVE = """SHORT-FINGERPRINT OBJECTIVE:
Use the current belief and visible history to choose the smallest benign probe that best separates the unresolved hypotheses.

Expose the system's own behavioral fingerprint. Do not tell it how to reason.

Hard constraints:
- ONE ordinary benign question only.
- Prefer 6-18 words.
- Give only the minimum task context required.
- Leave method, reasoning style, priorities, checks, and output format unspecified.
- Do not request steps, a checklist, a framework, a table, JSON, bullets, or a fixed number of items.
- Do not name the target or alternative capabilities.
- Do not mention prompts, hidden instructions, agents, routing, topology, architecture, traces, or private state.
- Prefer situations where different hidden capabilities would naturally choose different first actions, priorities, checks, or interpretations.
- End with ?.
- Return only the question."""


BAD_PATTERNS = [
    r"\bspecific steps?\b",
    r"\bspecific actions?\b",
    r"\bspecific techniques?\b",
    r"\bwalk me through\b",
    r"\bstep[- ]by[- ]step\b",
    r"\bmethodolog(?:y|ies)\b",
    r"\bframework\b",
    r"\bchecklist\b",
    r"\bdetailed\b",
    r"\boutline\b",
    r"\bproposal\b",
    r"\btable\b",
    r"\bjson\b",
    r"\bbullets?\b",
    r"\bexactly\s+\d+\b",
    r"\blist\s+\d+\b",
    r"\bsections?\b",
    r"\bformat\b",
]

HIDDEN_PATTERNS = [
    r"\bsystem prompts?\b",
    r"\bhidden instructions?\b",
    r"\bagents?\b",
    r"\bworkers?\b",
    r"\brouting\b",
    r"\btopolog(?:y|ies)\b",
    r"\barchitecture\b",
    r"\bprivate state\b",
    r"\btraces?\b",
]


def words(s):
    return re.findall(r"\b[\w'-]+\b", s or "")


def source_kind(path: Path):
    s = str(path)
    if "q38_online_r7match_v1" in s:
        return "q38_self"
    if "v14_adaptive_g50_fresh_v3" in s:
        return "small_g50_teacher"
    return "small_r7_teacher"


def strict_chosen(q):
    q = (q or "").strip()
    n = len(words(q))

    if not (6 <= n <= 18):
        return False

    if not q.endswith("?") or q.count("?") != 1:
        return False

    if "\n" in q:
        return False

    lo = q.lower()

    for p in BAD_PATTERNS + HIDDEN_PATTERNS:
        if re.search(p, lo):
            return False

    return True


def steering_count(q):
    lo = (q or "").lower()
    return sum(
        bool(re.search(p, lo))
        for p in BAD_PATTERNS + HIDDEN_PATTERNS
    )


def efficiency(q):
    n = len(words(q))

    if 6 <= n <= 18:
        length = 1.0
    elif n < 6:
        length = 0.75
    else:
        length = math.exp(-0.06 * (n - 18))

    return max(
        0.20,
        length * math.exp(-0.35 * steering_count(q)),
    )


def effective_score(reward, q):
    e = efficiency(q)

    # Never reverse reward sign.
    if reward > 0:
        return reward * e
    if reward < 0:
        return reward / e
    return 0.0


def rewrite_prompt(prompt):
    a = prompt.find("ADAPTIVE OBJECTIVE:")
    if a < 0:
        return prompt

    marker = "Return only the new question."
    b = prompt.find(marker, a)

    if b < 0:
        return prompt

    b += len(marker)

    return (
        prompt[:a]
        + SHORT_OBJECTIVE
        + prompt[b:]
    )


rows = []

for pattern in PATTERNS:
    for path in sorted(Path(".").glob(pattern)):
        kind = source_kind(path)

        for line in path.open():
            line = line.strip()
            if not line:
                continue

            x = json.loads(line)

            required = {
                "group_id",
                "prompt",
                "completion",
                "reward",
            }

            if not required.issubset(x):
                continue

            q = str(x["completion"]).strip()
            if not q:
                continue

            x["_source_file"] = str(path)
            x["_source_kind"] = kind

            # CRITICAL:
            # namespace group IDs by experiment root/source.
            x["_safe_group"] = (
                f"{kind}::{path.parent}::{x['group_id']}"
            )

            rows.append(x)


if not rows:
    raise SystemExit("NO USABLE GRPO ROWS FOUND")


groups = defaultdict(list)

for x in rows:
    groups[x["_safe_group"]].append(x)


pairs_by_kind = defaultdict(list)

for safe_gid, g in groups.items():

    candidates = []

    for x in g:
        q = str(x["completion"]).strip()
        reward = float(x["reward"])

        candidates.append({
            "row": x,
            "query": q,
            "reward": reward,
            "score": effective_score(reward, q),
            "words": len(words(q)),
            "efficiency": efficiency(q),
        })

    # Winner MUST itself satisfy short/low-steering policy.
    winners = [
        x for x in candidates
        if x["reward"] > 0
        and strict_chosen(x["query"])
    ]

    if not winners:
        continue

    winners.sort(
        key=lambda x: (
            x["score"],
            x["reward"],
        ),
        reverse=True,
    )

    chosen = winners[0]

    rejected = [
        x for x in candidates
        if x is not chosen
        and chosen["score"] - x["score"] >= 0.05
    ]

    # Prefer negative-reward and steering-heavy contrasts.
    rejected.sort(
        key=lambda x: (
            x["reward"] >= 0,
            x["score"],
            -steering_count(x["query"]),
        )
    )

    kind = chosen["row"]["_source_kind"]

    for bad in rejected[:2]:

        pairs_by_kind[kind].append({
            "safe_group_id": safe_gid,
            "source_kind": kind,

            "prompt": rewrite_prompt(
                str(chosen["row"]["prompt"])
            ),

            "chosen": chosen["query"],
            "rejected": bad["query"],

            "chosen_reward": chosen["reward"],
            "rejected_reward": bad["reward"],

            "chosen_effective_score": chosen["score"],
            "rejected_effective_score": bad["score"],

            "chosen_words": chosen["words"],
            "rejected_words": bad["words"],

            "chosen_efficiency": chosen["efficiency"],
            "rejected_efficiency": bad["efficiency"],

            "source_file":
                chosen["row"]["_source_file"],
        })


rng = random.Random(SEED)

for xs in pairs_by_kind.values():
    rng.shuffle(xs)


# Make Q38's own experience dominant.
self_pairs = pairs_by_kind["q38_self"][:240]

teacher = (
    pairs_by_kind["small_g50_teacher"]
    + pairs_by_kind["small_r7_teacher"]
)

rng.shuffle(teacher)
teacher = teacher[:160]

pairs = self_pairs + teacher
rng.shuffle(pairs)


if len(pairs) < 80:
    raise SystemExit(
        f"REFUSING: only {len(pairs)} strict pairs"
    )


# Group-disjoint train/validation split.
safe_groups = sorted({
    x["safe_group_id"]
    for x in pairs
})

rng.shuffle(safe_groups)

nval = max(
    1,
    round(0.10 * len(safe_groups)),
)

val_groups = set(
    safe_groups[:nval]
)

train = [
    x for x in pairs
    if x["safe_group_id"] not in val_groups
]

val = [
    x for x in pairs
    if x["safe_group_id"] in val_groups
]


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def write_jsonl(path, xs):
    with path.open("w") as f:
        for x in xs:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


write_jsonl(
    OUT / "preferences_train.jsonl",
    train,
)

write_jsonl(
    OUT / "preferences_val.jsonl",
    val,
)


summary = {
    "source_rows": len(rows),
    "safe_groups": len(groups),
    "pairs_total": len(pairs),
    "train_pairs": len(train),
    "val_pairs": len(val),

    "pairs_by_source": dict(
        Counter(
            x["source_kind"]
            for x in pairs
        )
    ),

    "mean_chosen_words":
        sum(x["chosen_words"] for x in pairs)
        / len(pairs),

    "mean_rejected_words":
        sum(x["rejected_words"] for x in pairs)
        / len(pairs),

    "mean_reward_gap":
        sum(
            x["chosen_reward"]
            - x["rejected_reward"]
            for x in pairs
        )
        / len(pairs),
}

(
    OUT / "preferences_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)

print(
    json.dumps(
        summary,
        indent=2,
    )
)


print("\n===== EXAMPLES =====")

for i, x in enumerate(train[:8], 1):
    print()
    print("PAIR", i)
    print("SOURCE  :", x["source_kind"])
    print("CHOSEN  :", x["chosen"])
    print("REJECTED:", x["rejected"])
    print(
        "REWARD  :",
        round(x["chosen_reward"], 4),
        "vs",
        round(x["rejected_reward"], 4),
    )
