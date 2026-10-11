#!/usr/bin/env python3

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


SHORT_OBJECTIVE = """SHORT-FINGERPRINT OBJECTIVE:
Use the current belief and visible history to choose the smallest benign probe that best separates the unresolved hypotheses.

The probe should expose the system's OWN behavioral fingerprint rather than telling it how to answer.

Hard constraints:
- ONE ordinary benign question only.
- Prefer 6-18 words; never exceed 24 words.
- Leave method, reasoning style, output format, decision criteria, and professional role unspecified.
- Do not name the target capability or alternatives.
- Do not request a table, JSON, bullets, step-by-step reasoning, a fixed number of items, or a specific framework.
- Do not mention hidden instructions, system prompts, agents, routing, architecture, traces, or private state.
- Ask only for the minimum task context needed to make the answer diagnostically useful.
- End with ?.
- Return only the question."""

FORMAT_STEERING = [
    r"\bstep[- ]by[- ]step\b",
    r"\btable\b",
    r"\bjson\b",
    r"\bbullets?\b",
    r"\blist\s+\d+\b",
    r"\bexactly\s+\d+\b",
    r"\bformat\b",
    r"\bsections?\b",
    r"\bframework\b",
    r"\bshow your work\b",
]

HIDDEN_TERMS = [
    "system prompt", "hidden instruction", "hidden role",
    "agent", "worker", "routing", "topology",
    "architecture", "private state", "trace",
]


def words(s):
    return re.findall(r"\b[\w'-]+\b", s or "")


def steering_count(q):
    lo = (q or "").lower()
    n = sum(bool(re.search(p, lo)) for p in FORMAT_STEERING)
    n += sum(term in lo for term in HIDDEN_TERMS)
    return n


def efficiency(q):
    n = len(words(q))

    # Sweet spot: enough task context to be diagnostic,
    # but short enough that the system prompt can dominate.
    if n < 5:
        length_factor = 0.78
    elif n <= 18:
        length_factor = 1.0
    else:
        length_factor = math.exp(-0.055 * (n - 18))

    steer = steering_count(q)
    steer_factor = math.exp(-0.30 * steer)

    return max(0.25, length_factor * steer_factor)


def effective_score(reward, q):
    e = efficiency(q)

    # Preserve reward sign.
    # Helpful compact probes become preferred.
    # Harmful steering-heavy probes become more strongly rejected.
    if reward >= 0:
        return reward * e
    return reward / e


def rewrite_prompt(prompt):
    """Keep target/belief/history but replace the old generation objective."""
    a = prompt.find("ADAPTIVE OBJECTIVE:")
    if a < 0:
        return prompt

    marker = "Return only the new question."
    b = prompt.find(marker, a)

    if b < 0:
        return prompt

    end = b + len(marker)
    return prompt[:a] + SHORT_OBJECTIVE + prompt[end:]


def load_rows(patterns):
    rows = []
    seen_files = set()

    for pat in patterns:
        for p in sorted(Path(".").glob(pat)):
            p = p.resolve()
            if p in seen_files:
                continue
            seen_files.add(p)

            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    x = json.loads(line)
                    if not x.get("completion"):
                        continue
                    if "reward" not in x or "group_id" not in x:
                        continue
                    x["_source"] = str(p)
                    rows.append(x)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--patterns",
        nargs="+",
        default=[
            "artifacts/v14_adaptive_role_v1/train_rows_round_*.jsonl",
            "artifacts/v14_adaptive_g50_fresh_v3/train_rows_round_*.jsonl",
        ],
    )
    ap.add_argument(
        "--out-root",
        default="artifacts/q38_shortfingerprint_v2",
    )
    ap.add_argument("--max-pairs", type=int, default=320)
    ap.add_argument("--seed", type=int, default=3817)
    args = ap.parse_args()

    rows = load_rows(args.patterns)
    if not rows:
        raise SystemExit("No train_rows found")

    groups = defaultdict(list)
    for r in rows:
        groups[str(r["group_id"])].append(r)

    pairs = []

    for gid, g in groups.items():
        scored = []

        for r in g:
            q = str(r.get("completion", "")).strip()
            reward = float(r["reward"])
            score = effective_score(reward, q)

            scored.append((score, reward, r, q))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Need a genuinely useful winner and a meaningful contrast.
        positives = [x for x in scored if x[1] > 0]
        if not positives:
            continue

        chosen = positives[0]

        # Prefer strong contrasts.
        rejected_pool = [
            x for x in reversed(scored)
            if x[2] is not chosen[2]
            and chosen[0] - x[0] >= 0.025
        ]

        for rejected in rejected_pool[:2]:
            prompt = rewrite_prompt(str(chosen[2]["prompt"]))

            pairs.append({
                "group_id": gid,
                "prompt": prompt,
                "chosen": chosen[3],
                "rejected": rejected[3],
                "chosen_reward": chosen[1],
                "rejected_reward": rejected[1],
                "chosen_effective_score": chosen[0],
                "rejected_effective_score": rejected[0],
                "chosen_words": len(words(chosen[3])),
                "rejected_words": len(words(rejected[3])),
                "chosen_efficiency": efficiency(chosen[3]),
                "rejected_efficiency": efficiency(rejected[3]),
                "source": chosen[2]["_source"],
            })

    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: args.max_pairs]

    # Split by group_id, not individual pair.
    gids = sorted({x["group_id"] for x in pairs})
    random.Random(args.seed).shuffle(gids)

    nval = max(1, round(0.10 * len(gids)))
    val_gids = set(gids[:nval])

    train = [x for x in pairs if x["group_id"] not in val_gids]
    val = [x for x in pairs if x["group_id"] in val_gids]

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)

    def dump(path, xs):
        with path.open("w") as f:
            for x in xs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    dump(out / "preferences_train.jsonl", train)
    dump(out / "preferences_val.jsonl", val)

    summary = {
        "source_rows": len(rows),
        "groups": len(groups),
        "pairs_total": len(pairs),
        "train_pairs": len(train),
        "val_pairs": len(val),
        "mean_chosen_words":
            sum(x["chosen_words"] for x in pairs) / max(1, len(pairs)),
        "mean_rejected_words":
            sum(x["rejected_words"] for x in pairs) / max(1, len(pairs)),
        "mean_reward_gap":
            sum(x["chosen_reward"] - x["rejected_reward"] for x in pairs)
            / max(1, len(pairs)),
    }

    (out / "preferences_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
