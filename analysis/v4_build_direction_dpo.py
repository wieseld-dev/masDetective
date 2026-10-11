from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("artifacts/v4_userlike_audit")
OUT = Path("artifacts/v4_direction_dpo")
OUT.mkdir(parents=True, exist_ok=True)

AUDIT = ROOT / "userlike_judgments.jsonl"
GOLD = ROOT / "gold_strict.jsonl"

PAIR_OUT = OUT / "direction_pairs.jsonl"
SUMMARY = OUT / "summary.json"


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p, rows):
    with Path(p).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


audit = read_jsonl(AUDIT)
gold = read_jsonl(GOLD)

assert len(gold) == 24, f"expected 24 strict Gold, got {len(gold)}"

gold_by = {
    int(x["gold_index"]): x
    for x in gold
}

samples = [
    x for x in audit
    if x.get("source") == "SAMPLE"
]

assert len(samples) == 327


# ------------------------------------------------------------
# Pure conversational-direction negatives.
#
# Keep META constant:
# rejected must be NON_META HIGH according to frozen META V3.
#
# Therefore preference should isolate:
# USER_TO_SYSTEM > SYSTEM_TO_USER
# ------------------------------------------------------------

neg = [
    x for x in samples
    if int(x["gold_index"]) in gold_by
    and x.get("user_like") == "ASSISTANT_LIKE"
    and x.get("direction") == "SYSTEM_TO_USER"
    and x.get("confidence") == "HIGH"
    and x.get("naturalness") == "NATURAL"
    and x.get("meta_label") == "NON_META"
    and x.get("meta_confidence") == "HIGH"
    and not x.get("internal_prompt_lexical_flag", False)
]


by_gold = defaultdict(list)

for x in neg:
    by_gold[int(x["gold_index"])].append(x)


pairs = []

for gi in sorted(by_gold):
    g = gold_by[gi]

    # Deduplicate exact normalized query text.
    seen = set()
    xs = []

    for x in sorted(
        by_gold[gi],
        key=lambda z: int(z["sample_index"])
    ):
        q = " ".join(str(x["query"]).split())

        if not q or q in seen:
            continue

        seen.add(q)
        xs.append(x)

    # ONE pair per Gold initially.
    # Prevent prompts with many negatives dominating.
    if not xs:
        continue

    # Prefer the clearest assistant-clarification example when available.
    xs.sort(
        key=lambda x: (
            0 if x.get(
                "assistant_clarification_lexical_flag",
                False
            ) else 1,
            int(x["sample_index"]),
        )
    )

    x = xs[0]

    assert str(g["prompt"]) == str(x["prompt"]), (
        f"PROMPT MISMATCH G{gi}"
    )

    chosen = str(g["completion"])
    rejected = str(x["completion"])

    assert chosen.strip()
    assert rejected.strip()
    assert chosen != rejected

    pairs.append({
        "gold_index": gi,
        "round": int(g["round"]),
        "session_id": str(g["session_id"]),
        "candidate_index": int(g["candidate_index"]),
        "sample_index": int(x["sample_index"]),

        "prompt": str(g["prompt"]),
        "chosen": chosen,
        "rejected": rejected,

        "chosen_query": str(g["query"]),
        "rejected_query": str(x["query"]),

        "chosen_user_like": g["user_like"],
        "chosen_direction": g["direction"],
        "chosen_naturalness": g["naturalness"],

        "rejected_user_like": x["user_like"],
        "rejected_direction": x["direction"],
        "rejected_naturalness": x["naturalness"],
        "rejected_meta_label": x["meta_label"],
        "rejected_meta_confidence": x["meta_confidence"],

        "gold_reward": float(g.get("gold_reward", 0.0)),
        "gold_advantage": float(g.get("gold_advantage", 0.0)),
    })


# ------------------------------------------------------------
# Strong invariants
# ------------------------------------------------------------

assert len({x["gold_index"] for x in pairs}) == len(pairs)

assert all(
    x["chosen_user_like"] == "USER_LIKE"
    and x["chosen_direction"] == "USER_TO_SYSTEM"
    and x["chosen_naturalness"] == "NATURAL"
    for x in pairs
)

assert all(
    x["rejected_user_like"] == "ASSISTANT_LIKE"
    and x["rejected_direction"] == "SYSTEM_TO_USER"
    and x["rejected_naturalness"] == "NATURAL"
    and x["rejected_meta_label"] == "NON_META"
    and x["rejected_meta_confidence"] == "HIGH"
    for x in pairs
)


write_jsonl(PAIR_OUT, pairs)

summary = {
    "strict_gold": len(gold),
    "pure_direction_negative_candidates": len(neg),
    "gold_with_direction_negative": len(pairs),
    "direction_pairs": len(pairs),
    "max_pairs_per_gold": 1,
}

SUMMARY.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)


print("======================================")
print(" PURE DIRECTION DPO SUMMARY")
print("======================================")
print("strict Gold               :", len(gold))
print("direction neg candidates  :", len(neg))
print(
    "Gold coverage             :",
    f"{len(pairs)}/{len(gold)}"
)
print("final direction pairs     :", len(pairs))


print()
print("=== ALL DIRECTION PAIRS ===")

for i, x in enumerate(pairs, 1):
    print(
        f"\n--- {i}/{len(pairs)} "
        f"| G{x['gold_index']} "
        f"r{x['round']} j{x['candidate_index']} ---"
    )
    print("CHOSEN  :", x["chosen_query"])
    print("REJECTED:", x["rejected_query"])


print()
print("wrote:", PAIR_OUT)
print("wrote:", SUMMARY)
