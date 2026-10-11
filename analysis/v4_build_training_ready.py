from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


GOLD_PATH = Path("artifacts/v4_userlike_audit/gold_strict.jsonl")
RANK_PATH = Path("artifacts/v4_final_dpo/topic_match_rankings.jsonl")
SAMPLES_PATH = Path("artifacts/v4_bulk_mining/samples.jsonl")

OUT = Path("artifacts/v4_training_ready")
OUT.mkdir(parents=True, exist_ok=True)

POS_OUT = OUT / "positives.jsonl"
PREF_OUT = OUT / "preferences.jsonl"
MANIFEST_OUT = OUT / "manifest.json"


# Human-audited positive Gold IDs.
KEEP_GOLD = {
    2, 6, 8, 9, 10, 16, 23, 24, 26,
    27, 32, 33, 35, 37, 39, 43, 44,
}

# Human-audited preference selections from topic-match report.
# Value = rank within that Gold/objective after descending topic-match score.
DIRECTION_SELECTION = {
    2: 1,
    8: 1,
    10: 3,
    16: 1,
    32: 1,
    43: 1,
}

META_SELECTION = {
    10: 2,
    32: 1,
}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


gold = read_jsonl(GOLD_PATH)
rankings = read_jsonl(RANK_PATH)
samples = read_jsonl(SAMPLES_PATH)

gold_by = {
    int(x["gold_index"]): x
    for x in gold
}

assert set(KEEP_GOLD) <= set(gold_by), (
    "Some manually approved Gold IDs are missing"
)

# ------------------------------------------------------------------
# Positive anchors
# ------------------------------------------------------------------

positives = []

for gi in sorted(KEEP_GOLD):
    g = gold_by[gi]

    positives.append({
        "gold_index": gi,
        "round": int(g["round"]),
        "session_id": str(g["session_id"]),
        "candidate_index": int(g["candidate_index"]),

        "prompt": str(g["prompt"]),
        "completion": str(g["completion"]),
        "query": str(g["query"]),

        "gold_reward": float(g.get("gold_reward", 0.0)),
        "gold_advantage": float(g.get("gold_advantage", 0.0)),

        "human_audit": "KEEP",
        "source": "V3_GOLD_INFORMATIVE",
    })

assert len(positives) == 17
assert len({x["gold_index"] for x in positives}) == 17


# ------------------------------------------------------------------
# Recover exact generated completions for preference negatives.
# Rankings only contain query text; training needs full completion.
# ------------------------------------------------------------------

sample_by_key = {
    (int(x["gold_index"]), int(x["sample_index"])): x
    for x in samples
}

rank_by_group = defaultdict(list)

for x in rankings:
    key = (str(x["kind"]), int(x["gold_index"]))
    rank_by_group[key].append(x)

for key in rank_by_group:
    rank_by_group[key].sort(
        key=lambda z: (
            float(z["score"]),
            float(z["semantic_similarity"]),
        ),
        reverse=True,
    )


def choose_rank(kind, gi, rank):
    xs = rank_by_group[(kind, gi)]

    assert len(xs) >= rank, (
        f"{kind} G{gi}: requested rank {rank}, "
        f"only {len(xs)} candidates"
    )

    return xs[rank - 1]


preferences = []


def add_pair(kind, gi, rank):
    assert gi in KEEP_GOLD, (
        f"Preference chosen Gold G{gi} is not an approved positive"
    )

    r = choose_rank(kind, gi, rank)

    sample_index = int(r["sample_index"])
    s = sample_by_key[(gi, sample_index)]
    g = gold_by[gi]

    # Critical identity checks.
    assert str(g["prompt"]) == str(s["prompt"]), (
        f"{kind} G{gi}: prompt mismatch"
    )

    assert str(g["query"]).strip() == str(r["chosen"]).strip(), (
        f"{kind} G{gi}: ranking chosen query mismatch"
    )

    assert str(s["query"]).strip() == str(r["rejected"]).strip(), (
        f"{kind} G{gi}: ranking rejected query mismatch"
    )

    assert str(g["completion"]).strip()
    assert str(s["completion"]).strip()
    assert str(g["completion"]) != str(s["completion"])

    preferences.append({
        "pair_type": kind,
        "gold_index": gi,
        "selected_rank": rank,

        "round": int(g["round"]),
        "session_id": str(g["session_id"]),
        "candidate_index": int(g["candidate_index"]),
        "sample_index": sample_index,

        # Exact same prompt for chosen/rejected.
        "prompt": str(g["prompt"]),

        # Full model completions for training.
        "chosen": str(g["completion"]),
        "rejected": str(s["completion"]),

        # Human-readable parsed queries for audit.
        "chosen_query": str(g["query"]),
        "rejected_query": str(s["query"]),

        "topic_score": float(r["score"]),
        "semantic_similarity": float(r["semantic_similarity"]),
        "word_jaccard": float(r["word_jaccard"]),

        "gold_reward": float(g.get("gold_reward", 0.0)),
        "gold_advantage": float(g.get("gold_advantage", 0.0)),

        "human_audit": "KEEP",
    })


for gi, rank in sorted(DIRECTION_SELECTION.items()):
    add_pair("DIRECTION", gi, rank)

for gi, rank in sorted(META_SELECTION.items()):
    add_pair("META", gi, rank)


# ------------------------------------------------------------------
# Hard integrity checks
# ------------------------------------------------------------------

assert len(preferences) == 8

assert sum(x["pair_type"] == "DIRECTION" for x in preferences) == 6
assert sum(x["pair_type"] == "META" for x in preferences) == 2

assert all(x["gold_index"] in KEEP_GOLD for x in preferences)

# All chosen completions must exactly correspond to an approved Gold.
for x in preferences:
    g = gold_by[x["gold_index"]]

    assert x["prompt"] == g["prompt"]
    assert x["chosen"] == g["completion"]
    assert x["chosen_query"] == g["query"]
    assert x["chosen"] != x["rejected"]

# Exact pair uniqueness.
pair_keys = {
    (
        x["pair_type"],
        x["gold_index"],
        x["sample_index"],
    )
    for x in preferences
}
assert len(pair_keys) == 8


write_jsonl(POS_OUT, positives)
write_jsonl(PREF_OUT, preferences)


manifest = {
    "base_adapter": "artifacts/robust_visible_generator_v1_epoch2/policy",

    "positive_count": len(positives),
    "positive_gold_ids": sorted(KEEP_GOLD),

    "preference_count": len(preferences),
    "direction_pairs": sum(
        x["pair_type"] == "DIRECTION"
        for x in preferences
    ),
    "meta_pairs": sum(
        x["pair_type"] == "META"
        for x in preferences
    ),

    "preference_gold_ids": sorted(
        set(x["gold_index"] for x in preferences)
    ),

    "unique_preference_gold": len(
        set(x["gold_index"] for x in preferences)
    ),

    "selection": {
        "direction": DIRECTION_SELECTION,
        "meta": META_SELECTION,
    },

    "notes": [
        "Positive anchors are V3 candidates with positive membership/elimination signal and passed manual audit.",
        "Preference pairs use byte-identical generator prompts.",
        "Chosen side is always one of the manually approved informative Gold positives.",
        "Rejected side passed automatic high-precision filtering and manual topic/direction or META audit.",
        "Synthetic clean alternatives without MAS reward are NOT used as positive anchors.",
    ],
}

MANIFEST_OUT.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)


print("========================================")
print(" V4 TRAINING-READY DATASET")
print("========================================")
print("positive anchors :", len(positives))
print("preference pairs :", len(preferences))
print(
    "  direction     :",
    sum(x["pair_type"] == "DIRECTION" for x in preferences),
)
print(
    "  META          :",
    sum(x["pair_type"] == "META" for x in preferences),
)
print(
    "unique pref Gold:",
    len(set(x["gold_index"] for x in preferences)),
)

print()
print("=== POSITIVES ===")
for x in positives:
    print(f"G{x['gold_index']:02d}: {x['query']}")

print()
print("=== PREFERENCES ===")
for i, x in enumerate(preferences, 1):
    print(
        f"\n{i:02d}. {x['pair_type']} "
        f"G{x['gold_index']} rank={x['selected_rank']}"
    )
    print(" +", x["chosen_query"])
    print(" -", x["rejected_query"])
    print(
        f"   topic={x['topic_score']:.3f} "
        f"sem={x['semantic_similarity']:.3f}"
    )

print()
print("INTEGRITY: PASS")
print("wrote:", POS_OUT)
print("wrote:", PREF_OUT)
print("wrote:", MANIFEST_OUT)
