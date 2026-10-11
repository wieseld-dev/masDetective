#!/usr/bin/env python3

import gc
import json
import random
import sys
from pathlib import Path
from collections import Counter, defaultdict

import torch

ROOT = Path("artifacts/v3_coverage_elimination_pilot")
JUDG = Path("artifacts/v4_generator/candidate_judgments.jsonl")

EPOCH2 = "artifacts/robust_visible_generator_v1_epoch2/policy"
V4 = "artifacts/v4_gold_generator_v1/policy"

OUT = Path("artifacts/v4_ab_generation")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path("analysis").resolve()))
import v3_generate as vg


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p, rows):
    with Path(p).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )


# ---------------------------------------------------------
# Recover exactly the 45 human-audited Gold examples
# ---------------------------------------------------------

judged = [
    x for x in read_jsonl(JUDG)
    if "error" not in x
]

gold48 = [
    x for x in judged
    if x["label"] == "NON_META"
    and float(x["delta_sz"]) > 0
    and float(x["delta_margin"]) > 0
    and float(x["maxfalse_drop"]) > 0
]

gold48.sort(
    key=lambda x: float(x["info_reward"]),
    reverse=True,
)

assert len(gold48) == 48, len(gold48)

EXCLUDE = {6, 20, 39}  # 2 META + 1 ambiguous

gold = [
    x for i, x in enumerate(gold48, 1)
    if i not in EXCLUDE
]

assert len(gold) == 45

gold_keys = {key(x) for x in gold}

gold_states = {
    (int(x["round"]), str(x["session_id"]))
    for x in gold
}

assert len(gold_states) == 29


# ---------------------------------------------------------
# Load ALL candidate contexts from those 29 states
# ---------------------------------------------------------

contexts = []

for rnd in range(1, 7):
    p = ROOT / f"train_rows_round_{rnd:02d}.jsonl"

    for x in read_jsonl(p):
        state_key = (
            int(x["round"]),
            str(x["session_id"]),
        )

        if state_key not in gold_states:
            continue

        contexts.append(x)

contexts.sort(
    key=lambda x: (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )
)

counts = Counter(
    (int(x["round"]), str(x["session_id"]))
    for x in contexts
)

print("=== CONTEXT INVENTORY ===")
print("gold states:", len(gold_states))
print("contexts:", len(contexts))
print(
    "states with exactly 4 candidates:",
    sum(v == 4 for v in counts.values()),
    "/",
    len(counts),
)
print(
    "Gold-trained contexts:",
    sum(key(x) in gold_keys for x in contexts),
)
print(
    "non-Gold contexts:",
    sum(key(x) not in gold_keys for x in contexts),
)

assert all(v == 4 for v in counts.values()), counts
assert len(contexts) == 29 * 4


# ---------------------------------------------------------
# Reconstruct public observation for each original context
# ---------------------------------------------------------

state_maps = {}
coverage_maps = {}

for rnd in sorted({int(x["round"]) for x in contexts}):

    raw_states = vg.read_jsonl(
        ROOT / f"states_train_round_{rnd-1:02d}.jsonl"
    )

    state_maps[rnd] = {
        str(x["session_id"]): x
        for x in raw_states
    }

    cov_ids, cov_mat = vg.load_matrix_npz(
        ROOT / f"coverage_train_round_{rnd-1:02d}.npz",
        "coverage",
    )

    coverage_maps[rnd] = {
        str(sid): cov_mat[i]
        for i, sid in enumerate(cov_ids)
    }


def make_obs(x):
    rnd = int(x["round"])
    sid = str(x["session_id"])
    j = int(x["candidate_index"])

    raw = state_maps[rnd][sid]

    st = vg.PublicBeliefState.from_public_dict(raw)
    st.validate(vg.TRAIN_ROLES, sorted(st.k_probs))

    vec = coverage_maps[rnd][sid]

    coverage = {
        r: float(vec[i])
        for i, r in enumerate(vg.TRAIN_ROLES)
    }

    return {
        "belief_state": st.to_public_dict(),
        "coverage": coverage,
        "candidate_index": j,
        "max_history_turns": 6,
    }


# ---------------------------------------------------------
# Recover exact original V3 generation seeds
# ---------------------------------------------------------

candidate_source_by_key = {}

for rnd in range(1, 7):
    p = ROOT / f"candidates_train_round_{rnd:02d}.jsonl"

    if not p.exists():
        raise FileNotFoundError(p)

    for x in read_jsonl(p):
        candidate_source_by_key[key(x)] = x

missing = [
    key(x)
    for x in contexts
    if key(x) not in candidate_source_by_key
]

assert not missing, f"missing original candidates: {missing[:5]}"

missing_seed = [
    key(x)
    for x in contexts
    if "seed" not in candidate_source_by_key[key(x)]
]

assert not missing_seed, f"missing seeds: {missing_seed[:5]}"

prompt_mismatch = [
    key(x)
    for x in contexts
    if str(x["prompt"])
    != str(candidate_source_by_key[key(x)]["prompt"])
]

assert not prompt_mismatch, (
    f"train/candidate prompt mismatch: {prompt_mismatch[:5]}"
)

print("original candidate seeds loaded:", len(contexts))
print("train/candidate prompts identical: YES")


# ---------------------------------------------------------
# Generate from one adapter
# ---------------------------------------------------------

def run_adapter(adapter, tag):

    print()
    print(f"=== LOADING {tag}: {adapter} ===", flush=True)

    policy = vg.CoverageEliminationProbePolicyV3(
        "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path=adapter,
        trainable=False,
        use_4bit=False,
        max_seq_length=4096,
    )

    out = []

    with torch.no_grad():

        for n, orig in enumerate(contexts, 1):

            obs = make_obs(orig)

            # Use the EXACT original V3 final generation seed.
            seed = int(candidate_source_by_key[key(orig)]["seed"])

            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            sample = policy.sample_probe(
                obs,
                temperature=0.85,
                top_p=0.95,
                max_new_tokens=192,
            )

            prompt = str(sample.prompt)
            q = str(sample.query or "").strip()

            # Critical causal-control check:
            # reconstructed context must be byte-identical
            # to the V3 prompt.
            if prompt != str(orig["prompt"]):
                raise RuntimeError(
                    f"PROMPT MISMATCH: "
                    f"round={orig['round']} "
                    f"sid={orig['session_id']} "
                    f"j={orig['candidate_index']}"
                )

            rec = {
                "round": int(orig["round"]),
                "session_id": str(orig["session_id"]),
                "candidate_index": int(orig["candidate_index"]),
                "is_gold_training_context": key(orig) in gold_keys,

                "seed": seed,
                "prompt": prompt,

                "original_query": str(candidate_source_by_key[key(orig)]["query"]),
                "query": q,

                "sample_valid": bool(sample.valid),

                "completion": str(sample.completion),
                "raw_completion": str(sample.raw_completion),

                "adapter": adapter,
                "tag": tag,
            }

            out.append(rec)

            if n % 20 == 0 or n == len(contexts):
                print(
                    f"{tag}: {n}/{len(contexts)}",
                    flush=True,
                )

    del policy
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


epoch2 = run_adapter(EPOCH2, "epoch2")
v4 = run_adapter(V4, "v4")

assert len(epoch2) == len(v4) == len(contexts)


# ---------------------------------------------------------
# Compare
# ---------------------------------------------------------

comparisons = []

same_prompt = 0
same_query = 0

seen_same = 0
seen_n = 0

unseen_same = 0
unseen_n = 0

epoch2_reproduced = 0

for a, b in zip(epoch2, v4):

    assert (
        a["round"],
        a["session_id"],
        a["candidate_index"],
    ) == (
        b["round"],
        b["session_id"],
        b["candidate_index"],
    )

    prompt_same = a["prompt"] == b["prompt"]
    query_same = a["query"] == b["query"]

    same_prompt += int(prompt_same)
    same_query += int(query_same)

    reproduced = (
        a["query"].strip()
        == a["original_query"].strip()
    )

    epoch2_reproduced += int(reproduced)

    seen = bool(a["is_gold_training_context"])

    if seen:
        seen_n += 1
        seen_same += int(query_same)
    else:
        unseen_n += 1
        unseen_same += int(query_same)

    comparisons.append({
        "round": a["round"],
        "session_id": a["session_id"],
        "candidate_index": a["candidate_index"],
        "is_gold_training_context": seen,
        "seed": a["seed"],

        "prompt_identical": prompt_same,
        "epoch2_reproduces_original": reproduced,

        "original_query": a["original_query"],
        "epoch2_query": a["query"],
        "v4_query": b["query"],

        "query_identical": query_same,

        "epoch2_valid": a["sample_valid"],
        "v4_valid": b["sample_valid"],
    })


write_jsonl(OUT / "epoch2.jsonl", epoch2)
write_jsonl(OUT / "v4.jsonl", v4)
write_jsonl(OUT / "comparison.jsonl", comparisons)


print()
print("=== A/B SUMMARY ===")
print("contexts:", len(comparisons))
print(
    "identical prompts:",
    f"{same_prompt}/{len(comparisons)}",
)

print(
    "epoch2 exact reproduction:",
    f"{epoch2_reproduced}/{len(comparisons)}",
    f"= {epoch2_reproduced/len(comparisons):.3f}",
)

print(
    "Epoch2 == V4 query:",
    f"{same_query}/{len(comparisons)}",
    f"= {same_query/len(comparisons):.3f}",
)

print(
    "Gold contexts changed:",
    f"{seen_n-seen_same}/{seen_n}",
    f"= {(seen_n-seen_same)/seen_n:.3f}",
)

print(
    "non-Gold contexts changed:",
    f"{unseen_n-unseen_same}/{unseen_n}",
    f"= {(unseen_n-unseen_same)/unseen_n:.3f}",
)

print(
    "Epoch2 valid:",
    sum(x["epoch2_valid"] for x in comparisons),
    "/",
    len(comparisons),
)

print(
    "V4 valid:",
    sum(x["v4_valid"] for x in comparisons),
    "/",
    len(comparisons),
)

print("wrote:", OUT)
