from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI

# IMPORTANT:
# Reuse the exact frozen V3 judge implementation/rubric.
import v4_meta_judge_v3 as frozen


SRC = Path("artifacts/v4_dpo_mining/same_prompt_samples.jsonl")
OUT = Path("artifacts/v4_dpo_mining")
JUDGED = OUT / "same_prompt_judgments.jsonl"
PAIRS = OUT / "provisional_dpo_pairs.jsonl"
SUMMARY = OUT / "pair_summary.json"


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


async def judge_one_safe(client, sem, model, src):
    # judge_one only needs the public query.
    inp = {
        "round": int(src["round"]),
        "session_id": str(src["session_id"]),
        "candidate_index": int(src["candidate_index"]),
        "query": str(src["sampled_query"]),
    }

    try:
        j = await frozen.judge_one(
            client,
            sem,
            model,
            inp,
        )
    except Exception as e:
        out = dict(src)
        out["error"] = f"{type(e).__name__}: {e}"
        return out

    out = dict(src)

    if not isinstance(j, dict):
        out["error"] = f"judge returned {type(j).__name__}"
        return out

    # Preserve mining row; attach all frozen-judge fields.
    for k, v in j.items():
        if k != "query":
            out[k] = v

    return out


async def run(args):
    OUT.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(SRC)

    assert len(rows) == 360, f"expected 360 mined rows, got {len(rows)}"

    valid = [x for x in rows if bool(x.get("valid"))]

    assert len(valid) == 327, (
        f"expected 327 valid mined rows, got {len(valid)}"
    )

    # Every alternative must already have passed exact same-prompt mining.
    assert all("prompt" in x for x in valid)
    assert all("chosen_completion_gold" in x for x in valid)
    assert all("sampled_completion" in x for x in valid)
    assert all(str(x["sampled_query"]).strip() for x in valid)

    print("=== FROZEN V3 JUDGE / DPO MINING ===", flush=True)
    print("mined rows :", len(rows), flush=True)
    print("valid rows :", len(valid), flush=True)
    print("model      :", args.model, flush=True)
    print("concurrency:", args.concurrency, flush=True)

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    judged = []

    # Batch so partial progress is preserved in the log.
    B = 25
    for start in range(0, len(valid), B):
        batch = valid[start:start+B]

        got = await asyncio.gather(*[
            judge_one_safe(client, sem, args.model, x)
            for x in batch
        ])

        judged.extend(got)

        print(
            f"[judge] {len(judged)}/{len(valid)}",
            flush=True,
        )

    write_jsonl(JUDGED, judged)

    good = [x for x in judged if "error" not in x]

    labels = Counter(str(x.get("label")) for x in good)
    confs = Counter(str(x.get("confidence")) for x in good)

    # Precision-first negative definition.
    #
    # We only want clear rejected examples:
    #   label == META
    #   confidence == HIGH
    #
    # If the frozen schema exposes asks_system, require it to be true.
    # This makes the negative set deliberately conservative.
    high_meta = []

    for x in good:
        if x.get("label") != "META":
            continue
        if x.get("confidence") != "HIGH":
            continue

        if "asks_system" in x and x["asks_system"] is not True:
            continue

        high_meta.append(x)

    by_gold = defaultdict(list)

    for x in high_meta:
        by_gold[int(x["gold_index"])].append(x)

    pairs = []

    for gold_index in sorted(by_gold):
        xs = by_gold[gold_index]

        # Deduplicate semantically identical generated query strings.
        uniq = []
        seen = set()

        for x in sorted(xs, key=lambda z: int(z["sample_index"])):
            q = " ".join(str(x["sampled_query"]).split())

            if not q or q in seen:
                continue

            seen.add(q)
            uniq.append(x)

        # Maximum 2 negatives per Gold prompt so one context cannot dominate.
        for rank, x in enumerate(uniq[:2], 1):
            chosen = str(x["chosen_completion_gold"])
            rejected = str(x["sampled_completion"])
            prompt = str(x["prompt"])

            assert chosen.strip()
            assert rejected.strip()
            assert chosen != rejected

            pairs.append({
                "gold_index": int(x["gold_index"]),
                "round": int(x["round"]),
                "session_id": str(x["session_id"]),
                "candidate_index": int(x["candidate_index"]),
                "negative_rank": rank,

                # DPO triple
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,

                # Human-readable query forms for audit
                "chosen_query_gold": str(
                    x.get("chosen_completion_gold", "")
                ),
                "rejected_query": str(x["sampled_query"]),

                # Frozen judge evidence
                "rejected_label": str(x["label"]),
                "rejected_confidence": str(x["confidence"]),
                "rejected_asks_system": x.get("asks_system"),
                "rejected_reason": x.get("reason"),

                "sample_index": int(x["sample_index"]),
                "seed": int(x["seed"]),
                "gold_reward": float(x.get("gold_reward", 0.0)),
                "gold_advantage": float(
                    x.get("gold_advantage", 0.0)
                ),
            })

    write_jsonl(PAIRS, pairs)

    prompts_with_1 = len(set(x["gold_index"] for x in pairs))

    pair_counts = Counter(x["gold_index"] for x in pairs)
    prompts_with_2 = sum(v >= 2 for v in pair_counts.values())

    summary = {
        "mined_total": len(rows),
        "judged_valid": len(valid),
        "successful_judgments": len(good),
        "failed_judgments": len(judged) - len(good),
        "labels": dict(labels),
        "confidences": dict(confs),
        "high_confidence_meta_negatives": len(high_meta),
        "gold_prompts_total": 45,
        "gold_prompts_with_at_least_1_negative": prompts_with_1,
        "gold_prompts_with_2_negatives": prompts_with_2,
        "dpo_pairs": len(pairs),
        "max_pairs_per_gold_prompt": 2,
        "judge_model": args.model,
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("======================================")
    print(" SAME-PROMPT DPO PAIR SUMMARY")
    print("======================================")
    print("successful judgments :", len(good))
    print("failed judgments     :", len(judged) - len(good))
    print("labels               :", dict(labels))
    print("confidence           :", dict(confs))
    print("HIGH META negatives  :", len(high_meta))
    print(
        "Gold prompts >=1 neg:",
        f"{prompts_with_1}/45"
    )
    print(
        "Gold prompts >=2 neg:",
        f"{prompts_with_2}/45"
    )
    print("DPO pairs            :", len(pairs))

    print()
    print("=== FIRST 15 PROVISIONAL PAIRS ===")

    for i, x in enumerate(pairs[:15], 1):
        print(
            f"\n--- {i} | gold={x['gold_index']} "
            f"r{x['round']} j{x['candidate_index']} ---"
        )
        print("CHOSEN  :", x["chosen_query_gold"])
        print("REJECTED:", x["rejected_query"])
        print(
            "judge   :",
            x["rejected_label"],
            x["rejected_confidence"],
            "asks_system=",
            x["rejected_asks_system"],
        )

    print()
    print("wrote:", JUDGED)
    print("wrote:", PAIRS)
    print("wrote:", SUMMARY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=12,
    )
    args = ap.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
