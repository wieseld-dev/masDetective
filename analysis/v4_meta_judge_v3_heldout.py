#!/usr/bin/env python3

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

# IMPORTANT:
# Reuse the exact frozen V3 prompt/schema.
from analysis.v4_meta_judge_v3 import SYSTEM_PROMPT, SCHEMA


DATA = Path("artifacts/v4_meta_heldout/heldout_set.jsonl")
HUMAN = Path("artifacts/v4_meta_heldout/human_labels.jsonl")

OUT = Path("artifacts/v4_meta_judge_v3_heldout")
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = OUT / "judgments.jsonl"
SUMMARY = OUT / "summary.json"


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )


async def judge_one(client, sem, model, row):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Classify this query:\n\n" + row["query"],
                },
            ],
            response_format=SCHEMA,
        )

        obj = json.loads(resp.choices[0].message.content)

        return {
            "round": row["round"],
            "session_id": row["session_id"],
            "candidate_index": row["candidate_index"],
            "query": row["query"],
            "bucket": row["bucket"],
            "reward": float(row["reward"]),
            **obj,
            "usage": {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        }


def metrics(pairs):
    tp = fn = fp = tn = 0

    for h, j in pairs:
        ht = h["human_label"]
        jp = j["label"]

        if ht == "CLEAR_META" and jp == "META":
            tp += 1
        elif ht == "CLEAR_META" and jp == "NON_META":
            fn += 1
        elif ht == "CLEAR_NON_META" and jp == "META":
            fp += 1
        elif ht == "CLEAR_NON_META" and jp == "NON_META":
            tn += 1

    def div(a, b):
        return a / b if b else float("nan")

    return {
        "N": len(pairs),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": div(tp + tn, len(pairs)),
        "meta_precision": div(tp, tp + fp),
        "meta_recall": div(tp, tp + fn),
        "nonmeta_precision": div(tn, tn + fn),
    }


def print_metrics(title, m):
    print()
    print(title)
    print(
        "                 Judge META   Judge NON_META"
    )
    print(
        f"Human META          {m['tp']:3d}             {m['fn']:3d}"
    )
    print(
        f"Human NON_META      {m['fp']:3d}             {m['tn']:3d}"
    )
    print()
    print(f"N                 : {m['N']}")
    print(f"accuracy          : {m['accuracy']:.3f}")
    print(f"META precision    : {m['meta_precision']:.3f}")
    print(f"META recall       : {m['meta_recall']:.3f}")
    print(f"NON_META precision: {m['nonmeta_precision']:.3f}")


def evaluate():
    data = read_jsonl(DATA)
    human = read_jsonl(HUMAN)
    judged = read_jsonl(RESULTS)

    dm = {key(x): x for x in data}
    hm = {key(x): x for x in human}
    jm = {key(x): x for x in judged if "error" not in x}

    assert len(dm) == 80, len(dm)
    assert len(hm) == 80, len(hm)
    assert len(jm) == 80, len(jm)

    pairs = []

    for k, h in hm.items():
        if h["human_label"] == "AMBIGUOUS":
            continue
        pairs.append((h, jm[k]))

    overall = metrics(pairs)
    print_metrics(
        "=== FROZEN V3 / HELD-OUT HUMAN TEST ===",
        overall
    )

    # Bucket-specific evaluation
    print()
    print("=== BY REWARD BUCKET ===")
    print(
        "Bucket  N | MetaN | Recall | PrecMeta | "
        "NonMetaPrec | FN"
    )
    print(
        "----------+-------+--------+----------+"
        "-------------+---"
    )

    bucket_stats = {}

    for bucket in ["TOP", "MIDDLE", "RANDOM", "BOTTOM"]:
        z = [
            (h, j)
            for h, j in pairs
            if h["bucket"] == bucket
        ]

        m = metrics(z)
        meta_n = m["tp"] + m["fn"]
        bucket_stats[bucket] = m

        print(
            f"{bucket:8s} {m['N']:2d} | "
            f"{meta_n:5d} | "
            f"{m['meta_recall']:.3f} | "
            f"{m['meta_precision']:.3f}    | "
            f"{m['nonmeta_precision']:.3f}       | "
            f"{m['fn']:2d}"
        )

    dangerous = []

    for h, j in pairs:
        if (
            h["human_label"] == "CLEAR_META"
            and j["label"] == "NON_META"
        ):
            dangerous.append((h, j))

    dangerous.sort(
        key=lambda z: float(z[0]["reward"]),
        reverse=True
    )

    print()
    print("=== DANGEROUS HELD-OUT FALSE NEGATIVES ===")
    print("Human META, frozen V3 allowed NON_META")

    if not dangerous:
        print("NONE")
    else:
        for h, j in dangerous:
            print()
            print(
                f"{h['bucket']} "
                f"reward={float(h['reward']):+.5f} "
                f"judge_conf={j['confidence']} "
                f"asks_system={j['asks_about_system_itself']}"
            )
            print(h["query"])
            print("reason:", j["reason"])

    false_pos = []

    for h, j in pairs:
        if (
            h["human_label"] == "CLEAR_NON_META"
            and j["label"] == "META"
        ):
            false_pos.append((h, j))

    false_pos.sort(
        key=lambda z: float(z[0]["reward"]),
        reverse=True
    )

    print()
    print("=== HELD-OUT FALSE POSITIVES ===")

    if not false_pos:
        print("NONE")
    else:
        for h, j in false_pos:
            print()
            print(
                f"{h['bucket']} "
                f"reward={float(h['reward']):+.5f} "
                f"conf={j['confidence']}"
            )
            print(h["query"])

    # Same conservative gate as dev.
    allowed = []
    rejected = []

    for h, j in pairs:
        reject = (
            j["label"] == "META"
            or (
                j["label"] == "NON_META"
                and j["confidence"] == "LOW"
            )
        )

        (rejected if reject else allowed).append((h, j))

    allowed_meta = sum(
        h["human_label"] == "CLEAR_META"
        for h, _ in allowed
    )

    allowed_nonmeta = sum(
        h["human_label"] == "CLEAR_NON_META"
        for h, _ in allowed
    )

    meta_total = sum(
        h["human_label"] == "CLEAR_META"
        for h, _ in pairs
    )

    blocked_meta = sum(
        h["human_label"] == "CLEAR_META"
        for h, _ in rejected
    )

    gate_precision = (
        allowed_nonmeta / (allowed_nonmeta + allowed_meta)
        if allowed_nonmeta + allowed_meta
        else float("nan")
    )

    gate_recall = (
        blocked_meta / meta_total
        if meta_total
        else float("nan")
    )

    print()
    print("=== CONSERVATIVE GATE / HELD-OUT ===")
    print("Reject META + LOW-confidence NON_META")
    print(f"allowed clear queries : {len(allowed)}")
    print(f"allowed true META     : {allowed_meta}")
    print(f"allowed precision     : {gate_precision:.3f}")
    print(f"META blocked recall   : {gate_recall:.3f}")

    total_in = sum(
        x.get("usage", {}).get("input_tokens", 0)
        for x in judged
    )
    total_out = sum(
        x.get("usage", {}).get("output_tokens", 0)
        for x in judged
    )

    summary = {
        "overall": overall,
        "by_bucket": bucket_stats,
        "dangerous_false_negatives": len(dangerous),
        "gate_allowed_true_meta": allowed_meta,
        "gate_precision": gate_precision,
        "gate_meta_recall": gate_recall,
        "input_tokens": total_in,
        "output_tokens": total_out,
    }

    with SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("tokens:", total_in, "input /", total_out, "output")
    print("wrote:", SUMMARY)


async def run(args):
    rows = read_jsonl(DATA)
    assert len(rows) == 80, len(rows)

    existing = read_jsonl(RESULTS)
    done = {
        key(x)
        for x in existing
        if "error" not in x
    }

    pending = [
        x for x in rows
        if key(x) not in done
    ]

    print("held-out examples:", len(rows))
    print("already judged:", len(done))
    print("pending:", len(pending))

    if pending:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)

        tasks = [
            judge_one(client, sem, args.model, row)
            for row in pending
        ]

        n = 0

        for fut in asyncio.as_completed(tasks):
            try:
                rec = await fut
            except Exception as e:
                rec = {"error": repr(e)}

            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(rec, ensure_ascii=False) + "\n"
                )

            n += 1
            if n % 10 == 0 or n == len(pending):
                print(f"completed {n}/{len(pending)}")

    evaluate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18"
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=12
    )

    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
