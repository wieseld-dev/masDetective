#!/usr/bin/env python3

import argparse
import asyncio
import json
from pathlib import Path
from collections import defaultdict

from openai import AsyncOpenAI

from analysis.v4_meta_judge_v3 import SYSTEM_PROMPT, SCHEMA

ROOT = Path("artifacts/v3_coverage_elimination_pilot")
RELABEL = ROOT / "membership_relabel_candidates.jsonl"

OUTDIR = Path("artifacts/v4_generator")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT = OUTDIR / "candidate_judgments.jsonl"
SUMMARY = OUTDIR / "candidate_audit.json"


def read_jsonl(p):
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )


def load_queries():
    qmap = {}

    for r in range(1, 7):
        p = ROOT / f"responses_train_round_{r:02d}.jsonl"
        for x in read_jsonl(p):
            qmap[key(x)] = str(x["query"]).strip()

    return qmap


async def judge_one(client, sem, model, x):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Classify this query:\n\n" + x["query"],
                },
            ],
            response_format=SCHEMA,
        )

        j = json.loads(resp.choices[0].message.content)

        return {
            **x,
            **j,
        }


async def main(args):
    relabel = read_jsonl(RELABEL)
    assert len(relabel) == 384, len(relabel)

    qmap = load_queries()

    rows = []

    for x in relabel:
        k = key(x)
        assert k in qmap, k

        dsz = float(x["delta_sz"])
        dm = float(x["delta_margin"])
        info = dsz + dm

        rows.append({
            "round": int(x["round"]),
            "session_id": str(x["session_id"]),
            "candidate_index": int(x["candidate_index"]),
            "query": qmap[k],
            "delta_sz": dsz,
            "delta_margin": dm,
            "maxfalse_drop": float(x["maxfalse_drop"]),
            "info_reward": info,
        })

    existing = read_jsonl(OUT)

    done = {
        key(x)
        for x in existing
        if "error" not in x
    }

    pending = [
        x for x in rows
        if key(x) not in done
    ]

    print("candidates:", len(rows))
    print("already judged:", len(done))
    print("pending:", len(pending))

    if pending:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)

        tasks = [
            judge_one(client, sem, args.model, x)
            for x in pending
        ]

        n = 0

        for fut in asyncio.as_completed(tasks):
            try:
                rec = await fut
            except Exception as e:
                rec = {"error": repr(e)}

            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n += 1
            if n % 25 == 0 or n == len(pending):
                print(f"completed {n}/{len(pending)}")

    judged = [
        x for x in read_jsonl(OUT)
        if "error" not in x
    ]

    assert len(judged) == 384, len(judged)

    groups = defaultdict(list)

    for x in judged:
        groups[(x["round"], x["session_id"])].append(x)

    assert len(groups) == 96, len(groups)

    meta_n = sum(x["label"] == "META" for x in judged)
    clean_n = len(judged) - meta_n

    pools_clean = []
    clean_counts = []

    all_best_info = []
    clean_best_info = []

    all_best_dm = []
    clean_best_dm = []

    all_best_maxf = []
    clean_best_maxf = []

    lost_all_clean = 0

    for g, xs in groups.items():
        clean = [x for x in xs if x["label"] == "NON_META"]

        clean_counts.append(len(clean))
        pools_clean.append(len(clean) > 0)

        ba = max(xs, key=lambda z: z["info_reward"])

        all_best_info.append(ba["info_reward"])
        all_best_dm.append(ba["delta_margin"])
        all_best_maxf.append(ba["maxfalse_drop"])

        if clean:
            bc = max(clean, key=lambda z: z["info_reward"])

            clean_best_info.append(bc["info_reward"])
            clean_best_dm.append(bc["delta_margin"])
            clean_best_maxf.append(bc["maxfalse_drop"])
        else:
            lost_all_clean += 1

    import statistics as st

    def mean(xs):
        return st.mean(xs) if xs else float("nan")

    print()
    print("=== FROZEN JUDGE V3 / ALL 384 ===")
    print(f"META candidates       : {meta_n}/384 = {meta_n/384:.3f}")
    print(f"NON_META candidates   : {clean_n}/384 = {clean_n/384:.3f}")
    print(f"pools >=1 clean       : {sum(pools_clean)}/96 = {mean(pools_clean):.3f}")
    print(f"mean clean / pool     : {mean(clean_counts):.3f}")
    print(f"zero-clean pools      : {lost_all_clean}")

    print()
    print("=== INFORMATION HEADROOM ===")
    print(f"ALL best info         : {mean(all_best_info):+.5f}")
    print(f"CLEAN best info       : {mean(clean_best_info):+.5f}")
    print(f"ALL best dMargin      : {mean(all_best_dm):+.5f}")
    print(f"CLEAN best dMargin    : {mean(clean_best_dm):+.5f}")
    print(f"ALL best maxF drop    : {mean(all_best_maxf):+.5f}")
    print(f"CLEAN best maxF drop  : {mean(clean_best_maxf):+.5f}")

    retention = (
        mean(clean_best_info) / mean(all_best_info)
        if mean(all_best_info) != 0
        else float("nan")
    )

    print(f"clean info retention  : {retention:.3f}")

    summary = {
        "meta_rate": meta_n / 384,
        "clean_rate": clean_n / 384,
        "pools_with_clean_rate": mean(pools_clean),
        "mean_clean_per_pool": mean(clean_counts),
        "zero_clean_pools": lost_all_clean,
        "all_best_info": mean(all_best_info),
        "clean_best_info": mean(clean_best_info),
        "all_best_margin": mean(all_best_dm),
        "clean_best_margin": mean(clean_best_dm),
        "all_best_maxfalse_drop": mean(all_best_maxf),
        "clean_best_maxfalse_drop": mean(clean_best_maxf),
        "clean_info_retention": retention,
    }

    with SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("wrote:", SUMMARY)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18"
    )
    ap.add_argument("--concurrency", type=int, default=12)

    asyncio.run(main(ap.parse_args()))
