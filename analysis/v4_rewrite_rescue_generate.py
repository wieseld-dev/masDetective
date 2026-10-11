#!/usr/bin/env python3

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI
from analysis.v4_meta_judge_v3 import SYSTEM_PROMPT, SCHEMA

INP = Path("artifacts/v4_generator/candidate_judgments.jsonl")
OUTDIR = Path("artifacts/v4_rewrite_rescue")
OUTDIR.mkdir(parents=True, exist_ok=True)

SOURCES = OUTDIR / "sources.jsonl"
RESULTS = OUTDIR / "rewrites.jsonl"


REWRITE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "natural_probe_rewrites",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "rewrite_1": {"type": "string"},
                "rewrite_2": {"type": "string"},
                "rewrite_3": {"type": "string"},
            },
            "required": [
                "rewrite_1",
                "rewrite_2",
                "rewrite_3",
            ],
            "additionalProperties": False,
        },
    },
}


REWRITE_PROMPT = r"""
You are rewriting diagnostic questions for a research experiment.

The original question may directly ask an AI/system about its capabilities,
experience, expertise, knowledge, workflow, or identity.

Rewrite it into THREE ordinary, realistic end-user domain/task questions.

The goal is to preserve the SAME underlying diagnostic pressure: a respondent
with the relevant skill should naturally produce a measurably different or
better answer. However, the rewritten query must look like a normal task or
domain question.

STRICT RULES:

1. Do NOT ask what the AI/system/assistant can do.
2. Do NOT ask about its capabilities, expertise, experience, knowledge,
   internal workflow, architecture, identity, roles, or processes.
3. Avoid formulations such as:
   "Do you...", "Can you..." when referring to the respondent,
   "your experience", "your expertise", "your capabilities",
   "these capabilities", "high-probability capabilities",
   "least-resolved capabilities".
4. Do NOT reveal that the purpose is to identify hidden agents or capabilities.
5. Create a concrete domain problem, decision, scenario, analysis request,
   explanation request, or task where the same underlying skill is useful.
6. Do not unnecessarily introduce unrelated skills.
7. Each rewrite must be standalone.
8. Keep each rewrite concise: preferably one or two sentences.

Example:

Original:
"Do you have experience designing databases?"

Good rewrite:
"A service must store customers, orders, and audit history while avoiding
duplicated data. How would you structure the main entities and relationships?"

Bad rewrite:
"Can you explain your database-design experience?"
"""


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def gkey(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
    )


def source_key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )


def prepare_sources():
    if SOURCES.exists():
        rows = read_jsonl(SOURCES)
        print("using existing sources:", len(rows))
        return rows

    rows = [
        x for x in read_jsonl(INP)
        if "error" not in x
    ]
    assert len(rows) == 384, len(rows)

    groups = defaultdict(list)
    for x in rows:
        groups[gkey(x)].append(x)

    candidates = []

    for g, xs in groups.items():
        winner = max(xs, key=lambda z: z["info_reward"])

        if winner["label"] != "META":
            continue

        clean = [
            x for x in xs
            if x["label"] == "NON_META"
        ]

        if clean:
            clean_best = max(
                clean,
                key=lambda z: z["info_reward"]
            )
            clean_score = float(
                clean_best["info_reward"]
            )
            gap = (
                float(winner["info_reward"])
                - clean_score
            )
            zero_clean = False
        else:
            clean_score = None
            gap = None
            zero_clean = True

        candidates.append({
            "round": int(winner["round"]),
            "session_id": str(winner["session_id"]),
            "candidate_index": int(
                winner["candidate_index"]
            ),
            "query": winner["query"],
            "info_reward": float(
                winner["info_reward"]
            ),
            "delta_sz": float(
                winner["delta_sz"]
            ),
            "delta_margin": float(
                winner["delta_margin"]
            ),
            "maxfalse_drop": float(
                winner["maxfalse_drop"]
            ),
            "zero_clean_pool": zero_clean,
            "best_clean_info": clean_score,
            "clean_gap": gap,
        })

    # First rescue the states with no clean candidate at all.
    zero = sorted(
        [
            x for x in candidates
            if x["zero_clean_pool"]
            and x["info_reward"] > 0
        ],
        key=lambda x: x["info_reward"],
        reverse=True,
    )

    selected = zero[:20]

    # Fill remaining slots with the biggest loss caused by the gate.
    if len(selected) < 20:
        used = {
            source_key(x)
            for x in selected
        }

        rest = [
            x for x in candidates
            if not x["zero_clean_pool"]
            and source_key(x) not in used
        ]

        rest.sort(
            key=lambda x: x["clean_gap"],
            reverse=True,
        )

        selected.extend(
            rest[:20 - len(selected)]
        )

    assert len(selected) == 20, len(selected)

    with SOURCES.open("w", encoding="utf-8") as f:
        for i, x in enumerate(selected, 1):
            x = {**x, "rescue_id": i}
            f.write(
                json.dumps(x, ensure_ascii=False)
                + "\n"
            )

    selected = read_jsonl(SOURCES)

    print("selected:", len(selected))
    print(
        "zero-clean selected:",
        sum(x["zero_clean_pool"] for x in selected),
    )

    return selected


async def judge(client, sem, q):
    async with sem:
        r = await client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content":
                    "Classify this query:\n\n" + q,
                },
            ],
            response_format=SCHEMA,
        )

        return json.loads(
            r.choices[0].message.content
        )


async def process_source(client, sem, model, src):
    async with sem:
        r = await client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": REWRITE_PROMPT,
                },
                {
                    "role": "user",
                    "content":
                    "Original diagnostic query:\n\n"
                    + src["query"],
                },
            ],
            response_format=REWRITE_SCHEMA,
        )

    obj = json.loads(
        r.choices[0].message.content
    )

    rewrites = [
        obj["rewrite_1"].strip(),
        obj["rewrite_2"].strip(),
        obj["rewrite_3"].strip(),
    ]

    judgments = await asyncio.gather(
        *[
            judge(client, sem, q)
            for q in rewrites
        ]
    )

    return {
        **src,
        "rewrites": [
            {
                "rewrite_index": i + 1,
                "query": q,
                **j,
            }
            for i, (q, j)
            in enumerate(zip(rewrites, judgments))
        ],
    }


async def main(args):
    sources = prepare_sources()

    existing = read_jsonl(RESULTS)
    done = {
        int(x["rescue_id"])
        for x in existing
        if "error" not in x
    }

    pending = [
        x for x in sources
        if int(x["rescue_id"]) not in done
    ]

    print("already done:", len(done))
    print("pending:", len(pending))

    client = AsyncOpenAI(
        timeout=30.0,
        max_retries=2,
    )

    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        process_source(
            client,
            sem,
            args.model,
            x,
        )
        for x in pending
    ]

    n = 0

    for fut in asyncio.as_completed(tasks):
        try:
            rec = await fut
        except Exception as e:
            rec = {"error": repr(e)}

        with RESULTS.open(
            "a",
            encoding="utf-8"
        ) as f:
            f.write(
                json.dumps(
                    rec,
                    ensure_ascii=False
                ) + "\n"
            )

        n += 1
        print(f"completed {n}/{len(pending)}")

    rows = [
        x for x in read_jsonl(RESULTS)
        if "error" not in x
    ]

    print()
    print("=== REWRITE RESCUE AUDIT ===")

    total = 0
    clean = 0
    at_least_1 = 0
    at_least_2 = 0
    all_3 = 0

    for x in rows:
        z = [
            r for r in x["rewrites"]
            if r["label"] == "NON_META"
        ]

        total += len(x["rewrites"])
        clean += len(z)

        at_least_1 += len(z) >= 1
        at_least_2 += len(z) >= 2
        all_3 += len(z) == 3

    print(
        f"source probes          : {len(rows)}"
    )
    print(
        f"clean rewrites         : "
        f"{clean}/{total} = {clean/total:.3f}"
    )
    print(
        f"sources >=1 clean      : "
        f"{at_least_1}/{len(rows)} = "
        f"{at_least_1/len(rows):.3f}"
    )
    print(
        f"sources >=2 clean      : "
        f"{at_least_2}/{len(rows)} = "
        f"{at_least_2/len(rows):.3f}"
    )
    print(
        f"sources all 3 clean    : "
        f"{all_3}/{len(rows)} = "
        f"{all_3/len(rows):.3f}"
    )

    print()
    print("=== ACCEPTED REWRITES ===")

    for x in sorted(
        rows,
        key=lambda z: z["rescue_id"]
    ):
        clean_rs = [
            r for r in x["rewrites"]
            if r["label"] == "NON_META"
        ]

        print()
        print(
            f"[{x['rescue_id']:02d}] "
            f"original R={x['info_reward']:+.5f}"
        )
        print("META:", x["query"])

        if not clean_rs:
            print("  -> NO CLEAN REWRITE")
            continue

        for r in clean_rs:
            print(
                f"  CLEAN {r['rewrite_index']}: "
                f"{r['query']}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18"
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=6
    )

    asyncio.run(
        main(ap.parse_args())
    )
