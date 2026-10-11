#!/usr/bin/env python3

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

import blackbox_commutator_poc as base
from operator3_poc import load_v11_probes
from router_r1 import ROLES


ANCHORS = ["R", "A", "B", "C"]


async def collect(args):

    tasks = load_v11_probes(args.n_prompts)

    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)

    done = {}

    if cache.exists():
        for line in cache.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[int(r["id"])] = r

    missing = [
        (i, q)
        for i, q in enumerate(tasks)
        if i not in done
    ]

    print(
        f"[anchor-data] total={len(tasks)} "
        f"cached={len(done)} "
        f"missing={len(missing)}"
    )

    if not missing:
        return [done[i] for i in range(len(tasks))]

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def call(role, task):
        return await base.call_agent(
            client,
            sem,
            args.model,
            ROLES[role],
            task,
            None,
            args.max_tokens,
        )

    async def one(i, task):

        outputs = {}
        total_in = 0
        total_out = 0

        rs = await asyncio.gather(
            *[
                call(role, task)
                for role in ANCHORS
            ]
        )

        for role, res in zip(ANCHORS, rs):
            outputs[role] = res["text"]

            total_in += res["input_tokens"]
            total_out += res["output_tokens"]

        row = {
            "id": i,
            "task": task,
            "outputs": outputs,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

        async with lock:
            with cache.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False
                    ) + "\n"
                )

        print(
            f"[anchor-api] {i+1}/{len(tasks)} "
            f"in={total_in} out={total_out}"
        )

        return row

    new = await asyncio.gather(
        *[
            one(i, task)
            for i, task in missing
        ]
    )

    for r in new:
        done[int(r["id"])] = r

    return [done[i] for i in range(len(tasks))]


async def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--n-prompts",
        type=int,
        default=60
    )

    ap.add_argument(
        "--cache",
        default="artifacts/router_anchor/anchors.jsonl"
    )

    ap.add_argument(
        "--model",
        default="gpt-4o-mini"
    )

    ap.add_argument(
        "--concurrency",
        type=int,
        default=16
    )

    ap.add_argument(
        "--max-tokens",
        type=int,
        default=300
    )

    args = ap.parse_args()

    rows = await collect(args)

    print()
    print("=" * 60)
    print("ANCHOR COLLECTION COMPLETE")
    print("=" * 60)

    print("DEV probes:", len(rows))

    print(
        "input tokens:",
        sum(r["input_tokens"] for r in rows)
    )

    print(
        "output tokens:",
        sum(r["output_tokens"] for r in rows)
    )


if __name__ == "__main__":
    asyncio.run(main())
