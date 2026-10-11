#!/usr/bin/env python3

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

import blackbox_commutator_poc as base
from operator3_poc import load_v11_probes


ROLES = {
"R": """You are a preprocessing and task-framing agent.
Read the user's task and rewrite the working answer so that the core problem,
constraints, ambiguities, and decision points are explicit.
Do not solve only one narrow aspect of the task.
Return only the revised working answer.
Keep it under 180 words.""",

"A": """You are a specialist analyst focused on requirements and constraints.
Given the user's task and possibly a previous draft:
- identify requirements, assumptions, dependencies, and edge cases,
- resolve missing or conflicting constraints,
- improve the draft while preserving useful reasoning already present.
Return only the revised answer.
Keep it under 180 words.""",

"B": """You are a specialist analyst focused on feasibility and failure risk.
Given the user's task and possibly a previous draft:
- examine whether the proposed reasoning is workable,
- identify failure modes, tradeoffs, dependencies, and weak assumptions,
- improve the draft while preserving useful reasoning already present.
Return only the revised answer.
Keep it under 180 words.""",

"C": """You are a specialist analyst focused on verification and consistency.
Given the user's task and possibly a previous draft:
- check claims, reasoning steps, assumptions, and internal consistency,
- identify contradictions, unsupported conclusions, and edge cases,
- improve the draft while preserving useful reasoning already present.
Return only the revised answer.
Keep it under 180 words.""",

"M": """You are a final synthesis agent.
Given the user's original task and a previous working answer:
- synthesize it into one coherent final answer,
- preserve the substantive information,
- resolve inconsistencies without introducing a new specialist perspective,
- do not mention agents, routing, drafts, or internal processing.
Return only the final answer.
Keep it under 180 words."""
}

# TRAIN compositions: only length-2 systems.
PAIRS = ["RA", "RB", "RC", "AM", "BM", "CM"]

# NEVER used for operator training.
PATHS = ["RAM", "RBM", "RCM"]


async def collect(args, tasks):
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)

    done = {}

    if cache.exists():
        for line in cache.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[int(r["id"])] = r

    missing = [(i, q) for i, q in enumerate(tasks) if i not in done]

    print(
        f"[data] total={len(tasks)} "
        f"cached={len(done)} "
        f"missing={len(missing)}"
    )

    if not missing:
        return [done[i] for i in range(len(tasks))]

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def call(role, task, previous):
        return await base.call_agent(
            client,
            sem,
            args.model,
            ROLES[role],
            task,
            previous,
            args.max_tokens,
        )

    async def run_comp(comp, task):
        previous = None
        tin = 0
        tout = 0

        for role in comp:
            res = await call(role, task, previous)
            previous = res["text"]
            tin += res["input_tokens"]
            tout += res["output_tokens"]

        return comp, previous, tin, tout

    async def one(i, task):
        outputs = {}
        total_in = 0
        total_out = 0

        # IMPORTANT:
        # Every composition is executed independently.
        # No intermediate output is reused across compositions.
        comps = PAIRS + PATHS

        rs = await asyncio.gather(
            *[run_comp(c, task) for c in comps]
        )

        for comp, text, tin, tout in rs:
            outputs[comp] = text
            total_in += tin
            total_out += tout

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
            f"[api] {i+1}/{len(tasks)} complete "
            f"in={total_in} out={total_out}"
        )

        return row

    new = await asyncio.gather(
        *[one(i, q) for i, q in missing]
    )

    for r in new:
        done[int(r["id"])] = r

    return [done[i] for i in range(len(tasks))]


async def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--n-prompts", type=int, default=217)
    ap.add_argument(
        "--cache",
        default="artifacts/router_r1/api_runs.jsonl"
    )
    ap.add_argument(
        "--model",
        default="gpt-4o-mini"
    )
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=300)

    args = ap.parse_args()

    tasks = load_v11_probes(args.n_prompts)

    rows = await collect(args, tasks)

    tin = sum(r["input_tokens"] for r in rows)
    tout = sum(r["output_tokens"] for r in rows)

    print()
    print("COLLECTION COMPLETE")
    print("probes:", len(rows))
    print("input tokens:", tin)
    print("output tokens:", tout)


if __name__ == "__main__":
    asyncio.run(main())
