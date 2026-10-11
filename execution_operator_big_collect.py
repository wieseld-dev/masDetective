#!/usr/bin/env python3

import argparse
import asyncio
import itertools
import json
import random
from pathlib import Path

from openai import AsyncOpenAI

import blackbox_commutator_poc as base
from operator3_poc import load_v11_probes
from router_r1 import ROLES


SPECIALISTS = "ABC"

PREFIX_SEQS = {
    k: ["".join(p) for p in itertools.permutations(SPECIALISTS, k)]
    for k in (1, 2, 3)
}


async def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--n-selected", type=int, default=160)
    ap.add_argument("--seed", type=int, default=20260916)

    ap.add_argument(
        "--cache",
        default="artifacts/execution_operator_big/api_runs.jsonl"
    )

    ap.add_argument(
        "--model",
        default="gpt-4o-mini"
    )

    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=300)

    args = ap.parse_args()

    all_tasks = load_v11_probes(217)

    ids = list(range(len(all_tasks)))

    rng = random.Random(args.seed)
    rng.shuffle(ids)

    ids = ids[:args.n_selected]

    selected = [
        (pos, src_id, all_tasks[src_id])
        for pos, src_id in enumerate(ids)
    ]

    split_file = Path(
        "artifacts/execution_operator_big/split.json"
    )

    split_file.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "selected_source_ids": ids,
                "train_positions": list(range(100)),
                "test_positions": list(range(100, args.n_selected)),
            },
            indent=2
        ),
        encoding="utf-8"
    )

    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)

    done = {}

    if cache.exists():
        for line in cache.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[int(r["position"])] = r

    missing = [
        x for x in selected
        if x[0] not in done
    ]

    print(
        f"[data] selected={len(selected)} "
        f"cached={len(done)} "
        f"missing={len(missing)}"
    )

    if not missing:
        print("COLLECTION ALREADY COMPLETE")
        return

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

    async def one(position, source_id, task):

        outputs = {}
        total_in = 0
        total_out = 0

        async def save_call(role, previous):

            nonlocal total_in, total_out

            r = await call(
                role,
                task,
                previous
            )

            total_in += r["input_tokens"]
            total_out += r["output_tokens"]

            return r["text"]

        # --------------------------------------------------
        # Level 0:
        # exact root / singleton states
        # --------------------------------------------------

        roots = await asyncio.gather(
            save_call("R", None),
            save_call("A", None),
            save_call("B", None),
            save_call("C", None),
        )

        for key, text in zip(
            ["R", "A", "B", "C"],
            roots
        ):
            outputs[key] = text

        # --------------------------------------------------
        # Exact shared-prefix trajectories after R
        # --------------------------------------------------

        for depth in (1, 2, 3):

            async def build_seq(seq):

                role = seq[-1]

                if depth == 1:
                    parent = "R"
                else:
                    parent = "R" + seq[:-1]

                key = "R" + seq

                text = await save_call(
                    role,
                    outputs[parent]
                )

                return key, text

            vals = await asyncio.gather(
                *[
                    build_seq(seq)
                    for seq in PREFIX_SEQS[depth]
                ]
            )

            for key, text in vals:
                outputs[key] = text

        # --------------------------------------------------
        # M transitions.
        #
        # Crucially, these consume the EXACT saved parents.
        # --------------------------------------------------

        m_parents = [
            "R",
            "A", "B", "C",
        ]

        for depth in (1, 2, 3):
            m_parents.extend(
                [
                    "R" + seq
                    for seq in PREFIX_SEQS[depth]
                ]
            )

        async def make_final(parent):

            text = await save_call(
                "M",
                outputs[parent]
            )

            return parent + "M", text

        vals = await asyncio.gather(
            *[
                make_final(parent)
                for parent in m_parents
            ]
        )

        for key, text in vals:
            outputs[key] = text

        row = {
            "position": position,
            "source_id": source_id,
            "task": task,
            "outputs": outputs,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

        async with lock:
            with cache.open(
                "a",
                encoding="utf-8"
            ) as f:

                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False
                    )
                    + "\n"
                )

        print(
            f"[api] pos={position+1}/{len(selected)} "
            f"source={source_id} "
            f"in={total_in} out={total_out}"
        )

        return row

    await asyncio.gather(
        *[
            one(*x)
            for x in missing
        ]
    )

    print()
    print("=" * 70)
    print("BIG CAUSAL TRAJECTORY COLLECTION COMPLETE")
    print("=" * 70)
    print("Selected prompts:", len(selected))
    print("Train: 100")
    print("Test:", len(selected) - 100)


if __name__ == "__main__":
    asyncio.run(main())
