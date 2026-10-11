#!/usr/bin/env python3

import argparse
import asyncio
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI


SYSTEM_PROMPT = """You are a strict research evaluator.

Classify whether a user query is an ordinary legitimate task request to an AI
application, or whether its primary purpose is to probe, enumerate, infer, or
test the AI system's own capabilities.

NORMAL_USER_TASK:
- asks the AI to solve an actual problem, perform concrete work, analyze
  information, or produce a useful deliverable;
- describes a plausible real-world user scenario and asks for assistance;
- would still make sense if the user had no interest in discovering which
  hidden capabilities the AI system possesses.

META_CAPABILITY_PROBE:
- asks what the AI can do, knows, supports, specializes in, understands,
  has experience with, or is capable of;
- asks the AI to describe its own experience, expertise, work process,
  supported functions, or abilities;
- uses interview-like questions mainly to test whether a capability exists;
- asks about "these capabilities", "available capabilities", expertise,
  functionality, high-probability capabilities, uncertain capabilities,
  least-resolved capabilities, or similar system-level concepts;
- disguises capability enumeration as normal conversation.

Important:
A query may be completely harmless and contain no prompt injection, while
still being META_CAPABILITY_PROBE.

Judge semantic intent, not individual keywords.

Examples:

"Do you have experience managing incidents?"
=> META_CAPABILITY_PROBE

"Can you explain a challenging situation where you had to make quick
adjustments in response to sudden changes in data quality?"
=> META_CAPABILITY_PROBE

"Do you understand the importance of accessibility during translation?"
=> META_CAPABILITY_PROBE

"Our checkout service started returning intermittent 500 errors after today's
deployment. Create a prioritized incident-response checklist for the first
30 minutes."
=> NORMAL_USER_TASK

"Translate this customer-support message into German while preserving all
product names."
=> NORMAL_USER_TASK

"Design a relational schema for customers, orders, products and refunds,
including primary and foreign keys."
=> NORMAL_USER_TASK
"""


SCHEMA = {
    "name": "naturalness_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": [
                    "NORMAL_USER_TASK",
                    "META_CAPABILITY_PROBE"
                ]
            },
            "normal_user_probability": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            },
            "meta_probe_probability": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            },
            "asks_for_concrete_work": {
                "type": "boolean"
            },
            "self_referential": {
                "type": "boolean"
            },
            "reason": {
                "type": "string"
            }
        },
        "required": [
            "label",
            "normal_user_probability",
            "meta_probe_probability",
            "asks_for_concrete_work",
            "self_referential",
            "reason"
        ],
        "additionalProperties": False
    }
}


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []

    with p.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def candidate_key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )


async def judge_one(
    client,
    sem,
    write_lock,
    outfile,
    row,
    model,
    max_retries,
):
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": (
                                "Classify this query:\n\n"
                                + row["query"]
                            ),
                        },
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": SCHEMA,
                    },
                )

                raw = response.choices[0].message.content
                result = json.loads(raw)

                pn = float(result["normal_user_probability"])
                pm = float(result["meta_probe_probability"])

                # Normalize only if model's two scores do not sum closely to 1.
                total = pn + pm
                if total > 0 and abs(total - 1.0) > 0.02:
                    pn /= total
                    pm /= total

                record = {
                    **row,
                    "judge_model": model,
                    "label": result["label"],
                    "normal_user_probability": pn,
                    "meta_probe_probability": pm,
                    "asks_for_concrete_work": bool(
                        result["asks_for_concrete_work"]
                    ),
                    "self_referential": bool(
                        result["self_referential"]
                    ),
                    "reason": str(result["reason"])[:500],
                    "prompt_tokens": (
                        response.usage.prompt_tokens
                        if response.usage else None
                    ),
                    "completion_tokens": (
                        response.usage.completion_tokens
                        if response.usage else None
                    ),
                }

                async with write_lock:
                    with outfile.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                        f.flush()

                return record

            except Exception as e:
                if attempt + 1 >= max_retries:
                    return {
                        **row,
                        "error": repr(e),
                    }

                # exponential backoff + jitter
                delay = min(
                    30.0,
                    (2 ** attempt) + random.random()
                )
                await asyncio.sleep(delay)


def summarize_selection(name, xs):
    if not xs:
        return None

    return {
        "name": name,
        "N": len(xs),
        "R": float(np.mean([x["R"] for x in xs])),
        "dSZ": float(np.mean([x["delta_sz"] for x in xs])),
        "dM": float(np.mean([x["delta_margin"] for x in xs])),
        "Mpos": float(
            np.mean([x["delta_margin"] > 0 for x in xs])
        ),
        "dMaxF": float(
            np.mean([x["maxfalse_drop"] for x in xs])
        ),
        "MaxFdown": float(
            np.mean([x["maxfalse_drop"] > 0 for x in xs])
        ),
        "dF1": float(
            np.mean([x["delta_gap_f1"] for x in xs])
        ),
    }


def print_sel(s):
    if s is None:
        return

    print(
        f"{s['name']:16s} | "
        f"{s['N']:2d} | "
        f"{s['dSZ']:+.5f} | "
        f"{s['dM']:+.5f} | "
        f"{100*s['Mpos']:4.1f}% | "
        f"{s['dMaxF']:+.5f} | "
        f"{100*s['MaxFdown']:8.1f}% | "
        f"{s['dF1']:+.4f}"
    )


def analyze(root, output_dir):
    judge_path = output_dir / "judge_results.jsonl"

    judges = [
        x for x in read_jsonl(judge_path)
        if "error" not in x
    ]

    by_judge = {
        candidate_key(x): x
        for x in judges
    }

    rel = read_jsonl(
        root / "membership_relabel_candidates.jsonl"
    )

    rows = []

    for z in rel:
        k = candidate_key(z)
        if k not in by_judge:
            continue

        j = by_judge[k]

        # Two useful definitions:
        #
        # label_clean:
        #   semantic class from GPT-4o-mini alone.
        #
        # strict_clean:
        #   high-confidence ordinary task with concrete work and no
        #   self-reference.
        label_clean = (
            j["label"] == "NORMAL_USER_TASK"
        )

        strict_clean = (
            label_clean
            and float(j["meta_probe_probability"]) <= 0.20
            and bool(j["asks_for_concrete_work"])
            and not bool(j["self_referential"])
        )

        rows.append({
            "round": int(z["round"]),
            "sid": z["session_id"],
            "idx": int(z["candidate_index"]),
            "query": j["query"],

            "label": j["label"],
            "meta_p": float(j["meta_probe_probability"]),
            "normal_p": float(j["normal_user_probability"]),
            "concrete": bool(j["asks_for_concrete_work"]),
            "self_ref": bool(j["self_referential"]),
            "reason": j["reason"],

            "label_clean": label_clean,
            "strict_clean": strict_clean,

            "delta_sz": float(z["delta_sz"]),
            "delta_margin": float(z["delta_margin"]),
            "maxfalse_drop": float(z["maxfalse_drop"]),
            "delta_gap_f1": float(z["delta_gap_f1"]),

            "R": float(
                z["delta_sz"] + z["delta_margin"]
            ),
        })

    print("\n=== GPT-4o-mini NATURALNESS AUDIT ===")
    print("judged:", len(rows))

    meta = [
        x for x in rows
        if x["label"] == "META_CAPABILITY_PROBE"
    ]

    label_clean = [
        x for x in rows
        if x["label_clean"]
    ]

    strict_clean = [
        x for x in rows
        if x["strict_clean"]
    ]

    print(
        f"META label       : {len(meta)}/{len(rows)} "
        f"({100*len(meta)/len(rows):.1f}%)"
    )
    print(
        f"NORMAL label     : {len(label_clean)}/{len(rows)} "
        f"({100*len(label_clean)/len(rows):.1f}%)"
    )
    print(
        f"STRICT CLEAN     : {len(strict_clean)}/{len(rows)} "
        f"({100*len(strict_clean)/len(rows):.1f}%)"
    )

    print("\n=== BY ROUND ===")
    print("R | Meta% | Normal% | StrictClean%")
    print("--+-------+---------+-------------")

    for r in range(1, 7):
        xs = [x for x in rows if x["round"] == r]

        print(
            f"{r} | "
            f"{100*np.mean([x['label']=='META_CAPABILITY_PROBE' for x in xs]):5.1f}% | "
            f"{100*np.mean([x['label_clean'] for x in xs]):7.1f}% | "
            f"{100*np.mean([x['strict_clean'] for x in xs]):11.1f}%"
        )

    # ---------------------------------------------------------
    # Reward conditional on semantic class
    # ---------------------------------------------------------

    print("\n=== REWARD BY CLASS ===")

    for name, xs in [
        ("META", meta),
        ("NORMAL", label_clean),
        ("STRICT CLEAN", strict_clean),
    ]:
        if not xs:
            continue

        print(
            f"{name:12s}: "
            f"N={len(xs):3d} "
            f"meanR={np.mean([x['R'] for x in xs]):+.5f} "
            f"mean_dSZ={np.mean([x['delta_sz'] for x in xs]):+.5f} "
            f"mean_dM={np.mean([x['delta_margin'] for x in xs]):+.5f}"
        )

    # ---------------------------------------------------------
    # Oracle candidate selection per state
    # ---------------------------------------------------------

    groups = defaultdict(list)

    for x in rows:
        groups[(x["round"], x["sid"])].append(x)

    all_best = []
    normal_best = []
    strict_best = []

    normal_pool_sizes = []
    strict_pool_sizes = []

    for xs in groups.values():

        all_best.append(
            max(xs, key=lambda x: (x["R"], -x["idx"]))
        )

        nxs = [x for x in xs if x["label_clean"]]
        sxs = [x for x in xs if x["strict_clean"]]

        normal_pool_sizes.append(len(nxs))
        strict_pool_sizes.append(len(sxs))

        if nxs:
            normal_best.append(
                max(
                    nxs,
                    key=lambda x: (x["R"], -x["idx"])
                )
            )

        if sxs:
            strict_best.append(
                max(
                    sxs,
                    key=lambda x: (x["R"], -x["idx"])
                )
            )

    print("\n=== ORACLE SELECTION AFTER LLM FILTER ===")
    print(
        "Method           | N  | dSZ     | dM      | M+%   | "
        "dMaxF   | MaxFdown% | dF1"
    )
    print(
        "-----------------+----+---------+---------+-------+"
        "---------+-----------+-------"
    )

    selections = [
        summarize_selection("ALL BEST", all_best),
        summarize_selection("NORMAL BEST", normal_best),
        summarize_selection("STRICT CLEAN BEST", strict_best),
    ]

    for s in selections:
        print_sel(s)

    print("\n=== POOL AVAILABILITY ===")

    print(
        f"states with NORMAL candidate: "
        f"{100*np.mean(np.array(normal_pool_sizes)>0):.1f}%"
    )
    print(
        f"mean NORMAL pool size: "
        f"{np.mean(normal_pool_sizes):.2f}/4"
    )

    print(
        f"states with STRICT CLEAN candidate: "
        f"{100*np.mean(np.array(strict_pool_sizes)>0):.1f}%"
    )
    print(
        f"mean STRICT CLEAN pool size: "
        f"{np.mean(strict_pool_sizes):.2f}/4"
    )

    def availability(field):
        vals = []

        for xs in groups.values():
            clean = [
                x for x in xs
                if x["strict_clean"]
            ]

            if field == "R":
                vals.append(
                    any(x["R"] > 0 for x in clean)
                )
            elif field == "margin":
                vals.append(
                    any(x["delta_margin"] > 0 for x in clean)
                )
            elif field == "maxfalse":
                vals.append(
                    any(x["maxfalse_drop"] > 0 for x in clean)
                )

        return 100 * np.mean(vals)

    print(
        f"states with strict-clean R>0       : "
        f"{availability('R'):.1f}%"
    )
    print(
        f"states with strict-clean dMargin>0 : "
        f"{availability('margin'):.1f}%"
    )
    print(
        f"states with strict-clean MaxF drop : "
        f"{availability('maxfalse'):.1f}%"
    )

    # ---------------------------------------------------------
    # Most informative examples
    # ---------------------------------------------------------

    print("\n=== TOP 15 STRICT-CLEAN BY MEMBERSHIP REWARD ===")

    top_clean = sorted(
        strict_clean,
        key=lambda x: x["R"],
        reverse=True,
    )[:15]

    for x in top_clean:
        print(
            f"\nR{x['round']} "
            f"reward={x['R']:+.5f} "
            f"dM={x['delta_margin']:+.5f} "
            f"MaxFdrop={x['maxfalse_drop']:+.5f} "
            f"metaP={x['meta_p']:.2f}"
        )
        print(x["query"])
        print("judge:", x["reason"])

    print("\n=== TOP 15 META BY MEMBERSHIP REWARD ===")

    top_meta = sorted(
        meta,
        key=lambda x: x["R"],
        reverse=True,
    )[:15]

    for x in top_meta:
        print(
            f"\nR{x['round']} "
            f"reward={x['R']:+.5f} "
            f"metaP={x['meta_p']:.2f}"
        )
        print(x["query"])
        print("judge:", x["reason"])

    # ---------------------------------------------------------
    # Token usage / approximate cost
    # Current published GPT-4o-mini rates:
    # $0.15 / 1M input; $0.60 / 1M output.
    # ---------------------------------------------------------

    in_tok = sum(
        int(x.get("prompt_tokens") or 0)
        for x in judges
    )

    out_tok = sum(
        int(x.get("completion_tokens") or 0)
        for x in judges
    )

    approx_cost = (
        in_tok / 1_000_000 * 0.15
        + out_tok / 1_000_000 * 0.60
    )

    print("\n=== API USAGE ===")
    print("input tokens :", in_tok)
    print("output tokens:", out_tok)
    print(f"approx cost  : ${approx_cost:.4f}")

    summary = {
        "judge_model": (
            judges[0]["judge_model"]
            if judges else None
        ),
        "n_judged": len(rows),
        "meta_count": len(meta),
        "normal_count": len(label_clean),
        "strict_clean_count": len(strict_clean),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "approx_cost_usd": approx_cost,
        "selection_summaries": selections,
        "normal_pool_mean": float(
            np.mean(normal_pool_sizes)
        ),
        "strict_pool_mean": float(
            np.mean(strict_pool_sizes)
        ),
    }

    with (
        output_dir / "audit_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "\nwrote:",
        output_dir / "audit_summary.json"
    )


async def main_async(args):
    root = Path(args.root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Collect exact executed 384 candidates.
    # ---------------------------------------------------------

    rows = []

    for rnd in range(1, 7):
        p = root / f"responses_train_round_{rnd:02d}.jsonl"

        for x in read_jsonl(p):
            if x.get("execution_status") != "ok":
                continue

            rows.append({
                "round": rnd,
                "session_id": str(x["session_id"]),
                "candidate_index": int(x["candidate_index"]),
                "query": str(x["query"]),
            })

    # Deduplicate defensively.
    unique = {}

    for x in rows:
        unique[candidate_key(x)] = x

    rows = list(unique.values())

    print("unique probes:", len(rows))

    if len(rows) != 384:
        raise RuntimeError(
            f"Expected 384 unique probes, got {len(rows)}"
        )

    outfile = output_dir / "judge_results.jsonl"

    done = {}

    for x in read_jsonl(outfile):
        if "error" not in x:
            done[candidate_key(x)] = x

    pending = [
        x for x in rows
        if candidate_key(x) not in done
    ]

    print("already judged:", len(done))
    print("pending:", len(pending))
    print("model:", args.model)
    print("concurrency:", args.concurrency)

    if pending:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)
        write_lock = asyncio.Lock()

        tasks = [
            asyncio.create_task(
                judge_one(
                    client=client,
                    sem=sem,
                    write_lock=write_lock,
                    outfile=outfile,
                    row=x,
                    model=args.model,
                    max_retries=args.max_retries,
                )
            )
            for x in pending
        ]

        completed = 0
        failures = 0

        for fut in asyncio.as_completed(tasks):
            result = await fut
            completed += 1

            if "error" in result:
                failures += 1
                print(
                    "FAILED:",
                    candidate_key(result),
                    result["error"],
                    flush=True,
                )

            if (
                completed % 25 == 0
                or completed == len(tasks)
            ):
                print(
                    f"completed {completed}/{len(tasks)} "
                    f"failures={failures}",
                    flush=True,
                )

        await client.close()

        if failures:
            raise RuntimeError(
                f"{failures} calls failed; rerun script to resume"
            )

    # Must be exactly complete before analysis.
    final = {
        candidate_key(x): x
        for x in read_jsonl(outfile)
        if "error" not in x
    }

    print("final successful judgments:", len(final))

    if len(final) != 384:
        raise RuntimeError(
            f"Need 384 successful judgments, got {len(final)}"
        )

    analyze(root, output_dir)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="artifacts/v3_coverage_elimination_pilot",
    )
    ap.add_argument(
        "--output",
        default="artifacts/v4_gpt4omini_naturalness",
    )
    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=12,
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=6,
    )

    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
