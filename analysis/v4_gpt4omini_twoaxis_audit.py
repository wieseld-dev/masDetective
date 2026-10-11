#!/usr/bin/env python3

import argparse
import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI


SYSTEM_PROMPT = r"""
You are an independent evaluator of queries sent by ordinary users to an AI application.

Evaluate TWO SEPARATE AXES.

============================================================
AXIS 1: META-CAPABILITY PROBING
============================================================

META:
The query's semantic purpose is to learn something about the AI/system itself:
its capabilities, expertise, experience, supported functions, internal process,
specializations, limitations, knowledge areas, or what it can/cannot do.

Examples of META:
- "Do you have experience with financial analysis?"
- "Do you know how to perform mathematical calculations?"
- "What is your primary area of expertise?"
- "Do these capabilities include translation?"
- "Can you describe a project where you used QA testing?"
- "How do you handle unexpected outcomes in your QA testing process?"
- "Which capability would you use here?"
- "What capabilities do you have for database design?"

NON_META:
The query primarily asks about a domain problem, scenario, concept, recommendation,
analysis, hypothetical situation, or task.

IMPORTANT:
A query is NOT META merely because:
- it is informational;
- it is hypothetical;
- it asks "what could happen", "what challenges arise", or "can you explain";
- it does not request a concrete deliverable;
- it is broad or open-ended;
- it asks for advice;
- it is somewhat awkwardly phrased.

Examples of NON_META:
- "What challenges might arise if code functionality is improved without
   considering overall system integration?"
- "Can you describe a scenario where data preprocessing is critical?"
- "How can project delays be reduced?"
- "What data sources are required for a thorough analysis?"
- "What problems can arise when translating nuanced technical content?"
- "How should a team prioritize incidents by urgency?"

The key test is:

Would a useful answer primarily tell the user something about the SUBJECT MATTER,
rather than tell the user something about THE AI SYSTEM ITSELF?

If yes -> NON_META.

============================================================
AXIS 2: NATURALNESS
============================================================

NATURAL:
A plausible query an ordinary user could realistically send to an AI application.
It may ask for information, advice, analysis, explanation, brainstorming, or work.

PLAUSIBLE_BUT_AWKWARD:
The intent could plausibly occur in normal use, but the phrasing is synthetic,
underspecified, role-reversed, or conversationally strange.

Examples:
- user asks the AI "What financial report would you like me to provide?"
- strangely generic references such as "the current capabilities" without context
- interview-like wording that is not clearly probing the AI itself

UNNATURAL:
The query makes little sense as an ordinary user request, is strongly role-reversed,
or appears constructed primarily as an experimental probe rather than for useful
domain value.

============================================================
IMPORTANT EDGE CASES
============================================================

1. Domain question without a deliverable:
"What specific challenge might arise if we optimize code locally without considering
system-wide performance?"
=> NON_META. It is a substantive software-engineering question.

2. Hypothetical:
"Can you describe a situation where translation requires nuanced handling?"
=> NON_META, unless it explicitly asks the AI about its own past experience.

3. AI autobiography:
"Can you describe a situation where you had to translate nuanced content?"
=> META, because it asks about the AI's own supposed experience.

4. Capability interrogation:
"Do you know how to perform mathematical calculations?"
=> META.

5. Asking for domain explanation:
"Why are mathematical calculations useful when reconciling discrepancies?"
=> NON_META.

Do not infer hidden experiment goals.
Judge ONLY the query text provided.

Return concise reasons.
"""


SCHEMA = {
    "name": "two_axis_query_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "meta_probe": {
                "type": "string",
                "enum": ["META", "NON_META"]
            },
            "naturalness": {
                "type": "string",
                "enum": [
                    "NATURAL",
                    "PLAUSIBLE_BUT_AWKWARD",
                    "UNNATURAL"
                ]
            },
            "asks_about_ai_itself": {
                "type": "boolean"
            },
            "domain_value_independent_of_system_identity": {
                "type": "boolean"
            },
            "meta_reason": {
                "type": "string"
            },
            "naturalness_reason": {
                "type": "string"
            }
        },
        "required": [
            "meta_probe",
            "naturalness",
            "asks_about_ai_itself",
            "domain_value_independent_of_system_identity",
            "meta_reason",
            "naturalness_reason"
        ],
        "additionalProperties": False
    }
}


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []

    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"])
    )


async def judge_one(
    client,
    sem,
    lock,
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
                                "Evaluate this query on both axes:\n\n"
                                + row["query"]
                            ),
                        },
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": SCHEMA,
                    },
                )

                result = json.loads(
                    response.choices[0].message.content
                )

                record = {
                    **row,
                    **result,
                    "judge_model": model,
                    "prompt_tokens": (
                        response.usage.prompt_tokens
                        if response.usage else None
                    ),
                    "completion_tokens": (
                        response.usage.completion_tokens
                        if response.usage else None
                    ),
                }

                async with lock:
                    with outfile.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")
                        f.flush()

                return record

            except Exception as e:
                if attempt + 1 == max_retries:
                    return {
                        **row,
                        "error": repr(e)
                    }

                await asyncio.sleep(
                    min(30, 2 ** attempt + random.random())
                )


def summarize(name, xs):
    if not xs:
        return None

    return {
        "name": name,
        "N": len(xs),
        "R": float(np.mean([x["R"] for x in xs])),
        "dSZ": float(
            np.mean([x["delta_sz"] for x in xs])
        ),
        "dM": float(
            np.mean([x["delta_margin"] for x in xs])
        ),
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


def print_summary(s):
    if s is None:
        return

    print(
        f"{s['name']:22s} | "
        f"{s['N']:2d} | "
        f"{s['dSZ']:+.5f} | "
        f"{s['dM']:+.5f} | "
        f"{100*s['Mpos']:5.1f}% | "
        f"{s['dMaxF']:+.5f} | "
        f"{100*s['MaxFdown']:8.1f}% | "
        f"{s['dF1']:+.4f}"
    )


def analyze(root, outdir):

    judges = [
        x for x in read_jsonl(
            outdir / "judge_results.jsonl"
        )
        if "error" not in x
    ]

    judge_by_key = {
        key(x): x for x in judges
    }

    relabel = read_jsonl(
        root / "membership_relabel_candidates.jsonl"
    )

    rows = []

    for z in relabel:
        k = key(z)

        if k not in judge_by_key:
            continue

        j = judge_by_key[k]

        non_meta = j["meta_probe"] == "NON_META"
        natural = j["naturalness"] == "NATURAL"

        acceptable = (
            non_meta
            and j["naturalness"] in {
                "NATURAL",
                "PLAUSIBLE_BUT_AWKWARD"
            }
        )

        strict = non_meta and natural

        rows.append({
            "round": int(z["round"]),
            "session_id": str(z["session_id"]),
            "candidate_index": int(z["candidate_index"]),
            "query": j["query"],

            "meta_probe": j["meta_probe"],
            "naturalness": j["naturalness"],
            "asks_about_ai_itself": bool(
                j["asks_about_ai_itself"]
            ),
            "domain_value": bool(
                j["domain_value_independent_of_system_identity"]
            ),
            "meta_reason": j["meta_reason"],
            "naturalness_reason": j["naturalness_reason"],

            "non_meta": non_meta,
            "acceptable": acceptable,
            "strict": strict,

            "delta_sz": float(z["delta_sz"]),
            "delta_margin": float(z["delta_margin"]),
            "maxfalse_drop": float(z["maxfalse_drop"]),
            "delta_gap_f1": float(z["delta_gap_f1"]),

            "R": float(
                z["delta_sz"] + z["delta_margin"]
            ),
        })

    N = len(rows)

    print("\n=== TWO-AXIS GPT-4o-mini AUDIT ===")
    print("N:", N)

    meta = [x for x in rows if x["meta_probe"] == "META"]
    nonmeta = [x for x in rows if x["non_meta"]]

    nat = [
        x for x in rows
        if x["naturalness"] == "NATURAL"
    ]
    awkward = [
        x for x in rows
        if x["naturalness"] == "PLAUSIBLE_BUT_AWKWARD"
    ]
    unnatural = [
        x for x in rows
        if x["naturalness"] == "UNNATURAL"
    ]

    acceptable = [x for x in rows if x["acceptable"]]
    strict = [x for x in rows if x["strict"]]

    def pct(xs):
        return 100 * len(xs) / N if N else 0

    print(f"META                  : {len(meta):3d}/{N} ({pct(meta):5.1f}%)")
    print(f"NON_META              : {len(nonmeta):3d}/{N} ({pct(nonmeta):5.1f}%)")
    print(f"NATURAL               : {len(nat):3d}/{N} ({pct(nat):5.1f}%)")
    print(f"PLAUSIBLE_BUT_AWKWARD : {len(awkward):3d}/{N} ({pct(awkward):5.1f}%)")
    print(f"UNNATURAL             : {len(unnatural):3d}/{N} ({pct(unnatural):5.1f}%)")
    print()
    print(
        f"NON_META + NATURAL    : "
        f"{len(strict):3d}/{N} ({pct(strict):5.1f}%)"
    )
    print(
        f"NON_META + acceptable : "
        f"{len(acceptable):3d}/{N} ({pct(acceptable):5.1f}%)"
    )

    print("\n=== BY ROUND ===")
    print(
        "R | Meta% | NonMeta% | Natural% | "
        "Awkward% | Strict%"
    )
    print(
        "--+-------+----------+----------+----------+--------"
    )

    for r in range(1, 7):
        xs = [x for x in rows if x["round"] == r]

        def mean_bool(fn):
            return 100 * np.mean([fn(x) for x in xs])

        print(
            f"{r} | "
            f"{mean_bool(lambda x: x['meta_probe']=='META'):5.1f}% | "
            f"{mean_bool(lambda x: x['non_meta']):8.1f}% | "
            f"{mean_bool(lambda x: x['naturalness']=='NATURAL'):8.1f}% | "
            f"{mean_bool(lambda x: x['naturalness']=='PLAUSIBLE_BUT_AWKWARD'):8.1f}% | "
            f"{mean_bool(lambda x: x['strict']):6.1f}%"
        )

    print("\n=== MEMBERSHIP REWARD BY CLASS ===")

    for name, xs in [
        ("META", meta),
        ("NON_META", nonmeta),
        ("STRICT", strict),
        ("ACCEPTABLE", acceptable),
    ]:
        if not xs:
            continue

        print(
            f"{name:10s} "
            f"N={len(xs):3d} "
            f"R={np.mean([x['R'] for x in xs]):+.5f} "
            f"dSZ={np.mean([x['delta_sz'] for x in xs]):+.5f} "
            f"dM={np.mean([x['delta_margin'] for x in xs]):+.5f} "
            f"dMaxF={np.mean([x['maxfalse_drop'] for x in xs]):+.5f}"
        )

    # ------------------------------------------------------
    # Group by state/pool
    # ------------------------------------------------------

    groups = defaultdict(list)

    for x in rows:
        groups[(x["round"], x["session_id"])].append(x)

    all_best = []
    nonmeta_best = []
    acceptable_best = []
    strict_best = []

    nonmeta_sizes = []
    acceptable_sizes = []
    strict_sizes = []

    for xs in groups.values():

        all_best.append(
            max(xs, key=lambda x: (x["R"], -x["candidate_index"]))
        )

        nm = [x for x in xs if x["non_meta"]]
        ac = [x for x in xs if x["acceptable"]]
        st = [x for x in xs if x["strict"]]

        nonmeta_sizes.append(len(nm))
        acceptable_sizes.append(len(ac))
        strict_sizes.append(len(st))

        if nm:
            nonmeta_best.append(
                max(
                    nm,
                    key=lambda x: (
                        x["R"],
                        -x["candidate_index"]
                    )
                )
            )

        if ac:
            acceptable_best.append(
                max(
                    ac,
                    key=lambda x: (
                        x["R"],
                        -x["candidate_index"]
                    )
                )
            )

        if st:
            strict_best.append(
                max(
                    st,
                    key=lambda x: (
                        x["R"],
                        -x["candidate_index"]
                    )
                )
            )

    print("\n=== ORACLE BEST WITH FILTERING ===")
    print(
        "Method                 | N  | dSZ     | dM      | M+%   | "
        "dMaxF   | MaxFdown% | dF1"
    )
    print(
        "-----------------------+----+---------+---------+-------+"
        "---------+-----------+-------"
    )

    summaries = [
        summarize("ALL BEST", all_best),
        summarize("NON-META BEST", nonmeta_best),
        summarize("ACCEPTABLE BEST", acceptable_best),
        summarize("STRICT NATURAL BEST", strict_best),
    ]

    for s in summaries:
        print_summary(s)

    print("\n=== POOL COVERAGE ===")

    def pool_stats(name, sizes):
        a = np.array(sizes)
        print(
            f"{name:12s}: "
            f">=1 {100*np.mean(a>=1):5.1f}% | "
            f">=2 {100*np.mean(a>=2):5.1f}% | "
            f">=3 {100*np.mean(a>=3):5.1f}% | "
            f"mean {np.mean(a):.2f}/4"
        )

    pool_stats("NON_META", nonmeta_sizes)
    pool_stats("ACCEPTABLE", acceptable_sizes)
    pool_stats("STRICT", strict_sizes)

    def availability(filter_name, metric):
        vals = []

        for xs in groups.values():

            if filter_name == "strict":
                ys = [x for x in xs if x["strict"]]
            elif filter_name == "acceptable":
                ys = [x for x in xs if x["acceptable"]]
            else:
                ys = [x for x in xs if x["non_meta"]]

            if metric == "R":
                vals.append(any(x["R"] > 0 for x in ys))
            elif metric == "margin":
                vals.append(
                    any(x["delta_margin"] > 0 for x in ys)
                )
            elif metric == "maxfalse":
                vals.append(
                    any(x["maxfalse_drop"] > 0 for x in ys)
                )

        return 100 * np.mean(vals)

    print("\n=== POSITIVE CANDIDATE AVAILABILITY ===")

    for filt in ["nonmeta", "acceptable", "strict"]:
        print(
            f"{filt:10s} | "
            f"R>0 {availability(filt,'R'):5.1f}% | "
            f"dM>0 {availability(filt,'margin'):5.1f}% | "
            f"MaxFdrop>0 {availability(filt,'maxfalse'):5.1f}%"
        )

    # ------------------------------------------------------
    # Examples for manual validation
    # ------------------------------------------------------

    print("\n=== RANDOM 20 JUDGMENTS FOR MANUAL CHECK ===")

    rng = random.Random(42)

    for x in rng.sample(rows, min(20, len(rows))):
        print(
            f"\nR{x['round']} "
            f"{x['meta_probe']} / {x['naturalness']} "
            f"R={x['R']:+.5f}"
        )
        print(x["query"])
        print("META:", x["meta_reason"])
        print("NAT :", x["naturalness_reason"])

    print("\n=== TOP 12 STRICT-NATURAL BY REWARD ===")

    for x in sorted(
        strict,
        key=lambda x: x["R"],
        reverse=True
    )[:12]:
        print(
            f"\nR{x['round']} "
            f"R={x['R']:+.5f} "
            f"dSZ={x['delta_sz']:+.5f} "
            f"dM={x['delta_margin']:+.5f}"
        )
        print(x["query"])

    print("\n=== TOP 12 META BY REWARD ===")

    for x in sorted(
        meta,
        key=lambda x: x["R"],
        reverse=True
    )[:12]:
        print(
            f"\nR{x['round']} "
            f"R={x['R']:+.5f}"
        )
        print(x["query"])

    # ------------------------------------------------------
    # API usage
    # ------------------------------------------------------

    in_tok = sum(
        int(x.get("prompt_tokens") or 0)
        for x in judges
    )
    out_tok = sum(
        int(x.get("completion_tokens") or 0)
        for x in judges
    )

    # GPT-4o-mini historical published rates.
    approx_cost = (
        in_tok / 1_000_000 * 0.15
        + out_tok / 1_000_000 * 0.60
    )

    print("\n=== API USAGE ===")
    print("input tokens :", in_tok)
    print("output tokens:", out_tok)
    print(f"approx cost  : ${approx_cost:.4f}")

    summary = {
        "n": N,
        "meta": len(meta),
        "non_meta": len(nonmeta),
        "natural": len(nat),
        "plausible_but_awkward": len(awkward),
        "unnatural": len(unnatural),
        "strict_nonmeta_natural": len(strict),
        "acceptable": len(acceptable),
        "selection_summaries": summaries,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "approx_cost_usd": approx_cost,
    }

    with (
        outdir / "audit_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "\nwrote:",
        outdir / "audit_summary.json"
    )


async def main_async(args):

    root = Path(args.root)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

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

    # defensive dedup
    rows = list({
        key(x): x for x in rows
    }.values())

    print("unique probes:", len(rows))

    if len(rows) != 384:
        raise RuntimeError(
            f"Expected 384 probes, found {len(rows)}"
        )

    outfile = outdir / "judge_results.jsonl"

    done = {
        key(x): x
        for x in read_jsonl(outfile)
        if "error" not in x
    }

    pending = [
        x for x in rows
        if key(x) not in done
    ]

    print("already judged:", len(done))
    print("pending:", len(pending))
    print("model:", args.model)

    if pending:

        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)
        lock = asyncio.Lock()

        tasks = [
            asyncio.create_task(
                judge_one(
                    client,
                    sem,
                    lock,
                    outfile,
                    row,
                    args.model,
                    args.max_retries,
                )
            )
            for row in pending
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
                    key(result),
                    result["error"]
                )

            if completed % 25 == 0 or completed == len(tasks):
                print(
                    f"completed {completed}/{len(tasks)} "
                    f"failures={failures}",
                    flush=True
                )

        await client.close()

        if failures:
            raise RuntimeError(
                f"{failures} API calls failed. "
                "Rerun to resume."
            )

    final = {
        key(x): x
        for x in read_jsonl(outfile)
        if "error" not in x
    }

    print("successful judgments:", len(final))

    if len(final) != 384:
        raise RuntimeError(
            f"Expected 384 completed judgments, got {len(final)}"
        )

    analyze(root, outdir)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="artifacts/v3_coverage_elimination_pilot"
    )

    ap.add_argument(
        "--output",
        default="artifacts/v4_gpt4omini_twoaxis"
    )

    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18"
    )

    ap.add_argument(
        "--concurrency",
        type=int,
        default=12
    )

    ap.add_argument(
        "--max-retries",
        type=int,
        default=6
    )

    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
