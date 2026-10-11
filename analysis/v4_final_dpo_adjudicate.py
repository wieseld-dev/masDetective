from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI


ROOT = Path("artifacts/v4_bulk_mining")
GOLD_PATH = Path("artifacts/v4_userlike_audit/gold_strict.jsonl")
OUT = Path("artifacts/v4_final_dpo")
OUT.mkdir(parents=True, exist_ok=True)

DIR_SRC = ROOT / "direction_negatives.jsonl"
META_SRC = ROOT / "meta_candidates.jsonl"

DIR_JUDGE = OUT / "direction_adjudications.jsonl"
META_JUDGE = OUT / "meta_adjudications.jsonl"

DIR_FINAL = OUT / "direction_final.jsonl"
META_FINAL = OUT / "meta_final.jsonl"
COMBINED = OUT / "combined_final.jsonl"
SUMMARY = OUT / "summary.json"


DIRECTION_SYSTEM = r"""
You are a precision adjudicator of conversational direction.

Determine whether the candidate query is CLEARLY something an AI assistant
would normally ask its USER for clarification, or something a USER could
naturally send to an AI system.

Return one of:

ASSISTANT_TO_USER:
The query directly asks the conversation partner to specify their need,
preference, desired assistance, missing information, requested scenario,
or what they want the assistant to investigate.

Examples:
"What specific scenario would you like assistance with?"
"What kind of problem are you currently facing?"
"Which part would you like me to examine?"
"What specifically do you need guidance on?"
"What information would you like me to gather first?"

USER_TO_SYSTEM:
The query can naturally be sent by a user to an AI asking for information,
analysis, explanation, or assessment.

Important:
"Do these users require assistance with incident management?"
is USER_TO_SYSTEM because "these users" is a third party.

"Do these tasks involve managing multiple versions of a document?"
is USER_TO_SYSTEM.

"Do these tasks require financial analysis?"
is USER_TO_SYSTEM.

Do not classify a question as ASSISTANT_TO_USER merely because it contains
"you". Resolve who is asking whom for what.

AMBIGUOUS:
Use when conversational direction is genuinely unclear.

Also classify strength:
EXPLICIT_CLARIFICATION
STRONG
WEAK

Precision matters more than recall. If uncertain, use AMBIGUOUS.
"""


DIRECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "direction_adjudication",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": [
                        "ASSISTANT_TO_USER",
                        "USER_TO_SYSTEM",
                        "AMBIGUOUS",
                    ],
                },
                "strength": {
                    "type": "string",
                    "enum": [
                        "EXPLICIT_CLARIFICATION",
                        "STRONG",
                        "WEAK",
                    ],
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                },
                "reason": {"type": "string"},
            },
            "required": [
                "direction",
                "strength",
                "confidence",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


META_SYSTEM = r"""
You are a precision adjudicator for benign black-box AI-system queries.

Determine whether the candidate CLEARLY asks about THIS AI SYSTEM itself.

SYSTEM_SELF:
The useful answer would reveal the AI/system's own capabilities, expertise,
experience, supported functions, specialization, limitations, behavior,
or how the AI/system itself performs a task.

Examples:
"Do you have experience with project management?"
"What is your primary area of expertise?"
"How do you handle cash flow projections?"
"Can you give examples of how you handle project management tasks?"
"How does the system handle multiple-language translations?"

EXTERNAL_DOMAIN:
The query asks about an external task, project, organization, workflow,
dataset, users, concept, or scenario.

Examples:
"How should cash-flow projections be calculated?"
"What problems arise in project management?"
"Do these tasks require financial analysis?"
"What factors matter when translating technical content?"

AMBIGUOUS:
It is genuinely unclear whether the referent is this AI or an external
person/system/scenario.

Do not classify SYSTEM_SELF merely because the query contains "you".
Precision matters more than recall.

Also classify system-self strength:
EXPLICIT
STRONG
WEAK
"""


META_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "meta_adjudication",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": [
                        "SYSTEM_SELF",
                        "EXTERNAL_DOMAIN",
                        "AMBIGUOUS",
                    ],
                },
                "strength": {
                    "type": "string",
                    "enum": ["EXPLICIT", "STRONG", "WEAK"],
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                },
                "reason": {"type": "string"},
            },
            "required": [
                "target",
                "strength",
                "confidence",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [
        json.loads(x)
        for x in p.open(encoding="utf-8")
        if x.strip()
    ]


def append_jsonl(p, x):
    with Path(p).open("a", encoding="utf-8") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


def write_jsonl(p, xs):
    with Path(p).open("w", encoding="utf-8") as f:
        for x in xs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def aid(kind, x):
    return (
        f"{kind}:{int(x['gold_index'])}:"
        f"{int(x['sample_index'])}"
    )


async def call_judge(client, sem, model, system, schema, kind, x):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": "Candidate query:\n\n" + str(x["query"]),
                },
            ],
            response_format=schema,
        )

        obj = json.loads(resp.choices[0].message.content)

        return {
            "audit_id": aid(kind, x),
            "gold_index": int(x["gold_index"]),
            "sample_index": int(x["sample_index"]),
            "query": str(x["query"]),
            **obj,
            "usage": {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        }


async def run_group(
    *,
    name,
    rows,
    outfile,
    client,
    sem,
    model,
    system,
    schema,
):
    existing = {
        x["audit_id"]: x
        for x in read_jsonl(outfile)
        if "error" not in x
    }

    pending = [
        x for x in rows
        if aid(name, x) not in existing
    ]

    print(
        f"{name}: total={len(rows)} "
        f"existing={len(existing)} pending={len(pending)}",
        flush=True,
    )

    B = 20

    for start in range(0, len(pending), B):
        batch = pending[start:start+B]

        got = await asyncio.gather(
            *[
                call_judge(
                    client,
                    sem,
                    model,
                    system,
                    schema,
                    name,
                    x,
                )
                for x in batch
            ],
            return_exceptions=True,
        )

        failures = []

        for src, res in zip(batch, got):
            if isinstance(res, Exception):
                failures.append(
                    f"{aid(name,src)} "
                    f"{type(res).__name__}: {res}"
                )
            else:
                # Save each paid result immediately.
                append_jsonl(outfile, res)
                existing[res["audit_id"]] = res

        print(
            f"[{name}] "
            f"{min(start+B,len(pending))}/{len(pending)}",
            flush=True,
        )

        if failures:
            for z in failures[:10]:
                print("FAIL:", z)
            raise RuntimeError(
                f"{name}: {len(failures)} failures"
            )

    missing = [
        aid(name, x)
        for x in rows
        if aid(name, x) not in existing
    ]

    if missing:
        raise RuntimeError(
            f"{name}: missing judgments {missing[:10]}"
        )

    return existing


def make_pair(g, x, pair_type, extra):
    assert str(g["prompt"]) == str(x["prompt"])

    chosen = str(g["completion"])
    rejected = str(x["completion"])

    assert chosen.strip()
    assert rejected.strip()
    assert chosen != rejected

    return {
        "pair_type": pair_type,
        "gold_index": int(g["gold_index"]),
        "round": int(g["round"]),
        "session_id": str(g["session_id"]),
        "candidate_index": int(g["candidate_index"]),
        "sample_index": int(x["sample_index"]),

        "prompt": str(g["prompt"]),
        "chosen": chosen,
        "rejected": rejected,

        "chosen_query": str(g["query"]),
        "rejected_query": str(x["query"]),

        "gold_reward": float(g.get("gold_reward", 0.0)),
        "gold_advantage": float(g.get("gold_advantage", 0.0)),

        **extra,
    }


async def main_async(args):
    gold = read_jsonl(GOLD_PATH)
    dirs = read_jsonl(DIR_SRC)
    metas = read_jsonl(META_SRC)

    assert len(gold) == 24, len(gold)
    assert len(dirs) == 83, len(dirs)
    assert len(metas) == 58, len(metas)

    gold_by = {
        int(x["gold_index"]): x
        for x in gold
    }

    print("======================================")
    print(" FINAL DPO ADJUDICATION")
    print("======================================")
    print("strict Gold         :", len(gold))
    print("direction candidates:", len(dirs))
    print("META candidates     :", len(metas))
    print("planned judgments   :", len(dirs)+len(metas))

    if args.dry_run:
        return

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    djud = await run_group(
        name="direction",
        rows=dirs,
        outfile=DIR_JUDGE,
        client=client,
        sem=sem,
        model=args.model,
        system=DIRECTION_SYSTEM,
        schema=DIRECTION_SCHEMA,
    )

    mjud = await run_group(
        name="meta",
        rows=metas,
        outfile=META_JUDGE,
        client=client,
        sem=sem,
        model=args.model,
        system=META_SYSTEM,
        schema=META_SCHEMA,
    )

    # -----------------------------
    # Direction final candidates
    # -----------------------------
    dir_good = []

    for x in dirs:
        j = djud[aid("direction", x)]

        if (
            j["direction"] == "ASSISTANT_TO_USER"
            and j["confidence"] == "HIGH"
            and j["strength"] in {
                "EXPLICIT_CLARIFICATION",
                "STRONG",
            }
        ):
            z = dict(x)
            z["_judge"] = j
            dir_good.append(z)

    # -----------------------------
    # META final candidates
    # -----------------------------
    meta_good = []

    for x in metas:
        j = mjud[aid("meta", x)]

        if (
            j["target"] == "SYSTEM_SELF"
            and j["confidence"] == "HIGH"
            and j["strength"] in {"EXPLICIT", "STRONG"}
        ):
            z = dict(x)
            z["_judge"] = j
            meta_good.append(z)

    # -----------------------------
    # One pair / Gold / objective
    # -----------------------------
    direction_by = defaultdict(list)
    meta_by = defaultdict(list)

    for x in dir_good:
        direction_by[int(x["gold_index"])].append(x)

    for x in meta_good:
        meta_by[int(x["gold_index"])].append(x)

    dir_final = []
    meta_final = []

    for gi in sorted(direction_by):
        g = gold_by[gi]

        xs = sorted(
            direction_by[gi],
            key=lambda x: (
                0 if x["_judge"]["strength"]
                == "EXPLICIT_CLARIFICATION"
                else 1,
                int(x["sample_index"]),
            )
        )

        x = xs[0]

        dir_final.append(
            make_pair(
                g,
                x,
                "DIRECTION",
                {
                    "negative_direction":
                        x["_judge"]["direction"],
                    "negative_strength":
                        x["_judge"]["strength"],
                    "negative_confidence":
                        x["_judge"]["confidence"],
                    "negative_reason":
                        x["_judge"]["reason"],
                },
            )
        )

    for gi in sorted(meta_by):
        g = gold_by[gi]

        xs = sorted(
            meta_by[gi],
            key=lambda x: (
                0 if x["_judge"]["strength"] == "EXPLICIT"
                else 1,
                int(x["sample_index"]),
            )
        )

        x = xs[0]

        meta_final.append(
            make_pair(
                g,
                x,
                "META",
                {
                    "negative_target": x["_judge"]["target"],
                    "negative_strength": x["_judge"]["strength"],
                    "negative_confidence":
                        x["_judge"]["confidence"],
                    "negative_reason":
                        x["_judge"]["reason"],
                },
            )
        )

    combined = dir_final + meta_final

    # Hard invariants.
    assert len({x["gold_index"] for x in dir_final}) == len(dir_final)
    assert len({x["gold_index"] for x in meta_final}) == len(meta_final)

    assert all(
        x["prompt"] == gold_by[x["gold_index"]]["prompt"]
        for x in combined
    )

    write_jsonl(DIR_FINAL, dir_final)
    write_jsonl(META_FINAL, meta_final)
    write_jsonl(COMBINED, combined)

    overlap = (
        set(x["gold_index"] for x in dir_final)
        & set(x["gold_index"] for x in meta_final)
    )

    summary = {
        "strict_gold": 24,

        "direction_input": len(dirs),
        "direction_survived_secondary": len(dir_good),
        "direction_final_pairs": len(dir_final),
        "direction_gold_coverage": len(dir_final),

        "meta_input": len(metas),
        "meta_survived_secondary": len(meta_good),
        "meta_final_pairs": len(meta_final),
        "meta_gold_coverage": len(meta_final),

        "combined_pairs": len(combined),
        "unique_gold_in_combined": len(
            set(x["gold_index"] for x in combined)
        ),
        "gold_with_both_objectives": len(overlap),
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("======================================")
    print(" FINAL PAIR SUMMARY")
    print("======================================")
    print(
        "Direction:",
        len(dir_good),
        "clear candidates ->",
        len(dir_final),
        "pairs | coverage",
        f"{len(dir_final)}/24",
    )
    print(
        "META     :",
        len(meta_good),
        "clear candidates ->",
        len(meta_final),
        "pairs | coverage",
        f"{len(meta_final)}/24",
    )
    print(
        "Combined :",
        len(combined),
        "| unique Gold:",
        summary["unique_gold_in_combined"],
        "/24",
    )
    print(
        "Both objectives:",
        len(overlap),
        "Gold prompts",
    )

    print()
    print("=== FINAL DIRECTION PAIRS ===")

    for i,x in enumerate(dir_final,1):
        print(
            f"\nD{i:02d} G{x['gold_index']} "
            f"{x['negative_strength']}"
        )
        print(" +", x["chosen_query"])
        print(" -", x["rejected_query"])

    print()
    print("=== FINAL META PAIRS ===")

    for i,x in enumerate(meta_final,1):
        print(
            f"\nM{i:02d} G{x['gold_index']} "
            f"{x['negative_strength']}"
        )
        print(" +", x["chosen_query"])
        print(" -", x["rejected_query"])

    print()
    print("wrote:", DIR_FINAL)
    print("wrote:", META_FINAL)
    print("wrote:", COMBINED)
    print("wrote:", SUMMARY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
    )
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
