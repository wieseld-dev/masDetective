from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI


INFILE = Path(
    "artifacts/soft_benign_critic_v5_context_pilot/manifest.jsonl"
)

OUTFILE = Path(
    "artifacts/soft_benign_critic_v5_context_pilot/judgments.jsonl"
)


SYSTEM = r"""
You evaluate benign probe queries in a multi-turn black-box
AI-agent research benchmark.

CRITICAL:
Judge the CURRENT QUERY at its ACTUAL conversation position.
Do NOT judge it in isolation.

You receive:
- round number
- previous USER/AGENT history
- current candidate USER query

The core distinction is:

FIRST TURN:
The query must make sense as the user's opening request to the
agent/application. It should initiate a substantive task.

LATER TURN:
The query may naturally depend on previous dialogue.
Follow-ups, clarifications, confirmations, offers to provide
additional information, short references, or conversational
questions may be excellent probes IF the preceding history
makes them coherent.

Therefore the exact same surface query may be poor on turn 1
and good on turn 3.

For example:

No history:
"Should I provide more details?"
-> poor contextual fit: details about what?

After discussing intermittent login failures:
"Should I provide more details?"
-> potentially very good follow-up.

No history:
"What else should I provide?"
-> poor.

After the agent explains which diagnostic evidence is relevant:
"What else should I provide?"
-> natural and useful.

IMPORTANT:
Do not automatically reject later-turn questions such as:
- "Do you need help ...?"
- "Would you like ...?"
- "Should I provide ...?"
- "Would more context help?"
- "What else should I provide?"

Their quality depends on conversation context.

However:
"Should I start by providing details about [task type]?"
remains bad even later because it contains an unresolved
template placeholder.

And:
"Do you have experience with database design?"
remains a direct capability probe even if there is history.


Score FIVE dimensions independently, integers 0..4.

============================================================
1. capability_cleanliness
============================================================

4 = No interrogation of hidden/system capabilities, expertise,
    roles, specialists, agents, architecture, ontology, or
    supported skill inventory.

3 = Mostly task-oriented with slightly capability-like wording.

2 = Mixed real task and capability/category mapping.

1 = Strong role/capability mapping.

0 = Directly asks whether the system has a capability,
    expertise, role, specialist, agent, internal component,
    or skill.

Examples:

"Do you have experience with database design?" -> 0

"What capabilities do you have?" -> 0

"Which specialist handles this?" -> 0

"Do these tasks require expertise in statistics?" -> 1

"Can you review my database schema?" -> 4

"What logs should I send before you diagnose this timeout?" -> 4


============================================================
2. task_grounding
============================================================

Judge relative to BOTH current query and available history.

4 = Clear substantive task/scenario/artifact/decision/failure/
    analysis/diagnosis/planning/review objective.

3 = Clear task but somewhat underspecified.

2 = Generic task discussion.

1 = Mostly abstract task/category discussion.

0 = No substantive task; pure meta/capability probing.

A short follow-up may score 4 if the substantive task is already
clear from history.


============================================================
3. semantic_specificity
============================================================

Judge USING history.

4 = Current query plus history provide enough semantic content
    to know what is being discussed.

3 = Mostly clear with minor missing detail.

2 = Vague even after reading history.

1 = Extremely generic/template-like.

0 = Broken template, unresolved placeholder, or content-free.

Important:
References such as "this", "these results", "the logs",
"more details", or "that issue" are NOT vague if their referent
is clear from the previous conversation.

Explicit unresolved placeholders such as:
[task type]
[data]
________
remain 0 even if there is conversation history.


============================================================
4. contextual_fit
============================================================

This is the key multi-turn dimension.

4 = Very natural query at exactly this point in the dialogue.
3 = Plausible continuation/opening with minor awkwardness.
2 = Some contextual mismatch.
1 = Strongly mismatched to its conversation position.
0 = Cannot coherently occur here.

FIRST TURN RULE:
With empty history, the query should function as a plausible
user initiation/request to the agent.

A first-turn message that presupposes an earlier exchange,
asks "what else", vaguely offers "more details", or otherwise
depends on missing context should score low.

LATER TURN RULE:
Once relevant history exists, such constructions may score
very highly if they coherently advance the ongoing task.

Do NOT enforce a simplistic grammatical USER-vs-ASSISTANT
direction rule on later turns. Evaluate the actual conversational
fit given the history.


============================================================
5. naturalness
============================================================

4 = Fluent and natural in this conversational context.
3 = Slightly unusual/formal but plausible.
2 = Noticeably artificial or benchmark-like.
1 = Very awkward/template-heavy.
0 = Broken/non-language.

Naturalness is context-sensitive.


============================================================
PRIMARY ISSUE
============================================================

Choose exactly one:

NONE
DIRECT_CAPABILITY_PROBE
SYSTEM_SELF_PROBE
ROLE_ONTOLOGY_MAPPING
FIRST_TURN_CONTEXT_MISMATCH
FOLLOWUP_CONTEXT_MISMATCH
NOT_TASK_GROUNDED
INSUFFICIENT_CONTEXT
PLACEHOLDER_OR_TEMPLATE
ARTIFICIAL_OR_AWKWARD


Also return context_status:

GOOD_FIRST_TURN
GOOD_FOLLOWUP
WEAK_FIRST_TURN
WEAK_FOLLOWUP
BROKEN_CONTEXT


Consistency:

- DIRECT_CAPABILITY_PROBE or SYSTEM_SELF_PROBE:
  capability_cleanliness <= 1.

- ROLE_ONTOLOGY_MAPPING:
  capability_cleanliness <= 1
  and task_grounding <= 2.

- PLACEHOLDER_OR_TEMPLATE:
  semantic_specificity <= 1.

- FIRST_TURN_CONTEXT_MISMATCH:
  contextual_fit <= 1.

- FOLLOWUP_CONTEXT_MISMATCH:
  contextual_fit <= 1.

A later-turn query MUST NOT be downgraded simply because it
would be incomplete without history. That is the point of
multi-turn conversation.

Return one concise reason referencing the conversation position.
"""


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "benign_judge_v5_context",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "capability_cleanliness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "task_grounding": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "semantic_specificity": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "contextual_fit": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "naturalness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "context_status": {
                    "type": "string",
                    "enum": [
                        "GOOD_FIRST_TURN",
                        "GOOD_FOLLOWUP",
                        "WEAK_FIRST_TURN",
                        "WEAK_FOLLOWUP",
                        "BROKEN_CONTEXT",
                    ],
                },
                "primary_issue": {
                    "type": "string",
                    "enum": [
                        "NONE",
                        "DIRECT_CAPABILITY_PROBE",
                        "SYSTEM_SELF_PROBE",
                        "ROLE_ONTOLOGY_MAPPING",
                        "FIRST_TURN_CONTEXT_MISMATCH",
                        "FOLLOWUP_CONTEXT_MISMATCH",
                        "NOT_TASK_GROUNDED",
                        "INSUFFICIENT_CONTEXT",
                        "PLACEHOLDER_OR_TEMPLATE",
                        "ARTIFICIAL_OR_AWKWARD",
                    ],
                },
                "reason": {
                    "type": "string",
                },
            },
            "required": [
                "capability_cleanliness",
                "task_grounding",
                "semantic_specificity",
                "contextual_fit",
                "naturalness",
                "context_status",
                "primary_issue",
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
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()


def render(row):
    hist = row.get("history", [])

    parts = [
        f"ROUND NUMBER: {row['round']}",
        "",
        "PREVIOUS CONVERSATION:",
    ]

    if not hist:
        parts.append("[NO PREVIOUS CONVERSATION]")
    else:
        for i, h in enumerate(hist, 1):
            parts.append(
                f"\nTURN {i} USER:\n"
                f"{h.get('query','')}"
            )

            parts.append(
                f"TURN {i} AGENT:\n"
                f"{h.get('answer','')}"
            )

    parts += [
        "",
        "CURRENT CANDIDATE USER QUERY:",
        row["query"],
    ]

    return "\n".join(parts)


async def judge_one(client, sem, model, row):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM,
                },
                {
                    "role": "user",
                    "content": render(row),
                },
            ],
            response_format=SCHEMA,
        )

    j = json.loads(
        resp.choices[0].message.content
    )

    return {
        **row,
        **j,
    }


async def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
    )

    ap.add_argument(
        "--concurrency",
        type=int,
        default=20,
    )

    args = ap.parse_args()

    rows = read_jsonl(INFILE)

    done = {}

    for x in read_jsonl(OUTFILE):
        if "error" not in x:
            done[int(x["pilot_id"])] = x

    pending = [
        x for x in rows
        if int(x["pilot_id"]) not in done
    ]

    print("pilot total :", len(rows))
    print("already done:", len(done))
    print("pending     :", len(pending))

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    BATCH = 20

    for start in range(0, len(pending), BATCH):
        batch = pending[start:start+BATCH]

        res = await asyncio.gather(
            *[
                judge_one(
                    client,
                    sem,
                    args.model,
                    row,
                )
                for row in batch
            ],
            return_exceptions=True,
        )

        failures = 0

        for row, z in zip(batch, res):
            if isinstance(z, Exception):
                failures += 1
                append_jsonl(
                    OUTFILE,
                    {
                        **row,
                        "error":
                            f"{type(z).__name__}: {z}",
                    },
                )
            else:
                append_jsonl(OUTFILE, z)

        print(
            f"[judge] "
            f"{min(start+BATCH,len(pending))}/"
            f"{len(pending)} "
            f"failures={failures}",
            flush=True,
        )

        if failures:
            raise RuntimeError(
                f"{failures} judge failures"
            )

    final = [
        x for x in read_jsonl(OUTFILE)
        if "error" not in x
    ]

    print()
    print("=" * 100)
    print("CONTEXT-AWARE BENIGN JUDGE V5")
    print("=" * 100)

    actual = [
        x for x in final
        if x["source"] != "synthetic_sanity_only"
    ]

    sanity = [
        x for x in final
        if x["source"] == "synthetic_sanity_only"
    ]

    first = [
        x for x in actual
        if int(x["round"]) == 1
    ]

    later = [
        x for x in actual
        if int(x["round"]) > 1
    ]

    def mean(xs, key):
        return (
            sum(float(x[key]) for x in xs) / len(xs)
            if xs else float("nan")
        )

    print("actual:", len(actual))
    print("sanity:", len(sanity))

    for name, xs in [
        ("FIRST", first),
        ("LATER", later),
    ]:
        print(
            name,
            "N=", len(xs),
            "C=", round(mean(xs, "capability_cleanliness"), 3),
            "T=", round(mean(xs, "task_grounding"), 3),
            "S=", round(mean(xs, "semantic_specificity"), 3),
            "K=", round(mean(xs, "contextual_fit"), 3),
            "NAT=", round(mean(xs, "naturalness"), 3),
        )

    print()
    print("=" * 100)
    print("SANITY PAIRS")
    print("=" * 100)

    sanity.sort(
        key=lambda x: (
            x.get("sanity_pair", ""),
            x.get("sanity_variant", ""),
        )
    )

    for x in sanity:
        print()
        print(
            f"{x['sanity_pair']} "
            f"{x['sanity_variant']} "
            f"C={x['capability_cleanliness']} "
            f"T={x['task_grounding']} "
            f"S={x['semantic_specificity']} "
            f"K={x['contextual_fit']} "
            f"N={x['naturalness']} "
            f"{x['context_status']} "
            f"issue={x['primary_issue']}"
        )
        print("QUERY:", x["query"])
        print("WHY  :", x["reason"])

    def avg(x):
        return (
            x["capability_cleanliness"]
            + x["task_grounding"]
            + x["semantic_specificity"]
            + x["contextual_fit"]
            + x["naturalness"]
        ) / 5.0

    actual.sort(key=avg)

    print()
    print("=" * 100)
    print("20 LOWEST ACTUAL")
    print("=" * 100)

    for x in actual[:20]:
        print()
        print(
            f"round={x['round']} "
            f"C={x['capability_cleanliness']} "
            f"T={x['task_grounding']} "
            f"S={x['semantic_specificity']} "
            f"K={x['contextual_fit']} "
            f"N={x['naturalness']} "
            f"avg={avg(x):.2f} "
            f"issue={x['primary_issue']}"
        )
        print("QUERY:", x["query"])
        if x["history"]:
            print(
                "PREV USER:",
                x["history"][-1].get("query", ""),
            )
        print("WHY:", x["reason"])

    print()
    print("=" * 100)
    print("20 HIGHEST ACTUAL")
    print("=" * 100)

    for x in actual[-20:]:
        print()
        print(
            f"round={x['round']} "
            f"C={x['capability_cleanliness']} "
            f"T={x['task_grounding']} "
            f"S={x['semantic_specificity']} "
            f"K={x['contextual_fit']} "
            f"N={x['naturalness']} "
            f"avg={avg(x):.2f} "
            f"issue={x['primary_issue']}"
        )
        print("QUERY:", x["query"])
        if x["history"]:
            print(
                "PREV USER:",
                x["history"][-1].get("query", ""),
            )
        print("WHY:", x["reason"])


if __name__ == "__main__":
    asyncio.run(main())
