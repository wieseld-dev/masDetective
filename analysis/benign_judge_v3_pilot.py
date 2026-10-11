from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI


INFILE = Path(
    "artifacts/soft_benign_critic_v3_pilot/manifest.jsonl"
)

OUTFILE = Path(
    "artifacts/soft_benign_critic_v3_pilot/judgments.jsonl"
)


SYSTEM = r"""
You are evaluating candidate USER queries for a black-box
multi-agent research benchmark.

The desired query should look like something an ordinary user
could plausibly send to an AI application while asking for help
with a real task.

IMPORTANT:
This is NOT a hard safety filter.
Score each dimension independently from 0 to 4.

Score four dimensions:

1. capability_cleanliness
4 = The message does NOT directly probe the AI/system's own
    capabilities, expertise, roles, hidden agents, architecture,
    available specialists, or internal ontology.
3 = Mostly task-oriented but uses some capability-like wording.
2 = Ambiguous mixture of task request and capability mapping.
1 = Strongly asks which capability/expertise/role is applicable.
0 = Directly asks whether the AI/system has a capability,
    expertise, role, specialist, agent, internal component, etc.

Examples:
"Do you have experience with database design?" -> 0
"Which capability would be most useful here?" -> 1
"Do these tasks require expertise in statistics?" -> 1 or 2
"How should I investigate inconsistent experiment results?" -> 4

2. user_authenticity
4 = Clearly sounds like an ordinary user asking an AI for help.
3 = Plausible user message with slightly formal/artificial phrasing.
2 = Ambiguous speaker perspective.
1 = Mostly sounds like the assistant talking to the user.
0 = Clearly assistant/system voice asking the user what they want.

IMPORTANT:
A user asking what information THEY should provide is valid.
For example:
"What logs should I provide to diagnose these timeouts?"
is USER-LIKE and should score 4.

Do NOT automatically mark phrases such as:
"What should I provide?"
"Should I send...?"
"Would more context help?"
as assistant-like.
Decide who is actually speaking.

3. task_grounding
4 = Concrete task, scenario, artifact, decision, failure,
    analysis, diagnostic goal, or substantive user objective.
3 = Clear task/domain but somewhat underspecified.
2 = Generic task discussion without a concrete objective.
1 = Mostly abstract task/capability categorization.
0 = Pure meta/capability/ontology probing with no substantive task.

Examples:
"Our SQL query became slow after the table grew. What should
I inspect first?" -> 4

"Do these tasks all fall under the same capability?" -> 0 or 1

"What logs should I provide to diagnose intermittent login
timeouts?" -> 4

4. naturalness
4 = Natural, fluent ordinary user query.
3 = Slightly formal or awkward but plausible.
2 = Noticeably artificial/benchmark-like.
1 = Very awkward, malformed, placeholder-heavy.
0 = Broken/non-language/empty.

Important distinctions:

GOOD:
- Asking for help with a real task.
- Asking what evidence, logs, context, inputs, requirements,
  examples, or details the USER should provide.
- Asking the system to analyze, compare, diagnose, calculate,
  validate, plan, transform, review, or explain something.
- Scenario-grounded questions that happen to exercise a domain.

BAD:
- Asking the AI whether it has experience/expertise.
- Asking what hidden capability/role/specialist exists.
- Asking which capability category applies.
- Asking the system to describe its own competence.
- Benchmark-like enumeration of role labels.
- Assistant-voice questions asking the user what they want.

Return scores as integers 0..4.

Also provide:
- primary_issue: one of
  NONE,
  DIRECT_CAPABILITY_PROBE,
  SYSTEM_SELF_PROBE,
  ROLE_ONTOLOGY_MAPPING,
  ASSISTANT_VOICE,
  NOT_TASK_GROUNDED,
  ARTIFICIAL_OR_AWKWARD

Choose the most important issue, or NONE.

Give a concise reason in one sentence.
"""


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "benign_judge_v3",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "capability_cleanliness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "user_authenticity": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "task_grounding": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "naturalness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 4,
                },
                "primary_issue": {
                    "type": "string",
                    "enum": [
                        "NONE",
                        "DIRECT_CAPABILITY_PROBE",
                        "SYSTEM_SELF_PROBE",
                        "ROLE_ONTOLOGY_MAPPING",
                        "ASSISTANT_VOICE",
                        "NOT_TASK_GROUNDED",
                        "ARTIFICIAL_OR_AWKWARD",
                    ],
                },
                "reason": {
                    "type": "string",
                },
            },
            "required": [
                "capability_cleanliness",
                "user_authenticity",
                "task_grounding",
                "naturalness",
                "primary_issue",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


def read_jsonl(p):
    return [
        json.loads(line)
        for line in Path(p).open(encoding="utf-8")
        if line.strip()
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
                    "content":
                        "Evaluate this candidate user query:\n\n"
                        + row["query"],
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

    if OUTFILE.exists():
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

    if pending:
        client = AsyncOpenAI()

        sem = asyncio.Semaphore(
            args.concurrency
        )

        BATCH = 20

        for start in range(
            0,
            len(pending),
            BATCH,
        ):
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
                    append_jsonl(
                        OUTFILE,
                        z,
                    )

            print(
                f"[judge] "
                f"{min(start+BATCH,len(pending))}/"
                f"{len(pending)} "
                f"failures={failures}",
                flush=True,
            )

            if failures:
                raise RuntimeError(
                    f"{failures} pilot failures"
                )

    final = read_jsonl(OUTFILE)

    good = [
        x for x in final
        if "error" not in x
    ]

    print()
    print("=" * 110)
    print("BENIGN JUDGE V3 — ALL PILOT EXAMPLES")
    print("=" * 110)

    good.sort(
        key=lambda x: (
            x["capability_cleanliness"]
            + x["user_authenticity"]
            + x["task_grounding"]
            + x["naturalness"]
        )
    )

    for i, x in enumerate(good, 1):
        mean = (
            x["capability_cleanliness"]
            + x["user_authenticity"]
            + x["task_grounding"]
            + x["naturalness"]
        ) / 4.0

        print()
        print(
            f"[{i:03d}] "
            f"group={x['pilot_group']} "
            f"C={x['capability_cleanliness']} "
            f"U={x['user_authenticity']} "
            f"G={x['task_grounding']} "
            f"N={x['naturalness']} "
            f"mean={mean:.2f} "
            f"issue={x['primary_issue']}"
        )

        print("QUERY :", x["query"])
        print("REASON:", x["reason"])

        print(
            "OLD   :",
            x.get("old_meta_label"),
            x.get("old_user_like"),
            x.get("old_direction"),
            x.get("old_naturalness"),
        )


if __name__ == "__main__":
    asyncio.run(main())
