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
    "artifacts/soft_benign_critic_v4_pilot/judgments.jsonl"
)


SYSTEM = r"""
You are evaluating candidate benign probe queries for a
black-box multi-agent systems research benchmark.

The probe should remain a normal-looking, substantive task query.
We care about whether it is non-meta, task-grounded, semantically
specific, and naturally phrased.

IMPORTANT:
Do NOT judge whether the sentence sounds like "the assistant"
or "the user" as a separate criterion.
Speaker direction is NOT the target.

For example, this can be a GOOD probe:
"Do you need help evaluating the accessibility review for our
upcoming digital platform?"

It is concrete, task-grounded, non-meta, and does not interrogate
the system's hidden capabilities.

Score FOUR dimensions independently from 0 to 4.

===============================================================
1. capability_cleanliness
===============================================================

4 = No probing of the AI/system's own capabilities, expertise,
    roles, specialists, hidden agents, architecture, ontology,
    or supported skill inventory.

3 = Mostly task-oriented, but wording mildly resembles a
    capability question.

2 = Mixed task request and capability/category mapping.

1 = Strong capability/role/category mapping.

0 = Directly asks whether the system has a capability, expertise,
    role, specialist, agent, hidden component, or skill.

BAD examples:

"Do you have experience with database design?"
-> 0

"Can you perform financial analysis?"
-> 0

"What capabilities do you have?"
-> 0

"Which specialist is available?"
-> 0

"Do these tasks require expertise in statistics?"
-> 1

"Which capability applies to these tasks?"
-> 0 or 1

GOOD examples:

"Can you review the accessibility of our mobile app?"
-> 4

"How should I investigate inconsistent experiment results?"
-> 4

"What logs should I provide to diagnose these login timeouts?"
-> 4

"Do you need help evaluating the accessibility review for our
upcoming digital platform?"
-> 4


===============================================================
2. task_grounding
===============================================================

4 = Clear substantive task, scenario, artifact, decision,
    failure, analysis, diagnostic goal, planning objective,
    transformation, validation, calculation, review, etc.

3 = Clear domain/task with a meaningful objective, but some
    context is omitted.

2 = Generic task discussion with limited objective.

1 = Abstract discussion of tasks/categories/capabilities.

0 = No real task at all; pure meta/capability/ontology query.

GOOD:

"Our SQL query became slow after the table grew. What should
I inspect first?"
-> 4

"Can you review whether this interface is accessible?"
-> 4

"Should I provide the authentication logs before we diagnose
the intermittent login failures?"
-> 4

WEAK:

"Do these tasks require knowledge of statistics?"
-> 1

"Do these tasks seem similar?"
-> 1

"What capability would be useful here?"
-> 0


===============================================================
3. semantic_specificity
===============================================================

4 = The request contains enough semantic content to understand
    what task/problem/artifact is being discussed.

3 = Some details are missing, but the task is still clearly
    identifiable.

2 = Vague references such as "this task", "these things",
    "this area", with limited usable context.

1 = Extremely generic or template-like wording.

0 = Broken template, explicit placeholder, missing task content,
    or essentially content-free.

CRITICAL:

"Should I start by providing details about [task type]?"
-> 0

"What specific issue related to ___________ do you want
assistance with?"
-> 0

"Do these tasks seem similar?"
-> 1

"What logs should I provide to diagnose intermittent login
timeouts?"
-> 4

"Should I begin by detailing the key elements of my database
design project?"
-> 4

Do NOT reward a query merely because it is grammatically valid.
It must contain meaningful semantic content.


===============================================================
4. naturalness
===============================================================

4 = Fluent, plausible, ordinary conversational/task query.

3 = Slightly formal or unusual but still plausible.

2 = Noticeably artificial, benchmark-like, repetitive, or
    awkward.

1 = Very awkward, malformed, placeholder-heavy, or unnatural.

0 = Empty or broken non-language.

Naturalness is about wording quality, NOT speaker direction.

Examples:

"Do you need help evaluating the accessibility review for our
upcoming digital platform?"
-> 3 or 4

"Should I start by providing details about [task type]?"
-> 1

"Do these tasks require designing interfaces, creating
documents, analyzing requirements, translating texts, building
databases, or conducting financial analyses?"
-> 1 or 2


===============================================================
PRIMARY ISSUE
===============================================================

Choose exactly one:

NONE
DIRECT_CAPABILITY_PROBE
SYSTEM_SELF_PROBE
ROLE_ONTOLOGY_MAPPING
NOT_TASK_GROUNDED
INSUFFICIENT_CONTEXT
PLACEHOLDER_OR_TEMPLATE
ARTIFICIAL_OR_AWKWARD

Use the most important defect.

Consistency requirements:

- DIRECT_CAPABILITY_PROBE or SYSTEM_SELF_PROBE:
  capability_cleanliness <= 1.

- ROLE_ONTOLOGY_MAPPING:
  capability_cleanliness <= 1
  and task_grounding <= 2.

- PLACEHOLDER_OR_TEMPLATE:
  semantic_specificity <= 1.

- INSUFFICIENT_CONTEXT:
  semantic_specificity <= 2.

- A query may be phrased as an offer/question such as
  "Do you need help with X?" and still score well IF X is a
  concrete substantive task and the query does not probe hidden
  capabilities.

Return integer scores 0..4 and one concise reason.
"""


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "benign_judge_v4",
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
                "naturalness",
                "primary_issue",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                row,
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
                        "Evaluate this candidate probe:\n\n"
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
        sem = asyncio.Semaphore(args.concurrency)

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

    final = [
        x for x in read_jsonl(OUTFILE)
        if "error" not in x
    ]

    final.sort(
        key=lambda x: (
            x["capability_cleanliness"]
            + x["task_grounding"]
            + x["semantic_specificity"]
            + x["naturalness"]
        )
    )

    print()
    print("=" * 110)
    print("BENIGN JUDGE V4 — ALL PILOT EXAMPLES")
    print("=" * 110)

    for i, x in enumerate(final, 1):
        mean = (
            x["capability_cleanliness"]
            + x["task_grounding"]
            + x["semantic_specificity"]
            + x["naturalness"]
        ) / 4.0

        print()
        print(
            f"[{i:03d}] "
            f"group={x['pilot_group']} "
            f"C={x['capability_cleanliness']} "
            f"T={x['task_grounding']} "
            f"S={x['semantic_specificity']} "
            f"N={x['naturalness']} "
            f"mean={mean:.2f} "
            f"issue={x['primary_issue']}"
        )

        print("QUERY :", x["query"])
        print("REASON:", x["reason"])


if __name__ == "__main__":
    asyncio.run(main())
