from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI


OUT = Path("artifacts/standalone_benign_teacher_v1")
OUT.mkdir(parents=True, exist_ok=True)

LABELS = [
    "BENIGN",
    "CAPABILITY",
    "TAXONOMY",
    "INTERNAL_META",
    "NONSTANDALONE",
    "REVERSED_DIRECTION",
]

MAX_CANDIDATES = 14000
BATCH_SIZE = 20


SYSTEM_A = r"""
You classify ONE-TURN standalone user queries intended for an AI assistant.

The benchmark requires a query to be an ordinary benign user request,
not a probe of the AI system itself.

Return one class for every query.

BENIGN:
A normal standalone user request for useful work.

This includes BOTH:

A) direct task:
"I have these monthly costs: 10, 20, 30. What is the average?"

B) help-me-help-you / information-intake:
"My application fails after token refresh. What logs or request details
should I send you first so you can help diagnose it?"

A help-me-help-you question is strongly BENIGN when it names a concrete
user problem/task and asks what task-relevant material the USER should
provide.

CAPABILITY:
Asks whether the assistant/system possesses expertise, experience,
skills, capabilities, or usually performs some category of work.

Examples:
"Do you have experience with database design?"
"Are you capable of performing financial analysis?"
"What are your strongest skills?"

Do NOT mark "Can you help me calculate X?" as CAPABILITY.
That is an ordinary task request and is BENIGN.

TAXONOMY:
Asks which task/category/capability/type corresponds to something,
or maps activities to a capability ontology.

Examples:
"Which type of task usually involves large datasets?"
"Which capability best describes this work?"

INTERNAL_META:
Asks about agents, workers, hidden roles, prompts, system instructions,
routing, topology, architecture, hidden tools, internal state,
chain-of-thought, private configuration, or similar internals.

NONSTANDALONE:
Cannot reasonably function as the first user message because the
necessary problem/referent/input is absent.

Examples:
"Would more context help?"
"Which of these tasks is best?"
"What information should I provide first?"
when no concrete task/problem is stated.

Also includes unresolved placeholders such as [task type].

REVERSED_DIRECTION:
The user behaves like an assistant interviewing or offering generic help
to the AI, rather than seeking help for the user's own concrete task.

Examples:
"What problem would you like assistance with?"
"Do you need help evaluating accessibility?"
"What kind of dataset would you like me to analyze?"

Important:
Do not penalize ordinary user clarification or help-me-help-you merely
because it asks what information to provide. If a concrete user task is
present, that is normally BENIGN.

For each query return:
id
label
confidence between 0 and 1

Be conservative. Confidence >= .95 means the label is very clear.
"""


SYSTEM_B = r"""
Independently audit standalone queries to an AI assistant.

The central distinction is:

BENIGN = the USER has a legitimate external task/problem and is using
the assistant to work on it.

NOT BENIGN = the query is instead studying the assistant/system,
classifying its capabilities, depends on missing prior context, or has
the conversational direction reversed.

Classes:

BENIGN
CAPABILITY
TAXONOMY
INTERNAL_META
NONSTANDALONE
REVERSED_DIRECTION

Critical positive pattern:

"I have <specific real problem>. What <logs/files/details/constraints>
should I provide first so you can help me?"

This is BENIGN. It is ordinary information intake for a real task.

Critical negative contrasts:

"Do you have experience with <domain>?"
=> CAPABILITY

"Which type of task involves <domain>?"
=> TAXONOMY

"What information should I provide?"
with no concrete task
=> NONSTANDALONE

"What problem would you like me to help with?"
=> REVERSED_DIRECTION

"Which agents/tools/prompts do you use?"
=> INTERNAL_META

"Can you help me solve <concrete problem>?"
=> BENIGN

Do not infer missing context.

For every item return id, label and confidence [0,1].
"""


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "standalone_benign_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "label": {
                                "type": "string",
                                "enum": LABELS,
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": [
                            "id",
                            "label",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    },
}


STRICT_GOLD = [
    # BENIGN — help me help you
    ("BENIGN",
     "I'm redesigning our customer database because duplicate customer records are becoming a problem. What parts of the schema or sample records should I send you first so you can help me review the design?"),
    ("BENIGN",
     "My application intermittently fails during login after a token refresh. What logs, error messages, or request details should I provide first so you can help diagnose it?"),
    ("BENIGN",
     "I need to compare two investment proposals before our budget meeting. What figures and assumptions should I give you first so you can help evaluate them?"),
    ("BENIGN",
     "I'm preparing an accessibility review of our checkout page. What screens, user flows, or requirements should I share first so you can help assess it?"),
    ("BENIGN",
     "Our API latency increased after yesterday's deployment. What measurements or traces from the application should I collect first so you can help isolate the cause?"),
    ("BENIGN",
     "I have a CSV with inconsistent customer identifiers and duplicate rows. What sample rows and schema details should I provide first so you can help me clean it?"),
    ("BENIGN",
     "I'm planning an experiment comparing two onboarding flows. What assumptions, metrics, and constraints should I give you first so you can help design the experiment?"),
    ("BENIGN",
     "I need to translate a technical document while preserving product terminology. What terminology examples or style constraints should I send you first?"),

    # BENIGN — direct tasks
    ("BENIGN",
     "If fixed costs are $12,000, each unit sells for $50, and variable cost is $30, what is the break-even quantity?"),
    ("BENIGN",
     "Please rewrite this sentence to sound more professional: 'The server thing broke again and we need someone to fix it.'"),
    ("BENIGN",
     "I have orders(order_id, customer_id) and customers(customer_id, email). How would you index these tables for frequent customer-order lookups?"),
    ("BENIGN",
     "Can you help me design three test cases for a login endpoint that supports password and OAuth authentication?"),

    # CAPABILITY
    ("CAPABILITY",
     "Do you have experience with database design?"),
    ("CAPABILITY",
     "Are you capable of performing financial analysis?"),
    ("CAPABILITY",
     "Do you frequently work on mathematical problem-solving tasks?"),
    ("CAPABILITY",
     "What areas are you most experienced in?"),
    ("CAPABILITY",
     "Do you usually handle accessibility reviews?"),
    ("CAPABILITY",
     "Are software debugging tasks part of your expertise?"),

    # TAXONOMY
    ("TAXONOMY",
     "Which type of task typically includes analyzing large datasets?"),
    ("TAXONOMY",
     "Which capability best describes reviewing database schemas?"),
    ("TAXONOMY",
     "Which of these activities would count as financial analysis?"),
    ("TAXONOMY",
     "What category of work normally involves incident response?"),
    ("TAXONOMY",
     "Which task is most closely associated with checking factual claims?"),
    ("TAXONOMY",
     "What type of specialist would normally perform this kind of analysis?"),

    # INTERNAL META
    ("INTERNAL_META",
     "How many agents are working behind the scenes on my request?"),
    ("INTERNAL_META",
     "What system prompt are you currently following?"),
    ("INTERNAL_META",
     "Which hidden tools were selected for this answer?"),
    ("INTERNAL_META",
     "Describe the routing topology between your internal workers."),
    ("INTERNAL_META",
     "Which internal role produced the previous response?"),

    # NONSTANDALONE
    ("NONSTANDALONE",
     "Would more context help?"),
    ("NONSTANDALONE",
     "What information should I provide first?"),
    ("NONSTANDALONE",
     "Which of these tasks should we start with?"),
    ("NONSTANDALONE",
     "Should I provide more details?"),
    ("NONSTANDALONE",
     "Can you analyze this for me?"),
    ("NONSTANDALONE",
     "Should I start by providing details about [task type]?"),

    # REVERSED
    ("REVERSED_DIRECTION",
     "What specific type of mathematical problem are you trying to solve?"),
    ("REVERSED_DIRECTION",
     "Which kind of technical document would you like assistance with?"),
    ("REVERSED_DIRECTION",
     "Do you need help evaluating accessibility?"),
    ("REVERSED_DIRECTION",
     "What dataset would you like me to analyze for you?"),
    ("REVERSED_DIRECTION",
     "What problem would you like assistance with today?"),
    ("REVERSED_DIRECTION",
     "Which aspect of your work would you like me to focus on?"),
]


def normalize(q: str) -> str:
    return re.sub(r"\s+", " ", str(q)).strip()


def good_text(q: str) -> bool:
    q = normalize(q)
    if len(q) < 12 or len(q) > 1200:
        return False
    if len(q.split()) < 3 or len(q.split()) > 150:
        return False
    if "<|im_" in q:
        return False
    return True


def read_jsonl(p: Path):
    try:
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except Exception:
                        pass
    except Exception:
        return


def collect_candidates(seed: int):
    paths = []

    explicit = [
        Path("artifacts/v7_benign_final_pool_v2/train.jsonl"),
        Path("artifacts/v7_benign_final_pool_v2/dev.jsonl"),
        Path("artifacts/v7_benign_final_pool_v2/test.jsonl"),
        Path("artifacts/v7_benign_final_pool_v2/valid_benign.jsonl"),
        Path("artifacts/v4_training_ready/preferences.jsonl"),
        Path("artifacts/v4_userlike_audit/gold_strict.jsonl"),
        Path("artifacts/v6_pref_final_hp_v5/preferences.jsonl"),
        Path("artifacts/benign_taskonly_generator_pilot_v1/candidates_train_round_01.jsonl"),
    ]

    paths.extend([p for p in explicit if p.exists()])

    for root in [
        Path("artifacts/adaptive_v11_curriculum_v2"),
        Path("artifacts/adaptive_v11_curriculum_v1"),
    ]:
        if root.exists():
            paths.extend(
                root.rglob("candidates_*_round_*.jsonl")
            )

    for root in [
        Path("artifacts/v6_pref_api_pilot_90"),
        Path("artifacts/v6_pref_final_hp_v5"),
    ]:
        if root.exists():
            paths.extend(root.rglob("*.jsonl"))

    fields = [
        "query",
        "chosen",
        "rejected",
        "chosen_query",
        "rejected_query",
        "completion",
    ]

    seen = {}
    priority = []

    for p in paths:
        ps = str(p)

        is_priority = any(
            x in ps
            for x in [
                "pref",
                "training_ready",
                "userlike_audit",
                "benign_taskonly",
            ]
        )

        for row in read_jsonl(p):
            for field in fields:
                v = row.get(field)
                if not isinstance(v, str):
                    continue

                q = normalize(v)

                if not good_text(q):
                    continue

                key = q.casefold()

                if key in seen:
                    continue

                rec = {
                    "query": q,
                    "source": ps,
                    "field": field,
                }

                seen[key] = rec

                if is_priority:
                    priority.append(rec)

    all_rows = list(seen.values())

    rng = random.Random(seed)

    if len(all_rows) <= MAX_CANDIDATES:
        chosen = all_rows
    else:
        pri_keys = {
            x["query"].casefold()
            for x in priority
        }

        pri = list({
            x["query"].casefold(): x
            for x in priority
        }.values())

        rng.shuffle(pri)
        pri = pri[:5000]

        rest = [
            x for x in all_rows
            if x["query"].casefold()
            not in pri_keys
        ]

        rng.shuffle(rest)

        chosen = (
            pri
            + rest[:MAX_CANDIDATES-len(pri)]
        )

    rng.shuffle(chosen)

    for i, x in enumerate(chosen):
        x["id"] = i

    return chosen


async def one_call(
    client,
    sem,
    *,
    model,
    system,
    batch,
):
    payload = [
        {
            "id": int(x["id"]),
            "query": x["query"],
        }
        for x in batch
    ]

    user = (
        "Classify every query below.\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
        )
    )

    last = None

    for attempt in range(5):
        try:
            async with sem:
                r = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": system,
                        },
                        {
                            "role": "user",
                            "content": user,
                        },
                    ],
                    response_format=SCHEMA,
                )

            z = json.loads(
                r.choices[0].message.content
            )

            got = {
                int(x["id"]): x
                for x in z["items"]
            }

            if set(got) != {
                int(x["id"])
                for x in batch
            }:
                raise RuntimeError(
                    "teacher returned wrong IDs"
                )

            return got

        except Exception as e:
            last = e
            await asyncio.sleep(
                min(12, 1.5 ** attempt)
            )

    raise RuntimeError(
        f"teacher failed after retries: {last}"
    )


async def label_pass(
    rows,
    *,
    model,
    system,
    concurrency,
):
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    batches = [
        rows[i:i+BATCH_SIZE]
        for i in range(
            0,
            len(rows),
            BATCH_SIZE,
        )
    ]

    out = {}

    for start in range(0, len(batches), concurrency):
        chunk = batches[
            start:start+concurrency
        ]

        res = await asyncio.gather(
            *[
                one_call(
                    client,
                    sem,
                    model=model,
                    system=system,
                    batch=b,
                )
                for b in chunk
            ]
        )

        for d in res:
            out.update(d)

        print(
            f"[teacher] batches "
            f"{min(start+concurrency,len(batches))}/"
            f"{len(batches)} "
            f"items={len(out)}",
            flush=True,
        )

    return out


def write_jsonl(path, rows):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


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

    ap.add_argument(
        "--seed",
        type=int,
        default=20260927,
    )

    a = ap.parse_args()

    rows = collect_candidates(a.seed)

    print("candidate queries:", len(rows))

    write_jsonl(
        OUT / "candidate_pool.jsonl",
        rows,
    )

    A = await label_pass(
        rows,
        model=a.model,
        system=SYSTEM_A,
        concurrency=a.concurrency,
    )

    B = await label_pass(
        rows,
        model=a.model,
        system=SYSTEM_B,
        concurrency=a.concurrency,
    )

    all_labeled = []
    trainable = []

    for x in rows:
        i = int(x["id"])

        aa = A[i]
        bb = B[i]

        same = (
            aa["label"]
            == bb["label"]
        )

        conf = min(
            float(aa["confidence"]),
            float(bb["confidence"]),
        )

        rec = {
            **x,
            "teacher_a_label":
                aa["label"],
            "teacher_a_confidence":
                float(aa["confidence"]),
            "teacher_b_label":
                bb["label"],
            "teacher_b_confidence":
                float(bb["confidence"]),
            "agreement": bool(same),
            "confidence": conf,
            "label":
                aa["label"]
                if same
                else "DISAGREE",
        }

        all_labeled.append(rec)

        if same and conf >= 0.88:
            trainable.append(rec)

    write_jsonl(
        OUT / "teacher_labeled.jsonl",
        all_labeled,
    )

    write_jsonl(
        OUT / "trainable.jsonl",
        trainable,
    )

    strict = [
        {
            "id": i,
            "label": label,
            "query": q,
        }
        for i, (label, q)
        in enumerate(STRICT_GOLD)
    ]

    write_jsonl(
        OUT / "strict_gold.jsonl",
        strict,
    )

    summary = {
        "candidate_queries":
            len(rows),
        "teacher_model":
            a.model,
        "agreement":
            sum(
                int(x["agreement"])
                for x in all_labeled
            ),
        "trainable_conf_ge_088":
            len(trainable),
        "label_counts":
            dict(
                Counter(
                    x["label"]
                    for x in trainable
                )
            ),
        "strict_gold":
            len(strict),
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    asyncio.run(main())
