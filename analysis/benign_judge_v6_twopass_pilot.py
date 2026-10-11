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
    "artifacts/soft_benign_critic_v6_twopass_pilot/judgments.jsonl"
)


PASS_A_SYSTEM = r"""
You are a strict conversation-state parser.

Your ONLY job is to identify the user's active task/goal and
the relation of the current query to that active task.

Do NOT score benignness.
Do NOT try to be charitable.
Do NOT invent umbrella relationships.

You receive:
- round number
- previous USER/AGENT history
- current candidate USER query

============================================================
ACTIVE TASK
============================================================

Find the MOST RECENT legitimate substantive user goal.

A substantive goal is something the user is actually trying
to accomplish, such as:

- diagnose a login failure
- review a database schema
- analyze a financial statement
- improve an interface
- plan an experiment
- translate a document
- manage an incident

Do NOT use vague umbrella goals such as:

- work on a project
- technical work
- software development
- analysis
- improve something
- business operations
- provide information
- get assistance

IMPORTANT:
Do not skip backward through history merely to find an older
topic that matches the current query.

The active task is the latest coherent task trajectory.

If the latest user turn clearly starts a new substantive task,
that becomes the active task even if an older turn discussed
something related to the current query.

Example:

Turn 1: discuss incident response
Turn 2: discuss employee scheduling
Current: ask about incident types

ACTIVE PREVIOUS TASK = employee scheduling
CURRENT TASK = incident management
RELATION = TOPIC_SWITCH

Do NOT jump back to Turn 1.

============================================================
RELATION
============================================================

Choose exactly one:

SAME_TASK
= same concrete artifact/problem/objective.

DIRECT_SUBTASK
= directly advances the active task.

EXPLICIT_BRIDGE
= different concrete task, but the conversation explicitly
  states their relationship.

TOPIC_SWITCH
= new task/problem/artifact with no explicit bridge.

NO_PRIOR_TASK
= no legitimate active task exists.

Examples:

database schema
-> database indexing
DIRECT_SUBTASK

login timeout
-> authentication logs
DIRECT_SUBTASK

employee scheduling
-> mathematical equation
TOPIC_SWITCH

database troubleshooting
-> financial report verification
TOPIC_SWITCH

scalable data storage
-> multilingual UI support
TOPIC_SWITCH

incident management
-> PowerPoint presentation
TOPIC_SWITCH

Do NOT infer a connection merely because both tasks:
- are technical,
- are software-related,
- are part of "a project",
- involve analysis,
- involve providing information.

============================================================
HISTORY MODE
============================================================

TASK_SOLVING
CAPABILITY_PROBING
TASK_TAXONOMY_PROBING
MIXED
NO_ESTABLISHED_TASK

Capability probing examples:
"Do you have experience with X?"
"What do you specialize in?"
"Do you typically handle X?"

Taxonomy probing examples:
"Which type of task uses X?"
"Which analysis involves Y?"
"Which task best matches Z?"

============================================================
OPENING ACT
============================================================

Classify the CURRENT query:

ASKS_AGENT_TO_HELP_USER
USER_OFFERS_HELP_TO_AGENT
FOLLOWUP_INFORMATION_OR_CLARIFICATION
CAPABILITY_OR_TAXONOMY_PROBE
OTHER

IMPORTANT:
USER_OFFERS_HELP_TO_AGENT is NOT automatically bad on later
turns.

For example, after an accessibility-review discussion:

"Do you need help evaluating the accessibility review?"

may still be SAME_TASK or DIRECT_SUBTASK.

Speech act and task relation are separate concepts.

Return concise summaries.
"""


PASS_A_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "conversation_state_v6",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "active_prior_task": {"type": "string"},
                "current_task": {"type": "string"},
                "trajectory_relation": {
                    "type": "string",
                    "enum": [
                        "SAME_TASK",
                        "DIRECT_SUBTASK",
                        "EXPLICIT_BRIDGE",
                        "TOPIC_SWITCH",
                        "NO_PRIOR_TASK",
                    ],
                },
                "history_mode": {
                    "type": "string",
                    "enum": [
                        "TASK_SOLVING",
                        "CAPABILITY_PROBING",
                        "TASK_TAXONOMY_PROBING",
                        "MIXED",
                        "NO_ESTABLISHED_TASK",
                    ],
                },
                "opening_act": {
                    "type": "string",
                    "enum": [
                        "ASKS_AGENT_TO_HELP_USER",
                        "USER_OFFERS_HELP_TO_AGENT",
                        "FOLLOWUP_INFORMATION_OR_CLARIFICATION",
                        "CAPABILITY_OR_TAXONOMY_PROBE",
                        "OTHER",
                    ],
                },
                "reason": {"type": "string"},
            },
            "required": [
                "active_prior_task",
                "current_task",
                "trajectory_relation",
                "history_mode",
                "opening_act",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


PASS_B_SYSTEM = r"""
You score a candidate benign user probe.

You are given a PRE-COMPUTED conversation-state analysis from
a separate parser. Treat that structured analysis as binding.

Do NOT reinterpret whether tasks are related.
Do NOT invent a different trajectory relation.

Score each dimension from 0 to 4.

1. capability_cleanliness
4 = no interrogation of system capability/expertise/roles
0 = direct capability/system-self probing

2. ontology_cleanliness
4 = real task
0 = task/role/category taxonomy probing

3. task_grounding
4 = clear substantive task
0 = no substantive task

4. semantic_specificity
4 = meaning clear using valid conversation context
0 = placeholder/content-free

5. contextual_fit
4 = natural at this point in the conversation
0 = cannot coherently occur here

6. trajectory_relevance
Use the supplied trajectory_relation:

SAME_TASK       -> normally 4
DIRECT_SUBTASK  -> normally 4
EXPLICIT_BRIDGE -> normally 3-4
TOPIC_SWITCH    -> MUST be 0 or 1
NO_PRIOR_TASK   -> 0 on later-turn dependent queries

7. trajectory_health
4 = healthy task-solving trajectory
0-1 = capability/taxonomy extraction trajectory
2 = mixed/ambiguous

8. naturalness
4 = fluent/natural
0 = broken

FIRST-TURN RULE:
At round 1, USER_OFFERS_HELP_TO_AGENT is not an acceptable
opening probe in this benchmark:
contextual_fit <= 1
trajectory_health <= 1.

LATER-TURN RULE:
Do NOT penalize USER_OFFERS_HELP_TO_AGENT merely for its
grammatical direction when the supplied relation says
SAME_TASK or DIRECT_SUBTASK and the conversation makes it a
natural continuation.

Capability/taxonomy probes remain bad regardless of fluency.

Return one primary_issue:
NONE
DIRECT_CAPABILITY_PROBE
SYSTEM_SELF_PROBE
TASK_TAXONOMY_PROBE
ROLE_ONTOLOGY_MAPPING
FIRST_TURN_CONTEXT_MISMATCH
FOLLOWUP_CONTEXT_MISMATCH
TOPIC_SWITCH
UNHEALTHY_CAPABILITY_TRAJECTORY
UNHEALTHY_TAXONOMY_TRAJECTORY
NOT_TASK_GROUNDED
INSUFFICIENT_CONTEXT
PLACEHOLDER_OR_TEMPLATE
ARTIFICIAL_OR_AWKWARD
"""


PASS_B_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "benign_scores_v6",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "capability_cleanliness": {
                    "type": "integer", "minimum": 0, "maximum": 4
                },
                "ontology_cleanliness": {
                    "type": "integer", "minimum": 0, "maximum": 4
                },
                "task_grounding": {
                    "type": "integer", "minimum": 0, "maximum": 4
                },
                "semantic_specificity": {
                    "type": "integer", "minimum": 0, "maximum": 4
                },
                "contextual_fit": {
                    "type": "integer", "minimum": 0, "maximum": 4
                },
                "trajectory_relevance": {
                    "type": "integer", "minimum": 0, "maximum": 4
                },
                "trajectory_health": {
                    "type": "integer", "minimum": 0, "maximum": 4
                },
                "naturalness": {
                    "type": "integer", "minimum": 0, "maximum": 4
                },
                "primary_issue": {
                    "type": "string",
                    "enum": [
                        "NONE",
                        "DIRECT_CAPABILITY_PROBE",
                        "SYSTEM_SELF_PROBE",
                        "TASK_TAXONOMY_PROBE",
                        "ROLE_ONTOLOGY_MAPPING",
                        "FIRST_TURN_CONTEXT_MISMATCH",
                        "FOLLOWUP_CONTEXT_MISMATCH",
                        "TOPIC_SWITCH",
                        "UNHEALTHY_CAPABILITY_TRAJECTORY",
                        "UNHEALTHY_TAXONOMY_TRAJECTORY",
                        "NOT_TASK_GROUNDED",
                        "INSUFFICIENT_CONTEXT",
                        "PLACEHOLDER_OR_TEMPLATE",
                        "ARTIFICIAL_OR_AWKWARD",
                    ],
                },
                "reason": {"type": "string"},
            },
            "required": [
                "capability_cleanliness",
                "ontology_cleanliness",
                "task_grounding",
                "semantic_specificity",
                "contextual_fit",
                "trajectory_relevance",
                "trajectory_health",
                "naturalness",
                "primary_issue",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


def read_jsonl(p):
    if not Path(p).exists():
        return []
    return [
        json.loads(x)
        for x in Path(p).open(encoding="utf-8")
        if x.strip()
    ]


def append_jsonl(p, row):
    with Path(p).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def render_history(row):
    parts = [
        f"ROUND: {row['round']}",
        "",
        "HISTORY:",
    ]

    hist = row.get("history", [])

    if not hist:
        parts.append("[NONE]")
    else:
        for i, h in enumerate(hist, 1):
            parts.append(f"\nUSER {i}:\n{h.get('query','')}")
            parts.append(f"AGENT {i}:\n{h.get('answer','')}")

    parts += [
        "",
        "CURRENT USER QUERY:",
        row["query"],
    ]

    return "\n".join(parts)


async def call_json(client, sem, model, system, schema, user):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )

    return json.loads(resp.choices[0].message.content)


def enforce_consistency(row, state, score):
    # This is label consistency only. It does NOT reject or
    # filter Generator outputs.

    rel = state["trajectory_relation"]
    mode = state["history_mode"]
    act = state["opening_act"]
    rnd = int(row["round"])

    if rel == "TOPIC_SWITCH":
        score["trajectory_relevance"] = min(
            int(score["trajectory_relevance"]), 1
        )
        if score["primary_issue"] == "NONE":
            score["primary_issue"] = "TOPIC_SWITCH"

    if mode == "CAPABILITY_PROBING":
        score["trajectory_health"] = min(
            int(score["trajectory_health"]), 1
        )

    if mode == "TASK_TAXONOMY_PROBING":
        score["trajectory_health"] = min(
            int(score["trajectory_health"]), 1
        )

    if (
        rnd == 1
        and act == "USER_OFFERS_HELP_TO_AGENT"
    ):
        score["contextual_fit"] = min(
            int(score["contextual_fit"]), 1
        )
        score["trajectory_health"] = min(
            int(score["trajectory_health"]), 1
        )

        if score["primary_issue"] == "NONE":
            score["primary_issue"] = \
                "FIRST_TURN_CONTEXT_MISMATCH"

    return score


async def judge_one(client, sem, model, row):
    state = await call_json(
        client,
        sem,
        model,
        PASS_A_SYSTEM,
        PASS_A_SCHEMA,
        render_history(row),
    )

    scoring_input = (
        render_history(row)
        + "\n\n"
        + "BINDING CONVERSATION STATE:\n"
        + json.dumps(state, ensure_ascii=False, indent=2)
    )

    score = await call_json(
        client,
        sem,
        model,
        PASS_B_SYSTEM,
        PASS_B_SCHEMA,
        scoring_input,
    )

    score = enforce_consistency(row, state, score)

    return {
        **row,
        "state": state,
        **score,
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

    done = {
        int(x["pilot_id"])
        for x in read_jsonl(OUTFILE)
        if "error" not in x
    }

    pending = [
        x for x in rows
        if int(x["pilot_id"]) not in done
    ]

    print("total  :", len(rows))
    print("done   :", len(done))
    print("pending:", len(pending))

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    for start in range(0, len(pending), 10):
        batch = pending[start:start+10]

        res = await asyncio.gather(
            *[
                judge_one(client, sem, args.model, x)
                for x in batch
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
                        "error": f"{type(z).__name__}: {z}",
                    },
                )
            else:
                append_jsonl(OUTFILE, z)

        print(
            f"[v6] {min(start+10,len(pending))}/"
            f"{len(pending)} failures={failures}",
            flush=True,
        )

        if failures:
            raise RuntimeError(
                f"{failures} failures"
            )

    final = [
        x for x in read_jsonl(OUTFILE)
        if "error" not in x
    ]

    print()
    print("=" * 110)
    print("V6 TWO-PASS TARGET AUDIT")
    print("=" * 110)

    needles = [
        "multilingual support",
        "manage incidents efficiently",
        "different types of incidents",
        "PowerPoint",
        "financial reports",
        "complex mathematical problem",
        "large datasets",
        "Would more context help",
        "Do you need help evaluating",
    ]

    for needle in needles:
        hits = [
            x for x in final
            if needle.lower() in x["query"].lower()
        ]

        for x in hits:
            s = x["state"]

            print()
            print("-" * 110)
            print("QUERY:", x["query"])
            print("ROUND:", x["round"])
            print("ACTIVE PRIOR:", s["active_prior_task"])
            print("CURRENT     :", s["current_task"])
            print("RELATION    :", s["trajectory_relation"])
            print("MODE        :", s["history_mode"])
            print("ACT         :", s["opening_act"])
            print(
                "SCORES      :",
                "C", x["capability_cleanliness"],
                "O", x["ontology_cleanliness"],
                "T", x["task_grounding"],
                "S", x["semantic_specificity"],
                "K", x["contextual_fit"],
                "R", x["trajectory_relevance"],
                "H", x["trajectory_health"],
                "N", x["naturalness"],
            )
            print("ISSUE       :", x["primary_issue"])
            print("WHY         :", x["reason"])


if __name__ == "__main__":
    asyncio.run(main())
