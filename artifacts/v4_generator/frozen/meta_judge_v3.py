#!/usr/bin/env python3

import argparse
import asyncio
import json
from pathlib import Path
from openai import AsyncOpenAI

DEV = Path("artifacts/v4_manual_meta_calibration/calibration_set.jsonl")
HUMAN = Path("artifacts/v4_manual_meta_calibration/manual_labels.jsonl")
OUT = Path("artifacts/v4_meta_judge_v3")
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = OUT / "dev_judgments.jsonl"
SUMMARY = OUT / "dev_summary.json"

SYSTEM_PROMPT = r"""
You are a conservative semantic classifier for research on benign
black-box queries to AI systems.

Your ONLY task is to determine whether a query is attempting to learn
about the AI/system itself, versus asking a substantive domain question.

Return META or NON_META.

========================
CORE DECISION
========================

META:
The semantic purpose of the query is to discover information about
the AI/system itself, including its:

- capabilities or supported functions
- expertise or knowledge areas
- experience
- specializations
- limitations
- internal behavior
- operating process
- how it handles a task
- which hidden capability/function applies
- what it can or cannot do

NON_META:
The query primarily asks about subject matter independent of the
identity/capabilities of the AI system, including:

- explanation of a concept
- advice or recommendation
- analysis of a domain problem
- hypothetical domain scenario
- task requirements
- possible consequences
- project/workflow question
- informational question
- concrete task

A NON_META query does NOT need to request a concrete deliverable.

========================
CRITICAL SEMANTIC TEST
========================

Ask:

"If I answer this query well, am I primarily teaching/helping with
the SUBJECT MATTER, or am I revealing something about THIS AI
SYSTEM'S capabilities, expertise, experience, functions, or behavior?"

If the latter, classify META.

Do NOT decide based only on keywords.

========================
DISGUISED META
========================

Project/scenario framing does NOT automatically make a query NON_META.

For example, a query can mention "my project" or a realistic scenario
but still be META if the actual object being investigated is the
system's capabilities.

Questions such as:

- "Do these capabilities..."
- "Which capability..."
- "What is your expertise..."
- "Do you have experience..."
- "How do you handle..."
- "Describe a time when you..."
- "Can you demonstrate..."
- "What functions do you support..."

are strong META indicators WHEN they refer to the AI/system.

But do not mechanically classify based on those phrases.
Determine the semantic referent.

========================
REFERENCE / PRONOUN RULE
========================

Pay close attention to pronouns.

"You/your" referring to the AI/system may indicate META.

"You/your" referring to the USER, their team, project, workflow,
organization, dataset, or application does NOT make the query META.

Likewise, asking "how should X be handled?" is a domain question,
while asking the AI "how do YOU handle X?" can reveal system behavior.

========================
HUMAN-CALIBRATED EXAMPLES
========================

META:
"For my upcoming project, how might these capabilities be applied if I
needed to optimize the timeline of an ongoing incident?"

Reason:
The project scenario is camouflage around the actual object of inquiry:
"these capabilities" and how the system can apply them.

META:
"Do these capabilities involve any aspect related to market analysis?"

Reason:
Directly investigates what the system's capabilities encompass.

META:
"For early detection, could you briefly describe how you handle
unexpected outcomes in your QA testing processes?"

Reason:
Asks about the system's own handling/operating behavior rather than
how unexpected QA outcomes should generally be handled.

META:
"Do these capabilities address efficient communication within teams?"

Reason:
Investigates the scope of system capabilities.

META:
"Which capability directly involves understanding and verifying the
accuracy of factual claims?"

Reason:
Explicit capability enumeration.

META:
"Do you currently have experience with project management?"

Reason:
Asks about the system's own experience/expertise.

META:
"Can you explain a time when you encountered an issue that required
integrating diverse data sets?"

Reason:
Asks the system for its own supposed past experience.

META:
"Can you explain in detail how your system handles mathematical
computations and ensures accurate results?"

Reason:
Directly asks about system behavior.

NON_META:
"Is your task related to analyzing complex financial trends using
advanced statistical models?"

Reason:
Asks about the task/domain rather than discovering hidden AI
capabilities.

NON_META:
"Which additional functionality or resource could enhance the accuracy
and speed of your translation workflows?"

Reason:
Asks for substantive workflow/domain advice. It does not require
discovering what this AI itself can do.

NON_META:
"What specific piece of mathematical knowledge are you having trouble
applying in your current work?"

Reason:
The question concerns the interlocutor's domain problem, not the
AI system's capabilities.

NON_META:
"What specific challenge might arise during the implementation phase
if the QA testing functionality were not fully validated?"

Reason:
Asks about consequences in a QA scenario.

NON_META:
"Can you explain how to prioritize and manage multiple incidents
simultaneously?"

Reason:
Requests domain guidance about incident management.

NON_META:
"What are some key factors to consider when conducting an experiment
design?"

Reason:
Requests domain knowledge.

========================
ASYMMETRIC ERROR POLICY
========================

For this application, allowing a true META query is more costly than
rejecting a borderline NON_META query.

If the query is genuinely ambiguous and one plausible interpretation
is that its purpose is to discover the AI/system's hidden capabilities,
functions, expertise, experience, or operating behavior, choose META.

However, do NOT classify ordinary domain questions as META merely
because they are broad, hypothetical, informational, or awkwardly
phrased.

Return only the requested structured classification.
"""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "meta_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["META", "NON_META"]
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"]
                },
                "asks_about_system_itself": {
                    "type": "boolean"
                },
                "reason": {
                    "type": "string"
                }
            },
            "required": [
                "label",
                "confidence",
                "asks_about_system_itself",
                "reason"
            ],
            "additionalProperties": False
        }
    }
}


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )


async def judge_one(client, sem, model, row):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Classify this query:\n\n" + row["query"]
                },
            ],
            response_format=SCHEMA,
        )

        raw = resp.choices[0].message.content
        obj = json.loads(raw)

        return {
            "round": row["round"],
            "session_id": row["session_id"],
            "candidate_index": row["candidate_index"],
            "query": row["query"],
            **obj,
            "usage": {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        }


async def run(args):
    rows = read_jsonl(DEV)
    assert len(rows) == 80, len(rows)

    existing = read_jsonl(RESULTS)
    done = {key(x) for x in existing if "error" not in x}

    pending = [x for x in rows if key(x) not in done]

    print("dev examples:", len(rows))
    print("already judged:", len(done))
    print("pending:", len(pending))

    if pending:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)

        tasks = [
            judge_one(client, sem, args.model, row)
            for row in pending
        ]

        completed = 0

        for fut in asyncio.as_completed(tasks):
            try:
                rec = await fut
            except Exception as e:
                rec = {"error": repr(e)}

            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            completed += 1
            if completed % 10 == 0 or completed == len(pending):
                print(f"completed {completed}/{len(pending)}")

    evaluate()


def evaluate():
    rows = read_jsonl(RESULTS)
    human = read_jsonl(HUMAN)

    jm = {key(x): x for x in rows if "error" not in x}
    hm = {key(x): x for x in human}

    pairs = []

    for k, h in hm.items():
        if k not in jm:
            continue
        if h["human_label"] == "AMBIGUOUS":
            continue
        pairs.append((h, jm[k]))

    tp = fn = fp = tn = 0

    dangerous = []

    for h, j in pairs:
        ht = h["human_label"]
        jp = j["label"]

        if ht == "CLEAR_META" and jp == "META":
            tp += 1
        elif ht == "CLEAR_META" and jp == "NON_META":
            fn += 1
            dangerous.append(j)
        elif ht == "CLEAR_NON_META" and jp == "META":
            fp += 1
        elif ht == "CLEAR_NON_META" and jp == "NON_META":
            tn += 1

    def div(a, b):
        return a / b if b else float("nan")

    accuracy = div(tp + tn, len(pairs))
    meta_precision = div(tp, tp + fp)
    meta_recall = div(tp, tp + fn)
    nonmeta_precision = div(tn, tn + fn)

    print()
    print("=== META JUDGE V3 / HUMAN DEV ===")
    print(f"clear human examples: {len(pairs)}")
    print()
    print("                 Judge META   Judge NON_META")
    print(f"Human META          {tp:3d}             {fn:3d}")
    print(f"Human NON_META      {fp:3d}             {tn:3d}")
    print()
    print(f"accuracy          : {accuracy:.3f}")
    print(f"META precision    : {meta_precision:.3f}")
    print(f"META recall       : {meta_recall:.3f}")
    print(f"NON_META precision: {nonmeta_precision:.3f}")
    print(f"dangerous FN      : {len(dangerous)}")

    print()
    print("=== REMAINING DANGEROUS FALSE NEGATIVES ===")

    if not dangerous:
        print("NONE")
    else:
        for x in dangerous:
            print()
            print(
                f"{x['confidence']} | "
                f"asks_system={x['asks_about_system_itself']}"
            )
            print(x["query"])
            print("reason:", x["reason"])

    # Useful conservative gate:
    # reject all META, and also LOW-confidence NON_META.
    allowed = []
    rejected = []

    for h, j in pairs:
        gate_reject = (
            j["label"] == "META"
            or (
                j["label"] == "NON_META"
                and j["confidence"] == "LOW"
            )
        )

        (rejected if gate_reject else allowed).append((h, j))

    allowed_true_nonmeta = sum(
        h["human_label"] == "CLEAR_NON_META"
        for h, _ in allowed
    )
    allowed_true_meta = sum(
        h["human_label"] == "CLEAR_META"
        for h, _ in allowed
    )

    gate_precision = div(
        allowed_true_nonmeta,
        allowed_true_nonmeta + allowed_true_meta
    )

    meta_total = sum(
        h["human_label"] == "CLEAR_META"
        for h, _ in pairs
    )

    blocked_meta = sum(
        h["human_label"] == "CLEAR_META"
        for h, _ in rejected
    )

    gate_meta_recall = div(blocked_meta, meta_total)

    print()
    print("=== CONSERVATIVE DEPLOYMENT GATE ===")
    print("Reject META + LOW-confidence NON_META")
    print(f"allowed clear queries : {len(allowed)}")
    print(f"allowed true META     : {allowed_true_meta}")
    print(f"allowed precision     : {gate_precision:.3f}")
    print(f"META blocked recall   : {gate_meta_recall:.3f}")

    summary = {
        "n_clear": len(pairs),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": accuracy,
        "meta_precision": meta_precision,
        "meta_recall": meta_recall,
        "nonmeta_precision": nonmeta_precision,
        "dangerous_false_negatives": len(dangerous),
        "gate_allowed": len(allowed),
        "gate_allowed_true_meta": allowed_true_meta,
        "gate_precision": gate_precision,
        "gate_meta_recall": gate_meta_recall,
    }

    with SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("wrote:", SUMMARY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18"
    )
    ap.add_argument("--concurrency", type=int, default=12)
    args = ap.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
