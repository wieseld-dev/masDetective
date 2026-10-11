from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI


ROOT = Path("artifacts/v4_userlike_audit")
OUT = Path("artifacts/v4_meta_adjudication")
OUT.mkdir(parents=True, exist_ok=True)

AUDIT = ROOT / "userlike_judgments.jsonl"
STRICT_GOLD = ROOT / "gold_strict.jsonl"

ADJ = OUT / "meta_adjudications.jsonl"
FINAL = OUT / "dpo_meta_clean.jsonl"
ASSIST = OUT / "assistant_direction_candidates.jsonl"
SUMMARY = OUT / "summary.json"


SYSTEM = r"""
You are performing conservative semantic adjudication for research on
benign black-box AI queries.

The candidate was already classified as META by another classifier.
Your job is to determine whether it CLEARLY asks about THIS AI SYSTEM
or assistant itself.

Classify TARGET as exactly one of:

SYSTEM_SELF:
The query clearly seeks information about the AI/system's own:
- capabilities
- supported functions
- expertise
- knowledge areas
- experience
- specialization
- limitations
- behavior
- process or way it handles tasks
- ability to perform some capability

Examples:
"Do you have experience with project management?"
"What is your primary area of expertise?"
"Do you know how to perform mathematical calculations?"
"How do you handle unexpected outcomes in QA testing?"
"Does your experience involve accessibility review?"

EXTERNAL_DOMAIN:
The query asks about an external problem, project, organization,
dataset, workflow, concept or task. It does NOT primarily reveal
information about this AI system.

Examples:
"What steps should be taken to ensure data accuracy?"
"What problems arise when translating technical material?"
"Do these tasks require financial analysis?"
"How should an organization prioritize incidents?"
"What factors matter when evaluating accessibility compliance?"

AMBIGUOUS:
The referent is genuinely unclear. For example, the wording does not
make clear whether "the analysis", "the system", "the work", or
"experience" belongs to the AI or to an external scenario.

CRITICAL:
- Do not classify a query SYSTEM_SELF merely because it uses "you".
- Resolve what the useful answer would reveal.
- Generic/domain advice is EXTERNAL_DOMAIN.
- Personal experience/expertise/capability of the respondent AI is SYSTEM_SELF.
- When uncertain, use AMBIGUOUS rather than SYSTEM_SELF.
- Precision is more important than recall here.

Also report SELF_TARGET_STRENGTH:
EXPLICIT: directly asks about AI ability/expertise/experience.
STRONG: clearly system-self, though less direct.
WEAK: indirect or ambiguous evidence.

Return only the schema.
"""

SCHEMA = {
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
                "self_target_strength": {
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
                "self_target_strength",
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


def append_jsonl(p, row):
    with Path(p).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(p, rows):
    with Path(p).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def aid(x):
    return (
        f"{int(x['gold_index'])}:"
        f"{int(x['sample_index'])}"
    )


async def judge_one(client, sem, model, x):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content":
                        "Adjudicate this candidate query:\n\n"
                        + str(x["query"]),
                },
            ],
            response_format=SCHEMA,
        )

        obj = json.loads(resp.choices[0].message.content)

        return {
            "audit_id": aid(x),
            "gold_index": int(x["gold_index"]),
            "sample_index": int(x["sample_index"]),
            "query": str(x["query"]),
            **obj,
            "usage": {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        }


async def run(args):
    audit = read_jsonl(AUDIT)
    gold = read_jsonl(STRICT_GOLD)

    assert audit, "missing userlike audit"
    assert gold, "strict Gold set is empty"

    samples = [
        x for x in audit
        if x.get("source") == "SAMPLE"
    ]

    assert len(samples) == 327, (
        f"expected 327 audited samples, got {len(samples)}"
    )

    # ---------------------------------------------------------
    # META-only candidate set with STYLE HELD CLEAN.
    #
    # This is deliberate:
    # We want DPO to learn META semantics, not merely
    # assistant-like vs user-like style.
    # ---------------------------------------------------------

    candidates = [
        x for x in samples
        if x.get("meta_label") == "META"
        and x.get("meta_confidence") == "HIGH"
        and x.get("user_like") == "USER_LIKE"
        and x.get("direction") == "USER_TO_SYSTEM"
        and x.get("naturalness") == "NATURAL"
        and x.get("confidence") == "HIGH"
        and not x.get("internal_prompt_lexical_flag", False)
        and not x.get(
            "assistant_clarification_lexical_flag",
            False,
        )
    ]

    # Separate diagnostic set for conversational-direction training.
    assistant = [
        x for x in samples
        if x.get("user_like") == "ASSISTANT_LIKE"
        and x.get("direction") == "SYSTEM_TO_USER"
        and x.get("confidence") == "HIGH"
    ]

    print("======================================")
    print(" META ADJUDICATION PREFLIGHT")
    print("======================================")
    print("strict Gold                 :", len(gold))
    print("all audited samples         :", len(samples))
    print("style-clean META candidates :", len(candidates))
    print("assistant-like diagnostics  :", len(assistant))

    assert len({int(x["gold_index"]) for x in gold}) == len(gold)

    gold_by = {
        int(x["gold_index"]): x
        for x in gold
    }

    # Only candidates whose positive Gold survived strict audit matter.
    candidates = [
        x for x in candidates
        if int(x["gold_index"]) in gold_by
    ]

    print(
        "META candidates with strict Gold:",
        len(candidates),
    )
    print(
        "Gold coverage before adjudication:",
        len(set(int(x["gold_index"]) for x in candidates)),
        "/",
        len(gold),
    )

    if args.dry_run:
        return

    # ---------------------------------------------------------
    # Resume-safe secondary adjudication
    # ---------------------------------------------------------

    existing = {
        x["audit_id"]: x
        for x in read_jsonl(ADJ)
        if "error" not in x
    }

    pending = [
        x for x in candidates
        if aid(x) not in existing
    ]

    print("already adjudicated         :", len(existing))
    print("pending API judgments       :", len(pending))

    if pending:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)

        B = 20

        for start in range(0, len(pending), B):
            batch = pending[start:start+B]

            results = await asyncio.gather(
                *[
                    judge_one(client, sem, args.model, x)
                    for x in batch
                ],
                return_exceptions=True,
            )

            failures = []

            for src, res in zip(batch, results):
                if isinstance(res, Exception):
                    failures.append(
                        (
                            aid(src),
                            f"{type(res).__name__}: {res}",
                        )
                    )
                    continue

                # Persist immediately: no paid response is lost
                # if a later batch fails.
                append_jsonl(ADJ, res)
                existing[res["audit_id"]] = res

            print(
                f"[adjudicate] "
                f"{min(start+B, len(pending))}/"
                f"{len(pending)}",
                flush=True,
            )

            if failures:
                for f in failures[:10]:
                    print("FAIL:", f)
                raise RuntimeError(
                    f"{len(failures)} API calls failed"
                )

    missing = [
        aid(x)
        for x in candidates
        if aid(x) not in existing
    ]

    if missing:
        raise RuntimeError(
            f"missing adjudications: {missing[:10]}"
        )

    # ---------------------------------------------------------
    # Final meta preference pairs
    # ---------------------------------------------------------

    adjudicated = []

    sample_by_id = {
        aid(x): x
        for x in candidates
    }

    for k, j in existing.items():
        if k not in sample_by_id:
            continue

        x = dict(sample_by_id[k])
        x["secondary_target"] = j["target"]
        x["secondary_strength"] = j[
            "self_target_strength"
        ]
        x["secondary_confidence"] = j["confidence"]
        x["secondary_reason"] = j["reason"]
        adjudicated.append(x)

    counts = Counter(
        (
            x["secondary_target"],
            x["secondary_strength"],
            x["secondary_confidence"],
        )
        for x in adjudicated
    )

    clear_meta = [
        x for x in adjudicated
        if x["secondary_target"] == "SYSTEM_SELF"
        and x["secondary_confidence"] == "HIGH"
        and x["secondary_strength"] in {
            "EXPLICIT",
            "STRONG",
        }
    ]

    by_gold = defaultdict(list)

    for x in clear_meta:
        by_gold[int(x["gold_index"])].append(x)

    final = []

    # ONE meta negative per Gold initially.
    # Avoid letting a few prompts dominate the DPO objective.
    for gi in sorted(by_gold):
        g = gold_by[gi]

        xs = sorted(
            by_gold[gi],
            key=lambda x: (
                0 if x["secondary_strength"] == "EXPLICIT" else 1,
                int(x["sample_index"]),
            )
        )

        x = xs[0]

        # Critical same-prompt invariant.
        assert str(g["prompt"]) == str(x["prompt"])

        chosen = str(g["completion"])
        rejected = str(x["completion"])

        assert chosen.strip()
        assert rejected.strip()
        assert chosen != rejected

        final.append({
            "gold_index": gi,
            "round": int(g["round"]),
            "session_id": str(g["session_id"]),
            "candidate_index": int(g["candidate_index"]),
            "sample_index": int(x["sample_index"]),

            "prompt": str(g["prompt"]),
            "chosen": chosen,
            "rejected": rejected,

            "chosen_query": str(g["query"]),
            "rejected_query": str(x["query"]),

            "negative_target": x["secondary_target"],
            "negative_strength": x["secondary_strength"],
            "negative_confidence": x[
                "secondary_confidence"
            ],
            "negative_reason": x["secondary_reason"],

            "gold_reward": float(
                g.get("gold_reward", 0.0)
            ),
            "gold_advantage": float(
                g.get("gold_advantage", 0.0)
            ),
        })

    write_jsonl(FINAL, final)
    write_jsonl(ASSIST, assistant)

    # Final integrity audit
    assert len({x["gold_index"] for x in final}) == len(final)
    assert all(x["prompt"] for x in final)
    assert all(
        x["negative_target"] == "SYSTEM_SELF"
        for x in final
    )
    assert all(
        x["negative_confidence"] == "HIGH"
        for x in final
    )

    input_tok = sum(
        int(x.get("usage", {}).get("input_tokens", 0))
        for x in existing.values()
    )
    output_tok = sum(
        int(x.get("usage", {}).get("output_tokens", 0))
        for x in existing.values()
    )

    summary = {
        "strict_gold": len(gold),
        "audited_samples": len(samples),
        "style_clean_frozen_meta_candidates": len(candidates),
        "secondary_counts": {
            "|".join(k): v
            for k, v in counts.items()
        },
        "clear_system_self_negatives": len(clear_meta),
        "final_dpo_pairs": len(final),
        "final_gold_coverage": len(final),
        "assistant_direction_candidates": len(assistant),
        "secondary_input_tokens": input_tok,
        "secondary_output_tokens": output_tok,
        "max_meta_pairs_per_gold": 1,
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("======================================")
    print(" FINAL CLEAN META-DPO SUMMARY")
    print("======================================")
    print("strict Gold              :", len(gold))
    print("style-clean META cand.   :", len(candidates))
    print("secondary classes        :")
    for k, v in sorted(counts.items()):
        print(" ", k, "=", v)
    print("clear SYSTEM_SELF        :", len(clear_meta))
    print(
        "FINAL DPO PAIRS         :",
        len(final),
        f"(coverage {len(final)}/{len(gold)} strict Gold)",
    )
    print(
        "assistant diagnostics   :",
        len(assistant),
    )

    print()
    print("=== ALL FINAL META PAIRS ===")

    for i, x in enumerate(final, 1):
        print(
            f"\n--- {i}/{len(final)} "
            f"| G{x['gold_index']} "
            f"| {x['negative_strength']} ---"
        )
        print("CHOSEN  :", x["chosen_query"])
        print("REJECTED:", x["rejected_query"])
        print("WHY     :", x["negative_reason"])

    print()
    print("wrote:", FINAL)
    print("wrote:", ASSIST)
    print("wrote:", SUMMARY)


def main():
    ap = argparse.ArgumentParser()

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
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
