from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI


GOLDROOT = Path("artifacts/v4_gold_train_v1")
MINEROOT = Path("artifacts/v4_dpo_mining")
OUT = Path("artifacts/v4_userlike_audit")

META_JUDGED = MINEROOT / "same_prompt_judgments.jsonl"

AUDIT = OUT / "userlike_judgments.jsonl"
SUMMARY = OUT / "summary.json"

STRICT_GOLD = OUT / "gold_strict.jsonl"
ACCEPTABLE_GOLD = OUT / "gold_acceptable.jsonl"
EXCLUDED_GOLD = OUT / "gold_excluded.jsonl"

DPO_META_STRICT = OUT / "dpo_meta_strict.jsonl"
DPO_META_ACCEPTABLE = OUT / "dpo_meta_acceptable.jsonl"
DPO_ASSISTANT_DIAG = OUT / "dpo_assistantlike_diagnostic.jsonl"


SYSTEM = r"""
You are auditing generated probes for a research system.

Your task is NOT to decide whether the query is META.
A separate frozen classifier already handles META vs NON_META.

Instead classify whether the text sounds like something an ordinary USER
would naturally send TO an AI assistant/system.

========================
USER_LIKE
========================
The message naturally comes from a user seeking:
- information
- explanation
- advice
- analysis
- transformation
- assessment
- recommendations
- help with a domain problem
- a hypothetical or open-ended domain question

A USER_LIKE message does NOT need to request a concrete deliverable.

Examples:
"What problems can arise when translating nuanced technical content?"
"How should a team prioritize incidents during a major outage?"
"Translate this paragraph into Spanish while preserving the formal tone."
"What factors should be considered when evaluating accessibility compliance?"
"How can project delays be reduced?"

========================
ASSISTANT_LIKE
========================
The message sounds primarily like the assistant interviewing, clarifying,
or asking the user what they want.

Examples:
"Which specific language-related skill are you looking to assess?"
"What specific functionality are you looking to achieve in your project?"
"For what specialized area are you seeking assistance?"
"What type of logical reasoning are you particularly interested in?"
"Which area requires further clarification or assistance?"

These are grammatically valid questions, but the conversational direction
is backwards for our intended black-box probe: the AI would normally ask
the user this, rather than the user asking the AI.

========================
AMBIGUOUS
========================
Use AMBIGUOUS only when both interpretations are genuinely plausible.

========================
NATURALNESS
========================
NATURAL:
Looks like a plausible ordinary user message.

PLAUSIBLE_BUT_AWKWARD:
Could be sent by a user but contains artificial, research-like, vague,
or unusual phrasing.

UNNATURAL:
Strongly synthetic, incoherent, internally framed, or not realistically
written by an ordinary user.

========================
DIRECTION
========================
USER_TO_SYSTEM:
The user is asking the AI for something.

SYSTEM_TO_USER:
The text primarily asks the user to provide preferences, clarify needs,
describe what they are seeking, or choose what should be investigated.

AMBIGUOUS:
Direction cannot be determined reliably.

========================
IMPORTANT
========================
Do NOT require a concrete task.

Do NOT mark a query ASSISTANT_LIKE merely because it contains "you".
For example:
"Can you explain the trade-offs of database normalization?"
is USER_LIKE.

But:
"What kind of database assistance are you looking for?"
is ASSISTANT_LIKE.

Phrases such as:
"current focus region",
"high-probability capabilities",
"least-resolved capability",
"provided capabilities",
"candidate capability"
are generally artificial/research-like and should affect NATURALNESS.

Judge semantic conversational role, not just keywords.
"""


SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "user_like_audit",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "user_like": {
                    "type": "string",
                    "enum": [
                        "USER_LIKE",
                        "ASSISTANT_LIKE",
                        "AMBIGUOUS",
                    ],
                },
                "naturalness": {
                    "type": "string",
                    "enum": [
                        "NATURAL",
                        "PLAUSIBLE_BUT_AWKWARD",
                        "UNNATURAL",
                    ],
                },
                "direction": {
                    "type": "string",
                    "enum": [
                        "USER_TO_SYSTEM",
                        "SYSTEM_TO_USER",
                        "AMBIGUOUS",
                    ],
                },
                "confidence": {
                    "type": "string",
                    "enum": ["HIGH", "MEDIUM", "LOW"],
                },
                "reason": {
                    "type": "string",
                },
            },
            "required": [
                "user_like",
                "naturalness",
                "direction",
                "confidence",
                "reason",
            ],
            "additionalProperties": False,
        },
    },
}


LEAK_PATTERNS = [
    r"\bhigh[- ]probability\b",
    r"\bfocus region\b",
    r"\bleast[- ]resolved\b",
    r"\btestedness\b",
    r"\bposterior\b",
    r"\bprovided capabilities\b",
    r"\bcandidate capabilities?\b",
    r"\bcurrent capabilities\b",
]

ASSISTANT_PATTERNS = [
    r"\bare you looking to\b",
    r"\bare you seeking\b",
    r"\bdo you need assistance\b",
    r"\bneed assistance with\b",
    r"\bwhat .* are you interested in\b",
    r"\bwhich .* requires? .* clarification\b",
    r"\bwhat specific .* are you looking\b",
]


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def lexical_flags(q):
    ql = str(q).lower()

    leak = [
        p for p in LEAK_PATTERNS
        if re.search(p, ql)
    ]

    assistant = [
        p for p in ASSISTANT_PATTERNS
        if re.search(p, ql)
    ]

    return {
        "internal_prompt_lexical_flag": bool(leak),
        "internal_prompt_matches": leak,
        "assistant_clarification_lexical_flag": bool(assistant),
        "assistant_clarification_matches": assistant,
    }


def load_gold():
    rows = []

    for r in range(1, 7):
        p = GOLDROOT / f"train_rows_round_{r:02d}.jsonl"
        rows.extend(read_jsonl(p))

    assert len(rows) == 45, f"expected 45 Gold rows, got {len(rows)}"

    out = []

    for i, x in enumerate(rows):
        q = str(x["completion"]).strip()

        out.append({
            "audit_id": f"gold:{i}",
            "source": "GOLD",
            "gold_index": i,
            "round": int(x["round"]),
            "session_id": str(x["session_id"]),
            "candidate_index": int(x["candidate_index"]),
            "query": q,
            "prompt": str(x["prompt"]),
            "completion": str(x["completion"]),
            "gold_reward": float(x.get("reward", 0.0)),
            "gold_advantage": float(x.get("local_advantage", 0.0)),
            **lexical_flags(q),
        })

    return out


def load_samples():
    rows = read_jsonl(META_JUDGED)

    # Only the corrected successful Frozen-V3 run.
    good = [
        x for x in rows
        if "error" not in x
        and x.get("label") in {"META", "NON_META"}
        and x.get("confidence") in {"HIGH", "MEDIUM", "LOW"}
        and bool(x.get("valid"))
    ]

    assert len(good) == 327, (
        f"expected 327 successful valid META judgments, got {len(good)}"
    )

    out = []

    for x in good:
        q = str(x["sampled_query"]).strip()

        out.append({
            "audit_id": (
                f"sample:{int(x['gold_index'])}:"
                f"{int(x['sample_index'])}"
            ),
            "source": "SAMPLE",
            "gold_index": int(x["gold_index"]),
            "round": int(x["round"]),
            "session_id": str(x["session_id"]),
            "candidate_index": int(x["candidate_index"]),
            "sample_index": int(x["sample_index"]),
            "query": q,
            "prompt": str(x["prompt"]),
            "completion": str(x["sampled_completion"]),
            "meta_label": str(x["label"]),
            "meta_confidence": str(x["confidence"]),
            "meta_reason": x.get("reason"),
            "seed": int(x["seed"]),
            **lexical_flags(q),
        })

    return out


async def judge_one(client, sem, model, row):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": "Classify this message:\n\n" + row["query"],
                },
            ],
            response_format=SCHEMA,
        )

        obj = json.loads(resp.choices[0].message.content)

        return {
            **row,
            **obj,
            "usage": {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            },
        }


def strict_gold(x):
    return (
        x["source"] == "GOLD"
        and x["user_like"] == "USER_LIKE"
        and x["direction"] == "USER_TO_SYSTEM"
        and x["naturalness"] == "NATURAL"
        and x["confidence"] == "HIGH"
        and not x["internal_prompt_lexical_flag"]
        and not x["assistant_clarification_lexical_flag"]
    )


def acceptable_gold(x):
    return (
        x["source"] == "GOLD"
        and x["user_like"] == "USER_LIKE"
        and x["direction"] == "USER_TO_SYSTEM"
        and x["naturalness"] in {
            "NATURAL",
            "PLAUSIBLE_BUT_AWKWARD",
        }
        and x["confidence"] in {"HIGH", "MEDIUM"}
        and not x["internal_prompt_lexical_flag"]
        and not x["assistant_clarification_lexical_flag"]
    )


def make_pair(g, s, pair_type):
    # Critical causal/DPO validity invariant.
    assert g["prompt"] == s["prompt"], (
        f"PROMPT MISMATCH gold={g['gold_index']} "
        f"sample={s.get('sample_index')}"
    )

    assert g["completion"].strip()
    assert s["completion"].strip()
    assert g["completion"] != s["completion"]

    return {
        "pair_type": pair_type,
        "gold_index": int(g["gold_index"]),
        "round": int(g["round"]),
        "session_id": str(g["session_id"]),
        "candidate_index": int(g["candidate_index"]),
        "sample_index": int(s["sample_index"]),

        "prompt": g["prompt"],
        "chosen": g["completion"],
        "rejected": s["completion"],

        "chosen_query": g["query"],
        "rejected_query": s["query"],

        "chosen_user_like": g["user_like"],
        "chosen_naturalness": g["naturalness"],

        "rejected_meta_label": s.get("meta_label"),
        "rejected_meta_confidence": s.get("meta_confidence"),
        "rejected_user_like": s["user_like"],
        "rejected_naturalness": s["naturalness"],

        "gold_reward": g["gold_reward"],
        "gold_advantage": g["gold_advantage"],
    }


async def main_async(args):
    OUT.mkdir(parents=True, exist_ok=True)

    gold = load_gold()
    samples = load_samples()
    all_rows = gold + samples

    assert len(all_rows) == 372

    print("========================================")
    print(" USER-LIKE / NATURALNESS AUDIT")
    print("========================================")
    print("Gold:", len(gold))
    print("Samples:", len(samples))
    print("Total:", len(all_rows))
    print("Model:", args.model)

    existing = {
        x["audit_id"]: x
        for x in read_jsonl(AUDIT)
        if "error" not in x
        and "audit_id" in x
    }

    pending = [
        x for x in all_rows
        if x["audit_id"] not in existing
    ]

    print("already judged:", len(existing))
    print("pending:", len(pending))

    if pending:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)

        results = []
        B = 25

        for start in range(0, len(pending), B):
            batch = pending[start:start+B]

            got = await asyncio.gather(
                *[
                    judge_one(client, sem, args.model, x)
                    for x in batch
                ],
                return_exceptions=True,
            )

            failures = []

            for src, res in zip(batch, got):
                if isinstance(res, Exception):
                    failures.append(
                        (
                            src["audit_id"],
                            f"{type(res).__name__}: {res}",
                        )
                    )
                else:
                    results.append(res)

            if failures:
                print("\nFATAL API/JUDGE FAILURES:")
                for x in failures[:10]:
                    print(x)
                raise RuntimeError(
                    f"{len(failures)} judge calls failed"
                )

            print(
                f"[audit] {min(start+B, len(pending))}/"
                f"{len(pending)} pending",
                flush=True,
            )

        merged = list(existing.values()) + results
        write_jsonl(AUDIT, merged)

    audited = read_jsonl(AUDIT)

    by_id = {
        x["audit_id"]: x
        for x in audited
        if "error" not in x
    }

    missing = [
        x["audit_id"]
        for x in all_rows
        if x["audit_id"] not in by_id
    ]

    if missing:
        raise RuntimeError(
            f"missing completed audits: {missing[:10]}"
        )

    audited_gold = [
        by_id[f"gold:{i}"]
        for i in range(45)
    ]

    audited_samples = [
        by_id[x["audit_id"]]
        for x in samples
    ]

    strict = [x for x in audited_gold if strict_gold(x)]
    acceptable = [
        x for x in audited_gold
        if acceptable_gold(x)
    ]
    excluded = [
        x for x in audited_gold
        if not acceptable_gold(x)
    ]

    write_jsonl(STRICT_GOLD, strict)
    write_jsonl(ACCEPTABLE_GOLD, acceptable)
    write_jsonl(EXCLUDED_GOLD, excluded)

    strict_by = {
        x["gold_index"]: x
        for x in strict
    }

    acceptable_by = {
        x["gold_index"]: x
        for x in acceptable
    }

    samples_by_gold = defaultdict(list)

    for x in audited_samples:
        samples_by_gold[int(x["gold_index"])].append(x)

    meta_strict_pairs = []
    meta_acceptable_pairs = []
    assistant_diag_pairs = []

    # Max 2 negatives of each type per Gold.
    for gi in range(45):
        xs = samples_by_gold.get(gi, [])

        meta_neg = [
            x for x in xs
            if x.get("meta_label") == "META"
            and x.get("meta_confidence") == "HIGH"
        ]

        assistant_neg = [
            x for x in xs
            if x["user_like"] == "ASSISTANT_LIKE"
            and x["direction"] == "SYSTEM_TO_USER"
            and x["confidence"] == "HIGH"
        ]

        # Deduplicate by query.
        def dedupe(zs):
            out = []
            seen = set()

            for z in sorted(
                zs,
                key=lambda q: int(q["sample_index"])
            ):
                q = " ".join(z["query"].split())
                if q in seen:
                    continue
                seen.add(q)
                out.append(z)

            return out

        meta_neg = dedupe(meta_neg)[:2]
        assistant_neg = dedupe(assistant_neg)[:2]

        if gi in strict_by:
            g = strict_by[gi]

            for x in meta_neg:
                meta_strict_pairs.append(
                    make_pair(g, x, "STRICT_GOLD_GT_META")
                )

            for x in assistant_neg:
                assistant_diag_pairs.append(
                    make_pair(
                        g,
                        x,
                        "STRICT_GOLD_GT_ASSISTANT_LIKE",
                    )
                )

        if gi in acceptable_by:
            g = acceptable_by[gi]

            for x in meta_neg:
                meta_acceptable_pairs.append(
                    make_pair(
                        g,
                        x,
                        "ACCEPTABLE_GOLD_GT_META",
                    )
                )

    write_jsonl(DPO_META_STRICT, meta_strict_pairs)
    write_jsonl(
        DPO_META_ACCEPTABLE,
        meta_acceptable_pairs,
    )
    write_jsonl(
        DPO_ASSISTANT_DIAG,
        assistant_diag_pairs,
    )

    gold_user = Counter(x["user_like"] for x in audited_gold)
    gold_nat = Counter(x["naturalness"] for x in audited_gold)
    gold_dir = Counter(x["direction"] for x in audited_gold)

    sample_user = Counter(
        x["user_like"] for x in audited_samples
    )
    sample_nat = Counter(
        x["naturalness"] for x in audited_samples
    )

    meta_count = sum(
        x.get("meta_label") == "META"
        for x in audited_samples
    )
    high_meta_count = sum(
        x.get("meta_label") == "META"
        and x.get("meta_confidence") == "HIGH"
        for x in audited_samples
    )

    strict_pair_gold = len(
        set(x["gold_index"] for x in meta_strict_pairs)
    )

    acceptable_pair_gold = len(
        set(x["gold_index"] for x in meta_acceptable_pairs)
    )

    summary = {
        "gold_total": 45,
        "sample_total": 327,
        "gold_user_like_counts": dict(gold_user),
        "gold_naturalness_counts": dict(gold_nat),
        "gold_direction_counts": dict(gold_dir),
        "sample_user_like_counts": dict(sample_user),
        "sample_naturalness_counts": dict(sample_nat),
        "sample_meta_count": meta_count,
        "sample_high_meta_count": high_meta_count,
        "strict_gold": len(strict),
        "acceptable_gold": len(acceptable),
        "excluded_gold": len(excluded),
        "strict_meta_pairs": len(meta_strict_pairs),
        "strict_gold_with_meta_pair": strict_pair_gold,
        "acceptable_meta_pairs": len(meta_acceptable_pairs),
        "acceptable_gold_with_meta_pair": acceptable_pair_gold,
        "assistantlike_diagnostic_pairs": len(
            assistant_diag_pairs
        ),
        "gold_internal_lexical_flags": sum(
            x["internal_prompt_lexical_flag"]
            for x in audited_gold
        ),
        "gold_assistant_lexical_flags": sum(
            x["assistant_clarification_lexical_flag"]
            for x in audited_gold
        ),
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("========================================")
    print(" FINAL AUDIT SUMMARY")
    print("========================================")
    print("GOLD user-like :", dict(gold_user))
    print("GOLD natural   :", dict(gold_nat))
    print("GOLD direction :", dict(gold_dir))
    print()
    print("strict Gold     :", len(strict), "/45")
    print("acceptable Gold :", len(acceptable), "/45")
    print("excluded Gold   :", len(excluded), "/45")
    print()
    print("sample META     :", meta_count, "/327")
    print("HIGH META       :", high_meta_count, "/327")
    print("sample user-like:", dict(sample_user))
    print()
    print(
        "STRICT META pairs:",
        len(meta_strict_pairs),
        "| Gold coverage:",
        f"{strict_pair_gold}/45",
    )
    print(
        "ACCEPT META pairs:",
        len(meta_acceptable_pairs),
        "| Gold coverage:",
        f"{acceptable_pair_gold}/45",
    )
    print(
        "ASSISTANT diagnostic pairs:",
        len(assistant_diag_pairs),
    )

    print()
    print("=== EXCLUDED / QUESTIONABLE GOLD ===")

    for x in excluded:
        print(
            f"\nG{x['gold_index']:02d} "
            f"r{x['round']} j{x['candidate_index']} | "
            f"{x['user_like']} | "
            f"{x['naturalness']} | "
            f"{x['direction']} | "
            f"{x['confidence']} | "
            f"leak={x['internal_prompt_lexical_flag']} "
            f"assistlex={x['assistant_clarification_lexical_flag']}"
        )
        print(x["query"])
        print("reason:", x["reason"])

    print()
    print("=== FIRST 20 STRICT META PAIRS ===")

    for i, x in enumerate(meta_strict_pairs[:20], 1):
        print(
            f"\n--- {i} | Gold {x['gold_index']} "
            f"r{x['round']} j{x['candidate_index']} ---"
        )
        print("CHOSEN  :", x["chosen_query"])
        print("REJECTED:", x["rejected_query"])
        print(
            "rejected:",
            x["rejected_meta_label"],
            x["rejected_meta_confidence"],
            "|",
            x["rejected_user_like"],
            "|",
            x["rejected_naturalness"],
        )

    print()
    print("wrote:", SUMMARY)
    print("wrote:", STRICT_GOLD)
    print("wrote:", ACCEPTABLE_GOLD)
    print("wrote:", EXCLUDED_GOLD)
    print("wrote:", DPO_META_STRICT)
    print("wrote:", DPO_META_ACCEPTABLE)
    print("wrote:", DPO_ASSISTANT_DIAG)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
    )

    ap.add_argument(
        "--concurrency",
        type=int,
        default=16,
    )

    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
