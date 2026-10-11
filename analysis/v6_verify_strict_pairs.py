from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI


SRC = Path("artifacts/v6_pref_strict_v2/preferences.jsonl")
OUT = Path("artifacts/v6_pref_verified_v3")
OUT.mkdir(parents=True, exist_ok=True)

RAW = OUT / "raw_responses.jsonl"
VERDICTS = OUT / "verdicts.jsonl"
FINAL = OUT / "preferences.jsonl"
SUMMARY = OUT / "summary.json"
REVIEW = OUT / "review.txt"

KEEP_FAMILIES = {"DIRECTION", "META"}
CAP = 8


BASE_SYSTEM = r"""
You are the FINAL ADVERSARIAL VERIFIER for pairwise preference data.

Be highly skeptical.

A candidate should PASS only if the rejected question creates a clean
contrast with the GOLD while preserving the same substantive domain/topic.

The GOLD is the preferred completion.

Reject a candidate if:
- the rejected question is actually a reasonable normal user question;
- the intended failure is ambiguous;
- the rejected question changes the substantive task too much;
- another defect besides the requested one dominates;
- the GOLD and rejected question are not a meaningful contrast;
- you would need generous interpretation to call it a failure.

This is a high-precision filter. False negatives are acceptable.
False positives are not.

Return JSON only.
"""


FAMILY_RULE = {
    "DIRECTION": r"""
Target failure = DIRECTION.

The rejected question must UNMISTAKABLY sound like an ASSISTANT asking
the USER for clarification, preferences, missing information, or what
help the user wants.

Use this reversal test:

"If a user typed this exact question into an AI chat, would it sound
clearly backwards because the AI should normally be the one asking it?"

PASS examples conceptually resemble:
- What part of X would you like help with?
- Which X issue are you currently struggling with?
- What information would you like me to clarify?

FAIL examples include normal user questions such as:
- Which aspects of X are most important?
- What challenges can arise in X?
- How should X be handled?

The second category may contain "you", but is still a normal user query.
""",

    "META": r"""
Target failure = META.

The rejected question must clearly interrogate the ADDRESSEE / AI /
application about its OWN capability, method, experience, process,
behavior, or way of handling the topic.

Use this test:

"Would a useful answer primarily describe the system/addressee itself,
rather than answer the underlying domain question?"

PASS examples conceptually resemble:
- How do you handle X?
- What is your approach to X?
- Does your system support X?
- Do you have experience with X?

FAIL normal domain questions such as:
- What details are important when doing X?
- What should I consider before X?
- What challenges arise in X?

Do not infer META merely because the sentence contains "you".
"""
}


def read_jsonl(path):
    if not path.exists():
        return []
    return [
        json.loads(x)
        for x in path.open(encoding="utf-8")
        if x.strip()
    ]


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def cid(x):
    raw = (
        str(x["gold_index"])
        + "\0" + x["family"]
        + "\0" + x["chosen"]
        + "\0" + x["rejected"]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def make_prompt(family, batch):
    items = [
        {
            "id": cid(x),
            "gold": x["chosen"],
            "rejected": x["rejected"],
        }
        for x in batch
    ]

    return (
        FAMILY_RULE[family]
        + "\n\nEvaluate every pair below:\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
        + r"""

For every item return:

{
  "results": [
    {
      "id": "...",

      "same_core_topic": true,
      "family_unmistakable": true,
      "target_is_primary_defect": true,
      "gold_clearly_preferred": true,

      "semantic_confound": "NONE",
      "confidence": "HIGH",

      "pass": true,
      "reason": "short concrete reason"
    }
  ]
}

semantic_confound must be one of:
NONE, MINOR, MAJOR

confidence must be:
LOW, MEDIUM, HIGH

PASS only if ALL FOUR booleans are true,
semantic_confound == NONE,
and confidence == HIGH.
"""
    )


def parse(text, requested):
    obj = json.loads(text)
    xs = obj.get("results")

    if not isinstance(xs, list):
        raise ValueError("missing results")

    requested = set(requested)
    out = {}

    for x in xs:
        if not isinstance(x, dict):
            continue

        i = str(x.get("id") or "")
        if i not in requested:
            continue

        same = x.get("same_core_topic") is True
        fam = x.get("family_unmistakable") is True
        primary = x.get("target_is_primary_defect") is True
        preferred = x.get("gold_clearly_preferred") is True

        confound = str(
            x.get("semantic_confound") or ""
        ).upper()

        confidence = str(
            x.get("confidence") or ""
        ).upper()

        hard_pass = (
            same
            and fam
            and primary
            and preferred
            and confound == "NONE"
            and confidence == "HIGH"
        )

        out[i] = {
            "id": i,
            "same_core_topic": same,
            "family_unmistakable": fam,
            "target_is_primary_defect": primary,
            "gold_clearly_preferred": preferred,
            "semantic_confound": confound,
            "confidence": confidence,
            "model_pass": x.get("pass") is True,
            "pass": hard_pass,
            "reason": str(x.get("reason") or ""),
        }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
    )
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_rows = read_jsonl(SRC)

    rows = [
        x for x in all_rows
        if x["family"] in KEEP_FAMILIES
    ]

    print("=" * 78)
    print("V6 FINAL ADVERSARIAL VERIFIER")
    print("=" * 78)
    print("input strict rows :", len(all_rows))
    print("direction+meta    :", len(rows))
    print("DANGLING dropped :", len(all_rows) - len(rows))
    print("cap/cell          :", CAP)

    assert len(rows) == 495
    assert len({cid(x) for x in rows}) == len(rows)

    if args.dry_run:
        print("DRY RUN: PASS")
        return

    source = {cid(x): x for x in rows}

    existing = {
        x["id"]: x
        for x in read_jsonl(VERDICTS)
    }

    client = OpenAI()

    for family in ["DIRECTION", "META"]:
        fam_rows = [
            x for x in rows
            if x["family"] == family
            and cid(x) not in existing
        ]

        batches = [
            fam_rows[i:i + args.batch_size]
            for i in range(0, len(fam_rows), args.batch_size)
        ]

        for bi, batch in enumerate(batches, 1):
            missing = {cid(x): x for x in batch}

            for attempt in range(1, args.max_retries + 1):
                if not missing:
                    break

                sub = list(missing.values())

                try:
                    resp = client.chat.completions.create(
                        model=args.model,
                        temperature=0,
                        max_tokens=5000,
                        response_format={
                            "type": "json_object"
                        },
                        messages=[
                            {
                                "role": "system",
                                "content": BASE_SYSTEM,
                            },
                            {
                                "role": "user",
                                "content": make_prompt(
                                    family, sub
                                ),
                            },
                        ],
                    )

                    text = (
                        resp.choices[0]
                        .message.content
                        or ""
                    )

                    append_jsonl(
                        RAW,
                        {
                            "family": family,
                            "batch": bi,
                            "attempt": attempt,
                            "requested_ids": list(missing),
                            "raw_content": text,
                        },
                    )

                    got = parse(
                        text,
                        list(missing),
                    )

                    for i, v in got.items():
                        src = source[i]

                        row = {
                            **v,
                            "gold_index":
                                int(src["gold_index"]),
                            "family":
                                src["family"],
                            "prompt":
                                src["prompt"],
                            "chosen":
                                src["chosen"],
                            "rejected":
                                src["rejected"],
                        }

                        append_jsonl(VERDICTS, row)
                        existing[i] = row
                        missing.pop(i, None)

                    print(
                        f"[{family}] "
                        f"batch={bi}/{len(batches)} "
                        f"attempt={attempt} "
                        f"got={len(got)} "
                        f"missing={len(missing)} "
                        f"total={len(existing)}/495",
                        flush=True,
                    )

                except Exception as e:
                    print(
                        f"[retry] {family} "
                        f"batch={bi} "
                        f"attempt={attempt}: "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )

                    time.sleep(
                        min(8, 2 ** (attempt - 1))
                        + random.random()
                    )

            if missing:
                raise RuntimeError(
                    f"unresolved IDs: {list(missing)[:5]}"
                )

    assert len(existing) == 495

    passed = [
        x for x in existing.values()
        if x["pass"]
    ]

    groups = defaultdict(list)

    for x in passed:
        groups[
            (
                int(x["gold_index"]),
                x["family"],
            )
        ].append(x)

    # We already ranked for closeness in the previous stage.
    # Here we only need a deterministic cap.
    selected = []

    for key in sorted(groups):
        xs = groups[key]

        xs.sort(
            key=lambda x: (
                x["gold_index"],
                x["rejected"],
            )
        )

        selected.extend(xs[:CAP])

    final = []

    for x in selected:
        final.append({
            "gold_index":
                int(x["gold_index"]),
            "family":
                x["family"],
            "prompt":
                x["prompt"],
            "chosen":
                x["chosen"],
            "rejected":
                x["rejected"],
        })

    with FINAL.open("w", encoding="utf-8") as f:
        for x in final:
            f.write(
                json.dumps(x, ensure_ascii=False)
                + "\n"
            )

    by_family = Counter(
        x["family"] for x in final
    )

    by_gold = Counter(
        x["gold_index"] for x in final
    )

    by_cell = Counter(
        (x["gold_index"], x["family"])
        for x in final
    )

    verdicts = list(existing.values())

    summary = {
        "verified": len(verdicts),
        "passed": len(passed),
        "pass_rate": len(passed) / len(verdicts),
        "selected": len(final),
        "selected_by_family":
            dict(by_family),
        "selected_by_gold":
            dict(by_gold),
        "cells": {
            f"G{g:02d}:{fam}":
                by_cell[(g, fam)]
            for g in sorted(
                {x["gold_index"] for x in final}
            )
            for fam in ["DIRECTION", "META"]
        },
        "failure_reasons": {
            "same_core_topic_false":
                sum(not x["same_core_topic"]
                    for x in verdicts),
            "family_unmistakable_false":
                sum(not x["family_unmistakable"]
                    for x in verdicts),
            "primary_defect_false":
                sum(not x["target_is_primary_defect"]
                    for x in verdicts),
            "gold_preferred_false":
                sum(not x["gold_clearly_preferred"]
                    for x in verdicts),
            "semantic_confound_non_none":
                sum(
                    x["semantic_confound"] != "NONE"
                    for x in verdicts
                ),
            "confidence_not_high":
                sum(
                    x["confidence"] != "HIGH"
                    for x in verdicts
                ),
        },
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Review weakest / most important examples:
    # show first two selected from every cell.
    lines = []

    for g in sorted(
        {x["gold_index"] for x in final}
    ):
        for fam in ["DIRECTION", "META"]:
            xs = [
                x for x in final
                if x["gold_index"] == g
                and x["family"] == fam
            ]

            if not xs:
                continue

            lines += [
                "",
                "=" * 100,
                f"G{g:02d} / {fam} / n={len(xs)}",
                "=" * 100,
                "GOLD:",
                xs[0]["chosen"],
            ]

            for x in xs[:3]:
                lines += [
                    "",
                    "NEG:",
                    x["rejected"],
                ]

    REVIEW.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("FINAL VERIFIED SUMMARY")
    print("=" * 78)
    print(json.dumps(summary, indent=2))
    print()
    print("preferences:", FINAL)
    print("review     :", REVIEW)


if __name__ == "__main__":
    main()
