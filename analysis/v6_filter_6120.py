from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI


ROOT = Path("artifacts/v6_pref_api_bulk_6120")
SRC = ROOT / "candidates.jsonl"

OUT = Path("artifacts/v6_pref_filtered_v1")
OUT.mkdir(parents=True, exist_ok=True)

RAW = OUT / "judge_raw.jsonl"
JUDGED = OUT / "judged.jsonl"
FINAL = OUT / "preferences.jsonl"
REVIEW = OUT / "review.txt"
SUMMARY = OUT / "summary.json"

TARGET_PER_GOLD_FAMILY = 15


SYSTEM = r"""
You are a precision evaluator for hard-negative preference data.

Each item contains:
- a GOLD question: the positive completion
- a candidate REJECTED question
- an intended failure family

The GOLD is known to be a useful informative probe.
Your job is NOT to judge informativeness.

Judge whether the rejected candidate is a high-quality HARD NEGATIVE
for the GOLD.

Failure families:

DIRECTION:
The candidate should primarily sound like an assistant asking a user
for clarification, preferences, missing information, or what help is
needed, rather than a normal user asking the application to do a task.

META:
The candidate should primarily ask about the AI/application/system's
own capability, method, experience, approach, behavior, or handling
of the same topic.

DANGLING:
The candidate should remain user-to-system and non-meta, but contain
an unresolved reference or missing antecedent that prevents it from
being fully standalone, such as "these tasks", "this area",
"the current setup", "this process", or equivalent.

A strong hard negative must:
1. preserve the GOLD's substantive topic/domain;
2. express the intended failure family clearly;
3. make that intended failure the MAIN defect;
4. avoid unrelated domain/task drift;
5. remain grammatical and plausible;
6. be reasonably close to the GOLD in content/specificity.

Use these scores:

topic_preservation:
  0 = different task/domain
  1 = weakly related
  2 = same broad topic but material drift
  3 = same substantive topic

family_strength:
  0 = intended failure absent
  1 = ambiguous
  2 = present
  3 = clear/unmistakable

isolation:
  0 = many other serious defects/domain drift
  1 = substantial confounds
  2 = mostly isolated
  3 = cleanly isolated

plausibility:
  0 = malformed/nonsensical
  1 = very awkward
  2 = plausible
  3 = natural plausible model output

hardness:
  0 = trivially unrelated
  1 = easy negative
  2 = reasonably close
  3 = strong near-miss/hard negative

PASS should be true only when:
- topic_preservation >= 2
- family_strength >= 2
- isolation >= 2
- plausibility >= 2
- hardness >= 2

Be conservative.
Return JSON only.
"""


def read_jsonl(p):
    if not p.exists():
        return []
    return [
        json.loads(x)
        for x in p.open(encoding="utf-8")
        if x.strip()
    ]


def append_jsonl(p, row):
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def norm(s):
    return " ".join(str(s).lower().split())


def candidate_id(r):
    return (
        f"G{int(r['gold_index']):02d}:"
        f"{r['family']}:"
        f"{r['request_key']}:"
        f"C{int(r['candidate_index']):02d}"
    )


def make_prompt(batch):
    items = []

    for r in batch:
        items.append({
            "id": candidate_id(r),
            "family": r["family"],
            "gold": r["chosen"],
            "rejected": r["rejected"],
        })

    return (
        "Evaluate every item below. Do not omit IDs.\n\n"
        + json.dumps(items, ensure_ascii=False, indent=2)
        + """

Return exactly:

{
  "results": [
    {
      "id": "...",
      "topic_preservation": 0,
      "family_strength": 0,
      "isolation": 0,
      "plausibility": 0,
      "hardness": 0,
      "pass": false,
      "reason": "brief explanation"
    }
  ]
}
"""
    )


def parse_response(text, requested):
    obj = json.loads(text)

    xs = obj.get("results")
    if not isinstance(xs, list):
        raise ValueError("missing results")

    requested = set(requested)
    out = {}

    for x in xs:
        if not isinstance(x, dict):
            continue

        cid = str(x.get("id") or "")

        if cid not in requested:
            continue

        try:
            vals = {
                k: int(x[k])
                for k in [
                    "topic_preservation",
                    "family_strength",
                    "isolation",
                    "plausibility",
                    "hardness",
                ]
            }
        except Exception:
            continue

        if not all(0 <= v <= 3 for v in vals.values()):
            continue

        passed = bool(x.get("pass"))

        # Enforce our gate ourselves; don't blindly trust boolean.
        hard_pass = (
            vals["topic_preservation"] >= 2
            and vals["family_strength"] >= 2
            and vals["isolation"] >= 2
            and vals["plausibility"] >= 2
            and vals["hardness"] >= 2
        )

        out[cid] = {
            "id": cid,
            **vals,
            "model_pass": passed,
            "pass": hard_pass,
            "reason": str(x.get("reason") or ""),
        }

    return out


def lexical_overlap(a, b):
    wa = set(re.findall(r"[a-zA-Z0-9_]+", a.lower()))
    wb = set(re.findall(r"[a-zA-Z0-9_]+", b.lower()))

    if not wa or not wb:
        return 0.0

    return len(wa & wb) / len(wa | wb)


def score(x):
    # Judge dominates. Lexical overlap is only a small tie-breaker.
    return (
        3.0 * x["topic_preservation"]
        + 3.0 * x["family_strength"]
        + 2.0 * x["isolation"]
        + 1.5 * x["plausibility"]
        + 2.5 * x["hardness"]
        + 1.0 * x["lexical_overlap"]
    )


def too_similar(a, b):
    # Cheap near-duplicate guard.
    sa = set(re.findall(r"[a-zA-Z0-9_]+", a.lower()))
    sb = set(re.findall(r"[a-zA-Z0-9_]+", b.lower()))

    if not sa or not sb:
        return False

    jac = len(sa & sb) / len(sa | sb)
    return jac >= 0.88


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--max-retries",
        type=int,
        default=5,
    )

    ap.add_argument("--dry-run", action="store_true")

    args = ap.parse_args()

    rows = read_jsonl(SRC)

    assert len(rows) == 6120

    for r in rows:
        r["cid"] = candidate_id(r)

    assert len({r["cid"] for r in rows}) == 6120

    print("=" * 72)
    print(" V6 HIGH-PRECISION FILTER")
    print("=" * 72)
    print("input candidates :", len(rows))
    print("batch size       :", args.batch_size)
    print(
        "planned calls    :",
        math.ceil(len(rows) / args.batch_size),
    )
    print(
        "target max final :",
        17 * 3 * TARGET_PER_GOLD_FAMILY,
    )
    print("model            :", args.model)

    if args.dry_run:
        print("DRY RUN: PASS")
        return

    # Resume support.
    judged = {
        x["id"]: x
        for x in read_jsonl(JUDGED)
    }

    print("already judged   :", len(judged))

    client = OpenAI()

    pending = [
        r for r in rows
        if r["cid"] not in judged
    ]

    batches = [
        pending[i:i + args.batch_size]
        for i in range(0, len(pending), args.batch_size)
    ]

    for bi, batch in enumerate(batches, 1):
        missing = {
            r["cid"]: r
            for r in batch
            if r["cid"] not in judged
        }

        attempt = 0

        while missing and attempt < args.max_retries:
            attempt += 1

            sub = list(missing.values())
            prompt = make_prompt(sub)

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
                            "content": SYSTEM,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )

                txt = (
                    resp.choices[0]
                    .message.content
                    or ""
                )

                # Save paid response BEFORE parsing.
                append_jsonl(
                    RAW,
                    {
                        "batch": bi,
                        "attempt": attempt,
                        "requested_ids":
                            list(missing),
                        "raw_content": txt,
                    },
                )

                parsed = parse_response(
                    txt,
                    list(missing),
                )

                for cid, j in parsed.items():
                    src = missing[cid]

                    out = {
                        **j,
                        "gold_index":
                            int(src["gold_index"]),
                        "family":
                            src["family"],
                        "gold":
                            src["chosen"],
                        "rejected":
                            src["rejected"],
                        "prompt":
                            src["prompt"],
                        "request_key":
                            src["request_key"],
                        "candidate_index":
                            int(src["candidate_index"]),
                    }

                    append_jsonl(JUDGED, out)
                    judged[cid] = out
                    missing.pop(cid, None)

                print(
                    f"[judge] batch={bi}/{len(batches)} "
                    f"attempt={attempt} "
                    f"accepted={len(parsed)} "
                    f"missing={len(missing)} "
                    f"total={len(judged)}/6120",
                    flush=True,
                )

                if missing:
                    time.sleep(0.5)

            except Exception as e:
                print(
                    f"[retry] batch={bi} "
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
                f"batch {bi}: unresolved IDs "
                f"after retries: {list(missing)[:5]}"
            )

    assert len(judged) == 6120

    # --------------------------------------------------------------
    # Selection.
    # --------------------------------------------------------------

    eligible = []

    for x in judged.values():
        x = dict(x)

        x["lexical_overlap"] = lexical_overlap(
            x["gold"],
            x["rejected"],
        )

        x["selection_score"] = score(x)

        if x["pass"]:
            eligible.append(x)

    groups = defaultdict(list)

    for x in eligible:
        groups[
            (
                int(x["gold_index"]),
                x["family"],
            )
        ].append(x)

    selected = []

    for key, xs in sorted(groups.items()):
        xs.sort(
            key=lambda z: z["selection_score"],
            reverse=True,
        )

        chosen = []

        for x in xs:
            if any(
                too_similar(
                    x["rejected"],
                    y["rejected"],
                )
                for y in chosen
            ):
                continue

            chosen.append(x)

            if len(chosen) >= TARGET_PER_GOLD_FAMILY:
                break

        selected.extend(chosen)

    # Final DPO format.
    pref_rows = []

    for x in selected:
        pref_rows.append({
            "gold_index":
                int(x["gold_index"]),
            "family":
                x["family"],
            "prompt":
                x["prompt"],
            "chosen":
                x["gold"],
            "rejected":
                x["rejected"],
            "weight":
                1.0,
            "judge": {
                "topic_preservation":
                    x["topic_preservation"],
                "family_strength":
                    x["family_strength"],
                "isolation":
                    x["isolation"],
                "plausibility":
                    x["plausibility"],
                "hardness":
                    x["hardness"],
                "lexical_overlap":
                    x["lexical_overlap"],
                "selection_score":
                    x["selection_score"],
            },
        })

    with FINAL.open("w", encoding="utf-8") as f:
        for x in pref_rows:
            f.write(
                json.dumps(x, ensure_ascii=False)
                + "\n"
            )

    by_gf = Counter(
        (x["gold_index"], x["family"])
        for x in pref_rows
    )

    by_family = Counter(
        x["family"]
        for x in pref_rows
    )

    by_gold = Counter(
        x["gold_index"]
        for x in pref_rows
    )

    pass_by_family = Counter(
        x["family"]
        for x in eligible
    )

    summary = {
        "input": len(rows),
        "judged": len(judged),
        "passed_gate": len(eligible),
        "selected": len(pref_rows),
        "passed_by_family":
            dict(pass_by_family),
        "selected_by_family":
            dict(by_family),
        "selected_per_gold":
            dict(by_gold),
        "selected_gold_family_min":
            min(by_gf.values()) if by_gf else 0,
        "selected_gold_family_max":
            max(by_gf.values()) if by_gf else 0,
        "target_per_gold_family":
            TARGET_PER_GOLD_FAMILY,
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Compact human audit:
    # best + weakest selected from every Gold/family.
    lines = []

    allkeys = sorted(by_gf)

    for key in allkeys:
        gi, fam = key

        xs = [
            x for x in selected
            if int(x["gold_index"]) == gi
            and x["family"] == fam
        ]

        xs.sort(
            key=lambda z: z["selection_score"],
            reverse=True,
        )

        if not xs:
            continue

        show = (
            [xs[0]]
            if len(xs) == 1
            else [xs[0], xs[-1]]
        )

        lines += [
            "",
            "=" * 100,
            f"G{gi:02d} / {fam} / "
            f"selected={len(xs)}",
            "=" * 100,
            "GOLD:",
            xs[0]["gold"],
        ]

        for tag, x in zip(
            ["BEST", "WEAKEST"],
            show,
        ):
            lines += [
                "",
                f"{tag} score="
                f"{x['selection_score']:.2f}",
                "REJECTED:",
                x["rejected"],
                (
                    "scores: "
                    f"T={x['topic_preservation']} "
                    f"F={x['family_strength']} "
                    f"I={x['isolation']} "
                    f"P={x['plausibility']} "
                    f"H={x['hardness']} "
                    f"lex={x['lexical_overlap']:.3f}"
                ),
                "judge:",
                x["reason"],
            ]

    REVIEW.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(" FINAL FILTER SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2))
    print()
    print("preferences:", FINAL)
    print("review     :", REVIEW)


if __name__ == "__main__":
    main()
