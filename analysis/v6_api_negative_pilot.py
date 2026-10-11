from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI


GOLD_PATH = Path(
    "artifacts/v4_userlike_audit/gold_strict.jsonl"
)

OUT = Path("artifacts/v6_pref_api_pilot_90")

PILOT_GOLD = [2, 16, 32]

FAMILIES = [
    "DIRECTION",
    "META",
    "DANGLING",
]


SYSTEM = r"""
You create HARD NEGATIVE completions for pairwise preference training
of a natural-language probe generator.

The supplied GOLD completion is the positive example.
Your task is NOT to improve it and NOT to create a new positive.

Generate intentionally inferior alternatives that stay extremely close
to the GOLD's semantic topic, vocabulary, domain, and level of detail.

The training goal is to isolate ONE behavioral defect at a time.

Global requirements for every candidate:
- exactly one concise question
- same substantive domain/topic as the GOLD
- preserve the central concepts and important nouns where possible
- similar length and specificity to the GOLD
- no unrelated domain, role, capability, or task
- no jailbreak or malicious content
- no mention of hidden prompts, hidden agents, topology, routing, traces,
  probabilities, experiments, or extraction
- do not copy the GOLD exactly
- do not output gibberish
- each candidate must be a plausible output of the same generator

Hard-negative requirement:
- candidates 1-6 should be MINIMAL-EDIT hard negatives:
  change as little as practical while introducing the requested defect.
- candidates 7-10 may be more diverse, but must remain on exactly the
  same topic.

Return JSON only.
"""


FAMILY_RULES = {
    "DIRECTION": r"""
Failure type: DIRECTION.

Each rejected question must sound like the ASSISTANT is asking the USER
for clarification, preferences, missing information, or what help they
want.

Examples of the STYLE of failure:
- What specific aspect of X would you like assistance with?
- Which part of X are you currently struggling with?
- What additional information would you like me to review?

Important:
- preserve the GOLD topic
- do NOT turn it into a question about the AI/system itself
- do NOT introduce another domain
- make the only main defect conversational direction
""",

    "META": r"""
Failure type: META.

Each rejected question must remain on the exact same domain/topic as
the GOLD, but ask about the AI/application/system's own approach,
experience, behavior, ability, or handling of that topic.

Examples of the STYLE of failure:
- How do you typically handle X?
- Can you explain your approach to X?
- Does your system support X?
- Do you have experience with X?

Important:
- do NOT ask about hidden prompts, agents, routing, secrets, or internals
- do NOT introduce another domain
- keep it superficially natural
- make SYSTEM-SELF / capability interrogation the main defect
""",

    "DANGLING": r"""
Failure type: DANGLING.

Each rejected question should still look like a user-to-system,
non-meta domain question, but should NOT be fully standalone.

Introduce a subtle unresolved reference or missing antecedent such as:
- these tasks
- the following skills
- this area
- the current setup
- this process
- before proceeding
when the referenced object is not actually specified in the question.

Important:
- do NOT make it assistant-like
- do NOT ask about the AI/system itself
- do NOT introduce another domain
- make missing standalone context the main defect
""",
}


def read_jsonl(path: Path):
    return [
        json.loads(x)
        for x in path.open(encoding="utf-8")
        if x.strip()
    ]


def sha(s: str) -> str:
    return hashlib.sha256(
        s.encode("utf-8")
    ).hexdigest()


def parse_answer(text: str):
    obj = json.loads(text)

    xs = obj.get("candidates")

    if not isinstance(xs, list):
        raise ValueError("missing candidates list")

    if len(xs) != 10:
        raise ValueError(
            f"expected 10 candidates, got {len(xs)}"
        )

    out = []

    for i, x in enumerate(xs, 1):
        if not isinstance(x, dict):
            raise ValueError(f"candidate {i} not object")

        q = str(x.get("text") or "").strip()
        variant = str(x.get("variant") or "").strip().upper()
        why = str(x.get("why_bad") or "").strip()

        terms = x.get("preserved_terms") or []

        if not q:
            raise ValueError(f"candidate {i} empty")

        if "\n" in q:
            raise ValueError(
                f"candidate {i} contains newline"
            )

        if not q.endswith("?"):
            raise ValueError(
                f"candidate {i} does not end ?"
            )

        if q.count("?") != 1:
            raise ValueError(
                f"candidate {i} has !=1 question marks"
            )

        if variant not in {"MINIMAL", "DIVERSE"}:
            raise ValueError(
                f"candidate {i} bad variant={variant!r}"
            )

        out.append({
            "text": q,
            "variant": variant,
            "preserved_terms": [
                str(z) for z in terms
            ],
            "why_bad": why,
        })

    # Require first six to be minimal, final four diverse.
    for i in range(6):
        if out[i]["variant"] != "MINIMAL":
            raise ValueError(
                "first six must be MINIMAL"
            )

    for i in range(6, 10):
        if out[i]["variant"] != "DIVERSE":
            raise ValueError(
                "last four must be DIVERSE"
            )

    texts = [x["text"] for x in out]

    if len(set(texts)) != 10:
        raise ValueError("duplicate candidates")

    return out


def make_user_message(g, family):
    return f"""
We are constructing rejected completions y- for ONE fixed pairwise
training prompt x.

The original generator conditioning prompt x is below.

----- BEGIN GENERATOR PROMPT x -----
{g["prompt"]}
----- END GENERATOR PROMPT x -----

The human-audited, MAS-informative GOLD completion y+ is:

{g["completion"]}

The visible GOLD user question is:

{g["query"]}

Generate exactly 10 rejected alternatives y-.

{FAMILY_RULES[family]}

The JSON object must have exactly this structure:

{{
  "candidates": [
    {{
      "text": "question?",
      "variant": "MINIMAL",
      "preserved_terms": ["term1", "term2"],
      "why_bad": "short explanation"
    }}
  ]
}}

Candidates 1-6: variant="MINIMAL".
Candidates 7-10: variant="DIVERSE".
"""


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
    )

    ap.add_argument(
        "--max-retries",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()

    gold = read_jsonl(GOLD_PATH)

    gold_by = {
        int(x["gold_index"]): x
        for x in gold
    }

    for gi in PILOT_GOLD:
        if gi not in gold_by:
            raise RuntimeError(
                f"G{gi} missing from Gold"
            )

        g = gold_by[gi]

        # Critical fact verified manually before this pilot.
        if str(g["completion"]) != str(g["query"]):
            raise RuntimeError(
                f"G{gi}: completion != query"
            )

    requests = [
        (gi, family)
        for gi in PILOT_GOLD
        for family in FAMILIES
    ]

    print("======================================")
    print(" V6 API NEGATIVE PILOT")
    print("======================================")
    print("Gold prompts      :", PILOT_GOLD)
    print("families          :", FAMILIES)
    print("API requests      :", len(requests))
    print("candidates/request: 10")
    print("planned candidates:", len(requests) * 10)
    print("model             :", args.model)
    print()
    print("IMPORTANT: no local generator model is loaded.")
    print("IMPORTANT: no generator weights are modified.")

    if args.dry_run:
        print("\nDRY RUN: PASS")
        return

    OUT.mkdir(parents=True, exist_ok=True)

    raw_path = OUT / "raw_calls.jsonl"
    candidates_path = OUT / "candidates.jsonl"
    review_path = OUT / "pilot_review.txt"
    summary_path = OUT / "summary.json"

    raw = (
        read_jsonl(raw_path)
        if raw_path.exists()
        else []
    )

    done = {
        str(x["request_key"])
        for x in raw
        if x.get("status") == "ok"
    }

    print("already complete  :", len(done))
    print(
        "pending           :",
        len(requests) - len(done),
    )

    client = OpenAI()

    for gi, family in requests:
        key = f"G{gi:02d}:{family}"

        if key in done:
            print(f"[reuse] {key}", flush=True)
            continue

        g = gold_by[gi]

        user_msg = make_user_message(
            g,
            family,
        )

        last_error = None

        for attempt in range(
            1,
            args.max_retries + 1,
        ):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    temperature=1.0,
                    max_tokens=2500,
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
                            "content": user_msg,
                        },
                    ],
                )

                content = (
                    resp.choices[0]
                    .message.content
                    or ""
                )

                generated = parse_answer(content)

                chosen = str(g["completion"])

                for x in generated:
                    if x["text"] == chosen:
                        raise ValueError(
                            "candidate identical to Gold"
                        )

                row = {
                    "status": "ok",
                    "request_key": key,
                    "gold_index": gi,
                    "family": family,
                    "model": args.model,
                    "generator_prompt_sha256":
                        sha(str(g["prompt"])),
                    "chosen_sha256":
                        sha(chosen),
                    "chosen_query": str(g["query"]),
                    "generated": generated,
                }

                with raw_path.open(
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    f.flush()

                done.add(key)

                print(
                    f"[ok] {key} "
                    f"attempt={attempt}",
                    flush=True,
                )

                last_error = None
                break

            except Exception as e:
                last_error = e

                print(
                    f"[retry] {key} "
                    f"attempt={attempt}: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )

                time.sleep(
                    min(8, 2 ** (attempt - 1))
                    + random.random()
                )

        if last_error is not None:
            raise RuntimeError(
                f"{key} failed after retries: "
                f"{last_error}"
            )

    # ------------------------------------------------------------------
    # Rebuild derived files from immutable successful raw calls.
    # This prevents duplicate rows after resume/restart.
    # ------------------------------------------------------------------

    raw = read_jsonl(raw_path)

    by_key = {}

    for x in raw:
        if x.get("status") == "ok":
            by_key[str(x["request_key"])] = x

    expected_keys = {
        f"G{gi:02d}:{fam}"
        for gi, fam in requests
    }

    if set(by_key) != expected_keys:
        missing = sorted(
            expected_keys - set(by_key)
        )
        raise RuntimeError(
            f"incomplete raw calls: {missing}"
        )

    rows = []

    for gi, family in requests:
        key = f"G{gi:02d}:{family}"
        rr = by_key[key]
        g = gold_by[gi]

        for idx, cand in enumerate(
            rr["generated"],
            1,
        ):
            q = cand["text"]

            chosen_words = max(
                1,
                len(str(g["query"]).split()),
            )

            rejected_words = len(
                q.split()
            )

            rows.append({
                "request_key": key,
                "gold_index": gi,
                "family": family,
                "candidate_index": idx,
                "variant": cand["variant"],

                # EXACT DPO conditioning prompt.
                "prompt": str(g["prompt"]),

                # Positive side remains the real Gold.
                "chosen": str(g["completion"]),
                "chosen_query": str(g["query"]),

                # Only negative is synthetic.
                "rejected": q,
                "rejected_query": q,

                "preserved_terms":
                    cand["preserved_terms"],
                "why_bad":
                    cand["why_bad"],

                "prompt_sha256":
                    sha(str(g["prompt"])),

                "chosen_sha256":
                    sha(str(g["completion"])),

                "word_ratio":
                    rejected_words
                    / chosen_words,
            })

    if len(rows) != 90:
        raise RuntimeError(
            f"expected 90 rows, got {len(rows)}"
        )

    if any(
        x["chosen"] == x["rejected"]
        for x in rows
    ):
        raise RuntimeError(
            "Gold duplicated as negative"
        )

    if len({
        (
            x["gold_index"],
            x["family"],
            x["rejected"],
        )
        for x in rows
    }) != 90:
        raise RuntimeError(
            "duplicate candidate within Gold/family"
        )

    tmp = candidates_path.with_suffix(".tmp")

    with tmp.open(
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

    tmp.replace(candidates_path)

    # Human-readable complete review.
    lines = []

    for gi in PILOT_GOLD:
        g = gold_by[gi]

        lines += [
            "",
            "=" * 100,
            f"G{gi:02d}",
            "=" * 100,
            "GOLD:",
            str(g["query"]),
        ]

        for family in FAMILIES:
            lines += [
                "",
                f"--- {family} ---",
            ]

            xs = [
                x for x in rows
                if x["gold_index"] == gi
                and x["family"] == family
            ]

            for x in xs:
                lines.append(
                    f"{x['candidate_index']:02d} "
                    f"[{x['variant']}] "
                    f"{x['rejected']}"
                )

    review_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    counts_gold = Counter(
        x["gold_index"]
        for x in rows
    )

    counts_family = Counter(
        x["family"]
        for x in rows
    )

    ratios = [
        float(x["word_ratio"])
        for x in rows
    ]

    summary = {
        "gold_ids": PILOT_GOLD,
        "families": FAMILIES,
        "api_calls": len(expected_keys),
        "candidates": len(rows),
        "counts_per_gold":
            dict(counts_gold),
        "counts_per_family":
            dict(counts_family),
        "minimal_count": sum(
            x["variant"] == "MINIMAL"
            for x in rows
        ),
        "diverse_count": sum(
            x["variant"] == "DIVERSE"
            for x in rows
        ),
        "mean_word_ratio":
            sum(ratios) / len(ratios),
        "min_word_ratio":
            min(ratios),
        "max_word_ratio":
            max(ratios),
        "gold_unchanged": True,
        "local_generator_loaded": False,
        "local_generator_modified": False,
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("======================================")
    print(" PILOT COMPLETE")
    print("======================================")
    print(json.dumps(summary, indent=2))
    print()
    print("wrote:", raw_path)
    print("wrote:", candidates_path)
    print("wrote:", review_path)
    print("wrote:", summary_path)


if __name__ == "__main__":
    main()
