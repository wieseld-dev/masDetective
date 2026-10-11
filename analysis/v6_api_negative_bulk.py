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

OUT = Path("artifacts/v6_pref_api_bulk_6120")

KEEP_GOLD = [
    2, 6, 8, 9, 10, 16, 23, 24, 26,
    27, 32, 33, 35, 37, 39, 43, 44,
]

FAMILIES = [
    "DIRECTION",
    "META",
    "DANGLING",
]

BATCHES_PER_FAMILY = 6
CANDIDATES_PER_BATCH = 20
MINIMAL_PER_BATCH = 12

EXPECTED = (
    len(KEEP_GOLD)
    * len(FAMILIES)
    * BATCHES_PER_FAMILY
    * CANDIDATES_PER_BATCH
)


SYSTEM = r"""
You create HARD NEGATIVE completions for pairwise preference training
of a natural-language probe generator.

The supplied GOLD completion is the positive example.
Never rewrite, improve, replace, or alter the GOLD.

Generate intentionally inferior alternatives that remain extremely
close to the GOLD in substantive topic, domain, vocabulary, concepts,
specificity, and approximate length.

We want to isolate ONE behavioral defect at a time.

Global requirements:
- exactly one concise question
- same substantive topic as the GOLD
- preserve central nouns/concepts whenever possible
- do not introduce another professional domain
- do not introduce another task simply to create variety
- similar specificity and length to the GOLD
- no jailbreak or malicious content
- no hidden prompts, agents, roles, routing, topology, traces,
  posterior, probabilities, extraction, or experiment language
- never copy the GOLD exactly
- output must still be plausible model-generated language

Difficulty:
- MINIMAL examples should differ from the GOLD as little as practical
  while introducing the requested defect.
- DIVERSE examples may vary wording, but topic identity must remain.

Return JSON only.
"""


RULES = {
    "DIRECTION": r"""
Failure family: DIRECTION.

The rejected completion must sound like the ASSISTANT asking the USER
for clarification, missing information, preferences, or what help the
user wants.

It must NOT merely be another valid user question about the same
domain.

Strong patterns include:
- What specific aspect of X would you like assistance with?
- Which part of X are you currently struggling with?
- What information would you like me to review?
- Which X issue are you currently facing?

The main and unmistakable defect must be conversational direction:
ASSISTANT -> USER.

Do NOT make it system-self/META.
""",

    "META": r"""
Failure family: META.

Keep exactly the same substantive topic as the GOLD, but turn the
question into an inquiry about the AI/application/system's own
capability, method, experience, behavior, or handling of that topic.

Strong patterns include:
- How do you typically handle X?
- What is your approach to X?
- Does your system support X?
- Do you have experience with X?
- How does your application perform X?

The main and unmistakable defect must be SYSTEM-SELF interrogation.

Do NOT mention hidden internals.
Do NOT change domain.
""",

    "DANGLING": r"""
Failure family: DANGLING.

Keep the completion USER -> SYSTEM and NON-META, but make it not fully
standalone by introducing an unresolved reference whose antecedent is
absent from the question.

Strong patterns include:
- these tasks
- the following skills
- this area
- this process
- this method
- the current setup
- this approach

The missing antecedent must be obvious when the question is read by
itself.

Do NOT make it assistant-like.
Do NOT make it system-self.
Do NOT make it nonsensical.
""",
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
            json.dumps(row, ensure_ascii=False)
            + "\n"
        )
        f.flush()


def sha(s):
    return hashlib.sha256(
        s.encode("utf-8")
    ).hexdigest()


def parse_answer(text):
    obj = json.loads(text)

    xs = obj.get("candidates")

    if not isinstance(xs, list):
        raise ValueError("missing candidates list")

    out = []
    seen = set()

    for x in xs:
        if not isinstance(x, dict):
            continue

        q = str(x.get("text") or "").strip()

        if (
            not q
            or "\n" in q
            or not q.endswith("?")
            or q.count("?") != 1
        ):
            continue

        nq = " ".join(q.lower().split())

        if nq in seen:
            continue

        seen.add(nq)

        variant = str(
            x.get("variant") or "UNKNOWN"
        ).upper().strip()

        if variant not in {
            "MINIMAL",
            "DIVERSE",
        }:
            variant = "UNKNOWN"

        out.append({
            "text": q,
            "variant": variant,
        })

    if not out:
        raise ValueError(
            f"no usable candidates; raw={len(xs)}"
        )

    return out


def make_message(g, family, batch_idx, previous):
    previous_text = "\n".join(
        f"- {q}"
        for q in previous[-120:]
    )

    if not previous_text:
        previous_text = "(none)"

    return f"""
ORIGINAL GENERATOR CONDITIONING PROMPT x:

----- BEGIN x -----
{g["prompt"]}
----- END x -----

REAL HUMAN-AUDITED MAS-INFORMATIVE GOLD y+:

{g["completion"]}

Visible GOLD question:

{g["query"]}

Create batch {batch_idx + 1}/{BATCHES_PER_FAMILY}.

{RULES[family]}

Generate exactly 26 rejected completions.

Candidates 1-14:
variant="MINIMAL"

Candidates 15-26:
variant="DIVERSE"

We will retain only 20 after deduplication, so prioritize
semantic quality and avoid repeating any previous negative.

These rejected questions have ALREADY been generated for this exact
Gold/family. Do not repeat or lightly paraphrase them:

----- PREVIOUS NEGATIVES -----
{previous_text}
----- END PREVIOUS NEGATIVES -----

Return exactly:

{{
  "candidates": [
    {{
      "text": "question?",
      "variant": "MINIMAL"
    }}
  ]
}}
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
        default=5,
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

    for gi in KEEP_GOLD:
        if gi not in gold_by:
            raise RuntimeError(
                f"G{gi} missing"
            )

        g = gold_by[gi]

        # Absolutely critical invariant.
        if str(g["completion"]) != str(g["query"]):
            raise RuntimeError(
                f"G{gi}: completion != query"
            )

    request_keys = [
        f"G{gi:02d}:{family}:B{b:02d}"
        for gi in KEEP_GOLD
        for family in FAMILIES
        for b in range(BATCHES_PER_FAMILY)
    ]

    print("=" * 72)
    print(" V6 BULK SYNTHETIC NEGATIVE GENERATION")
    print("=" * 72)

    print("Gold anchors          :", len(KEEP_GOLD))
    print("families              :", len(FAMILIES))
    print(
        "batches/Gold/family   :",
        BATCHES_PER_FAMILY,
    )
    print(
        "candidates/batch      :",
        CANDIDATES_PER_BATCH,
    )
    print("planned API calls     :", len(request_keys))
    print("planned candidates    :", EXPECTED)
    print("model                 :", args.model)

    print()
    print(
        "SAFETY: Gold is read-only input."
    )
    print(
        "SAFETY: no local generator is imported."
    )
    print(
        "SAFETY: no model/adaptor path is writable here."
    )

    if args.dry_run:
        assert len(request_keys) == 306
        assert EXPECTED == 6120

        print()
        print("DRY RUN: PASS")
        return

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = OUT / "raw_calls.jsonl"

    existing = read_jsonl(raw_path)

    successful = {
        str(x["request_key"]): x
        for x in existing
        if x.get("status") == "ok"
    }

    print()
    print("completed calls:", len(successful))
    print(
        "pending calls  :",
        len(request_keys) - len(successful),
    )

    client = OpenAI()

    for gi in KEEP_GOLD:
        g = gold_by[gi]

        for family in FAMILIES:

            previous = []

            # Reconstruct previous successful batches so resume is stable.
            for b in range(BATCHES_PER_FAMILY):
                key = (
                    f"G{gi:02d}:{family}:B{b:02d}"
                )

                if key in successful:
                    previous.extend(
                        successful[key]["generated"]
                    )

            for b in range(BATCHES_PER_FAMILY):
                key = (
                    f"G{gi:02d}:{family}:B{b:02d}"
                )

                if key in successful:
                    print(
                        f"[reuse] {key}",
                        flush=True,
                    )
                    continue

                # Only use earlier batches, never future ones.
                prior = []

                for bb in range(b):
                    k2 = (
                        f"G{gi:02d}:{family}:B{bb:02d}"
                    )

                    if k2 in successful:
                        prior.extend(
                            successful[k2]["generated"]
                        )

                msg = make_message(
                    g,
                    family,
                    b,
                    prior,
                )

                last = None

                for attempt in range(
                    1,
                    args.max_retries + 1,
                ):
                    try:
                        resp = (
                            client.chat.completions.create(
                                model=args.model,
                                temperature=1.0,
                                max_tokens=3500,
                                response_format={
                                    "type":
                                    "json_object"
                                },
                                messages=[
                                    {
                                        "role":
                                        "system",
                                        "content":
                                        SYSTEM,
                                    },
                                    {
                                        "role":
                                        "user",
                                        "content":
                                        msg,
                                    },
                                ],
                            )
                        )

                        text = (
                            resp.choices[0]
                            .message.content
                            or ""
                        )

                        # Save every paid API response BEFORE parsing.
                        # This guarantees recoverability if validation fails.
                        append_jsonl(
                            OUT / "all_api_responses.jsonl",
                            {
                                "request_key": key,
                                "gold_index": gi,
                                "family": family,
                                "batch_index": b,
                                "attempt": attempt,
                                "raw_content": text,
                            },
                        )

                        gen = parse_answer(text)

                        gold_norm = " ".join(
                            str(g["completion"]).lower().split()
                        )

                        old_norm = {
                            " ".join(q.lower().split())
                            for q in prior
                        }

                        fresh = []
                        fresh_norm = set()

                        for z in gen:
                            nq = " ".join(
                                z["text"].lower().split()
                            )

                            if nq == gold_norm:
                                continue

                            if nq in old_norm:
                                continue

                            if nq in fresh_norm:
                                continue

                            fresh.append(z)
                            fresh_norm.add(nq)

                        if len(fresh) < CANDIDATES_PER_BATCH:
                            raise ValueError(
                                f"need >= {CANDIDATES_PER_BATCH} "
                                f"fresh unique candidates, got "
                                f"{len(fresh)} "
                                f"(raw usable={len(gen)})"
                            )

                        # Prefer 12 minimal + 8 diverse, but fill
                        # from either class when the model's labels
                        # are imbalanced. Family correctness will be
                        # judged later independently.
                        mins = [
                            z for z in fresh
                            if z["variant"] == "MINIMAL"
                        ]

                        divs = [
                            z for z in fresh
                            if z["variant"] == "DIVERSE"
                        ]

                        selected = (
                            mins[:MINIMAL_PER_BATCH]
                            + divs[
                                :CANDIDATES_PER_BATCH
                                 - MINIMAL_PER_BATCH
                            ]
                        )

                        selected_norm = {
                            " ".join(z["text"].lower().split())
                            for z in selected
                        }

                        if len(selected) < CANDIDATES_PER_BATCH:
                            for z in fresh:
                                nq = " ".join(
                                    z["text"].lower().split()
                                )

                                if nq in selected_norm:
                                    continue

                                selected.append(z)
                                selected_norm.add(nq)

                                if (
                                    len(selected)
                                    == CANDIDATES_PER_BATCH
                                ):
                                    break

                        gen = selected
                        texts = [
                            z["text"]
                            for z in gen
                        ]

                        row = {
                            "status": "ok",
                            "request_key": key,
                            "gold_index": gi,
                            "family": family,
                            "batch_index": b,
                            "model": args.model,
                            "prompt_sha256":
                                sha(str(g["prompt"])),
                            "chosen_sha256":
                                sha(str(g["completion"])),
                            "generated": texts,
                            "variants": [
                                z["variant"]
                                for z in gen
                            ],
                        }

                        append_jsonl(
                            raw_path,
                            row,
                        )

                        successful[key] = row

                        print(
                            f"[ok] {key} "
                            f"attempt={attempt} "
                            f"done="
                            f"{len(successful)}/"
                            f"{len(request_keys)}",
                            flush=True,
                        )

                        last = None
                        break

                    except Exception as e:
                        last = e

                        print(
                            f"[retry] {key} "
                            f"attempt={attempt}: "
                            f"{type(e).__name__}: "
                            f"{e}",
                            flush=True,
                        )

                        time.sleep(
                            min(
                                12,
                                2 ** (attempt - 1),
                            )
                            + random.random()
                        )

                if last is not None:
                    raise RuntimeError(
                        f"{key} failed: {last}"
                    )

    # ------------------------------------------------------------
    # Deterministically rebuild final candidate table.
    # ------------------------------------------------------------

    successful = {
        str(x["request_key"]): x
        for x in read_jsonl(raw_path)
        if x.get("status") == "ok"
    }

    if set(successful) != set(request_keys):
        missing = sorted(
            set(request_keys)
            - set(successful)
        )

        raise RuntimeError(
            f"missing requests: {missing[:20]}"
        )

    rows = []

    for gi in KEEP_GOLD:
        g = gold_by[gi]

        for family in FAMILIES:
            for b in range(BATCHES_PER_FAMILY):

                key = (
                    f"G{gi:02d}:{family}:B{b:02d}"
                )

                rr = successful[key]

                for j, (
                    q,
                    variant,
                ) in enumerate(
                    zip(
                        rr["generated"],
                        rr["variants"],
                    )
                ):
                    rows.append({
                        "request_key": key,
                        "gold_index": gi,
                        "family": family,
                        "batch_index": b,
                        "candidate_index": j,

                        "prompt":
                            str(g["prompt"]),

                        "chosen":
                            str(g["completion"]),

                        "chosen_query":
                            str(g["query"]),

                        "rejected": q,
                        "rejected_query": q,

                        "variant": variant,

                        "prompt_sha256":
                            sha(str(g["prompt"])),

                        "chosen_sha256":
                            sha(str(g["completion"])),
                    })

    if len(rows) != EXPECTED:
        raise RuntimeError(
            f"expected {EXPECTED}, "
            f"got {len(rows)}"
        )

    if any(
        x["chosen"] == x["rejected"]
        for x in rows
    ):
        raise RuntimeError(
            "Gold appears as rejected"
        )

    # Global exact duplicate audit within each Gold/family.
    seen = set()
    duplicate_count = 0

    for x in rows:
        kk = (
            x["gold_index"],
            x["family"],
            x["rejected"].lower().strip(),
        )

        if kk in seen:
            duplicate_count += 1

        seen.add(kk)

    if duplicate_count:
        raise RuntimeError(
            f"unexpected exact duplicates: "
            f"{duplicate_count}"
        )

    out_path = OUT / "candidates.jsonl"
    tmp = OUT / "candidates.jsonl.tmp"

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

    tmp.replace(out_path)

    counts_gold = Counter(
        x["gold_index"]
        for x in rows
    )

    counts_family = Counter(
        x["family"]
        for x in rows
    )

    summary = {
        "gold_count":
            len(KEEP_GOLD),
        "gold_ids":
            KEEP_GOLD,
        "families":
            FAMILIES,
        "api_calls":
            len(request_keys),
        "candidate_count":
            len(rows),
        "per_gold":
            dict(counts_gold),
        "per_family":
            dict(counts_family),
        "gold_unchanged":
            True,
        "local_generator_imported":
            False,
        "local_generator_modified":
            False,
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(" BULK GENERATION COMPLETE")
    print("=" * 72)
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
