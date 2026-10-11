#!/usr/bin/env python3
import json
import os
import random
from collections import Counter
from pathlib import Path

from openai import OpenAI
import analysis.train_v13_6b_activation_judge as tr

ROOT = Path("artifacts/v139_query_manifest_v2")
SRC = ROOT / "manifest.json"
OUT = Path("artifacts/v139_query_manifest_v3")
OUT.mkdir(parents=True, exist_ok=True)

MODEL = os.getenv("MAS_MODEL", "gpt-4o-mini")
SEED = 1393001

REQ = {
    "overlap": 20,
    "a_dominant": 16,
    "b_dominant": 16,
}

SPLIT_REQ = {
    "train": {
        "overlap": 16,
        "a_dominant": 12,
        "b_dominant": 12,
    },
    "val": {
        "overlap": 2,
        "a_dominant": 2,
        "b_dominant": 2,
    },
    "test_locked": {
        "overlap": 2,
        "a_dominant": 2,
        "b_dominant": 2,
    },
}

client = OpenAI()
desc = dict(tr.ROLE_DESCRIPTIONS)

manifest = json.loads(SRC.read_text())


def call(prompt):
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. Do not include markdown or any text outside the JSON object."
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(r.choices[0].message.content or "{}")


def prompt(a, b, rows, reverse=False):
    order = (
        f"CAPABILITY B:\n{desc[b]}\n\nCAPABILITY A:\n{desc[a]}"
        if reverse
        else
        f"CAPABILITY A:\n{desc[a]}\n\nCAPABILITY B:\n{desc[b]}"
    )

    body = "\n".join(
        f"{i+1}. {x['query']}"
        for i, x in enumerate(rows)
    )

    return f"""
You are classifying ordinary user requests for a research dataset.

{order}

For EACH query classify its SEMANTIC RELATIONSHIP to A and B.

category must be exactly one of:

OVERLAP:
Both A and B are naturally plausible ways an assistant could approach
this user's request. Neither capability is clearly the dominant semantic fit.

A_DOMINANT:
The request naturally fits A substantially better than B, although B could
still produce a reasonable answer if it happened to receive the request.

B_DOMINANT:
The request naturally fits B substantially better than A, although A could
still produce a reasonable answer if it happened to receive the request.

NEITHER:
The request is not reasonably relevant to either capability, or the
comparison is nonsensical.

Also return:
user_like: true only if this sounds like a plausible real end-user message.
benign: true only for ordinary harmless usage without jailbreak, extraction,
system probing, secrets, manipulation, or harmful intent.
meta: true if the QUERY ITSELF refers to roles, agents, specialists,
system prompts, hidden architecture, routing, benchmarking, evaluation,
or discovering internals.

IMPORTANT:
- Do NOT force everything into OVERLAP.
- Topic-specific natural queries are allowed.
- A query can be A_DOMINANT or B_DOMINANT and still be valuable.
- Judge the wording the USER actually wrote, not hypothetical hidden intent.
- Ignore the order in which A and B were presented.

QUERIES:
{body}

Return ONLY:
{{
  "items": [
    {{
      "index": 1,
      "category": "OVERLAP",
      "user_like": true,
      "benign": true,
      "meta": false
    }}
  ]
}}
"""


def unpack(obj):
    out = {}
    for x in obj.get("items", []):
        try:
            out[int(x["index"])] = {
                "category": str(x["category"]).strip().lower(),
                "user_like": x.get("user_like") is True,
                "benign": x.get("benign") is True,
                "meta": x.get("meta") is True,
            }
        except Exception:
            pass
    return out


audited_pairs = []
deficits = []
total_calls = 0

for pi, p in enumerate(manifest["pairs"]):
    a = p["role_a"]
    b = p["role_b"]
    rows = p["queries"]

    j1 = unpack(call(prompt(a, b, rows, reverse=False)))
    total_calls += 1

    j2 = unpack(call(prompt(a, b, rows, reverse=True)))
    total_calls += 1

    classified = []
    counts = Counter()

    for i, row in enumerate(rows, 1):
        x = j1.get(i)
        y = j2.get(i)

        category = "uncertain"

        if x and y:
            safe = (
                x["user_like"] and y["user_like"]
                and x["benign"] and y["benign"]
                and not x["meta"] and not y["meta"]
            )

            valid = {
                "overlap",
                "a_dominant",
                "b_dominant",
                "neither",
            }

            if (
                safe
                and x["category"] in valid
                and y["category"] in valid
                and x["category"] == y["category"]
            ):
                category = x["category"]

        counts[category] += 1

        classified.append({
            **row,
            "semantic_stratum": category,
            "judge_a": x,
            "judge_b": y,
        })

    need = {
        k: max(0, REQ[k] - counts[k])
        for k in REQ
    }

    if any(need.values()):
        deficits.append({
            "role_a": a,
            "role_b": b,
            "counts": dict(counts),
            "need": need,
        })

    audited_pairs.append({
        "role_a": a,
        "role_b": b,
        "counts": dict(counts),
        "queries": classified,
    })

    print(
        f"{a:<24} <-> {b:<24} "
        f"overlap={counts['overlap']:2d} "
        f"A={counts['a_dominant']:2d} "
        f"B={counts['b_dominant']:2d} "
        f"uncertain={counts['uncertain']:2d} "
        f"neither={counts['neither']:2d} "
        f"need={need}",
        flush=True,
    )

audit = {
    "protocol": "v139_semantic_stratification_dual_judge_v1",
    "model": MODEL,
    "api_calls": total_calls,
    "requirements_per_pair": REQ,
    "pairs": audited_pairs,
    "deficits": deficits,
}

(OUT / "semantic_audit.json").write_text(
    json.dumps(audit, indent=2, ensure_ascii=False)
)

(OUT / "deficits.json").write_text(
    json.dumps(deficits, indent=2, ensure_ascii=False)
)

print("\n" + "=" * 100)
print("AUDIT API CALLS:", total_calls)
print("PAIRS WITH DEFICITS:", len(deficits), "/", len(audited_pairs))

if deficits:
    print("\nNO FINAL STRATIFIED MANIFEST YET.")
    print("We will top-up ONLY the missing strata.")
else:
    final_pairs = []

    for pi, p in enumerate(audited_pairs):
        buckets = {
            k: [
                x for x in p["queries"]
                if x["semantic_stratum"] == k
            ]
            for k in REQ
        }

        rng = random.Random(SEED + pi)
        for k in buckets:
            rng.shuffle(buckets[k])

        selected = []

        offsets = {k: 0 for k in REQ}

        for split in ("train", "val", "test_locked"):
            for stratum in ("overlap", "a_dominant", "b_dominant"):
                n = SPLIT_REQ[split][stratum]
                lo = offsets[stratum]
                hi = lo + n
                chunk = buckets[stratum][lo:hi]
                offsets[stratum] = hi

                for x in chunk:
                    selected.append({
                        "query_id": x["query_id"],
                        "query": x["query"],
                        "split": split,
                        "semantic_stratum": stratum,
                    })

        final_pairs.append({
            "role_a": p["role_a"],
            "role_b": p["role_b"],
            "queries": selected,
        })

    final = {
        "protocol": "v139_stratified_same_query_causal_v3",
        "pairs": final_pairs,
        "train_per_pair": 40,
        "val_per_pair": 6,
        "test_locked_per_pair": 6,
        "expected_worker_calls": 2 * sum(
            len(p["queries"]) for p in final_pairs
        ),
    }

    (OUT / "manifest.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False)
    )

    print("FINAL STRATIFIED MANIFEST WRITTEN.")
    print("NEXT WORKER CALLS:", final["expected_worker_calls"])
