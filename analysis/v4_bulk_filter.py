from __future__ import annotations

import argparse, asyncio, json, sys
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI

ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOTDIR / "analysis"))

import v4_meta_judge_v3 as meta
import v4_userlike_audit_and_pairs as ua


ROOT = Path("artifacts/v4_bulk_mining")
SRC = ROOT / "samples.jsonl"

META = ROOT / "meta.jsonl"
USER = ROOT / "userlike.jsonl"

DIRECTION = ROOT / "direction_negatives.jsonl"
CLEAN = ROOT / "clean_alternatives.jsonl"
META_CAND = ROOT / "meta_candidates.jsonl"
SUMMARY = ROOT / "filter_summary.json"


def read(p):
    if not Path(p).exists():
        return []
    return [
        json.loads(x)
        for x in Path(p).open(encoding="utf-8")
        if x.strip()
    ]


def append(p, x):
    with Path(p).open("a", encoding="utf-8") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


def write(p, xs):
    with Path(p).open("w", encoding="utf-8") as f:
        for x in xs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def aid(x):
    return f"{int(x['gold_index'])}:{int(x['sample_index'])}"


async def meta_one(client, sem, model, x):
    inp = {
        "round": int(x["round"]),
        "session_id": str(x["session_id"]),
        "candidate_index": int(x["candidate_index"]),
        "query": str(x["query"]),
    }

    r = await meta.judge_one(client, sem, model, inp)

    return {
        "audit_id": aid(x),
        "label": r["label"],
        "confidence": r["confidence"],
        "reason": r.get("reason"),
    }


async def user_one(client, sem, model, x):
    async with sem:
        r = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": ua.SYSTEM},
                {
                    "role": "user",
                    "content": "Classify this message:\n\n" + x["query"],
                },
            ],
            response_format=ua.SCHEMA,
        )

        obj = json.loads(r.choices[0].message.content)

        return {
            "audit_id": aid(x),
            **obj,
        }


async def run_phase(name, candidates, existing_path, fn, model, concurrency):
    existing = {
        x["audit_id"]: x
        for x in read(existing_path)
        if "error" not in x
    }

    pending = [
        x for x in candidates
        if aid(x) not in existing
    ]

    print(
        f"{name}: existing={len(existing)} pending={len(pending)}",
        flush=True,
    )

    if not pending:
        return existing

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)

    B = 24

    for start in range(0, len(pending), B):
        batch = pending[start:start+B]

        got = await asyncio.gather(
            *[
                fn(client, sem, model, x)
                for x in batch
            ],
            return_exceptions=True,
        )

        failures = []

        for src, r in zip(batch, got):
            if isinstance(r, Exception):
                failures.append(
                    f"{aid(src)} {type(r).__name__}: {r}"
                )
            else:
                append(existing_path, r)
                existing[r["audit_id"]] = r

        print(
            f"[{name}] {min(start+B,len(pending))}/{len(pending)}",
            flush=True,
        )

        if failures:
            for z in failures[:10]:
                print("FAIL", z)
            raise RuntimeError(
                f"{name}: {len(failures)} failures"
            )

    return existing


async def main_async(args):
    raw = read(SRC)

    assert len(raw) == 768, f"expected 768, got {len(raw)}"

    candidates = [
        x for x in raw
        if x["valid"] and x["unique_within_prompt"]
    ]

    print("raw:", len(raw))
    print("valid unique:", len(candidates))

    m = await run_phase(
        "META",
        candidates,
        META,
        meta_one,
        args.model,
        args.concurrency,
    )

    u = await run_phase(
        "USERLIKE",
        candidates,
        USER,
        user_one,
        args.model,
        args.concurrency,
    )

    merged = []

    for x in candidates:
        k = aid(x)

        if k not in m or k not in u:
            raise RuntimeError(f"missing judgment {k}")

        z = {
            **x,
            "meta_label": m[k]["label"],
            "meta_confidence": m[k]["confidence"],
            "meta_reason": m[k].get("reason"),

            "user_like": u[k]["user_like"],
            "naturalness": u[k]["naturalness"],
            "direction": u[k]["direction"],
            "user_confidence": u[k]["confidence"],
            "user_reason": u[k]["reason"],

            **ua.lexical_flags(x["query"]),
        }

        merged.append(z)

    # Pure direction negatives:
    direction = [
        x for x in merged
        if x["meta_label"] == "NON_META"
        and x["meta_confidence"] == "HIGH"
        and x["user_like"] == "ASSISTANT_LIKE"
        and x["direction"] == "SYSTEM_TO_USER"
        and x["user_confidence"] == "HIGH"
        and x["naturalness"] == "NATURAL"
        and not x["internal_prompt_lexical_flag"]
    ]

    # Clean alternatives: useful later as extra positives / sanity.
    clean = [
        x for x in merged
        if x["meta_label"] == "NON_META"
        and x["meta_confidence"] == "HIGH"
        and x["user_like"] == "USER_LIKE"
        and x["direction"] == "USER_TO_SYSTEM"
        and x["user_confidence"] == "HIGH"
        and x["naturalness"] == "NATURAL"
        and not x["internal_prompt_lexical_flag"]
    ]

    # These still require secondary system-self adjudication.
    meta_cand = [
        x for x in merged
        if x["meta_label"] == "META"
        and x["meta_confidence"] == "HIGH"
        and x["user_like"] == "USER_LIKE"
        and x["direction"] == "USER_TO_SYSTEM"
        and x["user_confidence"] == "HIGH"
        and x["naturalness"] == "NATURAL"
        and not x["internal_prompt_lexical_flag"]
    ]

    write(DIRECTION, direction)
    write(CLEAN, clean)
    write(META_CAND, meta_cand)

    def cov(xs):
        return len(set(int(x["gold_index"]) for x in xs))

    summary = {
        "raw": len(raw),
        "valid_unique": len(candidates),
        "meta_counts": dict(
            Counter(x["meta_label"] for x in merged)
        ),
        "userlike_counts": dict(
            Counter(x["user_like"] for x in merged)
        ),
        "direction_negatives": len(direction),
        "direction_gold_coverage": cov(direction),
        "clean_alternatives": len(clean),
        "clean_gold_coverage": cov(clean),
        "meta_candidates": len(meta_cand),
        "meta_candidate_gold_coverage": cov(meta_cand),
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("================================")
    print(" BULK FILTER SUMMARY")
    print("================================")
    print("valid unique           :", len(candidates))
    print("META labels            :", summary["meta_counts"])
    print("USERLIKE labels        :", summary["userlike_counts"])
    print()
    print(
        "DIRECTION negatives    :",
        len(direction),
        f"| coverage {cov(direction)}/24",
    )
    print(
        "CLEAN alternatives     :",
        len(clean),
        f"| coverage {cov(clean)}/24",
    )
    print(
        "META candidates        :",
        len(meta_cand),
        f"| coverage {cov(meta_cand)}/24",
    )

    print()
    print("=== DIRECTION NEGATIVE COUNTS PER GOLD ===")

    cc = Counter(int(x["gold_index"]) for x in direction)

    for gi in sorted(cc):
        print(f"G{gi:02d}: {cc[gi]}")

    print()
    print("wrote:", DIRECTION)
    print("wrote:", CLEAN)
    print("wrote:", META_CAND)
    print("wrote:", SUMMARY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
    )
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
