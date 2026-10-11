from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOTDIR / "analysis"))

import v4_meta_judge_v3 as meta
import v4_userlike_audit_and_pairs as userjudge


ROOT = Path("artifacts/v5_heldout_abc")
ARMS = {
    "epoch2": ROOT / "epoch2/candidates_dev_round_01.jsonl",
    "sft": ROOT / "sft/candidates_dev_round_01.jsonl",
    "combined": ROOT / "combined/candidates_dev_round_01.jsonl",
}

OUT = ROOT / "audit"
OUT.mkdir(parents=True, exist_ok=True)

JUDGMENTS = OUT / "query_judgments.jsonl"
SUMMARY = OUT / "summary.json"


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [
        json.loads(x)
        for x in p.open(encoding="utf-8")
        if x.strip()
    ]


def append_jsonl(p, x):
    with Path(p).open("a", encoding="utf-8") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


def key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )


async def judge_query(client, sem, model, q, exemplar):
    # Frozen META V3
    mrow = {
        "round": int(exemplar["round"]),
        "session_id": str(exemplar["session_id"]),
        "candidate_index": int(exemplar["candidate_index"]),
        "query": q,
    }

    mj = await meta.judge_one(
        client, sem, model, mrow
    )

    # Frozen USER_LIKE/NATURALNESS rubric
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": userjudge.SYSTEM,
                },
                {
                    "role": "user",
                    "content":
                        "Classify this message:\n\n" + q,
                },
            ],
            response_format=userjudge.SCHEMA,
        )

    uj = json.loads(resp.choices[0].message.content)

    return {
        "query": q,
        "meta_label": mj["label"],
        "meta_confidence": mj["confidence"],
        "meta_reason": mj.get("reason"),
        "user_like": uj["user_like"],
        "naturalness": uj["naturalness"],
        "direction": uj["direction"],
        "user_confidence": uj["confidence"],
        "user_reason": uj["reason"],
    }


def is_clean(j):
    return (
        j["meta_label"] == "NON_META"
        and j["user_like"] == "USER_LIKE"
        and j["direction"] == "USER_TO_SYSTEM"
        and j["naturalness"] == "NATURAL"
    )


def is_bad(j):
    return (
        j["meta_label"] == "META"
        or j["user_like"] == "ASSISTANT_LIKE"
        or j["direction"] == "SYSTEM_TO_USER"
    )


async def main_async(args):
    rows = {
        name: read_jsonl(path)
        for name, path in ARMS.items()
    }

    for name, xs in rows.items():
        assert len(xs) == 64, (
            f"{name}: expected 64 rows, got {len(xs)}"
        )

    maps = {
        name: {key(x): x for x in xs}
        for name, xs in rows.items()
    }

    keys = sorted(maps["epoch2"])

    assert set(keys) == set(maps["sft"])
    assert set(keys) == set(maps["combined"])
    assert len(keys) == 64

    # Controlled comparison invariants.
    prompt_match = 0
    seed_match = 0

    for k in keys:
        a = maps["epoch2"][k]
        b = maps["sft"][k]
        c = maps["combined"][k]

        assert a["prompt"] == b["prompt"] == c["prompt"]
        prompt_match += 1

        assert (
            int(a["seed"])
            == int(b["seed"])
            == int(c["seed"])
        )
        seed_match += 1

    print("======================================")
    print(" V5 HELD-OUT A/B/C PREFLIGHT")
    print("======================================")
    print("contexts             :", len(keys))
    print("identical prompts    :", f"{prompt_match}/64")
    print("identical seeds      :", f"{seed_match}/64")

    for name in ARMS:
        nvalid = sum(
            bool(x.get("valid"))
            for x in rows[name]
        )
        print(
            f"{name:10s} valid      : "
            f"{nvalid}/64 = {nvalid/64:.3f}"
        )

    # Judge only unique VALID queries across all arms.
    exemplar = {}

    for name in ARMS:
        for x in rows[name]:
            if not x.get("valid"):
                continue
            q = str(x.get("query") or "").strip()
            if q:
                exemplar.setdefault(q, x)

    print("unique valid queries :", len(exemplar))

    existing = {
        x["query"]: x
        for x in read_jsonl(JUDGMENTS)
        if "error" not in x
    }

    pending = [
        (q, x)
        for q, x in exemplar.items()
        if q not in existing
    ]

    print("already judged       :", len(existing))
    print("pending unique       :", len(pending))

    if pending:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)

        B = 16

        for start in range(0, len(pending), B):
            batch = pending[start:start+B]

            got = await asyncio.gather(
                *[
                    judge_query(
                        client,
                        sem,
                        args.model,
                        q,
                        x,
                    )
                    for q, x in batch
                ],
                return_exceptions=True,
            )

            failures = []

            for (q, _), res in zip(batch, got):
                if isinstance(res, Exception):
                    failures.append(
                        f"{q[:80]} :: "
                        f"{type(res).__name__}: {res}"
                    )
                else:
                    append_jsonl(JUDGMENTS, res)
                    existing[q] = res

            print(
                f"[judge] "
                f"{min(start+B,len(pending))}/"
                f"{len(pending)}",
                flush=True,
            )

            if failures:
                for z in failures[:5]:
                    print("FAIL:", z)
                raise RuntimeError(
                    f"{len(failures)} judge failures"
                )

    missing = [
        q for q in exemplar
        if q not in existing
    ]
    assert not missing, missing[:5]

    def arm_metrics(name):
        xs = rows[name]
        valid = [x for x in xs if x.get("valid")]

        js = [
            existing[str(x["query"]).strip()]
            for x in valid
        ]

        meta_n = sum(
            j["meta_label"] == "META"
            for j in js
        )
        assistant_n = sum(
            (
                j["user_like"] == "ASSISTANT_LIKE"
                or j["direction"] == "SYSTEM_TO_USER"
            )
            for j in js
        )
        clean_n = sum(is_clean(j) for j in js)
        bad_n = sum(is_bad(j) for j in js)

        return {
            "n": len(xs),
            "valid": len(valid),
            "valid_rate": len(valid) / len(xs),
            "meta": meta_n,
            "meta_rate_valid": meta_n / max(len(valid),1),
            "assistant_like": assistant_n,
            "assistant_like_rate_valid":
                assistant_n / max(len(valid),1),
            "clean": clean_n,
            "clean_rate_valid":
                clean_n / max(len(valid),1),
            "bad": bad_n,
            "bad_rate_valid":
                bad_n / max(len(valid),1),
        }

    metrics = {
        name: arm_metrics(name)
        for name in ARMS
    }

    def compare(base_name, arm_name):
        base = maps[base_name]
        arm = maps[arm_name]

        changed = 0
        comparable = 0

        clean_gain = 0
        clean_loss = 0
        bad_gain = 0
        bad_loss = 0

        changed_examples = []

        for k in keys:
            a = base[k]
            b = arm[k]

            qa = str(a.get("query") or "").strip()
            qb = str(b.get("query") or "").strip()

            if qa != qb:
                changed += 1

            if not (a.get("valid") and b.get("valid")):
                continue

            comparable += 1

            ja = existing[qa]
            jb = existing[qb]

            ca = is_clean(ja)
            cb = is_clean(jb)

            ba = is_bad(ja)
            bb = is_bad(jb)

            if not ca and cb:
                clean_gain += 1
            if ca and not cb:
                clean_loss += 1

            if ba and not bb:
                bad_gain += 1
            if not ba and bb:
                bad_loss += 1

            if qa != qb:
                changed_examples.append({
                    "key": k,
                    "base_query": qa,
                    "arm_query": qb,
                    "base_clean": ca,
                    "arm_clean": cb,
                    "base_bad": ba,
                    "arm_bad": bb,
                })

        return {
            "changed": changed,
            "changed_rate": changed / len(keys),
            "comparable_valid": comparable,
            "clean_gain": clean_gain,
            "clean_loss": clean_loss,
            "net_clean": clean_gain - clean_loss,
            "bad_removed": bad_gain,
            "bad_added": bad_loss,
            "net_bad_removed": bad_gain - bad_loss,
            "changed_examples": changed_examples,
        }

    comparisons = {
        "epoch2_to_sft":
            compare("epoch2", "sft"),
        "epoch2_to_combined":
            compare("epoch2", "combined"),
        "sft_to_combined":
            compare("sft", "combined"),
    }

    summary = {
        "contexts": len(keys),
        "prompt_identity": prompt_match,
        "seed_identity": seed_match,
        "unique_valid_queries": len(exemplar),
        "arms": metrics,
        "comparisons": {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk != "changed_examples"
            }
            for k, v in comparisons.items()
        },
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("======================================")
    print(" HELD-OUT BEHAVIOR RESULTS")
    print("======================================")

    for name in ["epoch2", "sft", "combined"]:
        m = metrics[name]

        print(
            f"{name:10s} | "
            f"valid={m['valid_rate']:.3f} | "
            f"META={m['meta_rate_valid']:.3f} | "
            f"assistant={m['assistant_like_rate_valid']:.3f} | "
            f"CLEAN={m['clean_rate_valid']:.3f} | "
            f"BAD={m['bad_rate_valid']:.3f}"
        )

    print()
    print("=== PAIRED TRANSITIONS ===")

    for name, x in comparisons.items():
        print(
            f"{name:22s} | "
            f"changed={x['changed']}/64 "
            f"({x['changed_rate']:.3f}) | "
            f"clean +{x['clean_gain']}/-{x['clean_loss']} "
            f"net={x['net_clean']:+d} | "
            f"bad removed={x['bad_removed']} "
            f"added={x['bad_added']} "
            f"net={x['net_bad_removed']:+d}"
        )

    print()
    print("=== COMBINED VS SFT: CHANGED EXAMPLES ===")

    for x in comparisons["sft_to_combined"]["changed_examples"][:30]:
        print()
        print(
            "KEY:",
            x["key"],
            "| clean",
            x["base_clean"],
            "->",
            x["arm_clean"],
            "| bad",
            x["base_bad"],
            "->",
            x["arm_bad"],
        )
        print("SFT :", x["base_query"])
        print("DPO :", x["arm_query"])

    print()
    print("wrote:", SUMMARY)
    print("wrote:", JUDGMENTS)


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
