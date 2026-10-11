from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

from openai import AsyncOpenAI

ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOTDIR / "analysis"))

import v4_meta_judge_v3 as meta
import v4_userlike_audit_and_pairs as userjudge


NEWROOT = Path("artifacts/v5_heldout_multiseed")
OLDROOT = Path("artifacts/v5_heldout_abc")

OUT = NEWROOT / "audit"
OUT.mkdir(parents=True, exist_ok=True)

JFILE = OUT / "query_judgments.jsonl"
SUMMARY = OUT / "summary.json"

DATA = {
    "20260924": {
        "sft": OLDROOT / "sft/candidates_dev_round_01.jsonl",
        "combined": OLDROOT / "combined/candidates_dev_round_01.jsonl",
    },
    "20260925": {
        "sft": NEWROOT / "seed_20260925/sft/candidates_dev_round_01.jsonl",
        "combined": NEWROOT / "seed_20260925/combined/candidates_dev_round_01.jsonl",
    },
    "20260926": {
        "sft": NEWROOT / "seed_20260926/sft/candidates_dev_round_01.jsonl",
        "combined": NEWROOT / "seed_20260926/combined/candidates_dev_round_01.jsonl",
    },
}


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [
        json.loads(line)
        for line in p.open(encoding="utf-8")
        if line.strip()
    ]


def append_jsonl(p, x):
    with Path(p).open("a", encoding="utf-8") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
        f.flush()


def k(row):
    return (
        int(row["round"]),
        str(row["session_id"]),
        int(row["candidate_index"]),
    )


def clean(j):
    return (
        j["meta_label"] == "NON_META"
        and j["user_like"] == "USER_LIKE"
        and j["direction"] == "USER_TO_SYSTEM"
        and j["naturalness"] == "NATURAL"
    )


def bad(j):
    return (
        j["meta_label"] == "META"
        or j["user_like"] == "ASSISTANT_LIKE"
        or j["direction"] == "SYSTEM_TO_USER"
    )


def exact_sign_p(a, b):
    # Two-sided exact binomial test for discordant pairs.
    n = a + b
    if n == 0:
        return 1.0

    m = min(a, b)
    prob = sum(
        math.comb(n, i) * (0.5 ** n)
        for i in range(m + 1)
    )
    return min(1.0, 2.0 * prob)


async def judge_one(client, sem, model, q, exemplar):
    mrow = {
        "round": int(exemplar["round"]),
        "session_id": str(exemplar["session_id"]),
        "candidate_index": int(exemplar["candidate_index"]),
        "query": q,
    }

    mj = await meta.judge_one(client, sem, model, mrow)

    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": userjudge.SYSTEM},
                {
                    "role": "user",
                    "content": "Classify this message:\n\n" + q,
                },
            ],
            response_format=userjudge.SCHEMA,
        )

    uj = json.loads(resp.choices[0].message.content)

    return {
        "query": q,
        "meta_label": mj["label"],
        "meta_confidence": mj["confidence"],
        "user_like": uj["user_like"],
        "naturalness": uj["naturalness"],
        "direction": uj["direction"],
        "user_confidence": uj["confidence"],
    }


def arm_stats(rows, judgments):
    valid = [x for x in rows if x.get("valid")]

    js = [
        judgments[str(x["query"]).strip()]
        for x in valid
    ]

    N = max(len(valid), 1)

    return {
        "rows": len(rows),
        "valid": len(valid),
        "valid_rate": len(valid) / len(rows),
        "clean": sum(clean(j) for j in js),
        "clean_rate": sum(clean(j) for j in js) / N,
        "bad": sum(bad(j) for j in js),
        "bad_rate": sum(bad(j) for j in js) / N,
        "meta": sum(j["meta_label"] == "META" for j in js),
        "meta_rate":
            sum(j["meta_label"] == "META" for j in js) / N,
        "assistant": sum(
            j["user_like"] == "ASSISTANT_LIKE"
            or j["direction"] == "SYSTEM_TO_USER"
            for j in js
        ),
        "assistant_rate": sum(
            j["user_like"] == "ASSISTANT_LIKE"
            or j["direction"] == "SYSTEM_TO_USER"
            for j in js
        ) / N,
    }


def paired(sft, combo, judgments):
    A = {k(x): x for x in sft}
    B = {k(x): x for x in combo}

    assert set(A) == set(B)
    assert len(A) == 64

    out = {
        "changed": 0,
        "both_valid": 0,
        "changed_both_valid": 0,
        "clean_gain": 0,
        "clean_loss": 0,
        "bad_removed": 0,
        "bad_added": 0,
    }

    examples = []

    for kk in sorted(A):
        a, b = A[kk], B[kk]

        assert a["prompt"] == b["prompt"]
        assert int(a["seed"]) == int(b["seed"])

        qa = str(a.get("query") or "").strip()
        qb = str(b.get("query") or "").strip()

        changed = qa != qb
        if changed:
            out["changed"] += 1

        if not (a.get("valid") and b.get("valid")):
            continue

        out["both_valid"] += 1

        if changed:
            out["changed_both_valid"] += 1

        ja = judgments[qa]
        jb = judgments[qb]

        ca, cb = clean(ja), clean(jb)
        ba, bb = bad(ja), bad(jb)

        if (not ca) and cb:
            out["clean_gain"] += 1
        if ca and (not cb):
            out["clean_loss"] += 1

        if ba and (not bb):
            out["bad_removed"] += 1
        if (not ba) and bb:
            out["bad_added"] += 1

        if changed:
            examples.append({
                "key": kk,
                "sft": qa,
                "combined": qb,
                "sft_clean": ca,
                "combined_clean": cb,
                "sft_bad": ba,
                "combined_bad": bb,
            })

    out["net_clean"] = (
        out["clean_gain"] - out["clean_loss"]
    )

    out["net_bad_removed"] = (
        out["bad_removed"] - out["bad_added"]
    )

    out["clean_sign_p"] = exact_sign_p(
        out["clean_gain"],
        out["clean_loss"],
    )

    out["bad_sign_p"] = exact_sign_p(
        out["bad_removed"],
        out["bad_added"],
    )

    out["examples"] = examples
    return out


async def main_async(args):
    rows = {}

    for seed, arms in DATA.items():
        rows[seed] = {}

        for arm, path in arms.items():
            xs = read_jsonl(path)
            assert len(xs) == 64, (
                f"{seed}/{arm}: expected 64, got {len(xs)}"
            )
            rows[seed][arm] = xs

        # Exact comparison integrity within each seed.
        A = {k(x): x for x in rows[seed]["sft"]}
        B = {k(x): x for x in rows[seed]["combined"]}

        assert set(A) == set(B)

        for kk in A:
            assert A[kk]["prompt"] == B[kk]["prompt"]
            assert int(A[kk]["seed"]) == int(B[kk]["seed"])

    print("PREFLIGHT: 3 seeds × 64 paired contexts = 192")
    print("prompt/seed identity: PASS")

    exemplar = {}

    for seed in rows:
        for arm in ("sft", "combined"):
            for x in rows[seed][arm]:
                if not x.get("valid"):
                    continue
                q = str(x.get("query") or "").strip()
                if q:
                    exemplar.setdefault(q, x)

    judgments = {}

    # Reuse previous paid judgments.
    old_j = OLDROOT / "audit/query_judgments.jsonl"

    for p in (old_j, JFILE):
        for x in read_jsonl(p):
            if "error" not in x:
                judgments[x["query"]] = x

    pending = [
        (q, x)
        for q, x in exemplar.items()
        if q not in judgments
    ]

    print("unique valid queries :", len(exemplar))
    print("reused judgments     :", len(exemplar) - len(pending))
    print("new queries to judge :", len(pending))

    if pending:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)

        BATCH = 16

        for start in range(0, len(pending), BATCH):
            batch = pending[start:start+BATCH]

            res = await asyncio.gather(
                *[
                    judge_one(
                        client, sem, args.model, q, x
                    )
                    for q, x in batch
                ],
                return_exceptions=True,
            )

            failures = []

            for (q, _), z in zip(batch, res):
                if isinstance(z, Exception):
                    failures.append(
                        f"{type(z).__name__}: {z}"
                    )
                else:
                    append_jsonl(JFILE, z)
                    judgments[q] = z

            print(
                f"[judge] "
                f"{min(start+BATCH,len(pending))}/"
                f"{len(pending)}",
                flush=True,
            )

            if failures:
                print("\n".join(failures[:5]))
                raise RuntimeError(
                    f"{len(failures)} judge failures"
                )

    assert all(q in judgments for q in exemplar)

    per_seed = {}
    pooled_sft = []
    pooled_combined = []

    print()
    print("=" * 78)
    print(" PER-SEED RESULTS")
    print("=" * 78)

    for seed in sorted(rows):
        sft = rows[seed]["sft"]
        com = rows[seed]["combined"]

        sm = arm_stats(sft, judgments)
        cm = arm_stats(com, judgments)
        pp = paired(sft, com, judgments)

        per_seed[seed] = {
            "sft": sm,
            "combined": cm,
            "paired": {
                k: v
                for k, v in pp.items()
                if k != "examples"
            },
        }

        pooled_sft.extend(sft)
        pooled_combined.extend(com)

        print(f"\nSEED {seed}")
        print(
            f"SFT      valid={sm['valid_rate']:.3f} "
            f"clean={sm['clean_rate']:.3f} "
            f"bad={sm['bad_rate']:.3f} "
            f"meta={sm['meta_rate']:.3f} "
            f"assistant={sm['assistant_rate']:.3f}"
        )
        print(
            f"COMBINED valid={cm['valid_rate']:.3f} "
            f"clean={cm['clean_rate']:.3f} "
            f"bad={cm['bad_rate']:.3f} "
            f"meta={cm['meta_rate']:.3f} "
            f"assistant={cm['assistant_rate']:.3f}"
        )
        print(
            f"PAIRED changed={pp['changed']}/64 "
            f"clean +{pp['clean_gain']}/-{pp['clean_loss']} "
            f"net={pp['net_clean']:+d} | "
            f"bad removed={pp['bad_removed']} "
            f"added={pp['bad_added']} "
            f"net={pp['net_bad_removed']:+d}"
        )

    # Pooled arm metrics.
    ps = arm_stats(pooled_sft, judgments)
    pc = arm_stats(pooled_combined, judgments)

    # Pooled paired transitions must be accumulated by seed because
    # session IDs repeat between different master seeds.
    pooled_pair = {
        "changed": 0,
        "both_valid": 0,
        "changed_both_valid": 0,
        "clean_gain": 0,
        "clean_loss": 0,
        "bad_removed": 0,
        "bad_added": 0,
    }

    all_examples = []

    for seed in sorted(rows):
        pp = paired(
            rows[seed]["sft"],
            rows[seed]["combined"],
            judgments,
        )

        for name in pooled_pair:
            pooled_pair[name] += pp[name]

        for x in pp["examples"]:
            z = dict(x)
            z["master_seed"] = seed
            all_examples.append(z)

    pooled_pair["net_clean"] = (
        pooled_pair["clean_gain"]
        - pooled_pair["clean_loss"]
    )

    pooled_pair["net_bad_removed"] = (
        pooled_pair["bad_removed"]
        - pooled_pair["bad_added"]
    )

    pooled_pair["clean_sign_p"] = exact_sign_p(
        pooled_pair["clean_gain"],
        pooled_pair["clean_loss"],
    )

    pooled_pair["bad_sign_p"] = exact_sign_p(
        pooled_pair["bad_removed"],
        pooled_pair["bad_added"],
    )

    print()
    print("=" * 78)
    print(" POOLED 3-SEED RESULT — 192 CONTEXTS")
    print("=" * 78)

    print(
        f"SFT      valid={ps['valid_rate']:.3f} "
        f"clean={ps['clean_rate']:.3f} "
        f"bad={ps['bad_rate']:.3f} "
        f"meta={ps['meta_rate']:.3f} "
        f"assistant={ps['assistant_rate']:.3f}"
    )

    print(
        f"COMBINED valid={pc['valid_rate']:.3f} "
        f"clean={pc['clean_rate']:.3f} "
        f"bad={pc['bad_rate']:.3f} "
        f"meta={pc['meta_rate']:.3f} "
        f"assistant={pc['assistant_rate']:.3f}"
    )

    print()
    print(
        f"changed           : "
        f"{pooled_pair['changed']}/192"
    )
    print(
        f"changed+both valid: "
        f"{pooled_pair['changed_both_valid']}"
    )

    print(
        f"CLEAN transitions : "
        f"+{pooled_pair['clean_gain']} "
        f"/-{pooled_pair['clean_loss']} "
        f"net={pooled_pair['net_clean']:+d} "
        f"exact-p={pooled_pair['clean_sign_p']:.4f}"
    )

    print(
        f"BAD transitions   : "
        f"removed={pooled_pair['bad_removed']} "
        f"added={pooled_pair['bad_added']} "
        f"net={pooled_pair['net_bad_removed']:+d} "
        f"exact-p={pooled_pair['bad_sign_p']:.4f}"
    )

    print()
    print("=== ALL CHANGED SFT -> COMBINED ===")

    for x in all_examples:
        print()
        print(
            f"seed={x['master_seed']} "
            f"key={x['key']} "
            f"clean {x['sft_clean']}->{x['combined_clean']} "
            f"bad {x['sft_bad']}->{x['combined_bad']}"
        )
        print("SFT:", x["sft"])
        print("DPO:", x["combined"])

    summary = {
        "contexts_total": 192,
        "per_seed": per_seed,
        "pooled": {
            "sft": ps,
            "combined": pc,
            "paired": pooled_pair,
        },
        "changed_examples": all_examples,
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
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
