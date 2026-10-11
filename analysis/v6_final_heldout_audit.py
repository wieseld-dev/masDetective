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


BASE = Path("artifacts/v6_final_heldout")
OUT = BASE / "audit"
OUT.mkdir(parents=True, exist_ok=True)

JFILE = OUT / "query_judgments.jsonl"
SUMMARY = OUT / "summary.json"

SEEDS = ["20261001", "20261002", "20261003"]
ARMS = ["epoch2", "sft", "dpo"]

DATA = {
    seed: {
        arm: (
            BASE
            / f"seed_{seed}"
            / arm
            / "candidates_dev_round_01.jsonl"
        )
        for arm in ARMS
    }
    for seed in SEEDS
}

OLD_JFILES = [
    Path("artifacts/v5_heldout_abc/audit/query_judgments.jsonl"),
    Path("artifacts/v5_heldout_multiseed/audit/query_judgments.jsonl"),
]


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


def write_jsonl(p, rows):
    with Path(p).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


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

    mj = await meta.judge_one(
        client,
        sem,
        model,
        mrow,
    )

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

    clean_n = sum(clean(j) for j in js)
    bad_n = sum(bad(j) for j in js)
    meta_n = sum(
        j["meta_label"] == "META"
        for j in js
    )
    assistant_n = sum(
        j["user_like"] == "ASSISTANT_LIKE"
        or j["direction"] == "SYSTEM_TO_USER"
        for j in js
    )

    nv = max(len(valid), 1)
    nr = max(len(rows), 1)

    return {
        "rows": len(rows),
        "valid": len(valid),
        "valid_rate": len(valid) / nr,

        "clean": clean_n,
        "clean_rate": clean_n / nv,
        "clean_overall_rate": clean_n / nr,

        "bad": bad_n,
        "bad_rate": bad_n / nv,

        "meta": meta_n,
        "meta_rate": meta_n / nv,

        "assistant": assistant_n,
        "assistant_rate": assistant_n / nv,
    }


def paired(a_rows, b_rows, judgments, a_name, b_name):
    A = {k(x): x for x in a_rows}
    B = {k(x): x for x in b_rows}

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

        # End-to-end CLEAN including validity.
        "usable_clean_gain": 0,
        "usable_clean_loss": 0,
    }

    examples = []

    for kk in sorted(A):
        a = A[kk]
        b = B[kk]

        assert a["prompt"] == b["prompt"]
        assert int(a["seed"]) == int(b["seed"])

        qa = str(a.get("query") or "").strip()
        qb = str(b.get("query") or "").strip()

        changed = qa != qb

        if changed:
            out["changed"] += 1

        ja = (
            judgments[qa]
            if a.get("valid")
            else None
        )
        jb = (
            judgments[qb]
            if b.get("valid")
            else None
        )

        usable_a = bool(
            a.get("valid")
            and ja is not None
            and clean(ja)
        )
        usable_b = bool(
            b.get("valid")
            and jb is not None
            and clean(jb)
        )

        if (not usable_a) and usable_b:
            out["usable_clean_gain"] += 1

        if usable_a and (not usable_b):
            out["usable_clean_loss"] += 1

        if a.get("valid") and b.get("valid"):
            out["both_valid"] += 1

            if changed:
                out["changed_both_valid"] += 1

            ca = clean(ja)
            cb = clean(jb)
            ba = bad(ja)
            bb = bad(jb)

            if (not ca) and cb:
                out["clean_gain"] += 1

            if ca and (not cb):
                out["clean_loss"] += 1

            if ba and (not bb):
                out["bad_removed"] += 1

            if (not ba) and bb:
                out["bad_added"] += 1
        else:
            ca = clean(ja) if ja is not None else None
            cb = clean(jb) if jb is not None else None
            ba = bad(ja) if ja is not None else None
            bb = bad(jb) if jb is not None else None

        if changed:
            examples.append({
                "key": kk,

                f"{a_name}_query": qa,
                f"{b_name}_query": qb,

                f"{a_name}_valid": bool(a.get("valid")),
                f"{b_name}_valid": bool(b.get("valid")),

                f"{a_name}_clean": ca,
                f"{b_name}_clean": cb,

                f"{a_name}_bad": ba,
                f"{b_name}_bad": bb,

                f"{a_name}_usable_clean": usable_a,
                f"{b_name}_usable_clean": usable_b,
            })

    out["net_clean"] = (
        out["clean_gain"]
        - out["clean_loss"]
    )

    out["net_bad_removed"] = (
        out["bad_removed"]
        - out["bad_added"]
    )

    out["net_usable_clean"] = (
        out["usable_clean_gain"]
        - out["usable_clean_loss"]
    )

    out["clean_sign_p"] = exact_sign_p(
        out["clean_gain"],
        out["clean_loss"],
    )

    out["bad_sign_p"] = exact_sign_p(
        out["bad_removed"],
        out["bad_added"],
    )

    out["usable_clean_sign_p"] = exact_sign_p(
        out["usable_clean_gain"],
        out["usable_clean_loss"],
    )

    out["examples"] = examples
    return out


def pool_pair(rows, judgments, a_name, b_name):
    count_fields = [
        "changed",
        "both_valid",
        "changed_both_valid",
        "clean_gain",
        "clean_loss",
        "bad_removed",
        "bad_added",
        "usable_clean_gain",
        "usable_clean_loss",
    ]

    out = {x: 0 for x in count_fields}
    examples = []

    for seed in SEEDS:
        pp = paired(
            rows[seed][a_name],
            rows[seed][b_name],
            judgments,
            a_name,
            b_name,
        )

        for name in count_fields:
            out[name] += pp[name]

        for x in pp["examples"]:
            z = dict(x)
            z["master_seed"] = seed
            examples.append(z)

    out["net_clean"] = (
        out["clean_gain"]
        - out["clean_loss"]
    )

    out["net_bad_removed"] = (
        out["bad_removed"]
        - out["bad_added"]
    )

    out["net_usable_clean"] = (
        out["usable_clean_gain"]
        - out["usable_clean_loss"]
    )

    out["clean_sign_p"] = exact_sign_p(
        out["clean_gain"],
        out["clean_loss"],
    )

    out["bad_sign_p"] = exact_sign_p(
        out["bad_removed"],
        out["bad_added"],
    )

    out["usable_clean_sign_p"] = exact_sign_p(
        out["usable_clean_gain"],
        out["usable_clean_loss"],
    )

    out["examples"] = examples
    return out


async def main_async(args):
    rows = {}

    # ------------------------------------------------------
    # Strict dataset / pairing integrity
    # ------------------------------------------------------
    for seed in SEEDS:
        rows[seed] = {}

        for arm in ARMS:
            path = DATA[seed][arm]
            xs = read_jsonl(path)

            assert len(xs) == 64, (
                seed,
                arm,
                len(xs),
            )

            by = {k(x): x for x in xs}
            assert len(by) == 64

            assert {
                int(x["candidate_index"])
                for x in xs
            } == set(range(8))

            assert {
                int(x["round"])
                for x in xs
            } == {1}

            assert {
                int(x["local_attempts"])
                for x in xs
            } == {1}

            rows[seed][arm] = xs

        maps = {
            arm: {
                k(x): x
                for x in rows[seed][arm]
            }
            for arm in ARMS
        }

        keys = set(maps["epoch2"])

        assert set(maps["sft"]) == keys
        assert set(maps["dpo"]) == keys

        for kk in keys:
            e = maps["epoch2"][kk]
            s = maps["sft"][kk]
            d = maps["dpo"][kk]

            assert (
                e["prompt"]
                == s["prompt"]
                == d["prompt"]
            )

            assert (
                int(e["seed"])
                == int(s["seed"])
                == int(d["seed"])
            )

    print(
        "PREFLIGHT: "
        "3 seeds × 64 contexts × 3 arms = 576 generations"
    )
    print("prompt/seed identity across all 3 arms: PASS")

    # ------------------------------------------------------
    # Build unique valid-query inventory
    # ------------------------------------------------------
    exemplar = {}

    for seed in SEEDS:
        for arm in ARMS:
            for x in rows[seed][arm]:
                if not x.get("valid"):
                    continue

                q = str(x.get("query") or "").strip()

                if q:
                    exemplar.setdefault(q, x)

    # ------------------------------------------------------
    # Reuse exact old judgments whenever possible
    # ------------------------------------------------------
    judgments = {}

    for p in OLD_JFILES + [JFILE]:
        for x in read_jsonl(p):
            if "error" not in x:
                judgments[x["query"]] = x

    pending = [
        (q, x)
        for q, x in exemplar.items()
        if q not in judgments
    ]

    print("unique valid queries :", len(exemplar))
    print(
        "reused judgments     :",
        len(exemplar) - len(pending),
    )
    print("new queries to judge :", len(pending))

    if args.preflight_only:
        print("PREFLIGHT-ONLY: no API calls made")
        return

    # ------------------------------------------------------
    # Frozen V5 judges
    # ------------------------------------------------------
    if pending:
        client = AsyncOpenAI()
        sem = asyncio.Semaphore(args.concurrency)

        BATCH = 16

        for start in range(0, len(pending), BATCH):
            batch = pending[start:start + BATCH]

            res = await asyncio.gather(
                *[
                    judge_one(
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
                f"{min(start+BATCH, len(pending))}/"
                f"{len(pending)}",
                flush=True,
            )

            if failures:
                print("\n".join(failures[:5]))
                raise RuntimeError(
                    f"{len(failures)} judge failures"
                )

    assert all(
        q in judgments
        for q in exemplar
    )

    # ------------------------------------------------------
    # Per-seed analysis
    # ------------------------------------------------------
    per_seed = {}

    print()
    print("=" * 86)
    print(" PER-SEED RESULTS")
    print("=" * 86)

    for seed in SEEDS:
        per_seed[seed] = {
            "arms": {},
            "pairs": {},
        }

        print(f"\nSEED {seed}")

        for arm in ARMS:
            st = arm_stats(
                rows[seed][arm],
                judgments,
            )

            per_seed[seed]["arms"][arm] = st

            print(
                f"{arm.upper():6s} "
                f"valid={st['valid_rate']:.3f} "
                f"clean|valid={st['clean_rate']:.3f} "
                f"clean/all={st['clean_overall_rate']:.3f} "
                f"bad={st['bad_rate']:.3f} "
                f"meta={st['meta_rate']:.3f} "
                f"assistant={st['assistant_rate']:.3f}"
            )

        for a, b in [
            ("epoch2", "sft"),
            ("epoch2", "dpo"),
            ("sft", "dpo"),
        ]:
            pp = paired(
                rows[seed][a],
                rows[seed][b],
                judgments,
                a,
                b,
            )

            per_seed[seed]["pairs"][f"{a}_vs_{b}"] = {
                k_: v
                for k_, v in pp.items()
                if k_ != "examples"
            }

        pp = paired(
            rows[seed]["sft"],
            rows[seed]["dpo"],
            judgments,
            "sft",
            "dpo",
        )

        print(
            "SFT->DPO "
            f"changed={pp['changed']}/64 | "
            f"CLEAN +{pp['clean_gain']}/-{pp['clean_loss']} "
            f"net={pp['net_clean']:+d} | "
            f"BAD removed={pp['bad_removed']} "
            f"added={pp['bad_added']} "
            f"net={pp['net_bad_removed']:+d} | "
            f"USABLE CLEAN +{pp['usable_clean_gain']}"
            f"/-{pp['usable_clean_loss']} "
            f"net={pp['net_usable_clean']:+d}"
        )

    # ------------------------------------------------------
    # Pooled arm metrics
    # ------------------------------------------------------
    pooled_rows = {
        arm: [
            x
            for seed in SEEDS
            for x in rows[seed][arm]
        ]
        for arm in ARMS
    }

    pooled_arms = {
        arm: arm_stats(
            pooled_rows[arm],
            judgments,
        )
        for arm in ARMS
    }

    pooled_pairs = {}

    for a, b in [
        ("epoch2", "sft"),
        ("epoch2", "dpo"),
        ("sft", "dpo"),
    ]:
        pp = pool_pair(
            rows,
            judgments,
            a,
            b,
        )

        pooled_pairs[f"{a}_vs_{b}"] = {
            k_: v
            for k_, v in pp.items()
            if k_ != "examples"
        }

        write_jsonl(
            OUT / f"changed_{a}_vs_{b}.jsonl",
            pp["examples"],
        )

    print()
    print("=" * 86)
    print(" POOLED 3-SEED RESULT — 192 CONTEXTS / ARM")
    print("=" * 86)

    for arm in ARMS:
        st = pooled_arms[arm]

        print(
            f"{arm.upper():6s} "
            f"valid={st['valid_rate']:.3f} "
            f"clean|valid={st['clean_rate']:.3f} "
            f"clean/all={st['clean_overall_rate']:.3f} "
            f"bad={st['bad_rate']:.3f} "
            f"meta={st['meta_rate']:.3f} "
            f"assistant={st['assistant_rate']:.3f}"
        )

    print()
    print("===== PRIMARY: SFT -> DPO =====")

    pp = pool_pair(
        rows,
        judgments,
        "sft",
        "dpo",
    )

    print(
        f"changed             : "
        f"{pp['changed']}/192"
    )

    print(
        f"changed + both valid: "
        f"{pp['changed_both_valid']}"
    )

    print(
        f"CLEAN transitions   : "
        f"+{pp['clean_gain']} "
        f"/-{pp['clean_loss']} "
        f"net={pp['net_clean']:+d} "
        f"exact-p={pp['clean_sign_p']:.6f}"
    )

    print(
        f"BAD transitions     : "
        f"removed={pp['bad_removed']} "
        f"added={pp['bad_added']} "
        f"net={pp['net_bad_removed']:+d} "
        f"exact-p={pp['bad_sign_p']:.6f}"
    )

    print(
        f"USABLE CLEAN        : "
        f"+{pp['usable_clean_gain']} "
        f"/-{pp['usable_clean_loss']} "
        f"net={pp['net_usable_clean']:+d} "
        f"exact-p={pp['usable_clean_sign_p']:.6f}"
    )

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------
    summary = {
        "model": args.model,
        "seeds": SEEDS,
        "contexts_per_seed": 64,
        "contexts_per_arm": 192,
        "generations_total": 576,
        "unique_valid_queries": len(exemplar),
        "pooled_arms": pooled_arms,
        "pooled_pairs": pooled_pairs,
        "per_seed": per_seed,
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("wrote:", SUMMARY)
    print(
        "changed examples:",
        OUT / "changed_sft_vs_dpo.jsonl",
    )


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

    ap.add_argument(
        "--preflight-only",
        action="store_true",
    )

    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
