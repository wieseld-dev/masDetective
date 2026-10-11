from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import v4_meta_judge_v3 as meta
import v4_userlike_audit_and_pairs as userjudge


MANIFEST = Path(
    "artifacts/soft_benign_critic_v2/label_manifest.jsonl"
)

OUT = Path(
    "artifacts/soft_benign_critic_v2"
)

LABELS = OUT / "labels.jsonl"


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
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )
        f.flush()


def norm(q):
    return re.sub(
        r"\s+",
        " ",
        str(q or "").strip(),
    )


def reusable_judgment_files():
    # Reuse every previous compatible judgment file.
    return sorted(
        Path("artifacts").rglob(
            "query_judgments.jsonl"
        )
    )


def load_existing():
    existing = {}

    files = reusable_judgment_files()

    # First load old paid audits.
    for p in files:
        for x in read_jsonl(p):
            q = norm(x.get("query"))

            if not q:
                continue

            needed = {
                "meta_label",
                "user_like",
                "naturalness",
                "direction",
            }

            if (
                needed <= set(x)
                and "error" not in x
            ):
                existing[q] = {
                    "query": q,
                    "meta_label":
                        x["meta_label"],
                    "meta_confidence":
                        x.get(
                            "meta_confidence"
                        ),
                    "user_like":
                        x["user_like"],
                    "naturalness":
                        x["naturalness"],
                    "direction":
                        x["direction"],
                    "user_confidence":
                        x.get(
                            "user_confidence"
                        ),
                    "label_source":
                        f"reuse:{p}",
                }

    # Resume our V2 labels last so they win.
    for x in read_jsonl(LABELS):
        q = norm(x.get("query"))

        if q and "error" not in x:
            existing[q] = x

    return existing, files


async def judge_one(
    client,
    sem,
    model,
    row,
):
    q = norm(
        row.get("query")
    )

    mrow = {
        "round":
            int(
                row.get("round")
                or 0
            ),
        "session_id":
            str(
                row.get("session_id")
                or row.get("example_id")
                or "v2"
            ),
        "candidate_index":
            int(
                row.get("candidate_index")
                or 0
            ),
        "query":
            q,
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
                    "role":
                        "system",
                    "content":
                        userjudge.SYSTEM,
                },
                {
                    "role":
                        "user",
                    "content":
                        "Classify this message:\n\n"
                        + q,
                },
            ],
            response_format=
                userjudge.SCHEMA,
        )

    uj = json.loads(
        resp.choices[0].message.content
    )

    return {
        "example_id":
            row.get("example_id"),
        "query":
            q,
        "source":
            row.get("source"),
        "meta_label":
            mj["label"],
        "meta_confidence":
            mj.get("confidence"),
        "user_like":
            uj["user_like"],
        "naturalness":
            uj["naturalness"],
        "direction":
            uj["direction"],
        "user_confidence":
            uj.get("confidence"),
        "label_source":
            "new_api_judgment",
    }


async def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default=
            "gpt-4o-mini-2024-07-18",
    )

    ap.add_argument(
        "--concurrency",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = ap.parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_jsonl(
        MANIFEST
    )

    if not rows:
        raise RuntimeError(
            f"No manifest rows: {MANIFEST}"
        )

    existing, files = (
        load_existing()
    )

    empty = []
    reusable = []
    pending = []

    for row in rows:
        q = norm(
            row.get("query")
        )

        if not q:
            empty.append(row)
        elif q in existing:
            reusable.append(row)
        else:
            pending.append(row)

    print("=" * 90)
    print("SOFT BENIGN CRITIC V2 LABEL AUDIT")
    print("=" * 90)

    print(
        "manifest rows       :",
        len(rows),
    )
    print(
        "existing judge files:",
        len(files),
    )
    print(
        "reusable labels     :",
        len(reusable),
    )
    print(
        "empty/parser-invalid:",
        len(empty),
    )
    print(
        "NEW API labels      :",
        len(pending),
    )

    print()
    print("Reused files:")

    for p in files:
        print(" ", p)

    if args.dry_run:
        return

    # Persist reused judgments into V2 dataset.
    already_v2 = {
        norm(x.get("query"))
        for x in read_jsonl(LABELS)
    }

    for row in reusable:
        q = norm(
            row.get("query")
        )

        if q not in already_v2:
            z = dict(
                existing[q]
            )

            z["example_id"] = (
                row.get("example_id")
            )

            z["source"] = (
                row.get("source")
            )

            append_jsonl(
                LABELS,
                z,
            )

            already_v2.add(q)

    # Empty output is not semantically judged.
    # It becomes an explicit zero-quality training case later.
    for row in empty:
        key = (
            f"__EMPTY__:"
            f"{row.get('example_id')}"
        )

        if key in already_v2:
            continue

        append_jsonl(
            LABELS,
            {
                "example_id":
                    row.get(
                        "example_id"
                    ),
                "query":
                    "",
                "source":
                    row.get("source"),
                "special_label":
                    "EMPTY_GENERATION",
                "label_source":
                    "deterministic_empty",
            },
        )

        already_v2.add(key)

    if not pending:
        print(
            "No new API judgments needed."
        )
        return

    client = AsyncOpenAI()

    sem = asyncio.Semaphore(
        args.concurrency
    )

    BATCH = 20

    failures = 0

    for start in range(
        0,
        len(pending),
        BATCH,
    ):
        batch = pending[
            start:start+BATCH
        ]

        results = await asyncio.gather(
            *[
                judge_one(
                    client,
                    sem,
                    args.model,
                    row,
                )
                for row in batch
            ],
            return_exceptions=True,
        )

        for row, z in zip(
            batch,
            results,
        ):
            if isinstance(
                z,
                Exception,
            ):
                failures += 1

                append_jsonl(
                    LABELS,
                    {
                        "example_id":
                            row.get(
                                "example_id"
                            ),
                        "query":
                            norm(
                                row.get(
                                    "query"
                                )
                            ),
                        "error":
                            (
                                f"{type(z).__name__}: "
                                f"{z}"
                            ),
                    },
                )
            else:
                append_jsonl(
                    LABELS,
                    z,
                )

        done = min(
            start + BATCH,
            len(pending),
        )

        print(
            f"[judge] "
            f"{done}/{len(pending)} "
            f"failures={failures}",
            flush=True,
        )

        if failures:
            raise RuntimeError(
                f"{failures} API "
                "judgment failures"
            )

    print()
    print(
        "V2 labeling complete."
    )


if __name__ == "__main__":
    asyncio.run(main())
