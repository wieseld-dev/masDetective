from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import v7_retention_execute_v1 as old


SRC = Path(
    "artifacts/v7_substantive_repair_v4/sources.jsonl"
)

OUT = Path(
    "artifacts/v7_substantive_repair_v4"
)


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def write_jsonl(p, rows):
    with Path(p).open(
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


def build_plan():
    sources = read_jsonl(SRC)

    assert len(sources) == 12

    plan = []

    for src in sources:
        pair_id = int(src["pair_id"])

        sysrow, manifest_path = (
            old.load_system(src)
        )

        common = {
            "pair_id": pair_id,
            "family": src["benign_label"],
            "session_id": str(src["session_id"]),
            "round": int(src["round"]),
            "candidate_index":
                int(src["candidate_index"]),
            "source_file":
                str(src["source_file"]),
            "manifest_path":
                manifest_path,
            "mode":
                str(sysrow["mode"]),
            "style":
                str(sysrow["style"]),
            "truth_roles":
                list(map(str, sysrow["roles"])),
            "k":
                int(sysrow["k"]),
            "historical_reward":
                float(src["reward"]),
            "historical_local_advantage":
                float(src["local_advantage"]),
            "bad_query":
                str(src["query"]),
            "v3_repair":
                str(src["v3_repair"]),
            "v4_repair":
                str(src["repair"]),
        }

        for rep in range(3):
            plan.append({
                **common,
                "kind": "repair_v4",
                "rep": rep,
                "query": str(src["repair"]),
            })

    assert len(plan) == 36

    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="gpt-4o-mini",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--plan-only",
        action="store_true",
    )
    args = ap.parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    plan = build_plan()

    write_jsonl(
        OUT / "execution_plan.jsonl",
        plan,
    )

    print("systems:", len(set(
        x["session_id"]
        for x in plan
    )))
    print("executions:", len(plan))

    for pair_id in range(12):
        x = next(
            z for z in plan
            if z["pair_id"] == pair_id
        )

        print("\n" + "=" * 100)
        print(
            f"PAIR {pair_id:02d}",
            x["family"],
            x["session_id"],
        )
        print("BAD:", x["bad_query"])
        print("V4 :", x["v4_repair"])
        print("truth:", x["truth_roles"])

    if args.plan_only:
        return

    results = []

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as ex:

        futs = [
            ex.submit(
                old.execute_one,
                x,
                args.model,
                args.retries,
            )
            for x in plan
        ]

        for n, fut in enumerate(
            as_completed(futs),
            1,
        ):
            z = fut.result()
            results.append(z)

            print(
                f"[{n:02d}/36] "
                f"pair={z['pair_id']:02d} "
                f"rep={z['rep']} "
                f"status={z['execution_status']} "
                f"selected="
                f"{z['selected_roles_eval_only']}",
                flush=True,
            )

    results.sort(
        key=lambda x: (
            int(x["pair_id"]),
            int(x["rep"]),
        )
    )

    write_jsonl(
        OUT / "executions.jsonl",
        results,
    )

    ok = sum(
        x["execution_status"] == "ok"
        for x in results
    )

    summary = {
        "systems": 12,
        "planned": 36,
        "successful": ok,
        "failed": 36 - ok,
        "model": args.model,
        "note":
            "V4 repair-only; BAD baseline reused "
            "from v7_retention_pilot_v1",
    }

    (
        OUT / "execution_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
