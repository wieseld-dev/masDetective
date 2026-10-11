from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Side-effect: register V11 roles/prompts.
from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)


ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
REPAIRS = Path("artifacts/v7_repair_pilot_v3/repairs.jsonl")


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p, rows):
    with Path(p).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def parse_stage_cycle(source_file):
    m = re.search(
        r"adaptive_v11_curriculum_v2/"
        r"(stage_\d+)/(cycle_\d+)/",
        str(source_file),
    )
    if not m:
        raise ValueError(source_file)

    stage = m.group(1)
    cycle = m.group(2)

    return stage, cycle


def load_system(src):
    stage, cycle = parse_stage_cycle(src["source_file"])

    manifest_path = (
        ROOT / stage /
        f"manifest_train_{cycle}.json"
    )

    mani = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    matches = [
        x for x in mani["systems"]
        if str(x["session_id"]) == str(src["session_id"])
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"{src['session_id']}: manifest matches={len(matches)} "
            f"in {manifest_path}"
        )

    return matches[0], str(manifest_path)


def eligible_pool():
    rows = [
        x for x in read_jsonl(REPAIRS)
        if x.get("eligible")
    ]

    # Remove mechanical phrases we already identified.
    bad_phrases = [
        "i'm working on working with",
        "i'm working on conducting",
        "i'm working on performing",
    ]

    rows = [
        x for x in rows
        if not any(
            z in x["repair"].lower()
            for z in bad_phrases
        )
    ]

    # Some DIRECTION cases were too generic.
    rows = [
        x for x in rows
        if x["repair"].lower().strip() not in {
            "i'm working on analysis. what information should i provide "
            "so you can help me determine what to focus on first?"
        }
    ]

    return rows


def select_sources():
    rows = eligible_pool()

    families = [
        "DIRECTION_BAD",
        "SYSTEM_SELF_BAD",
        "DANGLING_BAD",
    ]

    selected = []
    globally_used_sid = set()

    for fam in families:
        xs = [
            x for x in rows
            if x["benign_label"] == fam
        ]

        xs.sort(
            key=lambda x: (
                float(x["local_advantage"]),
                float(x["reward"]),
            ),
            reverse=True,
        )

        keep = []

        for x in xs:
            sid = str(x["session_id"])

            if sid in globally_used_sid:
                continue

            keep.append(x)
            globally_used_sid.add(sid)

            if len(keep) == 4:
                break

        if len(keep) != 4:
            raise RuntimeError(
                f"{fam}: only {len(keep)} usable unique systems"
            )

        selected.extend(keep)

    assert len(selected) == 12
    assert len({
        str(x["session_id"])
        for x in selected
    }) == 12

    return selected


def build_plan():
    sources = select_sources()

    plan = []

    for pair_id, src in enumerate(sources):
        sysrow, manifest_path = load_system(src)

        common = {
            "pair_id": pair_id,
            "family": src["benign_label"],
            "session_id": str(src["session_id"]),
            "round": int(src["round"]),
            "candidate_index": int(src["candidate_index"]),
            "source_file": str(src["source_file"]),
            "manifest_path": manifest_path,
            "mode": str(sysrow["mode"]),
            "style": str(sysrow["style"]),
            "truth_roles": list(map(str, sysrow["roles"])),
            "k": int(sysrow["k"]),
            "historical_reward": float(src["reward"]),
            "historical_local_advantage":
                float(src["local_advantage"]),
            "bad_query": str(src["query"]),
            "repair_query": str(src["repair"]),
        }

        # Interleaved repetitions conceptually:
        # bad0 repair0 bad1 repair1 bad2 repair2.
        for rep in range(3):
            plan.append({
                **common,
                "kind": "bad",
                "rep": rep,
                "query": str(src["query"]),
            })

            plan.append({
                **common,
                "kind": "repair",
                "rep": rep,
                "query": str(src["repair"]),
            })

    assert len(plan) == 72

    return sources, plan


def execute_one(x, model, retries):
    spec = SingleRouterSystemSpec(
        system_id=x["session_id"],
        split="adaptive_v11_train",
        execution_mode=x["mode"],
        style=x["style"],
        worker_roles=tuple(x["truth_roles"]),
    )

    last = None

    for attempt in range(retries + 1):
        try:
            target = SingleRouterFanoutMAS(
                spec,
                model=model,
            )

            answer = str(
                target.query(x["query"])
            )

            tr = getattr(
                target,
                "_last_trace",
                None,
            )

            selected = (
                list(map(
                    str,
                    getattr(tr, "selected_roles", ()),
                ))
                if tr is not None
                else []
            )

            return {
                **x,
                "final_output": answer,
                "selected_roles_eval_only": selected,
                "execution_status": "ok",
                "execution_error": "",
                "retries_used": attempt,
            }

        except Exception as e:
            last = e

            if attempt >= retries:
                break

            time.sleep(min(15.0, 2.0 ** attempt))

    return {
        **x,
        "final_output": "",
        "selected_roles_eval_only": [],
        "execution_status": "error",
        "execution_error": repr(last),
        "retries_used": retries,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="artifacts/v7_retention_pilot_v1",
    )
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

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources, plan = build_plan()

    write_jsonl(
        out / "selected_sources.jsonl",
        sources,
    )

    write_jsonl(
        out / "plan.jsonl",
        plan,
    )

    print("sources:", len(sources))
    print("systems:", len({
        x["session_id"] for x in sources
    }))
    print("executions:", len(plan))

    print("\n===== SOURCES =====")
    for x in sources:
        print("\n" + "=" * 100)
        print(
            x["benign_label"],
            x["session_id"],
            f"adv={x['local_advantage']:+.3f}",
            f"reward={x['reward']:+.3f}",
        )
        print("BAD   :", x["query"])
        print("REPAIR:", x["repair"])

    if args.plan_only:
        return

    results = []

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as ex:

        futs = [
            ex.submit(
                execute_one,
                x,
                args.model,
                args.retries,
            )
            for x in plan
        ]

        for n, f in enumerate(
            as_completed(futs),
            1,
        ):
            z = f.result()
            results.append(z)

            print(
                f"[{n:02d}/72] "
                f"pair={z['pair_id']:02d} "
                f"{z['kind']:<6} "
                f"rep={z['rep']} "
                f"status={z['execution_status']} "
                f"selected={z['selected_roles_eval_only']}",
                flush=True,
            )

    results.sort(
        key=lambda x: (
            x["pair_id"],
            x["rep"],
            0 if x["kind"] == "bad" else 1,
        )
    )

    write_jsonl(
        out / "executions.jsonl",
        results,
    )

    ok = sum(
        x["execution_status"] == "ok"
        for x in results
    )

    summary = {
        "sources": len(sources),
        "planned_executions": len(plan),
        "successful": ok,
        "failed": len(results) - ok,
        "model": args.model,
    }

    (out / "execution_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
