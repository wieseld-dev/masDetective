#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()
from targets.single_router_fanout_mas import SingleRouterFanoutMAS, SingleRouterSystemSpec


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path, rows):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    tmp.replace(p)


def key(row):
    return (
        str(row["session_id"]),
        int(row["round"]),
        int(row["candidate_index"]),
        str(row.get("query", "")),
    )


def execute_one(m, sid, q, model, retries):
    spec = SingleRouterSystemSpec(
        system_id=sid,
        split=f"adaptive_v11_{m['split']}",
        execution_mode=str(m["mode"]),
        style=str(m["style"]),
        worker_roles=tuple(map(str, m["roles"])),
    )
    last = None
    for attempt in range(retries + 1):
        try:
            target = SingleRouterFanoutMAS(spec, model=model)
            output = str(target.query(q))
            tr = getattr(target, "_last_trace", None)
            if tr is None:
                raise RuntimeError("SingleRouterFanoutMAS did not expose _last_trace")
            selected = list(map(str, getattr(tr, "selected_roles", ())))
            return output, selected, attempt
        except Exception as e:
            last = e
            if attempt >= retries:
                raise
            time.sleep(min(20.0, 2.0 ** attempt))
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    root = Path(args.root)
    candidates = read_jsonl(root / f"candidates_{args.split}_round_{args.round:02d}.jsonl")
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_id = {
        x["session_id"]: x
        for x in manifest["systems"]
        if x["split"] == args.split
    }

    out = root / f"responses_{args.split}_round_{args.round:02d}.jsonl"
    existing = {key(x): x for x in read_jsonl(out)}
    rows = dict(existing)

    pending = []
    for c in candidates:
        k = key(c)
        if k in rows:
            continue
        if not c.get("valid"):
            rows[k] = {
                **c,
                "system_id": c["session_id"],
                "final_output": "",
                "output": "",
                "answer": "",
                "selected_roles_training_only": [],
                "truth_roles_training_only": by_id[c["session_id"]]["roles"],
                "execution_status": "invalid",
                "retries_used": 0,
            }
            continue
        pending.append(c)

    print(f"[execute] existing={len(existing)} pending={len(pending)}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {}
        for c in pending:
            m = by_id[c["session_id"]]
            fut = ex.submit(
                execute_one, m, c["session_id"], c["query"], args.model, args.retries
            )
            futs[fut] = (c, m)

        done = 0
        for fut in as_completed(futs):
            c, m = futs[fut]
            try:
                output, selected, retries_used = fut.result()
                status = "ok"
                error = ""
            except Exception as e:
                output, selected, retries_used = "", [], args.retries
                status = "error"
                error = repr(e)

            rec = {
                **c,
                "system_id": c["session_id"],
                "final_output": output,
                "output": output,
                "answer": output,
                "selected_roles_training_only": selected,
                "truth_roles_training_only": list(map(str, m["roles"])),
                "mode": m["mode"],
                "execution_status": status,
                "retries_used": int(retries_used),
                "execution_error": error,
            }
            rows[key(c)] = rec
            done += 1
            # Crash-safe checkpointing.
            ordered = sorted(
                rows.values(),
                key=lambda x: (
                    str(x["session_id"]),
                    int(x["round"]),
                    int(x["candidate_index"]),
                ),
            )
            write_jsonl(out, ordered)
            print(
                f"[execute] {done}/{len(pending)} sid={c['session_id']} "
                f"cand={c['candidate_index']} status={status}",
                flush=True,
            )

    ordered = sorted(
        rows.values(),
        key=lambda x: (
            str(x["session_id"]),
            int(x["round"]),
            int(x["candidate_index"]),
        ),
    )
    write_jsonl(out, ordered)
    print(f"[done] {out} rows={len(ordered)}")


if __name__ == "__main__":
    main()
