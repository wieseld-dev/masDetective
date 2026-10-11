from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Fresh processes must register V11 EXTRA_ROLE_PROMPTS before constructing targets.
from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()
from targets.single_router_fanout_mas import SingleRouterFanoutMAS, SingleRouterSystemSpec

from adaptive_v11_active.io import read_jsonl, write_jsonl


def cache_key(session_id: str, query: str) -> str:
    return hashlib.sha256(f"{session_id}\0{query}".encode()).hexdigest()


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {str(r["key"]): r for r in read_jsonl(path)}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def _execute_one(m: dict, sid: str, q: str, model: str | None, retries: int):
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
                raise RuntimeError("SingleRouterFanoutMAS did not expose _last_trace after query")
            selected = list(map(str, getattr(tr, "selected_roles", ())))
            return output, selected, attempt
        except Exception as e:
            last = e
            if attempt >= retries:
                raise
            time.sleep(min(20.0, 2.0 ** attempt))
    raise last  # pragma: no cover


def main():
    ap = argparse.ArgumentParser(description="Execute Adaptive-V11 candidates on fresh black-box MAS systems")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--root", default="artifacts/adaptive_v11_active_v1")
    ap.add_argument("--manifest", default="artifacts/adaptive_v11_active_v1/manifest_privileged.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=4, help="Concurrent API requests; keep modest for rate limits")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    root = Path(args.root)
    cand_path = root / f"candidates_{args.split}_round_{args.round:02d}.jsonl"
    candidates = read_jsonl(cand_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    by_id = {r["session_id"]: r for r in manifest["systems"] if r["split"] == args.split}
    if not candidates:
        raise SystemExit(f"no candidates in {cand_path}")

    public_cache_path = root / "target_public_cache.jsonl"
    truth_cache_path = root / "target_truth_cache_training_only.jsonl"
    public_cache = load_cache(public_cache_path)
    truth_cache = load_cache(truth_cache_path)
    rows_by_pos: dict[int, dict] = {}
    pending = {}

    def make_row(c, m, output, selected, *, hit, status, retries_used=0):
        sid = str(c["session_id"])
        return {
            **c,
            "system_id": sid,
            "final_output": str(output),
            "output": str(output),
            "answer": str(output),
            "selected_roles_training_only": list(map(str, selected)),
            "truth_roles_training_only": list(map(str, m["roles"])),
            "manifest_sha256": manifest["sha256"],
            "mode": m["mode"],
            "execution_status": status,
            "cache_hit": bool(hit),
            "retries_used": int(retries_used),
            "action": {"target": f"adaptive_r{args.round:02d}", "competitor": None, "variant": int(c["candidate_index"])},
        }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for pos, c in enumerate(candidates):
            sid = str(c["session_id"])
            if sid not in by_id:
                raise RuntimeError(f"candidate references unknown session {sid}")
            m = dict(by_id[sid]); m["split"] = args.split
            q = str(c.get("query", ""))
            if not c.get("valid"):
                rows_by_pos[pos] = make_row(c, m, "", [], hit=False, status="invalid_generation")
                continue
            key = cache_key(sid, q)
            if key in public_cache and key in truth_cache:
                rows_by_pos[pos] = make_row(
                    c, m, public_cache[key]["output"], truth_cache[key].get("selected_roles", []),
                    hit=True, status="ok",
                )
                continue
            fut = ex.submit(_execute_one, m, sid, q, args.model, args.retries)
            pending[fut] = (pos, c, m, key)

        done = len(rows_by_pos)
        for fut in as_completed(pending):
            pos, c, m, key = pending[fut]
            sid = str(c["session_id"]); q = str(c["query"])
            output, selected, retries_used = fut.result()
            pub = {"key": key, "session_id": sid, "query": q, "output": output}
            prv = {
                "key": key, "session_id": sid,
                "query_hash": hashlib.sha256(q.encode()).hexdigest(),
                "selected_roles": selected, "true_roles": list(map(str, m["roles"])),
            }
            append_jsonl(public_cache_path, pub); append_jsonl(truth_cache_path, prv)
            public_cache[key] = pub; truth_cache[key] = prv
            rows_by_pos[pos] = make_row(c, m, output, selected, hit=False, status="ok", retries_used=retries_used)
            done += 1
            print(f"[execute] {done}/{len(candidates)} split={args.split} r={args.round} sid={sid} cand={c['candidate_index']} retries={retries_used}", flush=True)

    rows = [rows_by_pos[i] for i in range(len(candidates))]
    out = root / f"responses_{args.split}_round_{args.round:02d}.jsonl"
    write_jsonl(out, rows)
    hits = sum(bool(r.get("cache_hit")) for r in rows)
    print(f"[done] responses={out} rows={len(rows)} cache_hits={hits} workers={args.workers}", flush=True)

if __name__ == "__main__": main()
