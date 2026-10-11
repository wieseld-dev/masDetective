import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("artifacts/adaptive_v11_k_bootstrap_v1")

manifest = json.loads((ROOT / "manifest_privileged.json").read_text())
truth = {
    x["session_id"]: x
    for x in manifest["systems"]
    if x["split"] == "train"
}

def read_jsonl(p):
    if not p.exists():
        return []
    return [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ]

for r in range(1, 7):
    print("\n" + "=" * 90)
    print(f"ROUND {r}")

    candidates = read_jsonl(ROOT / f"candidates_train_round_{r:02d}.jsonl")
    responses  = read_jsonl(ROOT / f"responses_train_round_{r:02d}.jsonl")
    scored     = read_jsonl(ROOT / f"scored_train_round_{r:02d}.jsonl")
    states     = read_jsonl(ROOT / f"states_train_round_{r:02d}.jsonl")

    cand_by = defaultdict(list)
    for x in candidates:
        cand_by[str(x.get("session_id", ""))].append(x)

    resp_by = defaultdict(list)
    for x in responses:
        resp_by[str(x.get("session_id", x.get("system_id", "")))].append(x)

    scored_sids = set()
    for x in scored:
        sid = str(x.get("system_id", x.get("session_id", "")))
        if sid:
            scored_sids.add(sid)

    state_by = {str(x["session_id"]): x for x in states}

    counts = Counter()
    cell = defaultdict(Counter)
    reasons = Counter()

    for sid, meta in truth.items():
        k = int(meta["k"])
        mode = str(meta["mode"])
        key = (mode, k)

        cs = cand_by.get(sid, [])
        rs = resp_by.get(sid, [])
        st = state_by.get(sid)

        valid_cand = any(bool(x.get("valid")) for x in cs)
        exec_ok = any(
            bool(x.get("valid")) and x.get("execution_status") == "ok"
            for x in rs
        )
        got_score = sid in scored_sids

        hist_len = len((st or {}).get("history", []))
        full_prefix = hist_len >= r

        counts["systems"] += 1
        cell[key]["systems"] += 1

        for name, flag in [
            ("valid_candidate", valid_cand),
            ("execution_ok", exec_ok),
            ("scored", got_score),
            ("full_prefix", full_prefix),
        ]:
            if flag:
                counts[name] += 1
                cell[key][name] += 1

        if not valid_cand:
            for c in cs:
                reason = str(
                    c.get("invalid_reason")
                    or c.get("reason")
                    or "unknown"
                )
                reasons[reason] += 1

    print(
        f"systems={counts['systems']} "
        f"valid_candidate={counts['valid_candidate']} "
        f"execution_ok={counts['execution_ok']} "
        f"scored={counts['scored']} "
        f"full_prefix={counts['full_prefix']}"
    )

    print("\nBY MODE x K:")
    for mode in ("single", "fanout2", "fanout3", "mixed"):
        for k in (3,4,5,6):
            z = cell[(mode,k)]
            n = z["systems"]
            print(
                f"{mode:8s} K={k}: "
                f"n={n:2d} "
                f"cand={z['valid_candidate']:2d}/{n} "
                f"exec={z['execution_ok']:2d}/{n} "
                f"score={z['scored']:2d}/{n} "
                f"prefix={z['full_prefix']:2d}/{n}"
            )

    print("\nTOP INVALID REASONS:")
    for reason, n in reasons.most_common(12):
        print(f"{n:4d}  {reason}")
