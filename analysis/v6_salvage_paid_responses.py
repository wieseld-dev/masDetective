import json
import hashlib
from collections import defaultdict
from pathlib import Path

GOLD = Path("artifacts/v4_userlike_audit/gold_strict.jsonl")
ROOT = Path("artifacts/v6_pref_api_bulk_6120")

RAW_CALLS = ROOT / "raw_calls.jsonl"
API_RAW = ROOT / "all_api_responses.jsonl"

KEEP = [
    2, 6, 8, 9, 10, 16, 23, 24, 26,
    27, 32, 33, 35, 37, 39, 43, 44,
]

FAMILIES = ["DIRECTION", "META", "DANGLING"]
BATCHES = 6
NEED = 20


def read_jsonl(p):
    if not p.exists():
        return []
    return [
        json.loads(x)
        for x in p.open(encoding="utf-8")
        if x.strip()
    ]


def append_jsonl(p, x):
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
        f.flush()


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def norm(s):
    return " ".join(str(s).lower().split())


gold_rows = read_jsonl(GOLD)
gold = {int(x["gold_index"]): x for x in gold_rows}

existing_rows = read_jsonl(RAW_CALLS)

existing = {
    str(x["request_key"]): x
    for x in existing_rows
    if x.get("status") == "ok"
}

raw_by_key = defaultdict(list)

for x in read_jsonl(API_RAW):
    raw_by_key[str(x["request_key"])].append(x)

salvaged = 0

for gi in KEEP:
    g = gold[gi]

    if str(g["completion"]) != str(g["query"]):
        raise RuntimeError(f"G{gi}: completion != query")

    for family in FAMILIES:
        accepted = set()

        for b in range(BATCHES):
            key = f"G{gi:02d}:{family}:B{b:02d}"

            # Existing successful batch: preserve exactly.
            if key in existing:
                for q in existing[key]["generated"]:
                    accepted.add(norm(q))
                continue

            pool = []
            pool_seen = set()

            for rr in raw_by_key.get(key, []):
                try:
                    obj = json.loads(rr["raw_content"])
                except Exception:
                    continue

                xs = obj.get("candidates")
                if not isinstance(xs, list):
                    continue

                for item in xs:
                    if not isinstance(item, dict):
                        continue

                    q = str(item.get("text") or "").strip()

                    if (
                        not q
                        or "\n" in q
                        or not q.endswith("?")
                        or q.count("?") != 1
                    ):
                        continue

                    nq = norm(q)

                    if nq == norm(g["completion"]):
                        continue

                    if nq in accepted or nq in pool_seen:
                        continue

                    variant = str(
                        item.get("variant") or "UNKNOWN"
                    ).upper()

                    pool.append({
                        "text": q,
                        "variant": variant,
                    })

                    pool_seen.add(nq)

            if len(pool) < NEED:
                print(
                    f"[not enough] {key}: "
                    f"{len(pool)} unique usable"
                )
                continue

            # Prefer the intended 12/8 balance, but it is not mandatory.
            mins = [
                x for x in pool
                if x["variant"] == "MINIMAL"
            ]
            divs = [
                x for x in pool
                if x["variant"] == "DIVERSE"
            ]

            selected = mins[:12] + divs[:8]

            selected_norm = {
                norm(x["text"])
                for x in selected
            }

            if len(selected) < NEED:
                for x in pool:
                    if norm(x["text"]) in selected_norm:
                        continue

                    selected.append(x)
                    selected_norm.add(norm(x["text"]))

                    if len(selected) == NEED:
                        break

            assert len(selected) == NEED

            row = {
                "status": "ok",
                "request_key": key,
                "gold_index": gi,
                "family": family,
                "batch_index": b,
                "model": "gpt-4o-mini-2024-07-18",
                "prompt_sha256": sha(str(g["prompt"])),
                "chosen_sha256": sha(str(g["completion"])),
                "generated": [
                    x["text"] for x in selected
                ],
                "variants": [
                    x["variant"] for x in selected
                ],
                "recovered_from_paid_raw": True,
                "raw_responses_used": len(
                    raw_by_key.get(key, [])
                ),
            }

            append_jsonl(RAW_CALLS, row)
            existing[key] = row

            for x in selected:
                accepted.add(norm(x["text"]))

            salvaged += 1

            print(
                f"[SALVAGED] {key}: "
                f"pool={len(pool)} -> 20"
            )

print()
print("existing successful before/after :", len(existing_rows), "->", len(existing))
print("newly salvaged batches           :", salvaged)
print("remaining requests               :", 306 - len(existing))
